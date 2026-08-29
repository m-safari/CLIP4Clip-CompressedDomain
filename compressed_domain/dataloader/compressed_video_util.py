"""
Compressed-domain counterpart to RawVideoExtractorCV2 / MSRVTT_DataLoader.

Design goal: for every global frame index that the raw-RGB pipeline samples,
also fetch the aligned I-frame / motion-vector / residual triplet from the
compressed bitstream via coviar.load, so that "raw frame i" and
"iframe i / mv i / residual i" refer to the exact same instant in the video.

Assumes videos were encoded with a fixed, known GOP size (e.g. via
`ffmpeg -g 12 -keyint_min 12 -sc_threshold 0 ...`) matching self.gop_size.
coviar's frame_index argument is 0-indexed *within* a GOP, so a global frame
index is converted as: gop_index, frame_in_gop = divmod(global_idx, gop_size).
"""

import os
import numpy as np
import torch as th
from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, Normalize
import cv2

from coviar import load as coviar_load
from coviar import get_num_frames as coviar_get_num_frames

import sys
from pathlib import Path
project_root = Path.cwd().parents[1]
sys.path.insert(0, str(project_root))
from dataloaders.rawvideo_util import RawVideoExtractorCV2


class CoviarVideoExtractorCV2(RawVideoExtractorCV2):
    """
    Extends RawVideoExtractorCV2 with aligned compressed-domain retrieval.

    get_video_data() now returns a dict with five entries instead of one:
        {
          'video':    (T, 1, 3, H, W)  raw RGB, CLIP-preprocessed (unchanged)
          'iframe':   (T, 1, 3, H, W)  I-frame of each sampled frame's GOP,
                                       CLIP-preprocessed identically to 'video'
          'mv':       (T, 1, 2, H, W)  motion vectors, zero-centered
          'residual': (T, 1, 3, H, W)  prediction residual, zero-centered
          'valid':    (T,) bool        whether compressed retrieval succeeded
                                       for that frame (raw frame is kept
                                       either way; compressed falls back to
                                       zeros on failure)
        }
    """

    # coviar representation_type values
    REPR_IFRAME = 0
    REPR_MV = 1
    REPR_RESIDUAL = 2

    # With accumulate=True, coviar sums motion/residual all the way back to
    # each GOP's I-frame -- so magnitude grows with frame_in_gop (frames late
    # in a GOP accumulate more than frames right after the I-frame). Observed
    # on real data (int32, GOP=12): motion vectors up to ~[-253, 255],
    # residuals up to ~[-64, 78] but not a hard ceiling. A fixed linear clip
    # either saturates late-GOP frames or wastes resolution on early-GOP
    # ones, so we log-compress instead of hard-clipping by default -- it
    # preserves small-value resolution while still bounding outliers.
    # These *_SCALE constants set where the compression "knee" sits, not a
    # hard ceiling; raise them if late-GOP frames still look saturated.
    MV_SCALE = 255.0
    RESIDUAL_SCALE = 128.0
    USE_LOG_COMPRESSION = True

    def __init__(self, centercrop=False, size=224, framerate=-1,
                 gop_size=12, accumulate=True):
        super().__init__(centercrop=centercrop, size=size, framerate=framerate)
        self.gop_size = gop_size
        self.accumulate = accumulate

    # ---- numpy-domain resize/crop for non-RGB compressed signals ----------
    # torchvision's Resize/CenterCrop expect PIL images, which can't hold a
    # signed 2-channel motion-vector array. Do the equivalent transform by
    # hand with cv2, matching the parent's "shorter side -> n_px, then
    # center crop" behavior.

    def _resize_crop_np(self, arr, n_px):
        h, w = arr.shape[:2]
        if h <= w:
            new_h, new_w = n_px, max(n_px, int(round(w * n_px / h)))
        else:
            new_h, new_w = max(n_px, int(round(h * n_px / w))), n_px
        resized = cv2.resize(arr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        top = (new_h - n_px) // 2
        left = (new_w - n_px) // 2
        return resized[top:top + n_px, left:left + n_px, ...]

    def _iframe_to_tensor(self, arr):
        # coviar's I-frame comes out BGR (same convention as cv2.VideoCapture),
        # so treat it exactly like a raw decoded frame and reuse the CLIP
        # preprocessing pipeline for full consistency with 'video'.
        rgb = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
        return self.transform(Image.fromarray(rgb).convert("RGB"))

    def _normalize_signed(self, arr, scale):
        if self.USE_LOG_COMPRESSION:
            # sign(x) * log1p(|x|) / log1p(scale) -- maps 0 -> 0, keeps small
            # values roughly linear, and compresses large ones into (-1, 1)
            # without a hard saturation ceiling like a linear clip has.
            return np.sign(arr) * np.log1p(np.abs(arr)) / np.log1p(scale)
        return np.clip(arr, -scale, scale) / scale

    def _mv_to_tensor(self, arr, n_px):
        arr = self._resize_crop_np(arr, n_px).astype(np.float32)
        arr = self._normalize_signed(arr, self.MV_SCALE)
        # (H, W, 2) -> (2, H, W)
        return th.from_numpy(arr).permute(2, 0, 1).contiguous()

    def _residual_to_tensor(self, arr, n_px):
        arr = self._resize_crop_np(arr, n_px).astype(np.float32)
        arr = self._normalize_signed(arr, self.RESIDUAL_SCALE)
        return th.from_numpy(arr).permute(2, 0, 1).contiguous()

    # ---- shared frame-index computation (mirrors parent's inline logic) ---

    def _compute_sample_indices(self, cap, fps, frameCount, sample_fp, start_time, end_time):
        total_duration = (frameCount + fps - 1) // fps
        start_sec, end_sec = 0, total_duration
        if start_time is not None:
            start_sec = start_time
            end_sec = end_time if end_time <= total_duration else total_duration

        interval = 1
        if sample_fp > 0:
            interval = fps // sample_fp
        else:
            sample_fp = fps
        if interval == 0:
            interval = 1

        inds = [ind for ind in np.arange(0, fps, interval)]
        assert len(inds) >= sample_fp
        inds = inds[:sample_fp]

        global_indices = []
        for sec in np.arange(start_sec, end_sec + 1):
            sec_base = int(sec * fps)
            for ind in inds:
                global_indices.append(sec_base + ind)
        return global_indices

    # ---- overridden combined extraction ------------------------------------

    def video_to_tensor(self, video_file, preprocess, sample_fp=0, start_time=None, end_time=None):
        if start_time is not None or end_time is not None:
            assert isinstance(start_time, int) and isinstance(end_time, int) \
                   and start_time > -1 and end_time > start_time
        assert sample_fp > -1

        cap = cv2.VideoCapture(video_file)
        frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        if start_time is not None:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(start_time * fps))

        global_indices = self._compute_sample_indices(
            cap, fps, frameCount, sample_fp, start_time, end_time
        )

        try:
            n_compressed_frames = coviar_get_num_frames(video_file)
        except Exception as e:
            print("coviar get_num_frames error: {}. video: {}".format(e, video_file))
            n_compressed_frames = 0

        raw_images, iframe_images, mv_images, residual_images, valid_flags = [], [], [], [], []

        ret = True
        for global_idx in global_indices:
            if not ret:
                break
            cap.set(cv2.CAP_PROP_POS_FRAMES, global_idx)
            ret, frame = cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            raw_images.append(preprocess(Image.fromarray(frame_rgb).convert("RGB")))

            # Aligned compressed-domain triplet for this exact frame index.
            if global_idx < n_compressed_frames:
                gop_index, frame_in_gop = divmod(global_idx, self.gop_size)
                try:
                    iframe_arr = coviar_load(
                        video_file, gop_index, frame_in_gop, self.REPR_IFRAME, self.accumulate)
                    mv_arr = coviar_load(
                        video_file, gop_index, frame_in_gop, self.REPR_MV, self.accumulate)
                    residual_arr = coviar_load(
                        video_file, gop_index, frame_in_gop, self.REPR_RESIDUAL, self.accumulate)

                    iframe_images.append(self._iframe_to_tensor(iframe_arr))
                    mv_images.append(self._mv_to_tensor(mv_arr, self.size))
                    residual_images.append(self._residual_to_tensor(residual_arr, self.size))
                    valid_flags.append(True)
                    continue
                except Exception as e:
                    print("coviar load error: {} at gop={}, frame={}. video: {}".format(
                        e, gop_index, frame_in_gop, video_file))

            # Fallback: keep the raw frame, zero-fill compressed components,
            # so all tensors stay positionally aligned even on failure.
            iframe_images.append(th.zeros(3, self.size, self.size))
            mv_images.append(th.zeros(2, self.size, self.size))
            residual_images.append(th.zeros(3, self.size, self.size))
            valid_flags.append(False)

        cap.release()

        if len(raw_images) > 0:
            video_data = th.tensor(np.stack(raw_images))
            iframe_data = th.stack(iframe_images)
            mv_data = th.stack(mv_images)
            residual_data = th.stack(residual_images)
            valid = th.tensor(valid_flags, dtype=th.bool)
        else:
            video_data = th.zeros(1)
            iframe_data = th.zeros(1)
            mv_data = th.zeros(1)
            residual_data = th.zeros(1)
            valid = th.zeros(1, dtype=th.bool)

        return {
            'video': video_data,
            'iframe': iframe_data,
            'mv': mv_data,
            'residual': residual_data,
            'valid': valid,
        }

    def get_video_data(self, video_path, start_time=None, end_time=None):
        return self.video_to_tensor(
            video_path, self.transform, sample_fp=self.framerate,
            start_time=start_time, end_time=end_time,
        )


# Mirrors the bottom-of-file convention in rawvideo_util.py
CoviarVideoExtractor = CoviarVideoExtractorCV2
