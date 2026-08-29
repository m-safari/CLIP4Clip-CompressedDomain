import os
import json
import random
from collections import defaultdict

import coviar
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class MSRVTT_CompressedDataLoader(Dataset):
    """
    MSR-VTT compressed-domain dataloader.

    Each sampled temporal position contains:

        I-frame:
            actual I-frame only if this frame is an I-frame,
            otherwise zeros.

        Motion:
            actual accumulated motion vector for P-frames,
            otherwise zeros.

        Residual:
            actual accumulated residual for P-frames,
            otherwise zeros.

    Temporal sampling follows the same logic as CLIP4Clip's
    RawVideoExtractorCV2.

    IMPORTANT:
        gop_size must match the GOP structure of the encoded videos.
        This starter implementation assumes a fixed GOP size.
    """

    def __init__(
        self,
        csv_path,
        json_path,
        features_path,
        tokenizer,
        max_words=30,
        feature_framerate=1.0,
        max_frames=100,
        unfold_sentences=False,
        frame_order=0,
        slice_framepos=0,
        gop_size=50,
    ):
        self.csv = pd.read_csv(csv_path)
        self.data = json.load(open(json_path, "r"))

        self.features_path = features_path
        self.feature_framerate = feature_framerate
        self.max_words = max_words
        self.max_frames = max_frames
        self.tokenizer = tokenizer

        self.frame_order = frame_order
        assert self.frame_order in [0, 1, 2]

        self.slice_framepos = slice_framepos
        assert self.slice_framepos in [0, 1, 2]

        self.gop_size = gop_size

        self.unfold_sentences = unfold_sentences

        # ---------------------------------------------------------
        # Same sentence handling as original CLIP4Clip loader
        # ---------------------------------------------------------

        self.sample_len = 0

        if self.unfold_sentences:

            train_video_ids = list(self.csv["video_id"].values)

            self.sentences_dict = {}

            for itm in self.data["sentences"]:
                if itm["video_id"] in train_video_ids:
                    self.sentences_dict[len(self.sentences_dict)] = (
                        itm["video_id"],
                        itm["caption"],
                    )

            self.sample_len = len(self.sentences_dict)

        else:

            self.sentences = defaultdict(list)

            for itm in self.data["sentences"]:
                self.sentences[itm["video_id"]].append(
                    itm["caption"]
                )

            # Same bookkeeping as original loader.
            self.parent_ids = {}
            self.children_video_ids = defaultdict(list)

            for itm in self.data["videos"]:
                vid = itm["video_id"]

                url_suffix = itm["url"].split("?v=")[-1]

                self.parent_ids[vid] = url_suffix
                self.children_video_ids[url_suffix].append(vid)

            self.sample_len = len(self.csv)

        self.SPECIAL_TOKEN = {
            "CLS_TOKEN": "<|startoftext|>",
            "SEP_TOKEN": "<|endoftext|>",
            "MASK_TOKEN": "[MASK]",
            "UNK_TOKEN": "[UNK]",
            "PAD_TOKEN": "[PAD]",
        }

    # =============================================================
    # Dataset
    # =============================================================

    def __len__(self):
        return self.sample_len

    # =============================================================
    # Text
    # =============================================================

    def _get_text(self, video_id, caption=None):

        choice_video_ids = [video_id]

        pairs_text = np.zeros(
            (1, self.max_words),
            dtype=np.int64,
        )

        pairs_mask = np.zeros(
            (1, self.max_words),
            dtype=np.int64,
        )

        pairs_segment = np.zeros(
            (1, self.max_words),
            dtype=np.int64,
        )

        if caption is not None:
            words = self.tokenizer.tokenize(caption)
        else:
            words = self._get_single_text(video_id)

        words = [
            self.SPECIAL_TOKEN["CLS_TOKEN"]
        ] + words

        total_length_with_CLS = self.max_words - 1

        if len(words) > total_length_with_CLS:
            words = words[:total_length_with_CLS]

        words = words + [
            self.SPECIAL_TOKEN["SEP_TOKEN"]
        ]

        input_ids = self.tokenizer.convert_tokens_to_ids(words)

        input_mask = [1] * len(input_ids)
        segment_ids = [0] * len(input_ids)

        while len(input_ids) < self.max_words:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)

        pairs_text[0] = np.asarray(input_ids)
        pairs_mask[0] = np.asarray(input_mask)
        pairs_segment[0] = np.asarray(segment_ids)

        return (
            pairs_text,
            pairs_mask,
            pairs_segment,
            choice_video_ids,
        )

    def _get_single_text(self, video_id):

        caption = random.choice(
            self.sentences[video_id]
        )

        return self.tokenizer.tokenize(caption)

    # =============================================================
    # Exact temporal sampling from RawVideoExtractorCV2
    # =============================================================

    def _get_sample_indices(self, video_path):

        """
        Reproduce RawVideoExtractorCV2.video_to_tensor() sampling.

        For example, fps=30 and feature_framerate=1:

            0, 30, 60, 90, ...

        For fps=30 and feature_framerate=4:

            0, 7, 14, 21, ...

        This deliberately mirrors the original implementation.
        """

        # We use OpenCV ONLY to obtain FPS and frame count.
        # No RGB frames are decoded.
        import cv2

        cap = cv2.VideoCapture(video_path)

        frame_count = int(
            cap.get(cv2.CAP_PROP_FRAME_COUNT)
        )

        fps = int(
            cap.get(cv2.CAP_PROP_FPS)
        )

        cap.release()

        if fps <= 0:
            raise RuntimeError(
                f"Could not determine FPS for {video_path}"
            )

        if frame_count <= 0:
            raise RuntimeError(
                f"Could not determine frame count for {video_path}"
            )

        # ---------------------------------------------------------
        # Exactly the same logic as RawVideoExtractorCV2
        # ---------------------------------------------------------

        if self.feature_framerate > 0:

            interval = fps // int(
                self.feature_framerate
            )

        else:

            interval = 1

        if interval == 0:
            interval = 1

        sample_fp = (
            int(self.feature_framerate)
            if self.feature_framerate > 0
            else fps
        )

        # Frames selected within each one-second interval.
        inds = [
            ind
            for ind in np.arange(0, fps, interval)
        ]

        assert len(inds) >= sample_fp

        inds = inds[:sample_fp]

        # ---------------------------------------------------------
        # Original extractor loops over seconds:
        #
        # for sec in np.arange(start_sec, end_sec + 1)
        #
        # and produces sec * fps + ind.
        #
        # We reproduce that, but discard indices beyond the
        # actual video.
        # ---------------------------------------------------------

        total_duration = (
            frame_count + fps - 1
        ) // fps

        indices = []

        for sec in range(total_duration + 1):

            sec_base = sec * fps

            for ind in inds:

                frame_index = sec_base + ind

                if frame_index >= frame_count:
                    continue

                indices.append(frame_index)

        return np.asarray(
            indices,
            dtype=np.int64,
        )

    # =============================================================
    # Frame slicing
    # =============================================================

    def _process_frame_indices(self, indices):

        """
        Apply CLIP4Clip's max_frames/slice_framepos/frame_order
        logic after temporal sampling.
        """

        indices = np.asarray(
            indices,
            dtype=np.int64,
        )

        # ---------------------------------------------------------
        # max_frames
        # ---------------------------------------------------------

        if len(indices) > self.max_frames:

            if self.slice_framepos == 0:

                indices = indices[
                    :self.max_frames
                ]

            elif self.slice_framepos == 1:

                indices = indices[
                    -self.max_frames:
                ]

            else:

                sample_indx = np.linspace(
                    0,
                    len(indices) - 1,
                    num=self.max_frames,
                    dtype=int,
                )

                indices = indices[sample_indx]

        # ---------------------------------------------------------
        # frame order
        # ---------------------------------------------------------

        if self.frame_order == 1:

            indices = indices[::-1]

        elif self.frame_order == 2:

            indices = indices.copy()

            np.random.shuffle(indices)

        return indices

    # =============================================================
    # COVIAR frame indexing
    # =============================================================

    def _frame_to_gop(self, frame_index):

        """
        Convert absolute frame index into:

            GOP index
            frame index inside GOP

        Assumes fixed GOP size.
        """

        gop_index = frame_index // self.gop_size
        local_index = frame_index % self.gop_size

        return gop_index, local_index

    # =============================================================
    # Load one representation
    # =============================================================

    def _load_frame_components(
        self,
        video_path,
        frame_index,
    ):

        gop_index, local_index = self._frame_to_gop(
            int(frame_index)
        )

        # ---------------------------------------------------------
        # I-frame
        # ---------------------------------------------------------

        if local_index == 0:

            iframe = coviar.load(
                video_path,
                gop_index,
                local_index,
                0,
                True,
            )

            # No motion/residual at I-frame.
            motion = None
            residual = None

        # ---------------------------------------------------------
        # P-frame
        # ---------------------------------------------------------

        else:

            iframe = None

            motion = coviar.load(
                video_path,
                gop_index,
                local_index,
                1,
                True,
            )

            residual = coviar.load(
                video_path,
                gop_index,
                local_index,
                2,
                True,
            )

        return iframe, motion, residual

    # =============================================================
    # Compressed video
    # =============================================================

    def _get_compressed_video(
        self,
        choice_video_ids,
    ):

        all_iframes = []
        all_motion = []
        all_residual = []
        all_masks = []

        for video_id in choice_video_ids:

            video_path = os.path.join(
                self.features_path,
                f"{video_id}.mp4",
            )

            if not os.path.exists(video_path):

                video_path = video_path.replace(
                    ".mp4",
                    ".webm",
                )

            if not os.path.exists(video_path):
                raise FileNotFoundError(
                    video_path
                )

            # -----------------------------------------------------
            # COVIAR knows the number of frames.
            # -----------------------------------------------------

            num_frames = coviar.get_num_frames(
                video_path
            )

            # -----------------------------------------------------
            # Reproduce CLIP4Clip temporal sampling.
            # -----------------------------------------------------

            sampled_indices = self._get_sample_indices(
                video_path
            )

            sampled_indices = self._process_frame_indices(
                sampled_indices
            )

            # -----------------------------------------------------
            # Load components.
            # -----------------------------------------------------

            iframe_list = []
            motion_list = []
            residual_list = []

            for frame_index in sampled_indices:

                iframe, motion, residual = (
                    self._load_frame_components(
                        video_path,
                        frame_index,
                    )
                )

                iframe_list.append(iframe)
                motion_list.append(motion)
                residual_list.append(residual)

            # -----------------------------------------------------
            # Convert variable component lists into fixed arrays.
            #
            # We first determine spatial dimensions from whichever
            # representation is available.
            # -----------------------------------------------------

            reference = None

            for x in iframe_list:
                if x is not None:
                    reference = x
                    break

            if reference is None:
                for x in residual_list:
                    if x is not None:
                        reference = x
                        break

            if reference is None:
                for x in motion_list:
                    if x is not None:
                        reference = x
                        break

            if reference is None:
                raise RuntimeError(
                    f"No valid COVIAR representation for {video_path}"
                )

            height, width = reference.shape[:2]

            # -----------------------------------------------------
            # COVIAR convention:
            #
            # I-frame:    H x W x 3
            # residual:   H x W x 3
            # motion:     H x W x 2
            #
            # We store as:
            #
            # I:         T x H x W x 3
            # residual:  T x H x W x 3
            # motion:    T x H x W x 2
            # -----------------------------------------------------

            iframe_video = np.zeros(
                (
                    self.max_frames,
                    height,
                    width,
                    3,
                ),
                dtype=np.float32,
            )

            residual_video = np.zeros(
                (
                    self.max_frames,
                    height,
                    width,
                    3,
                ),
                dtype=np.float32,
            )

            motion_video = np.zeros(
                (
                    self.max_frames,
                    height,
                    width,
                    2,
                ),
                dtype=np.float32,
            )

            video_mask = np.zeros(
                self.max_frames,
                dtype=np.int64,
            )

            # -----------------------------------------------------
            # Fill valid temporal positions.
            # -----------------------------------------------------

            valid_length = min(
                len(sampled_indices),
                self.max_frames,
            )

            for t in range(valid_length):

                if iframe_list[t] is not None:

                    iframe_video[t] = (
                        iframe_list[t]
                    )

                if residual_list[t] is not None:

                    residual_video[t] = (
                        residual_list[t]
                    )

                if motion_list[t] is not None:

                    motion_video[t] = (
                        motion_list[t]
                    )

                video_mask[t] = 1

            all_iframes.append(
                iframe_video
            )

            all_motion.append(
                motion_video
            )

            all_residual.append(
                residual_video
            )

            all_masks.append(
                video_mask
            )

        return (
            np.stack(all_iframes),
            np.stack(all_motion),
            np.stack(all_residual),
            np.stack(all_masks),
            sampled_indices,
        )

    # =============================================================
    # __getitem__
    # =============================================================

    def __getitem__(self, idx):

        if self.unfold_sentences:

            video_id, caption = (
                self.sentences_dict[idx]
            )

        else:

            video_id = (
                self.csv["video_id"].values[idx]
            )

            caption = None

        (
            pairs_text,
            pairs_mask,
            pairs_segment,
            choice_video_ids,
        ) = self._get_text(
            video_id,
            caption,
        )

        (
            iframe,
            motion,
            residual,
            video_mask,
            frame_indices,
        ) = self._get_compressed_video(
            choice_video_ids
        )

        return {
            "video_id": video_id,

            "text": pairs_text,
            "text_mask": pairs_mask,
            "text_segment": pairs_segment,

            "iframe": iframe,
            "motion": motion,
            "residual": residual,

            "video_mask": video_mask,

            "frame_indices": frame_indices,
        }