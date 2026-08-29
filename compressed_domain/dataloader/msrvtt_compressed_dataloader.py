from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import os
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from collections import defaultdict
import json
import random

from .compressed_video_util import CoviarVideoExtractor


class MSRVTT_CompressedDataLoader(Dataset):
    """MSRVTT dataset loader with aligned raw + compressed-domain video.

    Same structure as MSRVTT_DataLoader, but each __getitem__ additionally
    returns I-frame / motion-vector / residual tensors aligned frame-for-
    frame with the raw RGB tensor.
    """
    def __init__(
            self,
            csv_path,
            features_path,
            tokenizer,
            max_words=30,
            feature_framerate=1.0,
            max_frames=100,
            image_resolution=224,
            frame_order=0,
            slice_framepos=0,
            gop_size=12,
            accumulate=True,
    ):
        self.data = pd.read_csv(csv_path)
        self.features_path = features_path
        self.feature_framerate = feature_framerate
        self.max_words = max_words
        self.max_frames = max_frames
        self.tokenizer = tokenizer
        self.frame_order = frame_order
        assert self.frame_order in [0, 1, 2]
        self.slice_framepos = slice_framepos
        assert self.slice_framepos in [0, 1, 2]

        self.videoExtractor = CoviarVideoExtractor(
            framerate=feature_framerate, size=image_resolution,
            gop_size=gop_size, accumulate=accumulate,
        )
        self.SPECIAL_TOKEN = {"CLS_TOKEN": "<|startoftext|>", "SEP_TOKEN": "<|endoftext|>",
                              "MASK_TOKEN": "[MASK]", "UNK_TOKEN": "[UNK]", "PAD_TOKEN": "[PAD]"}

    def __len__(self):
        return len(self.data)

    def _get_text(self, video_id, sentence):
        choice_video_ids = [video_id]
        k = len(choice_video_ids)
        pairs_text = np.zeros((k, self.max_words), dtype=np.int64)
        pairs_mask = np.zeros((k, self.max_words), dtype=np.int64)
        pairs_segment = np.zeros((k, self.max_words), dtype=np.int64)

        for i, video_id in enumerate(choice_video_ids):
            words = self.tokenizer.tokenize(sentence)
            words = [self.SPECIAL_TOKEN["CLS_TOKEN"]] + words
            total_length_with_CLS = self.max_words - 1
            if len(words) > total_length_with_CLS:
                words = words[:total_length_with_CLS]
            words = words + [self.SPECIAL_TOKEN["SEP_TOKEN"]]

            input_ids = self.tokenizer.convert_tokens_to_ids(words)
            input_mask = [1] * len(input_ids)
            segment_ids = [0] * len(input_ids)
            while len(input_ids) < self.max_words:
                input_ids.append(0)
                input_mask.append(0)
                segment_ids.append(0)
            assert len(input_ids) == self.max_words
            assert len(input_mask) == self.max_words
            assert len(segment_ids) == self.max_words

            pairs_text[i] = np.array(input_ids)
            pairs_mask[i] = np.array(input_mask)
            pairs_segment[i] = np.array(segment_ids)

        return pairs_text, pairs_mask, pairs_segment, choice_video_ids

    def _slice_and_order(self, tensor):
        """Apply the same max_frames slicing (head/tail/uniform) and
        frame_order permutation used for raw video, generically, to any
        (T, 1, C, H, W)-shaped modality tensor."""
        if self.max_frames < tensor.shape[0]:
            if self.slice_framepos == 0:
                sliced = tensor[:self.max_frames, ...]
            elif self.slice_framepos == 1:
                sliced = tensor[-self.max_frames:, ...]
            else:
                sample_indx = np.linspace(0, tensor.shape[0] - 1, num=self.max_frames, dtype=int)
                sliced = tensor[sample_indx, ...]
        else:
            sliced = tensor
        return self.videoExtractor.process_frame_order(sliced, frame_order=self.frame_order)

    def _get_compressed_video(self, choice_video_ids):
        n = len(choice_video_ids)
        video_mask = np.zeros((n, self.max_frames), dtype=np.int64)
        max_video_length = [0] * n

        shape5d = lambda c: np.zeros(
            (n, self.max_frames, 1, c, self.videoExtractor.size, self.videoExtractor.size),
            dtype=np.float32)
        video = shape5d(3)
        iframe = shape5d(3)
        mv = shape5d(2)
        residual = shape5d(3)

        for i, video_id in enumerate(choice_video_ids):
            video_path = os.path.join(self.features_path, "{}.mp4".format(video_id))
            if os.path.exists(video_path) is False:
                video_path = video_path.replace(".mp4", ".webm")

            raw = self.videoExtractor.get_video_data(video_path)

            if len(raw['video'].shape) <= 3:
                print("video path: {} error. video id: {}".format(video_path, video_id))
                continue

            def prep(key, channels):
                t = self.videoExtractor.process_raw_data(raw[key])
                return self._slice_and_order(t)

            video_slice = prep('video', 3)
            iframe_slice = prep('iframe', 3)
            mv_slice = prep('mv', 2)
            residual_slice = prep('residual', 3)

            slice_len = video_slice.shape[0]
            max_video_length[i] = max(max_video_length[i], slice_len)
            if slice_len < 1:
                continue

            video[i][:slice_len, ...] = video_slice
            iframe[i][:slice_len, ...] = iframe_slice
            mv[i][:slice_len, ...] = mv_slice
            residual[i][:slice_len, ...] = residual_slice

        for i, v_length in enumerate(max_video_length):
            video_mask[i][:v_length] = 1

        return video, iframe, mv, residual, video_mask

    def __getitem__(self, idx):
        video_id = self.data['video_id'].values[idx]
        sentence = self.data['sentence'].values[idx]

        pairs_text, pairs_mask, pairs_segment, choice_video_ids = self._get_text(video_id, sentence)
        video, iframe, mv, residual, video_mask = self._get_compressed_video(choice_video_ids)
        return pairs_text, pairs_mask, pairs_segment, video, iframe, mv, residual, video_mask


class MSRVTT_TrainCompressedDataLoader(Dataset):
    """Train-split counterpart, mirroring MSRVTT_TrainDataLoader's
    unfold_sentences / random-caption-per-epoch behavior, extended with
    aligned compressed-domain retrieval."""
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
            image_resolution=224,
            frame_order=0,
            slice_framepos=0,
            gop_size=12,
            accumulate=True,
    ):
        self.csv = pd.read_csv(csv_path)
        self.data = json.load(open(json_path, 'r'))
        self.features_path = features_path
        self.feature_framerate = feature_framerate
        self.max_words = max_words
        self.max_frames = max_frames
        self.tokenizer = tokenizer
        self.frame_order = frame_order
        assert self.frame_order in [0, 1, 2]
        self.slice_framepos = slice_framepos
        assert self.slice_framepos in [0, 1, 2]

        self.unfold_sentences = unfold_sentences
        self.sample_len = 0
        if self.unfold_sentences:
            train_video_ids = list(self.csv['video_id'].values)
            self.sentences_dict = {}
            for itm in self.data['sentences']:
                if itm['video_id'] in train_video_ids:
                    self.sentences_dict[len(self.sentences_dict)] = (itm['video_id'], itm['caption'])
            self.sample_len = len(self.sentences_dict)
        else:
            self.sentences = defaultdict(list)
            for itm in self.data['sentences']:
                self.sentences[itm['video_id']].append(itm['caption'])

            self.parent_ids = {}
            self.children_video_ids = defaultdict(list)
            for itm in self.data['videos']:
                vid = itm["video_id"]
                url_posfix = itm["url"].split("?v=")[-1]
                self.parent_ids[vid] = url_posfix
                self.children_video_ids[url_posfix].append(vid)
            self.sample_len = len(self.csv)

        self.videoExtractor = CoviarVideoExtractor(
            framerate=feature_framerate, size=image_resolution,
            gop_size=gop_size, accumulate=accumulate,
        )
        self.SPECIAL_TOKEN = {"CLS_TOKEN": "<|startoftext|>", "SEP_TOKEN": "<|endoftext|>",
                              "MASK_TOKEN": "[MASK]", "UNK_TOKEN": "[UNK]", "PAD_TOKEN": "[PAD]"}

    def __len__(self):
        return self.sample_len

    def _get_single_text(self, video_id):
        rind = random.randint(0, len(self.sentences[video_id]) - 1)
        return self.tokenizer.tokenize(self.sentences[video_id][rind])

    def _get_text(self, video_id, caption=None):
        choice_video_ids = [video_id]
        pairs_text = np.zeros((1, self.max_words), dtype=np.int64)
        pairs_mask = np.zeros((1, self.max_words), dtype=np.int64)
        pairs_segment = np.zeros((1, self.max_words), dtype=np.int64)

        for i, vid in enumerate(choice_video_ids):
            words = self.tokenizer.tokenize(caption) if caption is not None else self._get_single_text(vid)
            words = [self.SPECIAL_TOKEN["CLS_TOKEN"]] + words
            total_length_with_CLS = self.max_words - 1
            if len(words) > total_length_with_CLS:
                words = words[:total_length_with_CLS]
            words = words + [self.SPECIAL_TOKEN["SEP_TOKEN"]]

            input_ids = self.tokenizer.convert_tokens_to_ids(words)
            input_mask = [1] * len(input_ids)
            segment_ids = [0] * len(input_ids)
            while len(input_ids) < self.max_words:
                input_ids.append(0)
                input_mask.append(0)
                segment_ids.append(0)

            pairs_text[i] = np.array(input_ids)
            pairs_mask[i] = np.array(input_mask)
            pairs_segment[i] = np.array(segment_ids)

        return pairs_text, pairs_mask, pairs_segment, choice_video_ids

    # identical slicing/masking logic to the test loader above
    _slice_and_order = MSRVTT_CompressedDataLoader._slice_and_order
    _get_compressed_video = MSRVTT_CompressedDataLoader._get_compressed_video

    def __getitem__(self, idx):
        if self.unfold_sentences:
            video_id, caption = self.sentences_dict[idx]
        else:
            video_id, caption = self.csv['video_id'].values[idx], None
        pairs_text, pairs_mask, pairs_segment, choice_video_ids = self._get_text(video_id, caption)
        video, iframe, mv, residual, video_mask = self._get_compressed_video(choice_video_ids)
        return pairs_text, pairs_mask, pairs_segment, video, iframe, mv, residual, video_mask
