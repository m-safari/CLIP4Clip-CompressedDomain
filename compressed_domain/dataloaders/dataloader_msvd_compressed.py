"""
MSVD compressed-domain dataset for text-video retrieval.

Data layout (the CLIP4Clip msvd_data.zip, written by prepare_msvd_dataset.py):

    <data_path>/train_list.txt, val_list.txt, test_list.txt   one video id per line
    <data_path>/raw-captions.pkl                              {video_id: [[token, ...], ...]}
    <features_path>/**/<video_id>.mp4                         re-encoded with reencode.sh

MSVD has ~40 captions per video, so evaluation is multi-sentence: every caption
is a query and its video the one correct candidate. Unlike the RGB loader,
which decodes a video once per caption, an evaluation item here is ONE video,
decoded once; the captions are tokenized up front (caption_tokens) and encoded
separately by eval_epoch.

Decoding, sampling, truncation, padding and masking are inherited unchanged
from MSRVTT_Compressed_DataLoader, so an MSVD video reaches the model exactly
as an MSR-VTT video would.
"""
import os
import pickle

import numpy as np
from torch.utils.data import Dataset

from .dataloader_msrvtt_compressed import MSRVTT_Compressed_DataLoader, index_video_files

SUBSETS = ("train", "val", "test")
CAPTIONS_FILE = "raw-captions.pkl"


def read_video_list(path):
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def read_captions(path):
    """raw-captions.pkl -> {video_id: [caption, ...]}, each caption a plain string."""
    with open(path, "rb") as f:
        raw = pickle.load(f)
    return {video_id: [" ".join(tokens) if not isinstance(tokens, str) else tokens for tokens in caps]
            for video_id, caps in raw.items()}


class MSVD_Compressed_DataLoader(MSRVTT_Compressed_DataLoader):
    """
    train:      one item per (video, caption) pair, as in the RGB loader.
    val / test: one item per video (its first caption fills the text slot), and
                multi_sentence_per_video = True with
                    captions             (N_cap,) every caption of the subset
                    caption_video_index  (N_cap,) row in video_ids of its video
                    caption_tokens()     the captions as (ids, mask, segment)

    Videos with no re-encoded file under features_path are left out, together
    with their captions; their ids are in missing_video_ids.
    """

    def __init__(
        self,
        subset,
        data_path,
        features_path,
        tokenizer,
        max_words=30,
        iframe_sampling_rate=1,
        residual_sampling_rate=3,
        mv_sampling_rate=3,
        max_iframe_length=100,
        max_residual_length=100,
        max_mv_length=100,
        max_gops=None,
        gop_slice=2,
        image_resolution=224,
        gop_size=12,
        accumulate=True,
    ):
        # The parent's __init__ reads MSR-VTT's CSV and JSON; only the
        # attributes _get_text and _get_compressed_video use are set up here.
        Dataset.__init__(self)
        if subset not in SUBSETS:
            raise ValueError("subset must be one of {}, got {!r}".format(SUBSETS, subset))
        self.subset = subset

        self._init_video_side(
            features_path,
            tokenizer=tokenizer,
            max_words=max_words,
            iframe_sampling_rate=iframe_sampling_rate,
            residual_sampling_rate=residual_sampling_rate,
            mv_sampling_rate=mv_sampling_rate,
            max_iframe_length=max_iframe_length,
            max_residual_length=max_residual_length,
            max_mv_length=max_mv_length,
            max_gops=max_gops,
            gop_slice=gop_slice,
            image_resolution=image_resolution,
            gop_size=gop_size,
            accumulate=accumulate,
        )

        listed = read_video_list(os.path.join(data_path, "{}_list.txt".format(subset)))
        captions = read_captions(os.path.join(data_path, CAPTIONS_FILE))
        no_caption = [v for v in listed if v not in captions]
        if no_caption:
            raise ValueError("{} videos of {}_list.txt have no caption in {}, e.g. {}".format(
                len(no_caption), subset, CAPTIONS_FILE, no_caption[:3]))

        files = index_video_files(features_path)
        self.video_paths = {v: files[v] for v in listed if v in files}
        self.missing_video_ids = [v for v in listed if v not in files]
        self._video_ids = [v for v in listed if v in files]
        if not self._video_ids:
            raise ValueError("MSVD {}: none of the {} videos of {}_list.txt has a .mp4 under {}. Re-encode them "
                             "with dataloaders/prepare_msvd_dataset.py.".format(
                                 subset, len(listed), subset, features_path))
        if self.missing_video_ids:
            print("WARNING: MSVD {}: {} of {} videos have no .mp4 under {} and are skipped.".format(
                subset, len(self.missing_video_ids), len(listed), features_path))

        self.multi_sentence_per_video = subset != "train"
        self._captions, caption_video_index = [], []
        for row, video_id in enumerate(self._video_ids):
            for caption in captions[video_id]:
                self._captions.append(caption)
                caption_video_index.append(row)
        self.caption_video_index = np.asarray(caption_video_index, dtype=np.int64)
        self._first_caption = {v: captions[v][0] for v in self._video_ids}

        print("MSVD {}: {} videos, {} captions".format(subset, len(self._video_ids), len(self._captions)))

    def __len__(self):
        return len(self._video_ids) if self.multi_sentence_per_video else len(self._captions)

    @property
    def video_ids(self):
        return np.asarray(self._video_ids).astype(str)

    @property
    def captions(self):
        return np.asarray(self._captions).astype(str)

    def _video_path(self, video_id):
        return self.video_paths[video_id]

    def caption_tokens(self):
        """Every caption of the subset as (input_ids, input_mask, segment_ids), each (N_cap, max_words)."""
        ids, mask, segment = [], [], []
        for caption, row in zip(self._captions, self.caption_video_index):
            pairs_text, pairs_mask, pairs_segment, _ = self._get_text(self._video_ids[row], caption)
            ids.append(pairs_text[0])
            mask.append(pairs_mask[0])
            segment.append(pairs_segment[0])
        return np.stack(ids), np.stack(mask), np.stack(segment)

    def __getitem__(self, idx):
        if self.multi_sentence_per_video:
            video_id = self._video_ids[idx]
            caption = self._first_caption[video_id]
        else:
            caption = self._captions[idx]
            video_id = self._video_ids[self.caption_video_index[idx]]

        pairs_text, pairs_mask, pairs_segment, choice_video_ids = self._get_text(video_id, caption)
        video = self._get_compressed_video_or_empty(choice_video_ids)
        return (pairs_text, pairs_mask, pairs_segment) + tuple(video)
