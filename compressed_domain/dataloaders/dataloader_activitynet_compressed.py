"""
ActivityNet Captions compressed-domain dataset for paragraph-to-video retrieval.

Data layout (the ActivityNet Captions release, written by
prepare_activitynet_dataset.py):

    <data_path>/train_ids.json, val_ids.json   ["v_<youtube id>", ...]
    <data_path>/train.json, val_1.json         {"v_<id>": {"duration", "timestamps", "sentences"}}
    <features_path>/**/v_<id>.mp4 (or <id>.mp4) re-encoded with reencode.sh

As in CLIP4Clip, ActivityNet is video-paragraph retrieval: all sentences of a
video are joined into one paragraph and matched against the whole video, so
there is exactly one caption per video. The val split is val_1.json; there is
no separate test split.

Decoding, sampling, truncation, padding and masking are inherited unchanged
from MSRVTT_Compressed_DataLoader. --max_frames GOPs are spread uniformly
(--slice_framepos 2) over the whole video, whatever its length.
"""
import json
import os

import numpy as np
from torch.utils.data import Dataset

from .dataloader_msrvtt_compressed import MSRVTT_Compressed_DataLoader, index_video_files

SUBSETS = ("train", "val")
IDS_FILES = {"train": "train_ids.json", "val": "val_ids.json"}
CAPTION_FILES = {"train": "train.json", "val": "val_1.json"}


def read_paragraphs(path):
    """{"v_<id>": paragraph}: each video's sentences, stripped and joined by spaces."""
    with open(path) as f:
        data = json.load(f)
    return {video_id: " ".join(s.strip() for s in item["sentences"]) for video_id, item in data.items()}


def read_video_ids(ids_path, paragraphs):
    """Video ids in the order of the ids file, deduplicated; the caption file's keys if there is none."""
    if not os.path.exists(ids_path):
        return list(paragraphs)
    with open(ids_path) as f:
        return list(dict.fromkeys(json.load(f)))


def video_file_stems(video_id):
    """The file names a video may be stored under: "v_<id>" and "<id>"."""
    stems = [video_id]
    if video_id.startswith("v_"):
        stems.append(video_id[2:])
    return stems


class ActivityNet_Compressed_DataLoader(MSRVTT_Compressed_DataLoader):
    """
    One item per video (train and val alike): its paragraph and its three
    compressed modalities.

    Videos that are in the ids file but have no paragraph, or no re-encoded
    file under features_path, are left out; the latter are in missing_video_ids.
    ActivityNet is distributed per request and many YouTube originals are
    gone, so an incomplete copy is common.
    """

    multi_sentence_per_video = False

    def __init__(
        self,
        subset,
        data_path,
        features_path,
        tokenizer,
        max_words=64,
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

        paragraphs = read_paragraphs(os.path.join(data_path, CAPTION_FILES[subset]))
        listed = [v for v in read_video_ids(os.path.join(data_path, IDS_FILES[subset]), paragraphs)
                  if v in paragraphs]

        files = index_video_files(features_path)
        self.video_paths = {}
        for video_id in listed:
            path = next((files[s] for s in video_file_stems(video_id) if s in files), None)
            if path is not None:
                self.video_paths[video_id] = path
        self.missing_video_ids = [v for v in listed if v not in self.video_paths]
        self._video_ids = [v for v in listed if v in self.video_paths]
        self._captions = [paragraphs[v] for v in self._video_ids]
        if not self._video_ids:
            raise ValueError("ActivityNet {}: none of the {} videos of {} has a .mp4 under {}. Re-encode them with "
                             "dataloaders/prepare_activitynet_dataset.py --subsets {}.".format(
                                 subset, len(listed), CAPTION_FILES[subset], features_path, subset))
        if self.missing_video_ids:
            print("WARNING: ActivityNet {}: {} of {} videos have no .mp4 under {} and are skipped.".format(
                subset, len(self.missing_video_ids), len(listed), features_path))
        print("ActivityNet {}: {} videos / paragraphs".format(subset, len(self._video_ids)))

    def __len__(self):
        return len(self._video_ids)

    @property
    def video_ids(self):
        return np.asarray(self._video_ids).astype(str)

    @property
    def captions(self):
        return np.asarray(self._captions).astype(str)

    def _video_path(self, video_id):
        return self.video_paths[video_id]

    def __getitem__(self, idx):
        video_id = self._video_ids[idx]
        pairs_text, pairs_mask, pairs_segment, choice_video_ids = self._get_text(video_id, self._captions[idx])
        video = self._get_compressed_video_or_empty(choice_video_ids)
        return (pairs_text, pairs_mask, pairs_segment) + tuple(video)
