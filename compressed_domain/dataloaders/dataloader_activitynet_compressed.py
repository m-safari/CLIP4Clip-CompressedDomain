"""Full-video / paragraph retrieval with ActivityNet Captions and CoViAR."""
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .activitynet_annotations import index_videos, load_paragraphs
from .compressedvideo_util import CompressedVideoExtractor


class ActivityNet_Compressed_DataLoader(Dataset):
    """One paragraph/video pair per split row, in CSV order.

    Uses the same nine-element batch contract as MSR-VTT. Timestamps describe
    individual events; paragraph retrieval samples GOPs across the full video.
    Missing or unreadable videos keep their rows with all-zero masks so the
    training/evaluation loops can exclude and count them.
    """

    def __init__(self, csv_path, data_path, features_path, tokenizer, subset="train",
                 max_words=64, max_gops=12, gop_slice=2, residual_sampling_rate=1,
                 mv_sampling_rate=1, image_resolution=224, gop_size=12, accumulate=True):
        if subset not in ("train", "val", "test"):
            raise ValueError(f"Unknown ActivityNet subset: {subset}")
        self.max_words = max_words
        self.tokenizer = tokenizer
        if not 2 <= max_words <= 77:
            raise ValueError("max_words must be between 2 and CLIP's context length (77)")
        annotations = Path(data_path)
        if annotations.is_dir():
            annotations = annotations / ("train.json" if subset == "train" else "val_1.json")
        paragraphs = load_paragraphs(annotations)
        if csv_path:
            self.csv = pd.read_csv(csv_path, dtype={"video_id": str})
            if "video_id" not in self.csv:
                raise ValueError("ActivityNet split CSV must have a video_id column")
            if self.csv.video_id.isna().any() or self.csv.video_id.duplicated().any():
                raise ValueError("ActivityNet split must contain unique, nonempty video IDs")
            missing = set(self.csv.video_id) - paragraphs.keys()
            if missing:
                raise ValueError(f"Split IDs missing from {annotations}: {sorted(missing)[:5]}")
            # Paragraphs always come from annotations, never random event captions.
            self.csv["sentence"] = [paragraphs[v] for v in self.csv.video_id]
        else:
            self.csv = pd.DataFrame({"video_id": list(paragraphs), "sentence": list(paragraphs.values())})
        if self.csv.empty:
            raise ValueError("ActivityNet split contains no videos")
        self.csv_has_sentence = True  # eval_epoch's metadata/export contract
        self.video_paths = index_videos(features_path)
        self.extractor = CompressedVideoExtractor(
            gop_size=gop_size, image_resolution=image_resolution, max_gops=max_gops,
            gop_slice=gop_slice, residual_sampling_rate=residual_sampling_rate,
            mv_sampling_rate=mv_sampling_rate, accumulate=accumulate)
        self.lengths = (max_gops, max(1, max_gops * residual_sampling_rate),
                        max(1, max_gops * mv_sampling_rate))
        self.image_resolution = image_resolution

    def __len__(self):
        return len(self.csv)

    def __getitem__(self, idx):
        row = self.csv.iloc[idx]
        words = ["<|startoftext|>"] + self.tokenizer.tokenize(row.sentence)[:self.max_words - 2]
        words.append("<|endoftext|>")
        token_ids = self.tokenizer.convert_tokens_to_ids(words)
        ids = np.zeros((1, self.max_words), dtype=np.int64)
        text_mask = np.zeros_like(ids)
        ids[0, :len(token_ids)] = token_ids
        text_mask[0, :len(token_ids)] = 1
        output = []
        data = {}
        path = self.video_paths.get(row.video_id)
        if path is not None:
            try:
                data = self.extractor.video_to_compressed_tensors(str(path))
            except Exception as error:
                print(f"WARNING: failed to decode {row.video_id}: {error!r}")
        for key, channels, length in zip(("iframe", "residuals", "mv"), (3, 3, 2), self.lengths):
            frames = torch.zeros(1, length, channels, self.image_resolution, self.image_resolution)
            mask = torch.zeros(1, length, dtype=torch.long)
            if key in data:
                count = min(len(data[key]), length)
                frames[0, :count] = data[key][:count]
                mask[0, :count] = 1
            output.extend((frames, mask))
        return (ids, text_mask, np.zeros_like(ids), *output)
