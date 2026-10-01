"""
HMDB51 compressed-domain dataset for zero-shot action recognition.

Videos only: the "text" side of action recognition is a fixed set of class
prompts, encoded once (see action_recognition.py), so a dataset item is one
video's three compressed modalities plus its label.

Decoding, sampling, truncation, padding and masking are inherited unchanged
from MSRVTT_Compressed_DataLoader._get_compressed_video, so an HMDB51 video
reaches the model exactly as an MSR-VTT video would.
"""
import pandas as pd
import torch
from torch.utils.data import Dataset

from .compressedvideo_util import CompressedVideoExtractor
from .dataloader_msrvtt_compressed import MSRVTT_Compressed_DataLoader

# (frames key, mask key) of each modality, in the model's argument order.
MODALITIES = (("iframe", "iframe_mask"), ("residuals", "residuals_mask"), ("mv", "mv_mask"))


class HMDB51_Compressed_DataLoader(MSRVTT_Compressed_DataLoader):
    """
    One item per row of the index CSV written by prepare_hmdb51_dataset.py.

    `video_id` is "<class>/<stem>", so the inherited loader finds the video at
    <features_path>/<class>/<stem>.mp4.

    Each item is a dict:
        index, label_index                    : int
        iframe / residuals / mv               : (T_max, C, H, W) float
        iframe_mask / residuals_mask / mv_mask: (T_max,) int64
        decode_ok                             : bool -- False if CoViAR raised;
                                                all masks are then 0. Also False when
                                                CoViAR decoded no frames at all.
    """

    def __init__(
        self,
        index_csv,
        features_path,
        split=None,
        subset="test",
        iframe_sampling_rate=1,
        residual_sampling_rate=3,
        mv_sampling_rate=3,
        max_iframe_length=100,
        max_residual_length=100,
        max_mv_length=100,
        image_resolution=224,
        gop_size=12,
        accumulate=True,
    ):
        # The parent's __init__ reads MSR-VTT captions, which HMDB51 has none
        # of; only the attributes _get_compressed_video uses are set up here.
        Dataset.__init__(self)

        index = pd.read_csv(index_csv)
        if split is not None:
            tag = {"train": 1, "test": 2}[subset]
            index = index[index["split{}".format(split)] == tag]
        self.index = index.reset_index(drop=True)

        self.features_path = features_path
        self.max_iframe_length = max_iframe_length
        self.max_residual_length = max_residual_length
        self.max_mv_length = max_mv_length

        self.compressedVideoExtractor = CompressedVideoExtractor(
            gop_size=gop_size,
            image_resolution=image_resolution,
            iframe_sampling_rate=iframe_sampling_rate,
            residual_sampling_rate=residual_sampling_rate,
            mv_sampling_rate=mv_sampling_rate,
            accumulate=accumulate,
        )

    def __len__(self):
        return len(self.index)

    def _empty_video(self):
        # What _get_compressed_video returns for a video with no frames: all
        # zeros, every position masked out.
        res = self.compressedVideoExtractor.image_resolution
        lengths = (self.max_iframe_length, self.max_residual_length, self.max_mv_length)
        out = []
        for length, channels in zip(lengths, (3, 3, 2)):
            out.append(torch.zeros(1, length, channels, res, res))
            out.append(torch.zeros(1, length, dtype=torch.long))
        return tuple(out)

    def __getitem__(self, idx):
        row = self.index.iloc[idx]
        decode_ok = True
        try:
            video = self._get_compressed_video([row["video_id"]])
        except Exception:
            # A corrupt or missing file must not abort a 6.7k-video pass; it is
            # reported via decode_ok and yields an all-masked (NaN) embedding.
            decode_ok = False
            video = self._empty_video()
        # CoViAR does not raise on an unreadable file: it prints an error and
        # reports no frames, which leaves every mask at 0.
        decode_ok = decode_ok and any(bool(mask.any()) for mask in video[1::2])

        item = {"index": int(idx), "label_index": int(row["label_index"]), "decode_ok": decode_ok}
        for (frames_key, mask_key), frames, mask in zip(MODALITIES, video[0::2], video[1::2]):
            item[frames_key] = frames[0]     # drop the single-pair dimension
            item[mask_key] = mask[0]
        return item


def trim_padding_collate(items):
    """
    Default collation, then drop the trailing frame positions that are padding
    for every video in the batch (at least one position is always kept).

    Frames are encoded independently and padded positions are excluded from
    the masked mean pool, so embeddings are identical to the untrimmed batch's;
    only the encoder work on all-padding columns is saved.
    """
    batch = torch.utils.data.default_collate(items)
    for frames_key, mask_key in MODALITIES:
        mask = batch[mask_key]
        used = mask.any(dim=0).nonzero()
        keep = int(used.max()) + 1 if len(used) else 1
        batch[frames_key] = batch[frames_key][:, :keep]
        batch[mask_key] = mask[:, :keep]
    return batch


def dataloader_hmdb51(index_csv, features_path, batch_size=8, num_workers=0, split=None, subset="test",
                      **dataset_kwargs):
    dataset = HMDB51_Compressed_DataLoader(index_csv, features_path, split=split, subset=subset,
                                           **dataset_kwargs)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        collate_fn=trim_padding_collate, drop_last=False,
    )
    return loader, len(dataset)
