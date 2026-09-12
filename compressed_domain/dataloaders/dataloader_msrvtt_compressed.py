import os
import json
import random

import numpy as np
import pandas as pd
import torch

from torch.utils.data import Dataset

from .compressedvideo_util import (
    CompressedVideoExtractor,
)


class MSRVTT_Compressed_DataLoader(Dataset):
    """
    MSR-VTT dataset loader for compressed-domain video-text learning.

    Each dataset item contains:

        text
        ├── input_ids
        ├── input_mask
        └── segment_ids

        compressed video
        ├── iframe
        ├── iframe_mask
        ├── residuals
        ├── residuals_mask
        ├── mv
        └── mv_mask

    Unlike the original RawVideoExtractor-based loader, there is no
    single "video" tensor.

    Each compressed modality has its own temporal sequence and therefore
    its own maximum length and mask.
    """

    def __init__(
        self,
        csv_path,
        json_path,
        features_path,
        tokenizer,

        max_words=30,

        # ---------------------------------------------------------------
        # Sampling rates inside EACH GOP.
        # ---------------------------------------------------------------
        iframe_sampling_rate=1,
        residual_sampling_rate=3,
        mv_sampling_rate=3,

        # ---------------------------------------------------------------
        # Maximum sequence lengths for each modality.
        #
        # These are intentionally independent.
        # ---------------------------------------------------------------
        max_iframe_length=100,
        max_residual_length=300,
        max_mv_length=300,

        # ---------------------------------------------------------------
        # Spatial resolution.
        # ---------------------------------------------------------------
        image_resolution=224,

        # ---------------------------------------------------------------
        # GOP configuration.
        # ---------------------------------------------------------------
        gop_size=12,

        # ---------------------------------------------------------------
        # CoViAR accumulation.
        # ---------------------------------------------------------------
        accumulate=True,

        # ---------------------------------------------------------------
        # Caption handling.
        # ---------------------------------------------------------------
        unfold_sentences=False,
    ):

        self.csv = pd.read_csv(csv_path)
        self.data = json.load(open(json_path, "r"))

        self.features_path = features_path

        self.max_words = max_words

        self.max_iframe_length = max_iframe_length
        self.max_residual_length = max_residual_length
        self.max_mv_length = max_mv_length

        self.tokenizer = tokenizer

        self.unfold_sentences = unfold_sentences

        # ---------------------------------------------------------------
        # Create the compressed-video extractor.
        #
        # The extractor handles CoViAR and representation sampling.
        # ---------------------------------------------------------------
        self.compressedVideoExtractor = CompressedVideoExtractor(
            gop_size=gop_size,
            image_resolution=image_resolution,

            iframe_sampling_rate=iframe_sampling_rate,
            residual_sampling_rate=residual_sampling_rate,
            mv_sampling_rate=mv_sampling_rate,

            accumulate=accumulate,
        )

        # Same special-token convention as the original loader.
        self.SPECIAL_TOKEN = {
            "CLS_TOKEN": "<|startoftext|>",
            "SEP_TOKEN": "<|endoftext|>",
            "MASK_TOKEN": "[MASK]",
            "UNK_TOKEN": "[UNK]",
            "PAD_TOKEN": "[PAD]",
        }

        # ---------------------------------------------------------------
        # Caption bookkeeping.
        #
        # MSR-VTT contains multiple captions per video.
        # ---------------------------------------------------------------
        if self.unfold_sentences:

            train_video_ids = set(
                self.csv["video_id"].values
            )

            self.sentences_dict = {}

            for item in self.data["sentences"]:

                if item["video_id"] in train_video_ids:

                    self.sentences_dict[
                        len(self.sentences_dict)
                    ] = (
                        item["video_id"],
                        item["caption"],
                    )

            self.sample_len = len(self.sentences_dict)

        else:

            self.sentences = {}

            for item in self.data["sentences"]:

                video_id = item["video_id"]

                if video_id not in self.sentences:
                    self.sentences[video_id] = []

                self.sentences[video_id].append(
                    item["caption"]
                )

            # One dataset item per CSV row.
            self.sample_len = len(self.csv)

    def __len__(self):
        return self.sample_len

    # ==================================================================
    # TEXT SIDE
    #
    # This is intentionally almost identical to the original loader.
    # ==================================================================

    def _get_text(self, video_id, caption=None):

        # One video-text pair per dataset item.
        choice_video_ids = [video_id]

        k = 1

        pairs_text = np.zeros(
            (k, self.max_words),
            dtype=np.int64,
        )

        pairs_mask = np.zeros(
            (k, self.max_words),
            dtype=np.int64,
        )

        pairs_segment = np.zeros(
            (k, self.max_words),
            dtype=np.int64,
        )

        for i, _ in enumerate(choice_video_ids):

            # -----------------------------------------------------------
            # If a caption was explicitly provided, use it.
            #
            # Otherwise randomly select one caption belonging to
            # this video.
            # -----------------------------------------------------------
            if caption is not None:

                words = self.tokenizer.tokenize(
                    caption
                )

            else:

                words = self._get_single_text(
                    video_id
                )

            # -----------------------------------------------------------
            # Add start-of-text token.
            # -----------------------------------------------------------
            words = [
                self.SPECIAL_TOKEN["CLS_TOKEN"]
            ] + words

            # -----------------------------------------------------------
            # Reserve one position for end-of-text.
            # -----------------------------------------------------------
            total_length_with_CLS = (
                self.max_words - 1
            )

            if len(words) > total_length_with_CLS:

                words = words[
                    :total_length_with_CLS
                ]

            # -----------------------------------------------------------
            # Add end-of-text token.
            # -----------------------------------------------------------
            words = words + [
                self.SPECIAL_TOKEN["SEP_TOKEN"]
            ]

            # -----------------------------------------------------------
            # Convert tokens -> vocabulary IDs.
            # -----------------------------------------------------------
            input_ids = (
                self.tokenizer.convert_tokens_to_ids(
                    words
                )
            )

            # -----------------------------------------------------------
            # Real tokens receive mask=1.
            # -----------------------------------------------------------
            input_mask = [1] * len(input_ids)

            # -----------------------------------------------------------
            # This loader uses a single segment, so everything belongs
            # to segment 0.
            # -----------------------------------------------------------
            segment_ids = [0] * len(input_ids)

            # -----------------------------------------------------------
            # Pad to max_words.
            # -----------------------------------------------------------
            while len(input_ids) < self.max_words:

                input_ids.append(0)
                input_mask.append(0)
                segment_ids.append(0)

            pairs_text[i] = np.array(input_ids)
            pairs_mask[i] = np.array(input_mask)
            pairs_segment[i] = np.array(segment_ids)

        return (
            pairs_text,
            pairs_mask,
            pairs_segment,
            choice_video_ids,
        )

    def _get_single_text(self, video_id):
        """
        Randomly select one caption for a video.

        This is the same strategy as the original loader when
        unfold_sentences=False.
        """

        captions = self.sentences[video_id]

        caption = random.choice(captions)

        return self.tokenizer.tokenize(caption)

    # ==================================================================
    # COMPRESSED VIDEO SIDE
    # ==================================================================

    def _get_compressed_video(self, choice_video_ids):
        """
        Load, truncate, pad and mask all three compressed modalities.

        Each modality gets an independent maximum length.

        Returns
        -------

        iframe:
            [P, max_iframe_length, 3, H, W]

        iframe_mask:
            [P, max_iframe_length]

        residuals:
            [P, max_residual_length, 3, H, W]

        residuals_mask:
            [P, max_residual_length]

        mv:
            [P, max_mv_length, 2, H, W]

        mv_mask:
            [P, max_mv_length]

        where P is the number of video-text pairs.

        In this implementation P=1, just like the original MSR-VTT
        loader.
        """

        num_pairs = len(choice_video_ids)

        H = self.compressedVideoExtractor.image_resolution
        W = self.compressedVideoExtractor.image_resolution

        # ---------------------------------------------------------------
        # Allocate fixed-size output tensors.
        #
        # Notice that the temporal dimensions are different.
        # ---------------------------------------------------------------

        iframe = np.zeros(
            (
                num_pairs,
                self.max_iframe_length,
                3,
                H,
                W,
            ),
            dtype=np.float32,
        )

        residuals = np.zeros(
            (
                num_pairs,
                self.max_residual_length,
                3,
                H,
                W,
            ),
            dtype=np.float32,
        )

        mv = np.zeros(
            (
                num_pairs,
                self.max_mv_length,
                2,
                H,
                W,
            ),
            dtype=np.float32,
        )

        # ---------------------------------------------------------------
        # Independent masks.
        #
        # A 1 means:
        #     "this temporal position contains real data"
        #
        # A 0 means:
        #     "this position is padding"
        # ---------------------------------------------------------------

        iframe_mask = np.zeros(
            (
                num_pairs,
                self.max_iframe_length,
            ),
            dtype=np.int64,
        )

        residuals_mask = np.zeros(
            (
                num_pairs,
                self.max_residual_length,
            ),
            dtype=np.int64,
        )

        mv_mask = np.zeros(
            (
                num_pairs,
                self.max_mv_length,
            ),
            dtype=np.int64,
        )

        # ---------------------------------------------------------------
        # Process each video-text pair independently.
        # ---------------------------------------------------------------

        for i, video_id in enumerate(choice_video_ids):

            video_path = os.path.join(
                self.features_path,
                "{}.mp4".format(video_id),
            )

            if not os.path.exists(video_path):

                # Same .webm fallback used by the original loader.
                video_path = video_path.replace(
                    ".mp4",
                    ".webm",
                )

            # -----------------------------------------------------------
            # Extract compressed-domain sequences.
            #
            # The extractor returns variable-length tensors:
            #
            # iframe    -> [N_I, 3, H, W]
            # residuals -> [N_R, 3, H, W]
            # mv        -> [N_M, 2, H, W]
            # -----------------------------------------------------------

            compressed_data = (
                self.compressedVideoExtractor
                .video_to_compressed_tensors(
                    video_path
                )
            )

            raw_iframe = compressed_data["iframe"]
            raw_residuals = compressed_data["residuals"]
            raw_mv = compressed_data["mv"]

            # -----------------------------------------------------------
            # Apply independent truncation/padding.
            # -----------------------------------------------------------

            iframe_length = min(
                raw_iframe.shape[0],
                self.max_iframe_length,
            )

            residual_length = min(
                raw_residuals.shape[0],
                self.max_residual_length,
            )

            mv_length = min(
                raw_mv.shape[0],
                self.max_mv_length,
            )

            # -----------------------------------------------------------
            # Copy actual samples into the padded arrays.
            # -----------------------------------------------------------

            if iframe_length > 0:

                iframe[
                    i,
                    :iframe_length,
                ] = raw_iframe[
                    :iframe_length
                ].numpy()

                iframe_mask[
                    i,
                    :iframe_length
                ] = 1

            if residual_length > 0:

                residuals[
                    i,
                    :residual_length,
                ] = raw_residuals[
                    :residual_length
                ].numpy()

                residuals_mask[
                    i,
                    :residual_length
                ] = 1

            if mv_length > 0:

                mv[
                    i,
                    :mv_length,
                ] = raw_mv[
                    :mv_length
                ].numpy()

                mv_mask[
                    i,
                    :mv_length
                ] = 1

        return (
            torch.from_numpy(iframe),
            torch.from_numpy(iframe_mask),

            torch.from_numpy(residuals),
            torch.from_numpy(residuals_mask),

            torch.from_numpy(mv),
            torch.from_numpy(mv_mask),
        )

    # ==================================================================
    # DATASET ENTRY POINT
    # ==================================================================

    def __getitem__(self, idx):

        # ---------------------------------------------------------------
        # Determine video/caption according to the selected caption mode.
        # ---------------------------------------------------------------

        if self.unfold_sentences:

            video_id, caption = (
                self.sentences_dict[idx]
            )

        else:

            video_id = (
                self.csv["video_id"].values[idx]
            )

            caption = None

        # ---------------------------------------------------------------
        # Text preprocessing.
        # ---------------------------------------------------------------

        (
            pairs_text,
            pairs_mask,
            pairs_segment,
            choice_video_ids,
        ) = self._get_text(
            video_id,
            caption,
        )

        # ---------------------------------------------------------------
        # Compressed-domain video preprocessing.
        # ---------------------------------------------------------------

        (
            iframe,
            iframe_mask,

            residuals,
            residuals_mask,

            mv,
            mv_mask,

        ) = self._get_compressed_video(
            choice_video_ids
        )

        # ---------------------------------------------------------------
        # Final sample.
        #
        # No generic "video" tensor exists anymore.
        # Each modality is explicitly represented.
        # ---------------------------------------------------------------

        return (
            pairs_text,
            pairs_mask,
            pairs_segment,

            iframe,
            iframe_mask,

            residuals,
            residuals_mask,

            mv,
            mv_mask,
        )