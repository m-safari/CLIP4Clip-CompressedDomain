import torch
from torch.utils.data import DataLoader, Sampler
from .dataloader_msrvtt_compressed import MSRVTT_Compressed_DataLoader

# Positions of (frames, mask) for each modality in a dataset item:
# (text, text_mask, segment, iframe, iframe_mask, residuals, residuals_mask, mv, mv_mask)
_MODALITY_SLOTS = ((3, 4), (5, 6), (7, 8))


def trim_padding_collate(items):
    """
    Default collation, then drop the trailing frame positions that are padding
    for every video in the batch (at least one position is always kept).
    Padded positions never reach the encoders' output, so this changes nothing
    but the amount of memory moved around.
    """
    batch = list(torch.utils.data.default_collate(items))
    for frames_slot, mask_slot in _MODALITY_SLOTS:
        mask = batch[mask_slot]                               # (B, P, T)
        used = mask.reshape(-1, mask.shape[-1]).any(dim=0).nonzero()
        keep = int(used.max()) + 1 if len(used) else 1
        batch[frames_slot] = batch[frames_slot][:, :, :keep]  # (B, P, T, C, H, W)
        batch[mask_slot] = mask[..., :keep]
    return tuple(batch)


class EpochSubsetSampler(Sampler):
    """
    A shuffled pass over the dataset whose order depends only on (seed, epoch).

    That makes an epoch reproducible across sessions: resuming at epoch k draws
    exactly the samples epoch k would have drawn in an uninterrupted run, and
    every epoch draws a different order. With num_samples < len(dataset) each
    epoch is a different random subset of that size, which is how an "epoch" is
    sized to fit a time-limited session without always reusing the same rows.

    Call set_epoch(epoch) before iterating.
    """

    def __init__(self, data_len, num_samples=None, seed=0):
        self.data_len = data_len
        self.num_samples = data_len if not num_samples else min(num_samples, data_len)
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed * 1000003 + self.epoch)
        return iter(torch.randperm(self.data_len, generator=g)[:self.num_samples].tolist())

    def __len__(self):
        return self.num_samples


def _msrvtt_dataset(args, tokenizer, csv_path, unfold_sentences):
    return MSRVTT_Compressed_DataLoader(
        csv_path=csv_path,
        json_path=args.data_path,
        features_path=args.features_path,
        max_words=args.max_words,
        tokenizer=tokenizer,
        unfold_sentences=unfold_sentences,
        residual_sampling_rate=args.residual_per_gop,
        mv_sampling_rate=args.mv_per_gop,
        max_gops=args.max_frames,
        gop_slice=args.slice_framepos,
    )


def dataloader_msrvtt_train(args, tokenizer):
    msrvtt_dataset = _msrvtt_dataset(args, tokenizer, args.train_csv, args.expand_msrvtt_sentences)

    sampler = EpochSubsetSampler(len(msrvtt_dataset), num_samples=args.max_train_samples, seed=args.seed)

    dataloader = DataLoader(
        msrvtt_dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_thread_reader,
        pin_memory=True,
        drop_last=True,
        collate_fn=trim_padding_collate,
    )

    return dataloader, len(sampler)


def dataloader_msrvtt_eval(args, tokenizer, csv_path):
    """
    One item per CSV row, in CSV order, nothing dropped. If the CSV has a
    `sentence` column it is the query caption of that row's video.
    """
    msrvtt_dataset = _msrvtt_dataset(args, tokenizer, csv_path, unfold_sentences=False)

    dataloader = DataLoader(
        msrvtt_dataset,
        batch_size=args.batch_size_val,
        shuffle=False,
        num_workers=args.num_thread_reader,
        pin_memory=True,
        drop_last=False,
        collate_fn=trim_padding_collate,
    )

    return dataloader, len(msrvtt_dataset)


def dataloader_msrvtt_val(args, tokenizer):
    return dataloader_msrvtt_eval(args, tokenizer, args.val_csv)


def dataloader_msrvtt_test(args, tokenizer):
    return dataloader_msrvtt_eval(args, tokenizer, args.test_csv)


DATALOADER_DICT = {}
DATALOADER_DICT["msrvtt"] = {"train": dataloader_msrvtt_train, "val": dataloader_msrvtt_val,
                             "test": dataloader_msrvtt_test}


def _activitynet_loader(args, tokenizer, subset):
    from .dataloader_activitynet_compressed import ActivityNet_Compressed_DataLoader

    dataset = ActivityNet_Compressed_DataLoader(
        csv_path=getattr(args, subset + "_csv"), data_path=args.data_path,
        features_path=args.features_path, tokenizer=tokenizer, subset=subset,
        max_words=args.max_words, max_gops=args.max_frames, gop_slice=args.slice_framepos,
        residual_sampling_rate=args.residual_per_gop, mv_sampling_rate=args.mv_per_gop)
    train = subset == "train"
    sampler = EpochSubsetSampler(len(dataset), args.max_train_samples, args.seed) if train else None
    loader = DataLoader(dataset, batch_size=args.batch_size if train else args.batch_size_val,
                        sampler=sampler, shuffle=False, num_workers=args.num_thread_reader,
                        pin_memory=True, drop_last=train, collate_fn=trim_padding_collate)
    return loader, len(sampler) if train else len(dataset)


def dataloader_activitynet_train(args, tokenizer):
    return _activitynet_loader(args, tokenizer, "train")


def dataloader_activitynet_val(args, tokenizer):
    return _activitynet_loader(args, tokenizer, "val")


def dataloader_activitynet_test(args, tokenizer):
    return _activitynet_loader(args, tokenizer, "test")


DATALOADER_DICT["activitynet"] = {"train": dataloader_activitynet_train,
                                  "val": dataloader_activitynet_val,
                                  "test": dataloader_activitynet_test}
