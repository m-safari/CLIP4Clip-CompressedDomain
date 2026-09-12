import torch
from torch.utils.data import DataLoader
from .dataloader_msrvtt_compressed import MSRVTT_Compressed_DataLoader

def dataloader_msrvtt_train(args, tokenizer):
    msrvtt_dataset = MSRVTT_Compressed_DataLoader(
        csv_path=args.train_csv,
        json_path=args.data_path,
        features_path=args.features_path,
        max_words=args.max_words,
        tokenizer=tokenizer,
        unfold_sentences=args.expand_msrvtt_sentences,
    )

    dataloader = DataLoader(
        msrvtt_dataset,
        batch_size=args.batch_size,
        pin_memory=False,
        shuffle=(train_sampler is None),
        drop_last=True,
    )

    return dataloader, len(msrvtt_dataset)


def dataloader_msrvtt_test(args, tokenizer):
    msrvtt_dataset = MSRVTT_Compressed_DataLoader(
       csv_path=args.val_csv,
        json_path=args.data_path,
        features_path=args.features_path,
        max_words=args.max_words,
        tokenizer=tokenizer,
    )

    dataloader = DataLoader(
        msrvtt_dataset,
        batch_size=args.batch_size,
        pin_memory=False,
        shuffle=(train_sampler is None),
        drop_last=True,
    )

    return dataloader, len(msrvtt_dataset)


DATALOADER_DICT = {}
DATALOADER_DICT["msrvtt"] = {"train":dataloader_msrvtt_train, "val":dataloader_msrvtt_test, "test":None}