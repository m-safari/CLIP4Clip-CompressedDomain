"""
Carve a validation set out of the MSR-VTT training videos.

The split is made by VIDEO, never by caption, so no video (and none of its 20
captions) appears on both sides. The JSFUSION test set is not touched: it stays
held out for the final --do_eval.

    python -m dataloaders.make_msrvtt_val_split \
        --train_csv /storage/data/msrvtt/raw_data/MSRVTT_train.9k.csv \
        --data_path /storage/data/msrvtt/raw_data/MSRVTT_data.json

writes, next to the training CSV:

    MSRVTT_train.8k.csv   video_id                         -> --train_csv
    MSRVTT_val.1k.csv     key,vid_key,video_id,sentence    -> --val_csv

The validation CSV has the same columns as MSRVTT_JSFUSION_test.csv: one fixed
query caption per video (its first caption by sen_id), so validation metrics
are deterministic and directly comparable between epochs and runs.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd


def make_split(train_csv, data_path, num_val=1000, seed=42, out_dir=None, test_csv=None):
    out_dir = out_dir or os.path.dirname(os.path.abspath(train_csv))
    video_ids = pd.read_csv(train_csv)["video_id"].drop_duplicates().tolist()
    if not 0 < num_val < len(video_ids):
        raise ValueError("num_val must be between 1 and {}".format(len(video_ids) - 1))

    if test_csv:
        overlap = set(video_ids) & set(pd.read_csv(test_csv)["video_id"])
        if overlap:
            raise ValueError("{} training videos are also in the test CSV, e.g. {}".format(
                len(overlap), sorted(overlap)[:5]))

    order = np.random.default_rng(seed).permutation(len(video_ids))
    val_ids = sorted((video_ids[i] for i in order[:num_val]), key=video_ids.index)
    val_set = set(val_ids)
    train_ids = [v for v in video_ids if v not in val_set]

    with open(data_path, "r") as f:
        sentences = json.load(f)["sentences"]
    first_caption = {}
    for item in sorted(sentences, key=lambda s: s.get("sen_id", 0)):
        first_caption.setdefault(item["video_id"], item["caption"])
    missing = [v for v in val_ids if v not in first_caption]
    if missing:
        raise ValueError("{} validation videos have no caption in {}, e.g. {}".format(
            len(missing), data_path, missing[:5]))

    def name(count, kind):
        size = "{}k".format(count // 1000) if count % 1000 == 0 else str(count)
        return os.path.join(out_dir, "MSRVTT_{}.{}.csv".format(kind, size))

    train_out, val_out = name(len(train_ids), "train"), name(len(val_ids), "val")
    pd.DataFrame({"video_id": train_ids}).to_csv(train_out, index=False)
    pd.DataFrame({
        "key": ["val{}".format(i) for i in range(len(val_ids))],
        "vid_key": [v.replace("video", "msr") for v in val_ids],
        "video_id": val_ids,
        "sentence": [first_caption[v] for v in val_ids],
    }).to_csv(val_out, index=False)
    return train_out, val_out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split MSR-VTT training videos into train / validation.")
    parser.add_argument("--train_csv", required=True, help="MSRVTT_train.9k.csv")
    parser.add_argument("--data_path", required=True, help="MSRVTT_data.json")
    parser.add_argument("--test_csv", default=None, help="Optional: fail if any training video is in this CSV")
    parser.add_argument("--num_val", type=int, default=1000, help="Number of validation videos")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_dir", default=None, help="Default: the directory of --train_csv")
    args = parser.parse_args()

    train_out, val_out = make_split(args.train_csv, args.data_path, args.num_val, args.seed, args.out_dir,
                                    args.test_csv)
    print("train: {} ({} videos)".format(train_out, len(pd.read_csv(train_out))))
    print("val:   {} ({} videos)".format(val_out, len(pd.read_csv(val_out))))
