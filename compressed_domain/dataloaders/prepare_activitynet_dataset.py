"""Download ActivityNet Captions and prepare full-video paragraph retrieval.

Official captions: https://cs.stanford.edu/people/ranjaykrishna/densevid/
Videos are fetched from their YouTube IDs using optional yt-dlp, or supplied
locally with --videos-dir. Unavailable sources/failed encodes are reported.
No dataset is downloaded merely by importing this module.
"""
import argparse
import csv
import json
import importlib.util
import shutil
import random
import subprocess
import sys
from pathlib import Path
from urllib.request import urlretrieve
from zipfile import ZipFile

try:
    from .activitynet_annotations import index_videos, load_paragraphs
except ImportError:  # also support python dataloaders/prepare_activitynet_dataset.py
    from activitynet_annotations import index_videos, load_paragraphs

CAPTIONS_URL = "https://cs.stanford.edu/people/ranjaykrishna/densevid/captions.zip"


def download_annotations(root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    if all((root / name).exists() for name in ("train.json", "val_1.json")):
        return root
    archive = root / "captions.zip"
    if not archive.exists():
        partial = archive.with_suffix(".zip.partial")
        urlretrieve(CAPTIONS_URL, partial)
        partial.replace(archive)
    with ZipFile(archive) as zipped:
        for name in ("train.json", "val_1.json", "val_2.json"):
            members = [m for m in zipped.namelist() if Path(m).name == name]
            if not members and name != "val_2.json":
                raise ValueError(f"Caption archive is missing {name}")
            if members:
                # Copy only named annotation files, never extract arbitrary archive paths.
                (root / name).write_bytes(zipped.read(members[0]))
    return root


def download_video(video_id, output_dir):
    """Fetch one source with yt-dlp; subprocess errors are recorded by the caller."""
    youtube_id = video_id[2:] if video_id.startswith("v_") else video_id
    subprocess.run([
        sys.executable, "-m", "yt_dlp", "--no-playlist", "--no-progress",
        "--merge-output-format", "mp4", "-f", "bv*+ba/b",
        "-o", str(Path(output_dir) / (video_id + ".%(ext)s")),
        "--", "https://www.youtube.com/watch?v=" + youtube_id,
    ], check=True)
    candidates = [p for p in Path(output_dir).glob(video_id + ".*")
                  if p.suffix.lower() in (".mp4", ".mkv", ".webm", ".mov", ".avi")]
    if len(candidates) != 1:
        raise RuntimeError(f"Expected one downloaded video for {video_id}, found {candidates}")
    return candidates[0]


def reencode_video(source, destination):
    """MPEG-4 Part 2 elementary stream, fixed 12-frame GOPs, no B-frames for CoViAR."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size:
        return destination
    partial = destination.with_suffix(".partial")
    try:
        subprocess.run([
            "ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", str(source),
            "-an", "-vf", "scale=340:256,setsar=1:1", "-c:v", "mpeg4",
            "-q:v", "1", "-g", "12", "-bf", "0", "-sc_threshold", "0",
            "-f", "m4v", str(partial),
        ], check=True)
        if not partial.exists() or not partial.stat().st_size:
            raise RuntimeError(f"ffmpeg produced no video for {source}")
        partial.replace(destination)
    finally:
        partial.unlink(missing_ok=True)
    return destination


def write_splits(annotations_dir, video_dir, output_dir, num_val=1000, seed=42):
    """Reserve validation from train; use val_1 as held-out retrieval test.

    val_1 and val_2 may annotate the same videos: they are NOT independent
    validation/test sets. Only successfully prepared videos enter these CSVs.
    The merged JSON allows all three CSVs to share the same --data_path.
    """
    annotations_dir, output_dir = Path(annotations_dir), Path(output_dir)
    train = load_paragraphs(annotations_dir / "train.json")
    test = load_paragraphs(annotations_dir / "val_1.json")
    overlap = train.keys() & test.keys()
    if overlap:
        raise ValueError(f"Training and held-out annotations overlap: {sorted(overlap)[:5]}")
    available = index_videos(video_dir)
    train_ids = sorted(train.keys() & available.keys())
    test_ids = sorted(test.keys() & available.keys())
    if not 0 < num_val < len(train_ids):
        raise ValueError(f"num_val must leave nonempty train/val sets; found {len(train_ids)} train videos")
    if not test_ids:
        raise ValueError("No held-out val_1 videos were prepared")
    shuffled = train_ids.copy()
    random.Random(seed).shuffle(shuffled)
    val = set(shuffled[:num_val])
    splits = {"train": [v for v in train_ids if v not in val],
              "val": [v for v in train_ids if v in val], "test": test_ids}
    output_dir.mkdir(parents=True, exist_ok=True)
    paragraphs = {**train, **test}
    for split, ids in splits.items():
        with open(output_dir / f"ActivityNet_{split}.csv", "w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["video_id", "sentence"])
            writer.writerows((v, paragraphs[v]) for v in ids)
    # Keep official annotation structure so the loader can also read official JSONs directly.
    merged = {}
    for filename in ("train.json", "val_1.json"):
        with open(annotations_dir / filename) as handle:
            merged.update(json.load(handle))
    (output_dir / "ActivityNet_data.json").write_text(json.dumps(merged))
    report = {"seed": seed, "test_annotations": "val_1.json",
              "counts": {s: len(ids) for s, ids in splits.items()},
              "missing_train": sorted(train.keys() - available.keys()),
              "missing_test": sorted(test.keys() - available.keys())}
    (output_dir / "preparation_report.json").write_text(json.dumps(report, indent=2))
    return splits


def prepare_activitynet_dataset(download_path="/tmp/activitynet", videos_dir=None,
                                annotations_dir=None, num_val=1000, seed=42,
                                max_videos_per_split=None):
    if num_val < 1:
        raise ValueError("num_val must be positive")
    if max_videos_per_split is not None and max_videos_per_split < 1:
        raise ValueError("max_videos_per_split must be positive")
    if not shutil.which("ffmpeg"):
        raise RuntimeError("Install ffmpeg before preparing ActivityNet videos")
    if videos_dir is None and importlib.util.find_spec("yt_dlp") is None:
        raise RuntimeError("Install yt-dlp (pip install yt-dlp), or pass --videos-dir for local sources")
    root = Path(download_path)
    root.mkdir(parents=True, exist_ok=True)
    annotations = Path(annotations_dir) if annotations_dir else download_annotations(root / "annotations")
    source_dir = Path(videos_dir) if videos_dir else root / "videos"
    if videos_dir and not source_dir.is_dir():
        raise ValueError(f"Video directory does not exist: {source_dir}")
    source_dir.mkdir(parents=True, exist_ok=True)
    sources = index_videos(source_dir)
    output = root / "mpeg4_videos"
    output.mkdir(exist_ok=True)
    failures = []
    for filename in ("train.json", "val_1.json"):
        ids = sorted(load_paragraphs(annotations / filename))
        if max_videos_per_split:
            ids = ids[:max_videos_per_split]
        for index, video_id in enumerate(ids):
            destination = output / (video_id + ".mp4")
            if destination.exists() and destination.stat().st_size:
                continue
            try:
                source = sources.get(video_id)
                if source is None:
                    if videos_dir:
                        raise FileNotFoundError(f"No local source for {video_id}")
                    source = download_video(video_id, source_dir)
                reencode_video(source, destination)
            except (OSError, subprocess.SubprocessError, RuntimeError) as error:
                failures.append({"video_id": video_id, "error": str(error)})
                print(f"WARNING: {video_id}: {error}")
            if (index + 1) % 100 == 0:
                print(f"{filename}: processed {index + 1}/{len(ids)}")
    (root / "failures.json").write_text(json.dumps(failures, indent=2))
    splits_dir = root / "splits"
    splits = write_splits(annotations, output, splits_dir, num_val, seed)
    print(f"Prepared split sizes: { {s: len(v) for s, v in splits.items()} }")
    print(f"Videos: {output}\nSplits and merged annotations: {splits_dir}")
    return output, splits_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download-path", default="/tmp/activitynet", type=Path)
    parser.add_argument("--videos-dir", type=Path, help="Reuse local videos instead of downloading YouTube sources")
    parser.add_argument("--annotations-dir", type=Path, help="Reuse official train.json and val_1.json")
    parser.add_argument("--num-val", type=int, default=1000, help="Videos reserved from training for model selection")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-videos-per-split", type=int, help="Preparation smoke-test limit; also reduce --num-val")
    prepare_activitynet_dataset(**vars(parser.parse_args()))
