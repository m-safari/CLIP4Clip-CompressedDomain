"""
Download HMDB51 and prepare it for the compressed-domain pipeline.

Source: the official Serre Lab release on the Hugging Face Hub
(dataset Serrelab/hmdb51, CC-BY-4.0):

    hmdb51_org.rar          a RAR holding one RAR per action class
                            (<class>.rar -> <class>/<video>.avi)
    test_train_splits.rar   testTrainMulti_7030_splits/<class>_test_split{1,2,3}.txt

Steps, each skipped when its output is already on disk:

    1. download both archives (or reuse ones already in --download-path)
    2. extract them with whichever of unrar / 7z / bsdtar is installed
    3. re-encode every class directory to CoViAR's MPEG-4 GOP layout with
       reencode.sh, into <download-path>/mpeg4_videos/<class>/<video>.mp4
    4. write <download-path>/hmdb51_index.csv: one row per re-encoded video,
       video_id ("<class>/<stem>", the path under mpeg4_videos without .mp4),
       label, label_index, and split1..split3 (1 = train, 2 = test, 0 = unused,
       as in the official split files)

Usage:
    python dataloaders/prepare_hmdb51_dataset.py --download-path /tmp/hmdb51
    # smoke test: only the first 3 videos of each class
    python dataloaders/prepare_hmdb51_dataset.py --download-path /tmp/hmdb51 --max-videos-per-class 3
"""
import argparse
import csv
import shutil
import subprocess
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DOWNLOAD_PATH = "/tmp/hmdb51"

HF_REPO_ID = "Serrelab/hmdb51"
VIDEOS_ARCHIVE = "hmdb51_org.rar"
SPLITS_ARCHIVE = "test_train_splits.rar"
NUM_SPLITS = 3

# The 51 action classes, in the order used for label_index (alphabetical, which
# is also the order of the official per-class archives and split files).
HMDB51_CLASSES = (
    "brush_hair", "cartwheel", "catch", "chew", "clap", "climb", "climb_stairs", "dive",
    "draw_sword", "dribble", "drink", "eat", "fall_floor", "fencing", "flic_flac", "golf",
    "handstand", "hit", "hug", "jump", "kick", "kick_ball", "kiss", "laugh", "pick", "pour",
    "pullup", "punch", "push", "pushup", "ride_bike", "ride_horse", "run", "shake_hands",
    "shoot_ball", "shoot_bow", "shoot_gun", "sit", "situp", "smile", "smoke", "somersault",
    "stand", "swing_baseball", "sword", "sword_exercise", "talk", "throw", "turn", "walk", "wave",
)
assert len(HMDB51_CLASSES) == 51


# ---------------------------------------------------------------------- #
# Download
# ---------------------------------------------------------------------- #

def download_archive(filename, download_path):
    """Fetch one archive from the Hub, unless it is already in download_path."""
    target = Path(download_path) / filename
    if target.exists():
        print(f"Reusing {target}")
        return target

    from huggingface_hub import hf_hub_download
    path = hf_hub_download(repo_id=HF_REPO_ID, repo_type="dataset", filename=filename,
                           local_dir=str(download_path))
    return Path(path)


# ---------------------------------------------------------------------- #
# RAR extraction
# ---------------------------------------------------------------------- #

def rar_extract_command(archive, dest):
    """
    The command that extracts `archive` into `dest`, using the first available
    of unrar, 7z / 7za, bsdtar. Raises RuntimeError when none is installed.
    """
    archive, dest = str(archive), str(dest)
    if shutil.which("unrar"):
        return ["unrar", "x", "-o+", "-idq", archive, dest + "/"]
    for seven_zip in ("7z", "7za"):
        if shutil.which(seven_zip):
            return [seven_zip, "x", "-y", "-bd", "-o" + dest, archive]
    if shutil.which("bsdtar"):
        return ["bsdtar", "-xf", archive, "-C", dest]
    raise RuntimeError(
        "No RAR extractor found. Install one of: unrar, 7z (p7zip-full), bsdtar (libarchive-tools).")


def extract_rar(archive, dest):
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    subprocess.run(rar_extract_command(archive, dest), check=True)
    return dest


def extract_videos(videos_archive, raw_dir):
    """
    hmdb51_org.rar -> <raw_dir>/<class>/<video>.avi.

    The outer archive holds one <class>.rar per class; each of those is
    extracted and then removed, so only the .avi files are left.
    """
    raw_dir = Path(raw_dir)
    if raw_dir.exists() and all((raw_dir / c).is_dir() for c in HMDB51_CLASSES):
        print(f"Reusing extracted videos in {raw_dir}")
        return raw_dir

    class_archives_dir = raw_dir / "_class_archives"
    extract_rar(videos_archive, class_archives_dir)

    for class_archive in sorted(class_archives_dir.rglob("*.rar")):
        extract_rar(class_archive, raw_dir)
    shutil.rmtree(class_archives_dir)

    missing = [c for c in HMDB51_CLASSES if not (raw_dir / c).is_dir()]
    if missing:
        raise RuntimeError(f"Extraction left no directory for classes: {missing}")
    return raw_dir


def extract_splits(splits_archive, splits_dir):
    """test_train_splits.rar -> the directory holding the <class>_test_splitN.txt files."""
    splits_dir = Path(splits_dir)
    if not any(splits_dir.rglob("*_test_split1.txt")):
        extract_rar(splits_archive, splits_dir)
    found = sorted({p.parent for p in splits_dir.rglob("*_test_split1.txt")})
    if len(found) != 1:
        raise RuntimeError(f"Expected one directory of split files under {splits_dir}, found {found}")
    return found[0]


# ---------------------------------------------------------------------- #
# Re-encoding
# ---------------------------------------------------------------------- #

def reencode_classes(raw_dir, mpeg4_dir, max_videos_per_class=None, encoder_path=None):
    """Re-encode <raw_dir>/<class>/*.avi into <mpeg4_dir>/<class>/*.mp4 with reencode.sh."""
    encoder_path = Path(encoder_path or SCRIPT_DIR / "reencode.sh")
    for class_name in HMDB51_CLASSES:
        src = Path(raw_dir) / class_name
        dst = Path(mpeg4_dir) / class_name
        command = ["bash", str(encoder_path), str(src), str(dst)]
        if max_videos_per_class is not None:
            command.append(str(max_videos_per_class))
        print(f"Re-encoding {class_name}...")
        subprocess.run(command, check=True)
    return Path(mpeg4_dir)


# ---------------------------------------------------------------------- #
# Splits + index
# ---------------------------------------------------------------------- #

def read_split_file(path):
    """
    Parse one official split file: lines of "<video>.avi <tag>", where tag is
    1 (train), 2 (test) or 0 (not used in this split). Returns {stem: tag}.
    """
    assignment = {}
    with open(path) as f:
        for line in f:
            parts = line.split()
            if not parts:
                continue
            if len(parts) != 2 or parts[1] not in ("0", "1", "2"):
                raise ValueError(f"{path}: unexpected line {line!r}")
            assignment[Path(parts[0]).stem] = int(parts[1])
    return assignment


def read_splits(splits_dir):
    """{(class, stem): [tag_split1, tag_split2, tag_split3]} from the official split files."""
    splits = {}
    for class_name in HMDB51_CLASSES:
        for split in range(1, NUM_SPLITS + 1):
            path = Path(splits_dir) / f"{class_name}_test_split{split}.txt"
            for stem, tag in read_split_file(path).items():
                splits.setdefault((class_name, stem), [0] * NUM_SPLITS)[split - 1] = tag
    return splits


def build_index(mpeg4_dir, splits_dir, index_path):
    """
    Write the CSV the HMDB51 dataloader reads: one row per re-encoded video.
    Videos absent from the split files get split tags 0 (unused).
    """
    splits = read_splits(splits_dir) if splits_dir is not None else {}
    rows = []
    for label_index, class_name in enumerate(HMDB51_CLASSES):
        for video in sorted((Path(mpeg4_dir) / class_name).glob("*.mp4")):
            tags = splits.get((class_name, video.stem), [0] * NUM_SPLITS)
            rows.append([f"{class_name}/{video.stem}", class_name, label_index, *tags])

    with open(index_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["video_id", "label", "label_index"] + [f"split{s}" for s in range(1, NUM_SPLITS + 1)])
        writer.writerows(rows)
    return Path(index_path), len(rows)


# ---------------------------------------------------------------------- #
# Entry point
# ---------------------------------------------------------------------- #

def prepare_hmdb51_dataset(download_path=DEFAULT_DOWNLOAD_PATH, max_videos_per_class=None):
    download_path = Path(download_path)
    download_path.mkdir(parents=True, exist_ok=True)

    videos_archive = download_archive(VIDEOS_ARCHIVE, download_path)
    splits_archive = download_archive(SPLITS_ARCHIVE, download_path)

    raw_dir = extract_videos(videos_archive, download_path / "videos")
    splits_dir = extract_splits(splits_archive, download_path / "splits")

    mpeg4_dir = reencode_classes(raw_dir, download_path / "mpeg4_videos", max_videos_per_class)
    index_path, n = build_index(mpeg4_dir, splits_dir, download_path / "hmdb51_index.csv")

    print(f"MPEG-4 videos: {mpeg4_dir}")
    print(f"Index ({n} videos): {index_path}")
    return mpeg4_dir, index_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and prepare the HMDB51 dataset.")
    parser.add_argument("--download-path", type=Path, default=DEFAULT_DOWNLOAD_PATH,
                        help=f"Dataset directory (default: {DEFAULT_DOWNLOAD_PATH})")
    parser.add_argument("--max-videos-per-class", type=int, default=None,
                        help="Re-encode at most this many videos per class (for smoke tests)")
    args = parser.parse_args()

    prepare_hmdb51_dataset(args.download_path, args.max_videos_per_class)
