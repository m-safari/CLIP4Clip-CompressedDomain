"""
Download MSVD and prepare it for the compressed-domain pipeline.

Sources:

    msvd_data.zip      CLIP4Clip release v0.0: train_list.txt, val_list.txt,
                       test_list.txt and raw-captions.pkl (the splits of
                       collaborative-experts: 1200 / 100 / 670 videos)
    MSVD_Videos.zip    the 1970 original YouTube clips, <id>.avi, from the
                       Hugging Face mirror friedrichor/MSVD (the same uploader
                       as the MSR-VTT mirror prepare_msrvtt_dataset.py uses)

Steps, each skipped when its output is already on disk:

    1. download both archives (or reuse ones already in --download-path);
       with --videos-dir, the raw videos already on disk are used instead
       of MSVD_Videos.zip
    2. extract them into <download-path>/msvd_data and <download-path>/videos
    3. re-encode every video to CoViAR's MPEG-4 GOP layout with reencode.sh,
       into <download-path>/mpeg4_videos/<id>.mp4

Afterwards, train / evaluate with

    --datatype msvd --data_path <download-path>/msvd_data --features_path <download-path>/mpeg4_videos

Usage:
    python dataloaders/prepare_msvd_dataset.py --download-path /tmp/msvd
    # raw videos already on disk
    python dataloaders/prepare_msvd_dataset.py --download-path /tmp/msvd --videos-dir /data/YouTubeClips
    # smoke test: only the first 20 videos
    python dataloaders/prepare_msvd_dataset.py --download-path /tmp/msvd --max-videos 20
"""
import argparse
import shutil
import subprocess
from pathlib import Path
from urllib.request import urlretrieve
from zipfile import ZipFile

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DOWNLOAD_PATH = "/tmp/msvd"

DATA_URL = "https://github.com/ArrowLuo/CLIP4Clip/releases/download/v0.0/msvd_data.zip"
HF_REPO_ID = "friedrichor/MSVD"
VIDEOS_ARCHIVE = "MSVD_Videos.zip"
DATA_FILES = ("train_list.txt", "val_list.txt", "test_list.txt", "raw-captions.pkl")
RAW_EXTENSIONS = (".avi", ".mp4", ".mkv", ".webm")


# ---------------------------------------------------------------------- #
# Download
# ---------------------------------------------------------------------- #

def download(url, download_path):
    """Fetch `url` into download_path, unless a file of that name is already there."""
    target = Path(download_path) / url.rsplit("/", 1)[-1]
    if target.exists():
        print(f"Reusing {target}")
        return target
    print(f"Downloading {url} ...")
    partial = target.with_name(target.name + ".part")
    urlretrieve(url, partial)
    partial.rename(target)
    return target


def download_videos_archive(download_path):
    """Fetch MSVD_Videos.zip from the Hub, unless it is already in download_path."""
    target = Path(download_path) / VIDEOS_ARCHIVE
    if target.exists():
        print(f"Reusing {target}")
        return target

    from huggingface_hub import hf_hub_download
    path = hf_hub_download(repo_id=HF_REPO_ID, repo_type="dataset", filename=VIDEOS_ARCHIVE,
                           local_dir=str(download_path))
    return Path(path)


# ---------------------------------------------------------------------- #
# Extraction
# ---------------------------------------------------------------------- #

def find_data_dir(root):
    """The one directory under root that holds every file in DATA_FILES."""
    found = sorted({p.parent for p in Path(root).rglob("raw-captions.pkl")
                    if all((p.parent / name).exists() for name in DATA_FILES)})
    if len(found) != 1:
        raise RuntimeError(f"Expected one directory with {DATA_FILES} under {root}, found {found}")
    return found[0]


def extract_data(data_archive, data_dir):
    """msvd_data.zip -> <data_dir>/{train,val,test}_list.txt, raw-captions.pkl."""
    data_dir = Path(data_dir)
    if all((data_dir / name).exists() for name in DATA_FILES):
        print(f"Reusing splits and captions in {data_dir}")
        return data_dir

    staging = data_dir.parent / "_msvd_data_zip"
    with ZipFile(data_archive) as z:
        z.extractall(staging)
    data_dir.mkdir(parents=True, exist_ok=True)
    source = find_data_dir(staging)
    for name in DATA_FILES:
        shutil.move(str(source / name), str(data_dir / name))
    shutil.rmtree(staging)
    return data_dir


def find_videos_dir(root):
    """The directory under root (root included) holding the most raw videos."""
    counts = {}
    for path in Path(root).rglob("*"):
        if path.suffix.lower() in RAW_EXTENSIONS and path.is_file():
            counts[path.parent] = counts.get(path.parent, 0) + 1
    if not counts:
        raise RuntimeError(f"No {RAW_EXTENSIONS} videos under {root}")
    return max(counts, key=counts.get)


def extract_videos(videos_archive, raw_dir):
    """MSVD_Videos.zip -> the directory of <id>.avi files."""
    raw_dir = Path(raw_dir)
    if raw_dir.exists() and any(raw_dir.rglob("*.avi")):
        print(f"Reusing extracted videos in {raw_dir}")
        return find_videos_dir(raw_dir)

    raw_dir.mkdir(parents=True, exist_ok=True)
    with ZipFile(videos_archive) as z:
        z.extractall(raw_dir)
    return find_videos_dir(raw_dir)


# ---------------------------------------------------------------------- #
# Re-encoding
# ---------------------------------------------------------------------- #

def reencode_videos(raw_dir, mpeg4_dir, max_videos=None, encoder_path=None):
    """Re-encode the videos of raw_dir into <mpeg4_dir>/<id>.mp4 with reencode.sh."""
    encoder_path = Path(encoder_path or SCRIPT_DIR / "reencode.sh")
    command = ["bash", str(encoder_path), str(raw_dir), str(mpeg4_dir)]
    if max_videos is not None:
        command.append(str(max_videos))
    print(f"Re-encoding {raw_dir} ...")
    subprocess.run(command, check=True)
    return Path(mpeg4_dir)


# ---------------------------------------------------------------------- #
# Entry point
# ---------------------------------------------------------------------- #

def prepare_msvd_dataset(download_path=DEFAULT_DOWNLOAD_PATH, videos_dir=None, max_videos=None):
    download_path = Path(download_path)
    download_path.mkdir(parents=True, exist_ok=True)

    data_dir = extract_data(download(DATA_URL, download_path), download_path / "msvd_data")

    if videos_dir is not None:
        raw_dir = find_videos_dir(videos_dir)
    else:
        raw_dir = extract_videos(download_videos_archive(download_path), download_path / "videos")

    mpeg4_dir = reencode_videos(raw_dir, download_path / "mpeg4_videos", max_videos)

    print(f"Splits and captions (--data_path):  {data_dir}")
    print(f"MPEG-4 videos (--features_path):    {mpeg4_dir}")
    return data_dir, mpeg4_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and prepare the MSVD dataset.")
    parser.add_argument("--download-path", type=Path, default=DEFAULT_DOWNLOAD_PATH,
                        help=f"Dataset directory (default: {DEFAULT_DOWNLOAD_PATH})")
    parser.add_argument("--videos-dir", type=Path, default=None,
                        help="Raw MSVD videos already on disk (skips downloading MSVD_Videos.zip)")
    parser.add_argument("--max-videos", type=int, default=None,
                        help="Re-encode at most this many videos (for smoke tests)")
    args = parser.parse_args()

    prepare_msvd_dataset(args.download_path, args.videos_dir, args.max_videos)
