"""
Prepare ActivityNet Captions for the compressed-domain pipeline.

Sources (Hugging Face mirror friedrichor/ActivityNet_Captions, the same
uploader as the MSR-VTT mirror prepare_msrvtt_dataset.py uses):

    raw_data/train.json, val_1.json,      the official ActivityNet Captions
    train_ids.json, val_ids.json          annotations (Krishna et al.), as in
                                          Stanford's captions.zip
    ActivityNet_Videos.tar.part-000..007  the raw videos, ~42 GB; only fetched
                                          with --download-videos

Raw videos come from either
    --videos-dir DIR     a copy you already have (e.g. the official release,
                         http://activity-net.org/download.html), files named
                         v_<id>.<ext> or <id>.<ext> (.mp4 / .mkv / .webm), or
    --download-videos    the mirror's tar parts, downloaded and extracted into
                         <download-path>/videos (needs ~2x42 GB of disk).

Steps, each skipped when its output is already on disk:

    1. download the split / caption files into <download-path>/activitynet_data
    2. for every video of the chosen --subsets, find its raw file and link it
       into a staging directory
    3. re-encode the staged videos to CoViAR's MPEG-4 GOP layout with
       reencode.sh, into <download-path>/mpeg4_videos/v_<id>.mp4, and write
       <download-path>/missing_videos_<subset>.txt for videos not found

Only val (val_1.json, the split CLIP4Clip reports) is re-encoded by default;
add train to fine-tune.

Afterwards, train / evaluate with

    --datatype activity --data_path <download-path>/activitynet_data --features_path <download-path>/mpeg4_videos

Usage:
    python dataloaders/prepare_activitynet_dataset.py --download-path /tmp/activitynet --videos-dir /data/Activity_Videos
    python dataloaders/prepare_activitynet_dataset.py --download-path /tmp/activitynet --download-videos --subsets train val
    # smoke test: only the first 10 videos of each subset
    python dataloaders/prepare_activitynet_dataset.py --download-path /tmp/activitynet --videos-dir /data/Activity_Videos --max-videos 10
"""
import argparse
import io
import json
import os
import shutil
import subprocess
import tarfile
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DOWNLOAD_PATH = "/tmp/activitynet"

HF_REPO_ID = "friedrichor/ActivityNet_Captions"
HF_DATA_DIR = "raw_data"
DATA_FILES = ("train.json", "val_1.json", "train_ids.json", "val_ids.json")
VIDEO_PARTS = tuple("ActivityNet_Videos.tar.part-{:03d}".format(i) for i in range(8))
SUBSET_FILES = {"train": ("train_ids.json", "train.json"), "val": ("val_ids.json", "val_1.json")}
RAW_EXTENSIONS = (".mp4", ".mkv", ".webm", ".avi")


def _hf_download(filename, local_dir):
    from huggingface_hub import hf_hub_download
    return Path(hf_hub_download(repo_id=HF_REPO_ID, repo_type="dataset", filename=filename,
                                local_dir=str(local_dir)))


# ---------------------------------------------------------------------- #
# Captions and splits
# ---------------------------------------------------------------------- #

def download_captions(data_dir):
    """The official split / caption files -> <data_dir>/{train,val_1}.json, {train,val}_ids.json."""
    data_dir = Path(data_dir)
    if all((data_dir / name).exists() for name in DATA_FILES):
        print(f"Reusing captions and splits in {data_dir}")
        return data_dir

    staging = data_dir.parent / "_hf_raw_data"
    data_dir.mkdir(parents=True, exist_ok=True)
    for name in DATA_FILES:
        path = _hf_download(f"{HF_DATA_DIR}/{name}", staging)
        shutil.move(str(path), str(data_dir / name))
    shutil.rmtree(staging)
    return data_dir


def subset_video_ids(data_dir, subset):
    """The "v_<id>" videos of a subset that have captions, in ids-file order."""
    ids_name, captions_name = SUBSET_FILES[subset]
    with open(Path(data_dir) / captions_name) as f:
        captioned = json.load(f)
    with open(Path(data_dir) / ids_name) as f:
        ids = list(dict.fromkeys(json.load(f)))
    return [v for v in ids if v in captioned]


# ---------------------------------------------------------------------- #
# Raw videos
# ---------------------------------------------------------------------- #

class _ConcatenatedFiles(io.RawIOBase):
    """Read-only stream over several files back to back (`cat part-*`), for tarfile's stream mode."""

    def __init__(self, paths):
        self._paths = list(paths)
        self._current = None

    def readable(self):
        return True

    def readinto(self, buffer):
        while True:
            if self._current is None:
                if not self._paths:
                    return 0
                self._current = open(self._paths.pop(0), "rb")
            n = self._current.readinto(buffer)
            if n:
                return n
            self._current.close()
            self._current = None

    def close(self):
        if self._current is not None:
            self._current.close()
        super().close()


def download_videos(download_path):
    """
    The mirror's ActivityNet_Videos.tar.part-* -> <download-path>/videos, extracted
    as one tar stream (the parts are a split tar, not tars of their own). The
    parts are kept, so a failed extraction does not mean downloading again.
    """
    download_path = Path(download_path)
    raw_dir = download_path / "videos"
    done_marker = raw_dir / ".extracted"
    if done_marker.exists():
        print(f"Reusing extracted videos in {raw_dir}")
        return raw_dir

    parts = []
    for name in VIDEO_PARTS:
        target = download_path / name
        parts.append(target if target.exists() else _hf_download(name, download_path))

    raw_dir.mkdir(parents=True, exist_ok=True)
    # The "data" filter (Python >= 3.12, backported to 3.8.17+) refuses
    # absolute paths and links that escape raw_dir.
    extract_kwargs = {"filter": "data"} if hasattr(tarfile, "data_filter") else {}
    print(f"Extracting {len(parts)} parts into {raw_dir} ...")
    with tarfile.open(fileobj=io.BufferedReader(_ConcatenatedFiles(parts), 1 << 20), mode="r|") as tar:
        tar.extractall(raw_dir, **extract_kwargs)
    done_marker.touch()
    return raw_dir


def index_raw_videos(videos_dir):
    """{file stem: path} of every raw video under videos_dir (recursively)."""
    videos = {}
    for root, _, files in os.walk(videos_dir):
        for name in sorted(files):
            stem, ext = os.path.splitext(name)
            if ext.lower() in RAW_EXTENSIONS:
                videos.setdefault(stem, Path(root) / name)
    return videos


def match_raw_videos(video_ids, raw_index):
    """
    ({video_id: raw path}, [missing ids]). A video may be stored as v_<id> or <id>;
    either way the re-encoded file is named v_<id>, which the dataloader finds.
    """
    found, missing = {}, []
    for video_id in video_ids:
        stems = [video_id, video_id[2:]] if video_id.startswith("v_") else [video_id]
        path = next((raw_index[s] for s in stems if s in raw_index), None)
        if path is None:
            missing.append(video_id)
        else:
            found[video_id] = path
    return found, missing


def stage_videos(found, staging_dir):
    """
    Link each raw video into staging_dir as <video_id><ext>, so reencode.sh
    sees exactly the videos of the subset under the names the dataloader uses.
    Symlinks, or hard links where symlinks are not allowed.
    """
    staging_dir = Path(staging_dir)
    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    staging_dir.mkdir(parents=True)
    for video_id, path in found.items():
        link = staging_dir / (video_id + Path(path).suffix.lower())
        try:
            os.symlink(os.path.abspath(path), link)
        except OSError:
            os.link(path, link)
    return staging_dir


def reencode_videos(staging_dir, mpeg4_dir, max_videos=None, encoder_path=None):
    encoder_path = Path(encoder_path or SCRIPT_DIR / "reencode.sh")
    command = ["bash", str(encoder_path), str(staging_dir), str(mpeg4_dir)]
    if max_videos is not None:
        command.append(str(max_videos))
    subprocess.run(command, check=True)
    return Path(mpeg4_dir)


# ---------------------------------------------------------------------- #
# Entry point
# ---------------------------------------------------------------------- #

def prepare_activitynet_dataset(videos_dir=None, download_path=DEFAULT_DOWNLOAD_PATH, subsets=("val",),
                                max_videos=None, download_raw_videos=False):
    if (videos_dir is None) == (not download_raw_videos):
        raise ValueError("Give exactly one of videos_dir and download_raw_videos.")
    download_path = Path(download_path)
    download_path.mkdir(parents=True, exist_ok=True)

    data_dir = download_captions(download_path / "activitynet_data")
    if download_raw_videos:
        videos_dir = download_videos(download_path)
    mpeg4_dir = download_path / "mpeg4_videos"
    raw_index = index_raw_videos(videos_dir)
    print(f"{len(raw_index)} raw videos found in {videos_dir}")

    for subset in subsets:
        video_ids = subset_video_ids(data_dir, subset)
        found, missing = match_raw_videos(video_ids, raw_index)
        # Already re-encoded videos are not staged again.
        todo = {v: p for v, p in found.items() if not (mpeg4_dir / (v + ".mp4")).exists()}
        print(f"{subset}: {len(video_ids)} videos, {len(found)} on disk ({len(todo)} to re-encode), "
              f"{len(missing)} missing")

        missing_path = download_path / f"missing_videos_{subset}.txt"
        missing_path.write_text("".join(v + "\n" for v in missing))
        if missing:
            print(f"  missing video ids written to {missing_path}")

        if todo:
            staging = stage_videos(todo, download_path / f"_staging_{subset}")
            reencode_videos(staging, mpeg4_dir, max_videos)
            shutil.rmtree(staging)

    print(f"Captions and splits (--data_path):  {data_dir}")
    print(f"MPEG-4 videos (--features_path):    {mpeg4_dir}")
    return data_dir, mpeg4_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare the ActivityNet Captions dataset.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--videos-dir", type=Path, default=None,
                        help="Directory of raw ActivityNet videos you already have (searched recursively)")
    source.add_argument("--download-videos", action="store_true",
                        help=f"Download the raw videos (~42 GB) from the Hugging Face mirror {HF_REPO_ID}")
    parser.add_argument("--download-path", type=Path, default=DEFAULT_DOWNLOAD_PATH,
                        help=f"Output directory (default: {DEFAULT_DOWNLOAD_PATH})")
    parser.add_argument("--subsets", nargs="+", default=["val"], choices=sorted(SUBSET_FILES),
                        help="Which subsets' videos to re-encode (default: val)")
    parser.add_argument("--max-videos", type=int, default=None,
                        help="Re-encode at most this many videos per subset (for smoke tests)")
    args = parser.parse_args()

    prepare_activitynet_dataset(args.videos_dir, args.download_path, args.subsets, args.max_videos,
                                download_raw_videos=args.download_videos)
