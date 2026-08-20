"""Video discovery, sampling, and CLIP preprocessing."""

from __future__ import annotations

import csv
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, TypeVar

import numpy as np

# OpenCV is imported lazily inside the decoding functions: storage, packing,
# capacity planning and evaluation never touch a video file, and should not
# require a video library to be installed.

T = TypeVar("T")


VIDEO_EXTENSIONS = (".mp4", ".webm", ".mkv", ".avi", ".mov")
CLIP_MEAN = np.asarray((0.48145466, 0.4578275, 0.40821073), dtype=np.float32)
CLIP_STD = np.asarray((0.26862954, 0.26130258, 0.27577711), dtype=np.float32)


@dataclass(frozen=True)
class VideoItem:
    video_id: str
    path: Path
    caption: str | None = None


@dataclass
class DecodeResult:
    frames: np.ndarray | None
    decode_seconds: float
    error: str | None = None


def _find_video(videos_dir: Path, video_id: str, by_stem: dict[str, Path]) -> Path | None:
    direct = Path(video_id)
    if direct.suffix.lower() in VIDEO_EXTENSIONS:
        candidate = videos_dir / direct
        if candidate.is_file():
            return candidate.resolve()
    return by_stem.get(direct.stem)


def discover_videos(
    videos_dir: str | Path,
    split_csv: str | Path | None = None,
    id_column: str = "video_id",
    limit: int | None = None,
    caption_column: str = "sentence",
) -> tuple[list[VideoItem], list[str]]:
    """Return unique videos in deterministic order and missing IDs."""
    root = Path(videos_dir).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Video directory does not exist: {root}")

    files = sorted(
        (p.resolve() for p in root.rglob("*") if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS),
        key=lambda p: str(p).lower(),
    )
    by_stem: dict[str, Path] = {}
    for path in files:
        by_stem.setdefault(path.stem, path)

    missing: list[str] = []
    if split_csv is None:
        items = [VideoItem(path.stem, path) for path in files]
    else:
        csv_path = Path(split_csv).expanduser().resolve()
        with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames or id_column not in reader.fieldnames:
                raise ValueError(f"CSV must contain column {id_column!r}; columns={reader.fieldnames}")
            has_captions = caption_column in reader.fieldnames
            ordered_ids = []
            captions: dict[str, str] = {}
            seen = set()
            for row in reader:
                video_id = str(row[id_column]).strip()
                if video_id and video_id not in seen:
                    seen.add(video_id)
                    ordered_ids.append(video_id)
                    if has_captions:
                        captions[video_id] = str(row[caption_column]).strip()

        items = []
        # Distinct CSV ids can resolve to one file (for example "video1" and
        # "video1.mp4"), which would otherwise produce two rows sharing a path.
        claimed: set[Path] = set()
        for video_id in ordered_ids:
            path = _find_video(root, video_id, by_stem)
            if path is None:
                missing.append(video_id)
            elif path not in claimed:
                claimed.add(path)
                items.append(VideoItem(Path(video_id).stem, path, captions.get(video_id)))

    if limit is not None:
        if limit < 1:
            raise ValueError("limit must be a positive integer")
        items = items[:limit]
    return items, missing


def sampled_frame_indices(frame_count: int, fps: float, frame_rate: float, max_frames: int) -> np.ndarray:
    if frame_count < 1 or not math.isfinite(fps) or fps <= 0:
        return np.empty(0, dtype=np.int64)
    if frame_rate <= 0 or max_frames <= 0:
        raise ValueError("frame_rate and max_frames must be positive")

    interval = max(fps / frame_rate, 1.0)
    indices = np.floor(np.arange(0.0, float(frame_count), interval)).astype(np.int64)
    indices = np.unique(np.clip(indices, 0, frame_count - 1))
    if len(indices) > max_frames:
        positions = np.linspace(0, len(indices) - 1, max_frames, dtype=np.int64)
        indices = indices[positions]
    return indices


def preprocess_bgr_frame(frame: np.ndarray, size: int = 224) -> np.ndarray:
    """Match the CLIP resize-short-edge + center-crop preprocessing."""
    import cv2

    if frame is None or frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("Expected an HxWx3 BGR frame")
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    height, width = rgb.shape[:2]
    scale = size / min(height, width)
    new_height = max(size, int(round(height * scale)))
    new_width = max(size, int(round(width * scale)))
    resized = cv2.resize(rgb, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    top = (new_height - size) // 2
    left = (new_width - size) // 2
    cropped = resized[top : top + size, left : left + size].astype(np.float32) / 255.0
    cropped = (cropped - CLIP_MEAN) / CLIP_STD
    return np.transpose(cropped, (2, 0, 1)).astype(np.float32, copy=False)


def decode_video(
    item: VideoItem,
    frame_rate: float = 1.0,
    max_frames: int = 12,
    image_size: int = 224,
) -> DecodeResult:
    import cv2

    started = time.perf_counter()
    capture = cv2.VideoCapture(str(item.path), cv2.CAP_FFMPEG)
    try:
        if not capture.isOpened():
            return DecodeResult(None, time.perf_counter() - started, "OpenCV could not open the video")
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        indices = sampled_frame_indices(frame_count, fps, frame_rate, max_frames)
        if len(indices) == 0:
            return DecodeResult(None, time.perf_counter() - started, f"Invalid video metadata: frames={frame_count}, fps={fps}")

        frames = []
        for index in indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
            ok, frame = capture.read()
            if ok:
                frames.append(preprocess_bgr_frame(frame, image_size))
        if not frames:
            return DecodeResult(None, time.perf_counter() - started, "No sampled frame could be decoded")
        return DecodeResult(np.stack(frames), time.perf_counter() - started)
    except Exception as exc:  # one broken video should not abort a long job
        return DecodeResult(None, time.perf_counter() - started, f"{type(exc).__name__}: {exc}")
    finally:
        capture.release()


def batched(values: list[T], batch_size: int) -> Iterable[list[T]]:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    for offset in range(0, len(values), batch_size):
        yield values[offset : offset + batch_size]

