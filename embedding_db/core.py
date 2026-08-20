"""Storage and build pipeline for a compact NPY embedding database."""

from __future__ import annotations

import io
import json
import os
import platform
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
from numpy.lib import format as npy_format
from numpy.lib.format import open_memmap

from .model import Clip4ClipEncoder, mean_pool_frame_embeddings
from .pack import score_query
from .video import VideoItem, batched, decode_video, discover_videos


DEFAULT_MODEL_ID = "Searchium-ai/clip4clip-webvid150k"


def _json_dump(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def npy_bytes(count: int, dimension: int, dtype: str = "float32") -> int:
    """Exact size of the .npy file numpy would write for this array."""
    if count < 1 or dimension < 1:
        raise ValueError("count and dimension must be positive")
    descriptor = {
        "descr": npy_format.dtype_to_descr(np.dtype(dtype)),
        "fortran_order": False,
        "shape": (int(count), int(dimension)),
    }
    buffer = io.BytesIO()
    npy_format.write_array_header_1_0(buffer, descriptor)
    return buffer.tell() + count * dimension * np.dtype(dtype).itemsize


def release_memmaps(*arrays: Any) -> None:
    """Close numpy memmaps.

    Windows keeps a file locked while a mapping is open, so anything that
    finishes reading a database has to hand the mapping back explicitly.
    """
    for array in arrays:
        mapping = getattr(array, "_mmap", None)
        if mapping is not None:
            mapping.close()


def estimate_database_size(count: int, dimension: int = 512, dtype: str = "float32") -> dict[str, int | str]:
    if count < 1 or dimension < 1:
        raise ValueError("count and dimension must be positive")
    np_dtype = np.dtype(dtype)
    vector_bytes = count * dimension * np_dtype.itemsize
    return {
        "count": count,
        "dimension": dimension,
        "dtype": np_dtype.name,
        "raw_vector_bytes": vector_bytes,
        "estimated_npy_bytes": npy_bytes(count, dimension, dtype),
    }


@dataclass
class BuildConfig:
    model_id: str = DEFAULT_MODEL_ID
    device: str = "auto"
    precision: str = "auto"
    storage_dtype: str = "float32"
    frame_rate: float = 1.0
    max_frames: int = 12
    image_size: int = 224
    frame_batch_size: int = 128
    video_batch_size: int = 16
    decode_workers: int = 4


class EmbeddingDatabase:
    def __init__(self, directory: str | Path):
        self.directory = Path(directory).expanduser().resolve()
        self.manifest_path = self.directory / "manifest.json"
        self.vectors_path = self.directory / "embeddings.npy"
        self.completed_path = self.directory / "completed.npy"
        self.scales_path = self.directory / "scales.npy"
        self.items_path = self.directory / "items.jsonl"

    def _write_items(self, items: list[VideoItem]) -> None:
        with self.items_path.open("w", encoding="utf-8") as handle:
            for index, item in enumerate(items):
                row = {"index": index, "video_id": item.video_id, "path": str(item.path)}
                if item.caption is not None:
                    row["caption"] = item.caption
                print(json.dumps(row, ensure_ascii=False), file=handle)

    def _read_items(self) -> list[VideoItem]:
        items = []
        with self.items_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                items.append(VideoItem(row["video_id"], Path(row["path"]), row.get("caption")))
        return items

    def initialize(
        self,
        items: list[VideoItem],
        config: BuildConfig,
        embedding_dim: int,
        overwrite: bool = False,
        resume: bool = False,
    ) -> tuple[np.memmap, np.memmap]:
        if self.directory.exists() and overwrite:
            protected = {
                Path(self.directory.anchor).resolve(),
                Path.home().resolve(),
                Path.cwd().resolve(),
            }
            if self.directory in protected:
                raise ValueError(f"Refusing to overwrite protected directory: {self.directory}")
            shutil.rmtree(self.directory)
        elif self.directory.exists() and any(self.directory.iterdir()) and not self.manifest_path.exists():
            raise FileExistsError(
                f"Output directory is non-empty and is not an embedding database: {self.directory}"
            )
        self.directory.mkdir(parents=True, exist_ok=True)

        if self.manifest_path.exists():
            if not resume:
                raise FileExistsError(f"Database already exists: {self.directory}; pass --resume or --overwrite")
            manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            existing_items = self._read_items()
            if [(x.video_id, str(x.path)) for x in existing_items] != [(x.video_id, str(x.path)) for x in items]:
                raise ValueError("Resume input does not match the existing database item list")
            comparable = asdict(config)
            comparable.pop("device", None)
            comparable.pop("precision", None)
            stored = dict(manifest["config"])
            stored.pop("device", None)
            stored.pop("precision", None)
            if comparable != stored or embedding_dim != manifest["embedding_dim"]:
                raise ValueError("Resume configuration does not match the existing database")
            return (
                open_memmap(self.vectors_path, mode="r+"),
                open_memmap(self.completed_path, mode="r+"),
            )
        if resume:
            raise FileNotFoundError(f"Cannot resume; manifest.json does not exist in {self.directory}")

        self._write_items(items)
        vectors = open_memmap(
            self.vectors_path,
            mode="w+",
            dtype=np.dtype(config.storage_dtype),
            shape=(len(items), embedding_dim),
        )
        completed = open_memmap(self.completed_path, mode="w+", dtype=np.bool_, shape=(len(items),))
        completed[:] = False
        vectors.flush()
        completed.flush()
        manifest = {
            "format_version": 1,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "count": len(items),
            "embedding_dim": embedding_dim,
            "config": asdict(config),
            "status": "building",
        }
        _json_dump(self.manifest_path, manifest)
        return vectors, completed

    def load(self, mode: str = "r") -> tuple[np.ndarray, np.ndarray, list[VideoItem], dict[str, Any]]:
        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        return np.load(self.vectors_path, mmap_mode=mode), np.load(self.completed_path, mmap_mode=mode), self._read_items(), manifest

    def load_payload(self, mode: str = "r") -> tuple[dict[str, np.ndarray], str, np.ndarray, list[VideoItem], dict[str, Any]]:
        """Load the stored arrays together with the packing that decodes them."""
        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        payload = {"embeddings": np.load(self.vectors_path, mmap_mode=mode)}
        if self.scales_path.exists():
            payload["scales"] = np.load(self.scales_path, mmap_mode=mode)
        return payload, manifest.get("packing", "float32"), np.load(self.completed_path, mmap_mode=mode), self._read_items(), manifest

    def search_vector(self, query: np.ndarray, top_k: int = 10) -> list[dict[str, Any]]:
        if top_k < 1:
            raise ValueError("top_k must be positive")
        payload, packing, completed, items, manifest = self.load_payload()
        valid_indices = np.flatnonzero(completed)
        if len(valid_indices) == 0:
            return []
        subset = {name: np.asarray(array)[valid_indices] for name, array in payload.items()}
        release_memmaps(completed, *payload.values())
        scores = score_query(subset, packing, query, int(manifest["embedding_dim"]))
        k = min(top_k, len(scores))
        local = np.argpartition(scores, -k)[-k:]
        local = local[np.argsort(scores[local])[::-1]]
        hits = []
        for rank, pos in enumerate(local, 1):
            item = items[int(valid_indices[pos])]
            hit = {"rank": rank, "video_id": item.video_id, "path": str(item.path), "score": float(scores[pos])}
            if item.caption is not None:
                hit["caption"] = item.caption
            hits.append(hit)
        return hits


def _system_info(encoder: Clip4ClipEncoder) -> dict[str, Any]:
    torch = encoder.torch
    info: dict[str, Any] = {
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "device": str(encoder.device),
    }
    if encoder.device.type == "cuda":
        info["gpu"] = torch.cuda.get_device_name(encoder.device)
        info["gpu_memory_bytes"] = torch.cuda.get_device_properties(encoder.device).total_memory
    return info


def build_database(
    videos_dir: str | Path,
    output_dir: str | Path,
    config: BuildConfig,
    split_csv: str | Path | None = None,
    id_column: str = "video_id",
    limit: int | None = None,
    overwrite: bool = False,
    resume: bool = False,
    strict: bool = False,
    progress_callback: Callable[[dict[str, int]], None] | None = None,
) -> dict[str, Any]:
    items, missing = discover_videos(videos_dir, split_csv, id_column, limit)
    if not items:
        raise ValueError("No videos were found")
    if strict and missing:
        raise FileNotFoundError(f"{len(missing)} CSV video IDs are missing; first: {missing[:5]}")

    model_started = time.perf_counter()
    encoder = Clip4ClipEncoder(config.model_id, config.device, config.precision)
    model_load_seconds = time.perf_counter() - model_started
    database = EmbeddingDatabase(output_dir)
    vectors, completed = database.initialize(items, config, encoder.embedding_dim, overwrite, resume)
    failures_path = database.directory / "failures.jsonl"

    pending = [(index, item) for index, item in enumerate(items) if not completed[index]]
    attempted = successful = total_frames = 0
    decode_seconds = inference_seconds = 0.0
    processing_started = time.perf_counter()

    worker_count = max(1, config.decode_workers)
    with ThreadPoolExecutor(max_workers=worker_count) as executor, failures_path.open("a", encoding="utf-8") as failures:
        # Each group carries its own row indices, so two items that happen
        # to share a path can never overwrite one another's row.
        for group in batched(pending, config.video_batch_size):
            results = list(executor.map(
                lambda pair: decode_video(pair[1], config.frame_rate, config.max_frames, config.image_size),
                group,
            ))
            attempted += len(group)
            decode_seconds += sum(result.decode_seconds for result in results)
            valid = [(index, result) for (index, _), result in zip(group, results) if result.frames is not None]
            for (_, item), result in zip(group, results):
                if result.frames is None:
                    print(
                        json.dumps({"video_id": item.video_id, "path": str(item.path), "error": result.error}, ensure_ascii=False),
                        file=failures,
                    )
            if not valid:
                continue

            counts = [len(result.frames) for _, result in valid]
            all_frames = np.concatenate([result.frames for _, result in valid], axis=0)
            encoder.synchronize()
            inference_started = time.perf_counter()
            frame_embeddings = encoder.embed_frames(all_frames, config.frame_batch_size)
            encoder.synchronize()
            inference_seconds += time.perf_counter() - inference_started

            cursor = 0
            for (row_index, _), frame_count in zip(valid, counts):
                vectors[row_index] = mean_pool_frame_embeddings(frame_embeddings[cursor : cursor + frame_count]).astype(config.storage_dtype)
                completed[row_index] = True
                cursor += frame_count
                successful += 1
                total_frames += frame_count
            vectors.flush()
            completed.flush()
            if progress_callback is not None:
                progress_callback({
                    "attempted_this_run": attempted,
                    "pending_this_run": len(pending),
                    "completed_total": int(np.count_nonzero(completed)),
                    "failed_this_run": attempted - successful,
                })

    processing_seconds = time.perf_counter() - processing_started
    total_completed = int(np.count_nonzero(completed))
    report = {
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "requested_count": len(items),
        "completed_count": total_completed,
        "failed_this_run": attempted - successful,
        "missing_from_csv": len(missing),
        "model_load_seconds": model_load_seconds,
        "processing_seconds": processing_seconds,
        "decode_worker_seconds_sum": decode_seconds,
        "inference_seconds": inference_seconds,
        "frames_this_run": total_frames,
        "videos_per_second": attempted / processing_seconds if processing_seconds else None,
        "frames_per_second_inference": total_frames / inference_seconds if inference_seconds else None,
        "projected_processing_seconds": {
            str(count): processing_seconds / attempted * count for count in (1000, 2000)
        } if attempted else {},
        "database_files_bytes": sum(p.stat().st_size for p in database.directory.iterdir() if p.is_file()),
        "system": _system_info(encoder),
    }
    _json_dump(database.directory / "benchmark.json", report)
    manifest = json.loads(database.manifest_path.read_text(encoding="utf-8"))
    manifest["status"] = "complete" if total_completed == len(items) else "partial"
    manifest["completed_count"] = total_completed
    _json_dump(database.manifest_path, manifest)
    return report
