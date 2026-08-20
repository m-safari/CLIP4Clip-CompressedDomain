"""Capacity planning and format conversion for the embedding database.

`core.py` owns the build pipeline. This module answers the storage questions:
how many vectors fit in a byte budget, how to turn an existing vector matrix
into a real database, and what the cheaper packings actually cost in accuracy.
Nothing here loads a model.
"""

from __future__ import annotations

import csv
import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .core import BuildConfig, EmbeddingDatabase, _json_dump, npy_bytes, release_memmaps
from .pack import PACKINGS, bytes_per_vector, pack_vectors, score_matrix, unpack_vectors
from .video import VideoItem


BYTE_UNITS = {
    "b": 1,
    "kb": 10**3,
    "mb": 10**6,
    "gb": 10**9,
    "tb": 10**12,
    "kib": 2**10,
    "mib": 2**20,
    "gib": 2**30,
    "tib": 2**40,
}

PACKING_DTYPES = {"float32": "float32", "float16": "float16", "int8": "int8", "binary": "uint8"}


def parse_byte_budget(text: str | int) -> int:
    """Accept 2GB, 2GiB, 512MB, or a raw byte count."""
    if isinstance(text, int):
        value = text
        if value < 1:
            raise ValueError("budget must be positive")
        return value
    match = re.fullmatch(r"\s*([0-9]*\.?[0-9]+)\s*([a-zA-Z]*)\s*", str(text))
    if not match:
        raise ValueError(f"Cannot parse byte budget {text!r}")
    amount, unit = float(match.group(1)), match.group(2).lower() or "b"
    if unit not in BYTE_UNITS:
        raise ValueError(f"Unknown unit {unit!r}; expected one of {sorted(BYTE_UNITS)}")
    total = int(amount * BYTE_UNITS[unit])
    if total < 1:
        raise ValueError("budget must be at least one byte")
    return total


def human_bytes(value: float) -> str:
    amount = float(value)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(amount) < 1024 or unit == "TiB":
            return f"{amount:.2f} {unit}"
        amount /= 1024
    raise AssertionError


def measure_metadata_bytes_per_item(directory: str | Path) -> dict[str, float]:
    """Per-item cost of everything that is not the vector payload."""
    database = EmbeddingDatabase(directory)
    vectors, completed, _, manifest = database.load()
    release_memmaps(vectors, completed)
    count = int(manifest["count"])
    if count < 1:
        raise ValueError("database is empty")
    sidecars = {
        "items_jsonl_bytes": database.items_path.stat().st_size,
        "completed_npy_bytes": database.completed_path.stat().st_size,
        "manifest_json_bytes": database.manifest_path.stat().st_size,
    }
    total = sum(sidecars.values())
    return {**sidecars, "count": count, "total_bytes": total, "bytes_per_item": total / count}


def capacity_report(
    budget_bytes: int,
    dimension: int = 512,
    metadata_bytes_per_item: float = 0.0,
) -> dict[str, Any]:
    """How many vectors of each packing fit inside a byte budget."""
    if budget_bytes < 1:
        raise ValueError("budget_bytes must be positive")
    rows = []
    for packing in PACKINGS:
        per_item = bytes_per_vector(dimension, packing) + metadata_bytes_per_item
        capacity = int(budget_bytes // per_item)
        rows.append(
            {
                "packing": packing,
                "bytes_per_item": per_item,
                "max_items": capacity,
                "max_items_human": f"{capacity:,}",
                "bytes_for_1000": int(round(1000 * per_item)),
                "bytes_for_2000": int(round(2000 * per_item)),
                "percent_of_budget_at_2000": 2000 * per_item / budget_bytes * 100,
            }
        )
    return {
        "budget_bytes": budget_bytes,
        "budget_human": human_bytes(budget_bytes),
        "dimension": dimension,
        "metadata_bytes_per_item": metadata_bytes_per_item,
        "packings": rows,
    }


def read_split_csv(
    split_csv: str | Path,
    id_column: str = "video_id",
    caption_column: str = "sentence",
) -> tuple[list[str], dict[str, str]]:
    """Ordered, de-duplicated ids from a split CSV plus their captions."""
    path = Path(split_csv).expanduser().resolve()
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or id_column not in reader.fieldnames:
            raise ValueError(f"CSV must contain column {id_column!r}; columns={reader.fieldnames}")
        has_captions = caption_column in reader.fieldnames
        ordered, captions, seen = [], {}, set()
        for row in reader:
            video_id = str(row[id_column]).strip()
            if video_id and video_id not in seen:
                seen.add(video_id)
                ordered.append(video_id)
                if has_captions:
                    captions[video_id] = str(row[caption_column]).strip()
    return ordered, captions


def import_vectors(
    vectors_path: str | Path,
    output_dir: str | Path,
    split_csv: str | Path | None = None,
    id_column: str = "video_id",
    caption_column: str = "sentence",
    videos_dir: str | Path | None = None,
    model_id: str = "",
    source: str = "",
    normalize: bool = False,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Turn an existing vector matrix plus a split CSV into a real database.

    This is the no-inference path: the vectors already exist, and what is
    missing is the row mapping, the valid-row mask, and a manifest the rest of
    the module can read.
    """
    matrix = np.load(Path(vectors_path).expanduser().resolve())
    if matrix.ndim != 2:
        raise ValueError(f"Expected a 2-D vector matrix, got shape {matrix.shape}")
    matrix = matrix.astype(np.float32, copy=False)
    if not np.isfinite(matrix).all():
        raise ValueError("Vector matrix contains NaN or infinite values")

    count, dimension = matrix.shape
    norms = np.linalg.norm(matrix, axis=1)
    was_normalized = bool(np.allclose(norms, 1.0, atol=1e-4))
    if normalize and not was_normalized:
        matrix = matrix / np.maximum(norms, np.finfo(np.float32).eps)[:, None]

    if split_csv is None:
        ids = [f"row{index}" for index in range(count)]
        captions: dict[str, str] = {}
    else:
        ids, captions = read_split_csv(split_csv, id_column, caption_column)
        if len(ids) != count:
            raise ValueError(
                f"CSV has {len(ids)} unique ids but the matrix has {count} rows; they must line up"
            )

    root = Path(videos_dir).expanduser().resolve() if videos_dir is not None else None
    items = [
        VideoItem(
            video_id,
            (root / f"{video_id}.mp4") if root is not None else Path(video_id),
            captions.get(video_id),
        )
        for video_id in ids
    ]

    database = EmbeddingDatabase(output_dir)
    config = BuildConfig(model_id=model_id) if model_id else BuildConfig()
    vectors, completed = database.initialize(items, config, dimension, overwrite=overwrite)
    vectors[:] = matrix
    completed[:] = True
    vectors.flush()
    completed.flush()
    release_memmaps(vectors, completed)

    manifest = json.loads(database.manifest_path.read_text(encoding="utf-8"))
    manifest.update(
        {
            "status": "complete",
            "completed_count": count,
            "packing": "float32",
            "origin": "imported",
            "imported_at": datetime.now(timezone.utc).isoformat(),
            "source": source or str(vectors_path),
            "source_normalized": was_normalized,
            "renormalized_on_import": bool(normalize and not was_normalized),
            "paths_resolved": root is not None,
            "captions": bool(captions),
        }
    )
    _json_dump(database.manifest_path, manifest)

    return {
        "output_dir": str(database.directory),
        "count": count,
        "dimension": dimension,
        "source_normalized": was_normalized,
        "captions": len(captions),
        "paths_resolved": root is not None,
        "database_bytes": sum(p.stat().st_size for p in database.directory.iterdir() if p.is_file()),
    }


def pack_database(
    source_dir: str | Path,
    output_dir: str | Path,
    packing: str,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Rewrite a database in a cheaper vector layout, keeping its metadata."""
    if packing not in PACKINGS:
        raise ValueError(f"Unknown packing {packing!r}; expected one of {PACKINGS}")
    source = EmbeddingDatabase(source_dir)
    payload, source_packing, completed, items, manifest = source.load_payload()
    dimension = int(manifest["embedding_dim"])
    values = unpack_vectors({k: np.array(v) for k, v in payload.items()}, source_packing, dimension)
    mask = np.array(completed)
    release_memmaps(completed, *payload.values())

    target = EmbeddingDatabase(output_dir)
    if target.directory == source.directory:
        raise ValueError("Refusing to pack a database onto itself; choose another output directory")
    if target.directory.exists():
        if not overwrite:
            raise FileExistsError(f"Output already exists: {target.directory}; pass --overwrite")
        shutil.rmtree(target.directory)
    target.directory.mkdir(parents=True)

    packed = pack_vectors(values, packing)
    np.save(target.vectors_path, packed["embeddings"])
    if "scales" in packed:
        np.save(target.scales_path, packed["scales"])
    np.save(target.completed_path, mask)
    target._write_items(items)

    packed_manifest = dict(manifest)
    packed_manifest.update(
        {
            "packing": packing,
            "packed_at": datetime.now(timezone.utc).isoformat(),
            "packed_from": str(source.directory),
            "source_packing": source_packing,
        }
    )
    _json_dump(target.manifest_path, packed_manifest)

    files = {p.name: p.stat().st_size for p in target.directory.iterdir() if p.is_file()}
    return {
        "output_dir": str(target.directory),
        "packing": packing,
        "count": len(items),
        "dimension": dimension,
        "files": files,
        "database_bytes": sum(files.values()),
        "vector_bytes": files.get("embeddings.npy", 0) + files.get("scales.npy", 0),
        "bytes_per_item": bytes_per_vector(dimension, packing),
    }


def compare_packings(source_dir: str | Path, top_k: int = 10) -> dict[str, Any]:
    """Score every packing against the source vectors, with no model involved.

    Neighbour agreement uses the database's own rows as queries: for each row,
    how much of its float32 top-k neighbourhood survives the packing.
    """
    source = EmbeddingDatabase(source_dir)
    payload, source_packing, completed, items, manifest = source.load_payload()
    dimension = int(manifest["embedding_dim"])
    valid = np.flatnonzero(np.asarray(completed))
    reference = unpack_vectors(
        {k: np.asarray(v)[valid] for k, v in payload.items()}, source_packing, dimension
    )
    release_memmaps(completed, *payload.values())
    count = reference.shape[0]
    if count < 2:
        raise ValueError("need at least two completed rows to compare packings")
    k = min(top_k, count - 1)

    def neighbours(scores: np.ndarray) -> np.ndarray:
        # Drop each row's own match before taking the top-k.
        np.fill_diagonal(scores, -np.inf)
        top = np.argpartition(scores, -k, axis=1)[:, -k:]
        ordered = np.take_along_axis(scores, top, axis=1).argsort(axis=1)[:, ::-1]
        return np.take_along_axis(top, ordered, axis=1)

    baseline = neighbours(reference @ reference.T)

    rows = []
    for packing in PACKINGS:
        packed = pack_vectors(reference, packing)
        restored = unpack_vectors(packed, packing, dimension)
        agreement = np.mean(
            [
                len(set(baseline[i].tolist()) & set(candidate.tolist())) / k
                for i, candidate in enumerate(neighbours(score_matrix(packed, packing, reference, dimension)))
            ]
        )
        if packing == "binary":
            # Sign bits keep no magnitude, so a cosine to the original is meaningless.
            cosine = None
        else:
            restored_unit = restored / np.maximum(
                np.linalg.norm(restored, axis=1, keepdims=True), np.finfo(np.float32).eps
            )
            reference_unit = reference / np.maximum(
                np.linalg.norm(reference, axis=1, keepdims=True), np.finfo(np.float32).eps
            )
            cosine = float((restored_unit * reference_unit).sum(axis=1).mean())

        vector_bytes = npy_bytes(count, dimension, PACKING_DTYPES[packing]) if packing != "binary" else npy_bytes(
            count, (dimension + 7) // 8, "uint8"
        )
        if packing == "int8":
            vector_bytes += npy_bytes(count, 1, "float32")

        rows.append(
            {
                "packing": packing,
                "bytes_per_item": bytes_per_vector(dimension, packing),
                "vector_file_bytes": vector_bytes,
                "vector_file_human": human_bytes(vector_bytes),
                "mean_cosine_to_float32": cosine,
                f"neighbour_agreement_at_{k}": float(agreement),
            }
        )

    return {
        "source_dir": str(source.directory),
        "count": count,
        "dimension": dimension,
        "top_k": k,
        "packings": rows,
    }
