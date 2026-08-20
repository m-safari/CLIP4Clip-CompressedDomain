"""Text-to-video retrieval metrics computed from stored vectors alone.

Given a text-vector matrix whose row *i* describes database row *i*, the whole
evaluation is one matrix multiply. No model is loaded, so a packed database can
be scored as cheaply as the float32 original.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .core import EmbeddingDatabase, release_memmaps
from .pack import score_matrix


def ranking_metrics(scores: np.ndarray, recall_at: tuple[int, ...] = (1, 5, 10)) -> dict[str, float]:
    """Metrics for a [queries, rows] score matrix whose ground truth is the diagonal."""
    if scores.ndim != 2 or scores.shape[0] != scores.shape[1]:
        raise ValueError("scores must be a square [queries, rows] matrix aligned row-for-row")
    truth = np.diag(scores)[:, None]
    # Rank 1 means nothing outscored the correct row. Ties count against us.
    ranks = (scores > truth).sum(axis=1) + 1
    metrics = {f"R@{k}": float(np.mean(ranks <= k) * 100.0) for k in recall_at}
    metrics["MedianR"] = float(np.median(ranks))
    metrics["MeanR"] = float(np.mean(ranks))
    metrics["queries"] = int(scores.shape[0])
    return metrics


def load_text_vectors(path: str | Path, dimension: int) -> np.ndarray:
    matrix = np.load(Path(path).expanduser().resolve())
    if matrix.ndim != 2 or matrix.shape[1] != dimension:
        raise ValueError(f"text vectors must have shape [n, {dimension}], got {matrix.shape}")
    matrix = matrix.astype(np.float32, copy=False)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, np.finfo(np.float32).eps)


def evaluate_database(
    db_dir: str | Path,
    text_vectors_path: str | Path,
    recall_at: tuple[int, ...] = (1, 5, 10),
) -> dict[str, Any]:
    database = EmbeddingDatabase(db_dir)
    payload, packing, completed, items, manifest = database.load_payload()
    dimension = int(manifest["embedding_dim"])

    valid = np.flatnonzero(np.asarray(completed))
    if len(valid) != len(items):
        raise ValueError(
            f"{len(items) - len(valid)} rows are incomplete; evaluation needs a fully built database"
        )

    queries = load_text_vectors(text_vectors_path, dimension)
    if queries.shape[0] != len(items):
        raise ValueError(
            f"text vectors have {queries.shape[0]} rows but the database has {len(items)}; "
            "they must describe the same items in the same order"
        )

    stored = {k: np.array(v) for k, v in payload.items()}
    release_memmaps(completed, *payload.values())
    scores = score_matrix(stored, packing, queries, dimension)
    return {
        "db": str(database.directory),
        "packing": packing,
        "text_vectors": str(text_vectors_path),
        **ranking_metrics(scores, recall_at),
    }
