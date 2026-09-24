"""Vector packing formats and their exact on-disk cost.

The database is small enough that float32 is affordable, but the point of a
storage budget is knowing what the cheaper layouts buy and what they cost. Each
packing here is self-contained: it can be written, read back, and searched
without a model.
"""

from __future__ import annotations

import numpy as np


PACKINGS = ("float32", "float16", "int8", "binary")

# int8 rows are symmetric-quantized against a per-row scale stored as float32.
INT8_SCALE_BYTES = 4


def bytes_per_vector(dimension: int, packing: str) -> int:
    """Payload bytes for one vector, excluding NPY headers."""
    if dimension < 1:
        raise ValueError("dimension must be positive")
    if packing == "float32":
        return 4 * dimension
    if packing == "float16":
        return 2 * dimension
    if packing == "int8":
        return dimension + INT8_SCALE_BYTES
    if packing == "binary":
        return (dimension + 7) // 8
    raise ValueError(f"Unknown packing {packing!r}; expected one of {PACKINGS}")


def pack_vectors(vectors: np.ndarray, packing: str) -> dict[str, np.ndarray]:
    """Encode float32 rows into the arrays a packed database stores."""
    if vectors.ndim != 2:
        raise ValueError("vectors must have shape [rows, dimensions]")
    values = np.asarray(vectors, dtype=np.float32)

    if packing == "float32":
        return {"embeddings": values}
    if packing == "float16":
        return {"embeddings": values.astype(np.float16)}
    if packing == "int8":
        scales = np.abs(values).max(axis=1)
        # An all-zero row would divide by zero; its quantized form is zero anyway.
        safe = np.maximum(scales, np.finfo(np.float32).tiny)
        quantized = np.rint(values / safe[:, None] * 127.0)
        return {
            "embeddings": np.clip(quantized, -127, 127).astype(np.int8),
            "scales": (scales / 127.0).astype(np.float32),
        }
    if packing == "binary":
        return {"embeddings": np.packbits(values > 0, axis=1)}
    raise ValueError(f"Unknown packing {packing!r}; expected one of {PACKINGS}")


def unpack_vectors(payload: dict[str, np.ndarray], packing: str, dimension: int) -> np.ndarray:
    """Reconstruct float32 rows. Binary returns signs, which is all it kept."""
    embeddings = payload["embeddings"]
    if packing == "float32":
        return np.asarray(embeddings, dtype=np.float32)
    if packing == "float16":
        return np.asarray(embeddings, dtype=np.float32)
    if packing == "int8":
        return np.asarray(embeddings, dtype=np.float32) * np.asarray(payload["scales"], dtype=np.float32)[:, None]
    if packing == "binary":
        bits = np.unpackbits(np.asarray(embeddings, dtype=np.uint8), axis=1)[:, :dimension]
        return (bits.astype(np.float32) * 2.0) - 1.0
    raise ValueError(f"Unknown packing {packing!r}; expected one of {PACKINGS}")


def score_query(payload: dict[str, np.ndarray], packing: str, query: np.ndarray, dimension: int) -> np.ndarray:
    """Similarity of every stored row against one query. Higher ranks first."""
    vector = np.asarray(query, dtype=np.float32).reshape(-1)
    if vector.shape[0] != dimension:
        raise ValueError(f"query has {vector.shape[0]} dimensions, expected {dimension}")

    if packing == "binary":
        # Rank by agreeing sign bits, which is Hamming distance inverted.
        query_bits = np.packbits(vector > 0)
        matching = np.unpackbits(
            np.bitwise_xor(np.asarray(payload["embeddings"], dtype=np.uint8), query_bits), axis=1
        )[:, :dimension]
        return (dimension - matching.sum(axis=1)).astype(np.float32)
    return unpack_vectors(payload, packing, dimension) @ vector


def score_matrix(payload: dict[str, np.ndarray], packing: str, queries: np.ndarray, dimension: int) -> np.ndarray:
    """Similarity for many queries at once, shaped [queries, rows]."""
    matrix = np.asarray(queries, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[1] != dimension:
        raise ValueError(f"queries must have shape [n, {dimension}]")

    if packing == "binary":
        stored = np.unpackbits(np.asarray(payload["embeddings"], dtype=np.uint8), axis=1)[:, :dimension]
        stored_signs = stored.astype(np.int16) * 2 - 1
        query_signs = np.where(matrix > 0, 1, -1).astype(np.int16)
        # Agreeing bits per pair, expressed on the same scale as the loop above.
        return ((query_signs @ stored_signs.T).astype(np.float32) + dimension) / 2.0
    return matrix @ unpack_vectors(payload, packing, dimension).T
