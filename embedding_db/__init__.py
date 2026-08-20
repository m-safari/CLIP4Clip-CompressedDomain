"""Small, file-backed video embedding database for MSR-VTT."""

from .core import DEFAULT_MODEL_ID, EmbeddingDatabase, estimate_database_size, npy_bytes, release_memmaps
from .evaluate import evaluate_database, ranking_metrics
from .pack import PACKINGS, bytes_per_vector, pack_vectors, unpack_vectors
from .storage import (
    capacity_report,
    compare_packings,
    import_vectors,
    measure_metadata_bytes_per_item,
    pack_database,
    parse_byte_budget,
)

__all__ = [
    "DEFAULT_MODEL_ID",
    "PACKINGS",
    "EmbeddingDatabase",
    "bytes_per_vector",
    "capacity_report",
    "compare_packings",
    "estimate_database_size",
    "evaluate_database",
    "import_vectors",
    "measure_metadata_bytes_per_item",
    "npy_bytes",
    "pack_database",
    "pack_vectors",
    "parse_byte_budget",
    "ranking_metrics",
    "release_memmaps",
    "unpack_vectors",
]
