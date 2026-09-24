"""Small, file-backed video embedding database for MSR-VTT."""

from .config import (
    CLIP_IMAGE_MEAN,
    CLIP_IMAGE_STD,
    DEFAULT_ENCODER,
    DEFAULT_MODEL_ID,
    DEFAULT_POOLING,
    EncoderConfig,
    PreprocessSpec,
    encoder_config_from_dict,
    load_config_file,
)
from .core import (
    BuildConfig,
    EmbeddingDatabase,
    build_database,
    encoder_config_from_manifest,
    estimate_database_size,
    npy_bytes,
    release_memmaps,
)
from .encoders import (
    FrameEncoder,
    TextEncoder,
    available_encoders,
    create_encoder,
    register_encoder,
)
from .evaluate import evaluate_database, ranking_metrics
from .model import available_poolings, get_pooling, register_pooling
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
    "CLIP_IMAGE_MEAN",
    "CLIP_IMAGE_STD",
    "DEFAULT_ENCODER",
    "DEFAULT_MODEL_ID",
    "DEFAULT_POOLING",
    "PACKINGS",
    "BuildConfig",
    "EmbeddingDatabase",
    "EncoderConfig",
    "FrameEncoder",
    "PreprocessSpec",
    "TextEncoder",
    "available_encoders",
    "available_poolings",
    "build_database",
    "bytes_per_vector",
    "capacity_report",
    "compare_packings",
    "create_encoder",
    "encoder_config_from_dict",
    "encoder_config_from_manifest",
    "estimate_database_size",
    "evaluate_database",
    "get_pooling",
    "import_vectors",
    "load_config_file",
    "measure_metadata_bytes_per_item",
    "npy_bytes",
    "pack_database",
    "pack_vectors",
    "parse_byte_budget",
    "ranking_metrics",
    "register_encoder",
    "register_pooling",
    "release_memmaps",
    "unpack_vectors",
]
