"""Frame-to-video pooling.

Pure NumPy: no torch, no checkpoint. Model loading lives in `encoders.py`, so
importing this module stays cheap for the storage-only code paths.
"""

from __future__ import annotations

from typing import Callable

import numpy as np


def l2_normalize(values: np.ndarray, axis: int = -1) -> np.ndarray:
    norms = np.linalg.norm(values, axis=axis, keepdims=True)
    return values / np.maximum(norms, np.finfo(np.float32).eps)


POOLINGS: dict[str, Callable[[np.ndarray], np.ndarray]] = {}


def register_pooling(name: str) -> Callable[[Callable], Callable]:
    """Register a frames -> one vector reduction under a CLI-visible name."""

    def decorator(function: Callable[[np.ndarray], np.ndarray]) -> Callable[[np.ndarray], np.ndarray]:
        key = name.lower()
        if key in POOLINGS and POOLINGS[key] is not function:
            raise ValueError(f"Pooling {name!r} is already registered")
        POOLINGS[key] = function
        return function

    return decorator


def available_poolings() -> list[str]:
    return sorted(POOLINGS)


def get_pooling(name: str) -> Callable[[np.ndarray], np.ndarray]:
    key = name.lower()
    if key not in POOLINGS:
        raise ValueError(f"Unknown pooling {name!r}; available poolings are {available_poolings()}")
    return POOLINGS[key]


@register_pooling("meanp")
def mean_pool_frame_embeddings(frame_embeddings: np.ndarray) -> np.ndarray:
    """CLIP4Clip meanP: normalize frames, average, normalize the video."""
    if frame_embeddings.ndim != 2 or frame_embeddings.shape[0] == 0:
        raise ValueError("frame_embeddings must have shape [frames, dimensions]")
    normalized_frames = l2_normalize(frame_embeddings.astype(np.float32, copy=False))
    return l2_normalize(normalized_frames.mean(axis=0)).astype(np.float32, copy=False)


def encode_text_query(model_id: str, query: str, device: str = "auto") -> np.ndarray:
    """Encode one query with a Hugging Face checkpoint.

    Kept as a convenience wrapper; anything needing more control should build an
    encoder from an EncoderConfig directly.
    """
    from .config import EncoderConfig
    from .encoders import create_encoder

    encoder = create_encoder(EncoderConfig(model=model_id, device=device))
    return encoder.encode_text([query])[0]
