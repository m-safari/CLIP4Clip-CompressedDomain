"""Pluggable frame/text encoders.

The build pipeline talks to the `FrameEncoder` protocol and never names a
concrete model. A new backend — the repository's own CLIP4Clip, a compressed-
domain model, a stub for tests — is a class plus one decorator, with no changes
to the pipeline.
"""

from __future__ import annotations

import os
from contextlib import nullcontext
from typing import Any, Callable, Protocol, runtime_checkable

import numpy as np

from .config import CLIP_IMAGE_MEAN, CLIP_IMAGE_STD, EncoderConfig, PreprocessSpec
from .model import l2_normalize

# This is a PyTorch-only module. Prevent Transformers from importing an
# unrelated TensorFlow installation (which can also have a different NumPy ABI).
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")


@runtime_checkable
class FrameEncoder(Protocol):
    """Turns preprocessed frames into vectors."""

    embedding_dim: int
    preprocess: PreprocessSpec

    def embed_frames(self, frames: np.ndarray, batch_size: int = 128) -> np.ndarray:
        """frames: [N, C, H, W] -> [N, embedding_dim]."""

    def synchronize(self) -> None:
        """Block until queued device work is done, so timings are truthful."""

    def describe(self) -> dict[str, Any]:
        """Provenance for the manifest and the benchmark report."""


@runtime_checkable
class TextEncoder(Protocol):
    """Turns query strings into vectors in the same space as the frames."""

    def encode_text(self, queries: list[str]) -> np.ndarray:
        """-> [len(queries), embedding_dim], L2-normalized."""


ENCODERS: dict[str, Callable[[EncoderConfig], FrameEncoder]] = {}


def register_encoder(name: str) -> Callable[[type], type]:
    """Register an encoder implementation under a CLI-visible name."""

    def decorator(cls: type) -> type:
        key = name.lower()
        if key in ENCODERS and ENCODERS[key] is not cls:
            raise ValueError(f"Encoder {name!r} is already registered to {ENCODERS[key]!r}")
        ENCODERS[key] = cls
        cls.encoder_name = key
        return cls

    return decorator


def available_encoders() -> list[str]:
    return sorted(ENCODERS)


def create_encoder(config: EncoderConfig) -> FrameEncoder:
    key = config.name.lower()
    if key not in ENCODERS:
        raise ValueError(f"Unknown encoder {config.name!r}; available encoders are {available_encoders()}")
    return ENCODERS[key](config)


def resolve_device(device: str) -> str:
    """Turn 'auto' into a real device name and reject impossible requests."""
    import torch

    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False")
    return device


def resolve_precision(precision: str, device: str) -> str:
    if precision == "auto":
        return "float16" if device.startswith("cuda") else "float32"
    if precision == "float16" and not device.startswith("cuda"):
        raise ValueError("float16 inference is only supported on CUDA; use float32 on CPU")
    return precision


@register_encoder("hf-clip")
class HFClipEncoder:
    """A CLIP-style checkpoint loaded through Hugging Face Transformers.

    `model` is anything `from_pretrained` accepts: a hub id or a local
    directory. The vision branch loads eagerly; the text branch is only built
    when a query actually has to be encoded, so `build` never pays for it.
    """

    def __init__(self, config: EncoderConfig):
        import torch
        from transformers import CLIPVisionModelWithProjection

        self.config = config
        self.torch = torch
        self.device = torch.device(resolve_device(config.device))
        self.precision = resolve_precision(config.precision, self.device.type)

        self.model = CLIPVisionModelWithProjection.from_pretrained(
            config.model, **self._load_kwargs()
        ).eval().to(self.device)
        self.embedding_dim = int(self.model.config.projection_dim)
        self.preprocess = config.apply_overrides(self._checkpoint_preprocess())
        self._text_model = None
        self._tokenizer = None

    def _load_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {}
        if self.config.revision is not None:
            kwargs["revision"] = self.config.revision
        if self.config.cache_dir is not None:
            kwargs["cache_dir"] = self.config.cache_dir
        if self.config.local_files_only:
            kwargs["local_files_only"] = True
        return kwargs

    def _checkpoint_preprocess(self) -> PreprocessSpec:
        """Ask the checkpoint how it wants frames, falling back to CLIP defaults."""
        try:
            from transformers import CLIPImageProcessor

            processor = CLIPImageProcessor.from_pretrained(self.config.model, **self._load_kwargs())
        except Exception:
            # A checkpoint without a preprocessor_config.json is normal; the
            # CLIP defaults are the right answer for CLIP-architecture weights.
            return PreprocessSpec(image_size=self.model.config.image_size)

        size = getattr(processor, "crop_size", None) or getattr(processor, "size", None)
        if isinstance(size, dict):
            image_size = int(size.get("height") or size.get("shortest_edge") or self.model.config.image_size)
        else:
            image_size = int(size or self.model.config.image_size)

        return PreprocessSpec(
            image_size=image_size,
            mean=tuple(getattr(processor, "image_mean", None) or CLIP_IMAGE_MEAN),
            std=tuple(getattr(processor, "image_std", None) or CLIP_IMAGE_STD),
        )

    def synchronize(self) -> None:
        if self.device.type == "cuda":
            self.torch.cuda.synchronize(self.device)

    def _autocast(self):
        return (
            self.torch.autocast(device_type="cuda", dtype=self.torch.float16)
            if self.precision == "float16"
            else nullcontext()
        )

    def embed_frames(self, frames: np.ndarray, batch_size: int = 128) -> np.ndarray:
        if frames.ndim != 4:
            raise ValueError("frames must have shape [N, C, H, W]")
        if frames.shape[1] != self.preprocess.channels:
            raise ValueError(
                f"frames have {frames.shape[1]} channels but this encoder expects "
                f"{self.preprocess.channels}"
            )
        torch = self.torch
        outputs = []
        for offset in range(0, len(frames), batch_size):
            tensor = torch.from_numpy(frames[offset : offset + batch_size]).to(self.device)
            with torch.inference_mode(), self._autocast():
                encoded = self.model(pixel_values=tensor).image_embeds
            outputs.append(encoded.float().cpu().numpy())
        return np.concatenate(outputs, axis=0)

    def _ensure_text_branch(self) -> None:
        if self._text_model is not None:
            return
        from transformers import CLIPTextModelWithProjection, CLIPTokenizer

        self._tokenizer = CLIPTokenizer.from_pretrained(self.config.model, **self._load_kwargs())
        self._text_model = (
            CLIPTextModelWithProjection.from_pretrained(self.config.model, **self._load_kwargs())
            .eval()
            .to(self.device)
        )

    def encode_text(self, queries: list[str]) -> np.ndarray:
        if not queries:
            raise ValueError("queries must not be empty")
        self._ensure_text_branch()
        torch = self.torch
        inputs = self._tokenizer(text=list(queries), padding=True, return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with torch.inference_mode():
            output = self._text_model(**inputs).text_embeds.float().cpu().numpy()

        norms = np.linalg.norm(output, axis=-1)
        if not np.isfinite(norms).all() or (norms <= np.finfo(np.float32).eps).any():
            raise RuntimeError("The checkpoint produced an invalid all-zero text embedding")
        return l2_normalize(output).astype(np.float32, copy=False)

    def describe(self) -> dict[str, Any]:
        info: dict[str, Any] = {
            "encoder": self.encoder_name,
            "model": self.config.model,
            "revision": self.config.revision,
            "device": str(self.device),
            "precision": self.precision,
            "embedding_dim": self.embedding_dim,
            "preprocess": self.preprocess.to_dict(),
            "torch": self.torch.__version__,
        }
        if self.device.type == "cuda":
            info["gpu"] = self.torch.cuda.get_device_name(self.device)
            info["gpu_memory_bytes"] = self.torch.cuda.get_device_properties(self.device).total_memory
        return info
