"""Configuration for the model-facing side of the pipeline.

These are plain data types with no heavy imports, so both the encoders and the
video preprocessing can depend on them without pulling in torch.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Any


DEFAULT_MODEL_ID = "Searchium-ai/clip4clip-webvid150k"
DEFAULT_ENCODER = "hf-clip"
DEFAULT_POOLING = "meanp"

# OpenAI CLIP's published normalization. Used only when a checkpoint does not
# ship a preprocessor configuration of its own.
CLIP_IMAGE_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_IMAGE_STD = (0.26862954, 0.26130258, 0.27577711)

RESIZE_MODES = ("short-edge-center-crop", "stretch")


@dataclass(frozen=True)
class PreprocessSpec:
    """How a model wants its frames. Owned by the encoder, not by the decoder."""

    image_size: int = 224
    mean: tuple[float, ...] = CLIP_IMAGE_MEAN
    std: tuple[float, ...] = CLIP_IMAGE_STD
    resize_mode: str = "short-edge-center-crop"
    channels: int = 3

    def __post_init__(self) -> None:
        # Frozen dataclass: assign through object.__setattr__ to coerce.
        object.__setattr__(self, "mean", tuple(float(x) for x in self.mean))
        object.__setattr__(self, "std", tuple(float(x) for x in self.std))
        if self.image_size < 1:
            raise ValueError("image_size must be positive")
        if self.channels < 1:
            raise ValueError("channels must be positive")
        if len(self.mean) != self.channels or len(self.std) != self.channels:
            raise ValueError(
                f"mean and std must have {self.channels} values to match channels, "
                f"got {len(self.mean)} and {len(self.std)}"
            )
        if any(value <= 0 for value in self.std):
            raise ValueError("std values must be positive")
        if self.resize_mode not in RESIZE_MODES:
            raise ValueError(f"resize_mode must be one of {RESIZE_MODES}, got {self.resize_mode!r}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EncoderConfig:
    """Everything needed to construct an encoder, and nothing else.

    `image_size`, `image_mean` and `image_std` are overrides: left as None, the
    encoder reports what its own checkpoint asks for.
    """

    name: str = DEFAULT_ENCODER
    model: str = DEFAULT_MODEL_ID
    revision: str | None = None
    cache_dir: str | None = None
    local_files_only: bool = False
    device: str = "auto"
    precision: str = "auto"
    pooling: str = DEFAULT_POOLING
    image_size: int | None = None
    image_mean: tuple[float, ...] | None = None
    image_std: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        if self.image_mean is not None:
            self.image_mean = tuple(float(x) for x in self.image_mean)
        if self.image_std is not None:
            self.image_std = tuple(float(x) for x in self.image_std)

    def apply_overrides(self, spec: PreprocessSpec) -> PreprocessSpec:
        """Layer any explicit preprocessing overrides onto what a model reported."""
        changes: dict[str, Any] = {}
        if self.image_size is not None:
            changes["image_size"] = self.image_size
        if self.image_mean is not None:
            changes["mean"] = self.image_mean
        if self.image_std is not None:
            changes["std"] = self.image_std
        return replace(spec, **changes) if changes else spec


def _field_names(cls: type) -> tuple[str, ...]:
    return tuple(f.name for f in fields(cls))


def encoder_config_from_dict(data: dict[str, Any], base: EncoderConfig | None = None) -> EncoderConfig:
    """Build an EncoderConfig from a mapping, rejecting unknown keys loudly.

    A silently ignored key in a config file is a setting the user believes is
    applied and is not, so this fails instead of guessing.
    """
    valid = _field_names(EncoderConfig)
    unknown = sorted(set(data) - set(valid))
    if unknown:
        raise ValueError(f"Unknown encoder settings {unknown}; valid settings are {sorted(valid)}")
    merged = asdict(base) if base is not None else asdict(EncoderConfig())
    merged.update({key: value for key, value in data.items() if value is not None})
    return EncoderConfig(**merged)


def load_config_file(path: str | Path) -> dict[str, Any]:
    """Load a JSON (always) or YAML (when PyYAML is installed) config file.

    The package's core dependency is NumPy alone, so YAML stays optional and
    says so plainly rather than failing with an import error.
    """
    file_path = Path(path).expanduser()
    if not file_path.is_file():
        raise FileNotFoundError(f"Config file does not exist: {file_path}")
    text = file_path.read_text(encoding="utf-8")

    if file_path.suffix.lower() in (".yaml", ".yml"):
        try:
            import yaml
        except ImportError as exc:
            raise ValueError(
                f"{file_path.name} is YAML, but PyYAML is not installed. "
                "Install it with 'pip install pyyaml', or use a .json config file."
            ) from exc
        data = yaml.safe_load(text)
    else:
        data = json.loads(text)

    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise ValueError(f"Config file must contain a mapping at the top level: {file_path}")

    unknown = sorted(set(data) - {"encoder", "build"})
    if unknown:
        raise ValueError(f"Unknown config sections {unknown}; expected 'encoder' and/or 'build'")
    return data
