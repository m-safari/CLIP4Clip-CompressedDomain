"""Thin Hugging Face adapter for the CLIP4Clip-compatible checkpoint."""

from __future__ import annotations

import os
from contextlib import nullcontext

import numpy as np

# This is a PyTorch-only module. Prevent Transformers from importing an
# unrelated TensorFlow installation (which can also have a different NumPy ABI).
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")


def l2_normalize(values: np.ndarray, axis: int = -1) -> np.ndarray:
    norms = np.linalg.norm(values, axis=axis, keepdims=True)
    return values / np.maximum(norms, np.finfo(np.float32).eps)


def mean_pool_frame_embeddings(frame_embeddings: np.ndarray) -> np.ndarray:
    """CLIP4Clip meanP: normalize frames, average, normalize the video."""
    if frame_embeddings.ndim != 2 or frame_embeddings.shape[0] == 0:
        raise ValueError("frame_embeddings must have shape [frames, dimensions]")
    normalized_frames = l2_normalize(frame_embeddings.astype(np.float32, copy=False))
    return l2_normalize(normalized_frames.mean(axis=0)).astype(np.float32, copy=False)


class Clip4ClipEncoder:
    def __init__(self, model_id: str, device: str = "auto", precision: str = "auto"):
        import torch
        from transformers import CLIPVisionModelWithProjection

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False")
        if precision == "auto":
            precision = "float16" if device.startswith("cuda") else "float32"
        if precision == "float16" and not device.startswith("cuda"):
            raise ValueError("float16 inference is only supported on CUDA; use float32 on CPU")

        self.torch = torch
        self.device = torch.device(device)
        self.precision = precision
        self.model_id = model_id
        self.model = CLIPVisionModelWithProjection.from_pretrained(model_id).eval().to(self.device)
        self.embedding_dim = int(self.model.config.projection_dim)

    def synchronize(self) -> None:
        if self.device.type == "cuda":
            self.torch.cuda.synchronize(self.device)

    def embed_frames(self, frames: np.ndarray, batch_size: int = 128) -> np.ndarray:
        if frames.ndim != 4:
            raise ValueError("frames must have shape [N, 3, H, W]")
        outputs = []
        torch = self.torch
        for offset in range(0, len(frames), batch_size):
            tensor = torch.from_numpy(frames[offset : offset + batch_size]).to(self.device)
            amp = (
                torch.autocast(device_type="cuda", dtype=torch.float16)
                if self.precision == "float16"
                else nullcontext()
            )
            with torch.inference_mode(), amp:
                encoded = self.model(pixel_values=tensor).image_embeds
            outputs.append(encoded.float().cpu().numpy())
        return np.concatenate(outputs, axis=0)


def encode_text_query(model_id: str, query: str, device: str = "auto") -> np.ndarray:
    import torch
    from transformers import CLIPTextModelWithProjection, CLIPTokenizer

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = CLIPTokenizer.from_pretrained(model_id)
    model = CLIPTextModelWithProjection.from_pretrained(model_id).eval().to(device)
    inputs = tokenizer(text=[query], padding=True, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.inference_mode():
        output = model(**inputs).text_embeds.float().cpu().numpy()[0]
    norm = float(np.linalg.norm(output))
    if not np.isfinite(norm) or norm <= np.finfo(np.float32).eps:
        raise RuntimeError("The checkpoint produced an invalid all-zero text embedding")
    return l2_normalize(output).astype(np.float32, copy=False)
