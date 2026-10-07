"""
Shared fixtures for the compressed_domain model tests.

Nothing here downloads a checkpoint: `tiny_clip_state_dict` builds a CLIP that is
small enough to run on CPU in milliseconds, and `patched_clip_config` makes
CLIP.get_config hand that back instead of loading ViT-B-32.pt.
"""
import os
import sys
import types
from contextlib import contextmanager
from unittest import mock

# The model code imports itself as `modules.*` / `dataloaders.*`, relative to
# compressed_domain/, so put that directory on the path the way the training
# script's working directory does.
COMPRESSED_DOMAIN = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if COMPRESSED_DOMAIN not in sys.path:
    sys.path.insert(0, COMPRESSED_DOMAIN)

import torch  # noqa: E402

from modules.module_clip import CLIP  # noqa: E402

# Tiny but structurally complete: 1-layer ViT (width 64 = one head, patch 32 on a
# 64x64 image = 4 patches + CLS) and a 1-layer text transformer.
EMBED_DIM = 32
IMAGE_RESOLUTION = 64
CONTEXT_LENGTH = 16
VOCAB_SIZE = 100


def tiny_clip_state_dict(seed=0):
    torch.manual_seed(seed)
    clip = CLIP(
        embed_dim=EMBED_DIM, image_resolution=IMAGE_RESOLUTION, vision_layers=1, vision_width=64,
        vision_patch_size=32, context_length=CONTEXT_LENGTH, vocab_size=VOCAB_SIZE,
        transformer_width=64, transformer_heads=1, transformer_layers=1,
    )
    return {k: v.clone() for k, v in clip.state_dict().items()}


@contextmanager
def patched_clip_config(state_dict=None):
    """Make CLIP.get_config return a tiny CLIP's weights instead of ViT-B/32."""
    state_dict = tiny_clip_state_dict() if state_dict is None else state_dict
    fake = staticmethod(lambda pretrained_clip_name="ViT-B/32": {k: v.clone() for k, v in state_dict.items()})
    with mock.patch.object(CLIP, "get_config", fake):
        yield state_dict


def task_config(visual_branches=None, **kwargs):
    config = types.SimpleNamespace(pretrained_clip_name="ViT-B/32", **kwargs)
    if visual_branches is not None:
        config.visual_branches = visual_branches
    return config


def build_model(visual_branches=None, state_dict=None, seed=0):
    """A CPU, fp32 CLIP4ClipCompressed built on the tiny CLIP."""
    from modules.modeling import CLIP4ClipCompressed

    with patched_clip_config():
        torch.manual_seed(seed)
        model = CLIP4ClipCompressed.from_pretrained(
            state_dict=state_dict, task_config=task_config(visual_branches))
    return model.float()


def fake_batch(batch_size=3, n_iframe=2, n_residual=3, n_mv=4, seed=0):
    """A random batch in the dataloader's layout, with some padded positions."""
    g = torch.Generator().manual_seed(seed)
    res = IMAGE_RESOLUTION
    input_ids = torch.zeros(batch_size, CONTEXT_LENGTH, dtype=torch.long)
    attention_mask = torch.zeros(batch_size, CONTEXT_LENGTH, dtype=torch.long)
    for b in range(batch_size):
        length = 4 + b
        input_ids[b, :length] = torch.randint(1, VOCAB_SIZE - 1, (length,), generator=g)
        input_ids[b, length - 1] = VOCAB_SIZE - 1     # end-of-text: the highest id, as CLIP expects
        attention_mask[b, :length] = 1

    def frames(n, channels):
        x = torch.randn(batch_size, n, channels, res, res, generator=g)
        mask = torch.ones(batch_size, n, dtype=torch.long)
        mask[-1, n // 2 + 1:] = 0                      # last sample is shorter
        return x, mask

    iframe, iframe_mask = frames(n_iframe, 3)
    residuals, residuals_mask = frames(n_residual, 3)
    mv, mv_mask = frames(n_mv, 2)
    return dict(
        input_ids=input_ids, token_type_ids=torch.zeros_like(input_ids), attention_mask=attention_mask,
        iframe=iframe, iframe_mask=iframe_mask, residuals=residuals, residuals_mask=residuals_mask,
        mv=mv, mv_mask=mv_mask,
    )
