"""
Compact CNN encoder for the residual and motion-vector branches.

Residuals and motion vectors are sparse, edge-like, block-structured signals,
not natural images. CoViAR (Wu et al., CVPR 2018) models them with a ResNet-18,
a small network, while the I-frames get a much bigger one. This module is that
idea: a ResNet-18-style CNN (~11M parameters) instead of a 12-layer ViT-B/32
(~88M).

It is a drop-in for `module_clip.VisualTransformer` as far as
`CLIP4ClipCompressed._encode_modality` is concerned. It exposes the same
attributes and the same call convention:

    conv1               first conv layer (its dtype is the input dtype)
    ln_post, proj       final LayerNorm and (width, embed_dim) projection
    forward(x, video_frame=-1) -> (N, L, width), the summary token at index 0

Here L == 1: the "summary token" is the globally average-pooled feature map.
The caller applies ln_post and proj to it, exactly as for the ViT's CLS token,
so the output is one embed_dim vector per frame in the shared CLIP space.
"""
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from modules.module_clip import LayerNorm


class _GroupNorm(nn.GroupNorm):
    """GroupNorm computed in fp32 (like module_clip.LayerNorm), so it also works with fp16 weights/inputs."""

    def forward(self, x):
        return super().forward(x.type(torch.float32)).type(x.dtype)


def _norm(channels):
    # GroupNorm rather than BatchNorm: no running statistics to get wrong when
    # frames of one video share a batch, when failed videos are dropped from it,
    # or when evaluation uses a different batch size than training.
    return _GroupNorm(32, channels)


class _BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.norm1 = _norm(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False)
        self.norm2 = _norm(out_channels)
        self.act = nn.ReLU()
        self.shortcut = None
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                _norm(out_channels),
            )

    def forward(self, x):
        out = self.act(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        identity = x if self.shortcut is None else self.shortcut(x)
        return self.act(out + identity)


class CompactResNetEncoder(nn.Module):
    """
    ResNet-18-style encoder: 7x7/2 stem, 3x3/2 max-pool, four stages of two
    basic blocks (64-128-256-512 channels, strides 1-2-2-2), global average pool.
    Any input resolution works (the pooling is global).

    input_channels: 3 for residuals, 2 for motion vectors (dx, dy).
    output_dim:     embedding size of the shared CLIP space (512 for ViT-B/32).
    """

    def __init__(self, input_channels, output_dim, layers=(2, 2, 2, 2), widths=(64, 128, 256, 512)):
        super().__init__()
        self.input_channels = input_channels
        self.output_dim = output_dim

        self.conv1 = nn.Conv2d(input_channels, widths[0], 7, stride=2, padding=3, bias=False)
        self.norm1 = _norm(widths[0])
        self.act = nn.ReLU()
        self.pool = nn.MaxPool2d(3, stride=2, padding=1)

        stages, in_channels = [], widths[0]
        for i, (n_blocks, out_channels) in enumerate(zip(layers, widths)):
            blocks = []
            for b in range(n_blocks):
                stride = 2 if (b == 0 and i > 0) else 1
                blocks.append(_BasicBlock(in_channels, out_channels, stride))
                in_channels = out_channels
            stages.append(nn.Sequential(*blocks))
        self.stages = nn.ModuleList(stages)

        width = widths[-1]
        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(width ** -0.5 * torch.randn(width, output_dim))

        # Same switch the ViT's Transformer has: recompute stage activations in backward.
        self.grad_checkpointing = False

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.GroupNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        # Each residual block starts as the identity: easier optimisation from scratch.
        for m in self.modules():
            if isinstance(m, _BasicBlock):
                nn.init.zeros_(m.norm2.weight)

    def forward(self, x, video_frame=-1):
        # video_frame is accepted for interface parity with VisualTransformer and ignored.
        x = self.pool(self.act(self.norm1(self.conv1(x))))
        for stage in self.stages:
            if self.grad_checkpointing and self.training and x.requires_grad:
                x = checkpoint(stage, x, use_reentrant=False)
            else:
                x = stage(x)
        return x.mean(dim=(2, 3)).unsqueeze(1)      # (N, 1, width)