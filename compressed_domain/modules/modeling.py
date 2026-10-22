from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

import torch
from torch import nn

from modules.until_module import PreTrainedModel, CrossEn
from modules.until_config import PretrainedConfig
from modules.module_clip import CLIP, VisualTransformer, convert_weights
from modules.module_compact import CompactResNetEncoder

logger = logging.getLogger(__name__)

# The visual branches a CLIP4ClipCompressed can be built with, in the order their
# frame embeddings are concatenated before pooling.
VISUAL_BRANCHES = ("iframe", "residual", "mv")


def resolve_visual_branches(task_config):
    """
    Which visual branches to build, read from `task_config.visual_branches`.

    Absent or None means all three, so every existing caller keeps the full model.
    Any non-empty subset is allowed -- that is how ablated variants are built, e.g.
    ("residual", "mv") for a model with no I-frame tower at all. Returned in
    VISUAL_BRANCHES order, whatever order they were requested in.
    """
    requested = getattr(task_config, "visual_branches", None)
    if requested is None:
        return VISUAL_BRANCHES
    if isinstance(requested, str):
        requested = [requested]
    requested = list(requested)
    if not requested:
        raise ValueError("visual_branches must name at least one of {}".format(VISUAL_BRANCHES))
    unknown = sorted(set(requested) - set(VISUAL_BRANCHES))
    if unknown:
        raise ValueError("Unknown visual_branches {}; valid branches are {}".format(unknown, VISUAL_BRANCHES))
    if len(set(requested)) != len(requested):
        raise ValueError("visual_branches has duplicates: {}".format(requested))
    return tuple(branch for branch in VISUAL_BRANCHES if branch in requested)


# How the frame embeddings of the built branches become one video embedding.
#   "modality": mean over the frames of each modality, then mean over the
#               modalities. Every modality weighs the same however many frames
#               it contributes.
#   "frame":    one mean over all frames of all modalities (the original
#               behaviour); a modality's weight is its share of the frames.
VISUAL_POOLINGS = ("modality", "frame")

# Initialisation of the motion-vector encoder (see from_pretrained).
MV_INITS = ("random", "clip")

# Architecture of the residual / motion-vector encoders (the I-frame tower is always CLIP's ViT).
#   "vit":       a full copy of CLIP's visual tower (12 layers, ~88M parameters). The original.
#   "vit_small": the same ViT cut to its first `<branch>_layers` blocks (default 4, ~31M).
#                CLIP-initialisable: the patch embedding and the first blocks (low-level,
#                edge-like features) transfer, which suits residuals.
#   "cnn":       ResNet-18-style CNN (~11M), as CoViAR uses for residuals and motion vectors.
#                No CLIP weights apply to it; it trains from scratch.
# All three expose the interface _encode_modality needs, so inputs, outputs, pooling,
# similarity and loss are identical whatever the choice.
BRANCH_ARCHS = ("vit", "vit_small", "cnn")
DEFAULT_SMALL_VIT_LAYERS = 4


def resolve_branch_arch(task_config, branch, full_layers):
    """(arch, n_transformer_layers) of the residual / mv encoder, from task_config.<branch>_arch / _layers."""
    arch = getattr(task_config, branch + "_arch", None) or "vit"
    if arch not in BRANCH_ARCHS:
        raise ValueError("Unknown {}_arch {!r}; valid values are {}".format(branch, arch, BRANCH_ARCHS))
    layers = full_layers
    if arch == "vit_small":
        layers = int(getattr(task_config, branch + "_layers", None) or DEFAULT_SMALL_VIT_LAYERS)
        if not 1 <= layers <= full_layers:
            raise ValueError("{}_layers must be in [1, {}], got {}".format(branch, full_layers, layers))
    return arch, layers


def _clip_visual_layers(clip_state_dict):
    return len([k for k in clip_state_dict if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])


def resolve_visual_pooling(task_config):
    pooling = getattr(task_config, "visual_pooling", None) or "modality"
    if pooling not in VISUAL_POOLINGS:
        raise ValueError("Unknown visual_pooling {!r}; valid values are {}".format(pooling, VISUAL_POOLINGS))
    return pooling


def _safe_normalize(x, eps=1e-6):
    # Rows that are exactly zero (masked-out frames / absent modalities) stay
    # zero instead of becoming NaN.
    return x / x.norm(dim=-1, keepdim=True).clamp_min(eps)


class _EmptyConfig(PretrainedConfig):
    """
    Placeholder config. `until_module.PreTrainedModel.__init__` requires a
    `PretrainedConfig` instance, and `init_weights` reads `initializer_range`
    off it for Linear/Embedding init. Nothing else in this model touches the
    config, so this is hardcoded to CLIP4Clip's standard value rather than
    threaded through as a real option — revisit if/when you want it tunable.
    """
    def __init__(self):
        self.initializer_range = 0.02


class CLIP4ClipPreTrainedModel(PreTrainedModel, nn.Module):
    """
    Thin base class providing the weight-loading plumbing (`init_weights`,
    `init_preweight`, `from_pretrained`) inherited from `until_module.PreTrainedModel`.
    Unlike the original CLIP4Clip, there is no `cross` submodule here — the tight
    cross-attention similarity path (module_cross.CrossModel) has been removed.
    """
    def __init__(self, config, *inputs, **kwargs):
        super(CLIP4ClipPreTrainedModel, self).__init__(config)
        self.clip = None

    @classmethod
    def from_pretrained(cls, state_dict=None, task_config=None, *inputs, **kwargs):
        if state_dict is None:
            state_dict = {}

        pretrained_clip_name = getattr(task_config, "pretrained_clip_name", "ViT-B/32")
        clip_state_dict = CLIP.get_config(pretrained_clip_name=pretrained_clip_name)

        # I-frame + text branch: identical to stock CLIP, loaded under the "clip." prefix.
        for key, val in clip_state_dict.items():
            new_key = "clip." + key
            if new_key not in state_dict:
                state_dict[new_key] = val.clone()

        # Keys the caller supplied, before the CLIP defaults below are filled in; used at the
        # end to catch a checkpoint saved with a different residual / mv architecture.
        supplied_keys = set(state_dict.keys())
        branches = resolve_visual_branches(task_config)
        full_layers = _clip_visual_layers(clip_state_dict)

        def copy_clip_visual(prefix, layers, mv_conv1=False):
            # CLIP's visual tower (its first `layers` transformer blocks) under `prefix`.
            for key, val in clip_state_dict.items():
                if not key.startswith("visual."):
                    continue
                sub = key[len("visual."):]
                if sub.startswith("transformer.resblocks.") and int(sub.split(".")[2]) >= layers:
                    continue
                new_key = prefix + sub
                if new_key not in state_dict:
                    if mv_conv1 and key == "visual.conv1.weight":
                        val = val.mean(dim=1, keepdim=True).repeat(1, 2, 1, 1)
                    state_dict[new_key] = val.clone()

        # Residual branch: same 3-channel, 224x224 input shape as I-frames, so the
        # pretrained CLIP visual weights transfer directly (shape-for-shape). With
        # residual_arch == "vit_small" only the first residual_layers blocks are copied;
        # with "cnn" there is nothing to copy.
        if "residual" in branches:
            arch, layers = resolve_branch_arch(task_config, "residual", full_layers)
            if arch != "cnn":
                copy_clip_visual("residual_encoder.", layers)

        # Motion-vector branch. By default it keeps its random initialization: its
        # 2-channel (dx, dy) input has no correspondence to RGB. With
        # task_config.mv_init == "clip" it is warm-started like the residual branch:
        # CLIP's visual transformer is copied, and the 2-channel patch embedding is
        # the mean of CLIP's RGB filters (CoViAR likewise starts its motion-vector
        # network from image-pretrained weights). A checkpoint's own mv_encoder
        # weights always take precedence. mv_init does not apply to the CNN.
        mv_init = getattr(task_config, "mv_init", None) or "random"
        if mv_init not in MV_INITS:
            raise ValueError("Unknown mv_init {!r}; valid values are {}".format(mv_init, MV_INITS))
        if "mv" in branches:
            arch, layers = resolve_branch_arch(task_config, "mv", full_layers)
            if mv_init == "clip":
                if arch == "cnn":
                    raise ValueError("mv_init='clip' needs a ViT motion-vector encoder (mv_arch vit or "
                                     "vit_small); the CNN has no CLIP weights to start from.")
                copy_clip_visual("mv_encoder.", layers, mv_conv1=True)

        config = _EmptyConfig()
        model = cls(config, clip_state_dict, task_config)
        _check_branch_arch_matches(model, state_dict, supplied_keys)
        model = cls.init_preweight(model, state_dict, task_config=task_config)
        return model


def _check_branch_arch_matches(model, state_dict, supplied_keys):
    """
    init_preweight only logs weights that do not fit, it does not fail, so a checkpoint saved with
    another residual / mv architecture would silently leave that branch at its random init. Refuse it.
    Only keys the caller supplied are checked; CLIP defaults filled in by from_pretrained always fit.
    A branch the model was built without is skipped: loading a full checkpoint into an ablated model
    is supported, and that branch's weights are simply not used.
    """
    model_state = model.state_dict()
    for branch, prefix in (("residual", "residual_encoder."), ("mv", "mv_encoder.")):
        if branch not in model.visual_branches:
            continue
        unexpected = sorted(k for k in supplied_keys if k.startswith(prefix) and k not in model_state)
        mismatched = sorted(k for k in supplied_keys
                            if k in model_state and k.startswith(prefix) and state_dict[k].shape != model_state[k].shape)
        if unexpected or mismatched:
            raise ValueError(
                "The checkpoint's {0} encoder does not match the model being built ({1} unexpected keys, "
                "{2} shape mismatches; e.g. {3}). It was probably trained with a different --{0}_arch or "
                "--{0}_layers: pass the same values here.".format(
                    branch, len(unexpected), len(mismatched), (unexpected + mismatched)[0]))


def show_log(task_config, info):
    logger.warning(info)


class CLIP4ClipCompressed(CLIP4ClipPreTrainedModel):
    """
    Compressed-domain CLIP4Clip baseline.

    Visual branch: three independent CLIP-ViT encoders (I-frame, residual, motion
    vector) each produce one CLS embedding per frame, in the same joint space as
    the text encoder. With `task_config.visual_pooling == "modality"` (default)
    each modality's frames are mean-pooled first and the modality embeddings are
    then averaged, so every modality has the same weight. With "frame", all
    frames of all modalities share one masked mean (the original behaviour).

    Similarity: loose (cosine) similarity only. No tight/cross-attention path.
    Loss: CrossEn (the same symmetric contrastive loss as stock CLIP4Clip).

    Ablation: `task_config.visual_branches` (see resolve_visual_branches) builds
    any non-empty subset of the three encoders. A dropped branch has no weights at
    all; its frames are still accepted (and ignored) by forward() and
    get_visual_output(), so the dataloader and training/eval loops need no changes.

    Assumes the dataloader already yields tensors shaped:
        input_ids / attention_mask / token_type_ids : (B, L)
        iframe / residuals                            : (B, T, 3, H, W)
        mv                                             : (B, T, 2, H, W)
        *_mask                                         : (B, T)
    (an extra "n_pair" dimension after B is flattened into the batch).
    """

    def __init__(self, config, clip_state_dict, task_config):
        super(CLIP4ClipCompressed, self).__init__(config)
        self.task_config = task_config

        vit = "visual.proj" in clip_state_dict
        assert vit, "Only ViT-based CLIP checkpoints are supported."

        vision_width = clip_state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len([k for k in clip_state_dict.keys()
                              if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        vision_patch_size = clip_state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round((clip_state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_resolution = vision_patch_size * grid_size

        embed_dim = clip_state_dict["text_projection"].shape[1]
        context_length = clip_state_dict["positional_embedding"].shape[0]
        vocab_size = clip_state_dict["token_embedding.weight"].shape[0]
        transformer_width = clip_state_dict["ln_final.weight"].shape[0]
        transformer_heads = transformer_width // 64
        transformer_layers = len(set(k.split(".")[2] for k in clip_state_dict
                                      if k.startswith("transformer.resblocks")))

        show_log(task_config, "embed_dim: {}".format(embed_dim))
        show_log(task_config, "image_resolution: {}".format(image_resolution))
        show_log(task_config, "vision_layers: {}".format(vision_layers))
        show_log(task_config, "vision_width: {}".format(vision_width))
        show_log(task_config, "vision_patch_size: {}".format(vision_patch_size))
        show_log(task_config, "context_length: {}".format(context_length))
        show_log(task_config, "transformer_width: {}".format(transformer_width))
        show_log(task_config, "transformer_layers: {}".format(transformer_layers))

        self.visual_branches = resolve_visual_branches(task_config)
        show_log(task_config, "visual_branches: {}".format(", ".join(self.visual_branches)))

        # --- Text encoder + I-frame visual encoder: stock CLIP ---
        self.clip = CLIP(
            embed_dim, image_resolution, vision_layers, vision_width, vision_patch_size,
            context_length, vocab_size, transformer_width, transformer_heads, transformer_layers,
            linear_patch='2d',
        ).float()
        if "iframe" not in self.visual_branches:
            # The text encoder lives in the same CLIP module, so only its visual tower is cut.
            self.clip.visual = None
        convert_weights(self.clip)

        vision_heads = vision_width // 64

        # --- Residual / motion-vector encoders: any of BRANCH_ARCHS, own weights ---
        # Residual: 3-channel input like the I-frames (CLIP-initialised unless "cnn").
        # Motion vector: 2-channel (dx, dy) input, random init unless mv_init == "clip".
        def build_encoder(branch, input_channels):
            arch, layers = resolve_branch_arch(task_config, branch, vision_layers)
            if arch == "cnn":
                encoder = CompactResNetEncoder(input_channels=input_channels, output_dim=embed_dim).float()
            else:
                encoder = VisualTransformer(
                    input_resolution=image_resolution, patch_size=vision_patch_size, width=vision_width,
                    layers=layers, heads=vision_heads, output_dim=embed_dim,
                    linear_patch='2d', input_channels=input_channels,
                ).float()
            convert_weights(encoder)
            show_log(task_config, "{} encoder: {}{}, {:.1f}M parameters".format(
                branch, arch, " ({} layers)".format(layers) if arch != "cnn" else "",
                sum(p.numel() for p in encoder.parameters()) / 1e6))
            return encoder

        self.residual_encoder = build_encoder("residual", 3) if "residual" in self.visual_branches else None
        self.mv_encoder = build_encoder("mv", 2) if "mv" in self.visual_branches else None

        self.visual_pooling = resolve_visual_pooling(task_config)
        show_log(task_config, "visual_pooling: {}".format(self.visual_pooling))

        # Weight of the per-branch auxiliary losses (see forward). 0 = off.
        self.modality_loss_weight = float(getattr(task_config, "modality_loss_weight", 0.) or 0.)
        show_log(task_config, "modality_loss_weight: {}".format(self.modality_loss_weight))

        if getattr(task_config, "grad_checkpoint", False):
            for encoder in (self.clip.visual, self.residual_encoder, self.mv_encoder):
                if encoder is not None:
                    # ViTs checkpoint their transformer blocks, the CNN its stages.
                    getattr(encoder, "transformer", encoder).grad_checkpointing = True

        self.loss_fct = CrossEn()

        self.apply(self.init_weights)

    # ---------------------------------------------------------------- #
    # Encoding
    # ---------------------------------------------------------------- #

    def get_sequence_output(self, input_ids, token_type_ids=None, attention_mask=None):
        # token_type_ids / attention_mask are accepted for interface parity with the
        # dataloader batch, but (as in stock CLIP4Clip) CLIP's text encoder doesn't
        # use them: it builds its own causal attention mask internally.
        input_ids = input_ids.view(-1, input_ids.shape[-1])
        bs_pair = input_ids.size(0)
        sequence_hidden = self.clip.encode_text(input_ids).float()
        sequence_hidden = sequence_hidden.view(bs_pair, -1, sequence_hidden.size(-1))
        return sequence_hidden

    def _encode_modality(self, encoder, frames, mask):
        """
        Run one modality's frames through its VisualTransformer and return one
        CLS embedding per frame, in the shared embed_dim space. Mirrors
        CLIP.encode_image's post-processing (ln_post -> proj on the CLS token),
        generalized to any VisualTransformer instance.

        Only real frames (mask == 1) are encoded; padded positions come out as
        zero vectors and cost no encoder memory.

        frames: (..., T, C, H, W) -- any number of leading dims (batch, n_pair, ...)
        are flattened into one batch dim, the same way get_sequence_output flattens
        text's leading dims via .view(-1, last_dim).
        mask:   (B_flat, T)
        Returns (B_flat, T, embed_dim), float32.
        """
        frames = frames.contiguous().view(-1, *frames.shape[-4:])
        b, t = frames.shape[:2]
        valid = mask.bool()

        out = torch.zeros(b, t, encoder.proj.shape[1], dtype=torch.float32, device=frames.device)
        if valid.any():
            x = frames[valid].type(encoder.conv1.weight.dtype)      # (N_valid, C, H, W)
            hidden = encoder(x, video_frame=t)                       # (N_valid, tokens, width)
            cls = encoder.ln_post(hidden[:, 0, :]) @ encoder.proj    # (N_valid, embed_dim)
            out[valid] = cls.float()
        return out

    def get_visual_output(self, iframe, iframe_mask, residuals, residuals_mask, mv, mv_mask):
        """
        Returns (visual_output, video_mask) to be pooled by pool_video():

            visual_pooling == "modality": (B, M, D) one embedding per built
                modality, and (B, M) = 1 where that modality has any real frame.
            visual_pooling == "frame":    (B, T_total, D) one embedding per frame
                of every built modality, and the concatenated frame masks.
        """
        # Every modality is always passed in; only the branches this model was
        # built with are encoded. A dropped branch's frames are ignored.
        inputs = {
            "iframe": (self.clip.visual, iframe, iframe_mask),
            "residual": (self.residual_encoder, residuals, residuals_mask),
            "mv": (self.mv_encoder, mv, mv_mask),
        }

        features, masks = [], []
        for branch in self.visual_branches:
            encoder, frames, mask = inputs[branch]
            # Masks get the same leading-dim flattening as their frame tensors so the
            # two stay aligned (e.g. (B, n_pair, T) -> (B*n_pair, T)).
            mask = mask.contiguous().view(-1, mask.shape[-1])
            feature = self._encode_modality(encoder, frames, mask)
            if self.visual_pooling == "modality":
                feature, mask = self._pool_frames(feature, mask)
            features.append(feature)
            masks.append(mask)

        visual_output = torch.cat(features, dim=1)
        video_mask = torch.cat(masks, dim=1)
        return visual_output, video_mask

    def valid_video_rows(self, iframe_mask, residuals_mask, mv_mask):
        """
        (B,) bool: True where a video has at least one real frame in a branch this
        model was built with. False means the video failed to decode (or has no
        frame of any built modality); it has no embedding and must be excluded.
        """
        masks = {"iframe": iframe_mask, "residual": residuals_mask, "mv": mv_mask}
        valid = None
        for branch in self.visual_branches:
            mask = masks[branch]
            has_frames = mask.contiguous().view(-1, mask.shape[-1]).sum(dim=-1) > 0
            valid = has_frames if valid is None else (valid | has_frames)
        return valid

    def get_sequence_visual_output(self, input_ids, token_type_ids, attention_mask,
                                    iframe, iframe_mask, residuals, residuals_mask, mv, mv_mask):
        sequence_output = self.get_sequence_output(input_ids, token_type_ids, attention_mask)
        visual_output, video_mask = self.get_visual_output(
            iframe, iframe_mask, residuals, residuals_mask, mv, mv_mask)
        return sequence_output, visual_output, video_mask

    # ---------------------------------------------------------------- #
    # Pooling + similarity (meanP / loose only)
    # ---------------------------------------------------------------- #

    def _mean_pooling_for_similarity_sequence(self, sequence_output, attention_mask):
        attention_mask_un = attention_mask.to(dtype=torch.float).unsqueeze(-1)
        attention_mask_un[:, 0, :] = 0.
        sequence_output = sequence_output * attention_mask_un
        text_out = torch.sum(sequence_output, dim=1) / torch.sum(attention_mask_un, dim=1, dtype=torch.float)
        return text_out

    def _mean_pooling_for_similarity_visual(self, visual_output, video_mask):
        video_mask_un = video_mask.to(dtype=torch.float).unsqueeze(-1)
        visual_output = visual_output * video_mask_un
        video_mask_un_sum = torch.sum(video_mask_un, dim=1, dtype=torch.float)
        video_mask_un_sum[video_mask_un_sum == 0.] = 1.
        video_out = torch.sum(visual_output, dim=1) / video_mask_un_sum
        return video_out

    @staticmethod
    def _pool_frames(frame_features, frame_mask):
        """(B, T, D) frame embeddings of ONE modality -> ((B, 1, D), (B, 1))."""
        mask = frame_mask.to(dtype=torch.float).unsqueeze(-1)
        pooled = torch.sum(_safe_normalize(frame_features) * mask, dim=1) / mask.sum(dim=1).clamp_min(1.)
        return pooled.unsqueeze(1), (frame_mask.sum(dim=1, keepdim=True) > 0).to(frame_mask.dtype)

    def pool_video(self, visual_output, video_mask):
        """
        get_visual_output()'s result -> (B, D) L2-normalised video embeddings.
        A video with nothing unmasked has no embedding and comes out as NaN;
        callers exclude such videos (see valid_video_rows).
        """
        visual_output = _safe_normalize(visual_output.contiguous())
        visual_output = self._mean_pooling_for_similarity_visual(visual_output, video_mask)
        return visual_output / visual_output.norm(dim=-1, keepdim=True)

    def _loose_similarity(self, sequence_output, visual_output, attention_mask, video_mask):
        sequence_output = sequence_output.contiguous()
        visual_output = self.pool_video(visual_output, video_mask)

        sequence_output = sequence_output.squeeze(1)
        sequence_output = sequence_output / sequence_output.norm(dim=-1, keepdim=True)

        logit_scale = self.clip.logit_scale.exp()
        retrieve_logits = logit_scale * torch.matmul(sequence_output, visual_output.t())
        return retrieve_logits

    def get_similarity_logits(self, sequence_output, visual_output, attention_mask, video_mask):
        return self._loose_similarity(sequence_output, visual_output, attention_mask, video_mask)

    def _modality_loss(self, sequence_output, visual_output, video_mask):
        """
        Mean, over the built branches, of the same symmetric contrastive loss
        computed between the text and THAT branch's embedding alone.

        The main loss only sees the average of the branch embeddings, so a branch
        that already solves the task (the CLIP-pretrained I-frame tower) leaves
        the others with little to learn. This term makes every branch align with
        the text on its own. Needs per-branch embeddings, i.e. modality pooling
        with more than one branch; otherwise it is 0.
        """
        zero = visual_output.new_zeros(())
        if self.visual_pooling != "modality" or visual_output.shape[1] < 2:
            return zero

        text = sequence_output.contiguous().squeeze(1)
        text = text / text.norm(dim=-1, keepdim=True)
        logit_scale = self.clip.logit_scale.exp()

        losses = []
        for m in range(visual_output.shape[1]):
            rows = video_mask[:, m] > 0              # videos that have this branch
            if int(rows.sum()) < 2:
                continue
            branch = _safe_normalize(visual_output[rows, m])
            sim = logit_scale * torch.matmul(text[rows], branch.t())
            losses.append((self.loss_fct(sim) + self.loss_fct(sim.T)) / 2)
        return torch.stack(losses).mean() if losses else zero

    # ---------------------------------------------------------------- #
    # Forward
    # ---------------------------------------------------------------- #

    def forward(self, input_ids, token_type_ids, attention_mask,
                iframe, iframe_mask, residuals, residuals_mask, mv, mv_mask):
        input_ids = input_ids.view(-1, input_ids.shape[-1])
        attention_mask = attention_mask.view(-1, attention_mask.shape[-1])

        sequence_output, visual_output, video_mask = self.get_sequence_visual_output(
            input_ids, token_type_ids, attention_mask,
            iframe, iframe_mask, residuals, residuals_mask, mv, mv_mask)

        if self.training:
            sim_matrix = self.get_similarity_logits(sequence_output, visual_output, attention_mask, video_mask)
            sim_loss1 = self.loss_fct(sim_matrix)
            sim_loss2 = self.loss_fct(sim_matrix.T)
            loss = (sim_loss1 + sim_loss2) / 2
            if self.modality_loss_weight > 0:
                loss = loss + self.modality_loss_weight * self._modality_loss(
                    sequence_output, visual_output, video_mask)
            return loss

        return None