from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

import torch
from torch import nn

from modules.until_module import PreTrainedModel, CrossEn
from modules.until_config import PretrainedConfig
from modules.module_clip import CLIP, VisualTransformer, convert_weights

logger = logging.getLogger(__name__)


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

        # Residual branch: same 3-channel, 224x224 input shape as I-frames, so the
        # pretrained CLIP visual weights transfer directly (shape-for-shape).
        for key, val in clip_state_dict.items():
            if key.startswith("visual."):
                new_key = "residual_encoder." + key[len("visual."):]
                if new_key not in state_dict:
                    state_dict[new_key] = val.clone()

        # Motion-vector branch is intentionally left with its random initialization:
        # its 2-channel (dx, dy) input has no correspondence to RGB, so warm-starting
        # it from CLIP's RGB conv1 filters would be a speculative, ungrounded choice.

        config = PretrainedConfig.from_dict({})
        model = cls(config, clip_state_dict, task_config)
        model = cls.init_preweight(model, state_dict, task_config=task_config)
        return model


def show_log(task_config, info):
    logger.warning(info)


class CLIP4ClipCompressed(CLIP4ClipPreTrainedModel):
    """
    Compressed-domain CLIP4Clip baseline.

    Visual branch: three independent CLIP-ViT encoders (I-frame, residual, motion
    vector) each produce one CLS embedding per frame, in the same joint space as
    the text encoder. All frames from all three modalities are concatenated and
    pooled with a single masked mean (meanP) -- exactly the pooling CLIP4Clip
    already applies across frames of one modality, just applied across a longer,
    mixed-modality frame sequence.

    Similarity: loose (cosine) similarity only. No tight/cross-attention path.
    Loss: CrossEn (the same symmetric contrastive loss as stock CLIP4Clip).

    Assumes the dataloader already yields tensors shaped:
        input_ids / attention_mask / token_type_ids : (B, L)
        iframe / residuals                            : (B, T, 3, H, W)
        mv                                             : (B, T, 2, H, W)
        *_mask                                         : (B, T)
    with no extra "n_pair" dimension.
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

        # --- Text encoder + I-frame visual encoder: stock CLIP ---
        self.clip = CLIP(
            embed_dim, image_resolution, vision_layers, vision_width, vision_patch_size,
            context_length, vocab_size, transformer_width, transformer_heads, transformer_layers,
            linear_patch='2d',
        ).float()
        convert_weights(self.clip)

        vision_heads = vision_width // 64

        # --- Residual encoder: same architecture/channels as I-frame branch, own weights ---
        self.residual_encoder = VisualTransformer(
            input_resolution=image_resolution, patch_size=vision_patch_size, width=vision_width,
            layers=vision_layers, heads=vision_heads, output_dim=embed_dim,
            linear_patch='2d', input_channels=3,
        ).float()
        convert_weights(self.residual_encoder)

        # --- Motion-vector encoder: 2-channel (dx, dy) input, trained from scratch ---
        self.mv_encoder = VisualTransformer(
            input_resolution=image_resolution, patch_size=vision_patch_size, width=vision_width,
            layers=vision_layers, heads=vision_heads, output_dim=embed_dim,
            linear_patch='2d', input_channels=2,
        ).float()
        convert_weights(self.mv_encoder)

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

    def _encode_modality(self, encoder, frames, video_frame):
        """
        Run one modality's frames through its VisualTransformer and return one
        CLS embedding per frame, in the shared embed_dim space. Mirrors
        CLIP.encode_image's post-processing (ln_post -> proj -> take CLS token),
        generalized to any VisualTransformer instance.

        frames: (B, T, C, H, W)  ->  returns (B, T, embed_dim)
        """
        b, t, c, h, w = frames.shape
        frames = frames.contiguous().view(b * t, c, h, w)
        frames = frames.type(encoder.conv1.weight.dtype)

        hidden = encoder(frames, video_frame=video_frame)          # (B*T, tokens, width)
        hidden = encoder.ln_post(hidden) @ encoder.proj             # (B*T, tokens, embed_dim)
        cls = hidden[:, 0, :].float()                                # (B*T, embed_dim)
        cls = cls.view(b, t, -1)
        return cls

    def get_visual_output(self, iframe, iframe_mask, residuals, residuals_mask, mv, mv_mask):
        iframe_feat = self._encode_modality(self.clip.visual, iframe, iframe.shape[1])
        residual_feat = self._encode_modality(self.residual_encoder, residuals, residuals.shape[1])
        mv_feat = self._encode_modality(self.mv_encoder, mv, mv.shape[1])

        # All frames from all three modalities become one pooled set of "frames",
        # the same way CLIP4Clip already pools frames within a single modality.
        visual_output = torch.cat([iframe_feat, residual_feat, mv_feat], dim=1)   # (B, T_total, D)
        video_mask = torch.cat([iframe_mask, residuals_mask, mv_mask], dim=1)      # (B, T_total)
        return visual_output, video_mask

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

    def _loose_similarity(self, sequence_output, visual_output, attention_mask, video_mask):
        sequence_output = sequence_output.contiguous()
        visual_output = visual_output.contiguous()

        visual_output = visual_output / visual_output.norm(dim=-1, keepdim=True)
        visual_output = self._mean_pooling_for_similarity_visual(visual_output, video_mask)
        visual_output = visual_output / visual_output.norm(dim=-1, keepdim=True)

        sequence_output = sequence_output.squeeze(1)
        sequence_output = sequence_output / sequence_output.norm(dim=-1, keepdim=True)

        logit_scale = self.clip.logit_scale.exp()
        retrieve_logits = logit_scale * torch.matmul(sequence_output, visual_output.t())
        return retrieve_logits

    def get_similarity_logits(self, sequence_output, visual_output, attention_mask, video_mask):
        return self._loose_similarity(sequence_output, visual_output, attention_mask, video_mask)

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
            return loss

        return None
