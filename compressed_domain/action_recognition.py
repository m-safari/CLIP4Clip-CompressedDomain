"""
Zero-shot action recognition with a (possibly ablated) CLIP4ClipCompressed.

Each HMDB51 class becomes a text prompt; a video is assigned the class whose
prompt embedding is most similar to its video embedding, using exactly the
similarity the retrieval model was trained with (_loose_similarity: per-frame
L2 normalisation, masked mean over all frames of all built branches, L2
normalisation, cosine).

Used by reports/hmdb51_action_recognition.ipynb; nothing here is specific to
a notebook.
"""
import types

import numpy as np
import torch

from dataloaders.prepare_hmdb51_dataset import HMDB51_CLASSES
from modules.modeling import CLIP4ClipCompressed, VISUAL_BRANCHES

# ---------------------------------------------------------------------- #
# Prompts
# ---------------------------------------------------------------------- #

# Each class label as a short natural-language phrase (present participle),
# following the HMDB51 class descriptions.
CLASS_PHRASES = {
    "brush_hair": "brushing hair",
    "cartwheel": "doing a cartwheel",
    "catch": "catching a ball",
    "chew": "chewing",
    "clap": "clapping",
    "climb": "climbing",
    "climb_stairs": "climbing stairs",
    "dive": "diving",
    "draw_sword": "drawing a sword",
    "dribble": "dribbling a ball",
    "drink": "drinking",
    "eat": "eating",
    "fall_floor": "falling on the floor",
    "fencing": "fencing",
    "flic_flac": "doing a back handspring",
    "golf": "swinging a golf club",
    "handstand": "doing a handstand",
    "hit": "hitting something",
    "hug": "hugging someone",
    "jump": "jumping",
    "kick": "kicking",
    "kick_ball": "kicking a ball",
    "kiss": "kissing someone",
    "laugh": "laughing",
    "pick": "picking something up",
    "pour": "pouring a drink",
    "pullup": "doing pull ups",
    "punch": "punching",
    "push": "pushing something",
    "pushup": "doing push ups",
    "ride_bike": "riding a bike",
    "ride_horse": "riding a horse",
    "run": "running",
    "shake_hands": "shaking hands",
    "shoot_ball": "shooting a basketball",
    "shoot_bow": "shooting a bow and arrow",
    "shoot_gun": "shooting a gun",
    "sit": "sitting down",
    "situp": "doing sit ups",
    "smile": "smiling",
    "smoke": "smoking",
    "somersault": "doing a somersault",
    "stand": "standing up",
    "swing_baseball": "swinging a baseball bat",
    "sword": "sword fighting",
    "sword_exercise": "practicing with a sword",
    "talk": "talking",
    "throw": "throwing",
    "turn": "turning around",
    "walk": "walking",
    "wave": "waving",
}
assert set(CLASS_PHRASES) == set(HMDB51_CLASSES)

# Prompt sets compared in the report. "label" is the class name with
# underscores as spaces ("brush hair"), the closest a prompt gets to the raw label.
PROMPT_SETS = {
    "label": lambda c: c.replace("_", " "),
    "phrase": lambda c: CLASS_PHRASES[c],
    "a person is {phrase}": lambda c: "a person is " + CLASS_PHRASES[c],
}


def class_prompts(prompt_set, classes=HMDB51_CLASSES):
    return [PROMPT_SETS[prompt_set](c) for c in classes]


# Same special tokens as MSRVTT_Compressed_DataLoader._get_text.
CLS_TOKEN = "<|startoftext|>"
SEP_TOKEN = "<|endoftext|>"


def tokenize_prompts(prompts, tokenizer, max_words=32):
    """
    Token ids / mask / segment ids for a list of prompts, built exactly as the
    MSR-VTT loader's _get_text builds a caption: [CLS] + tokens, truncated to
    max_words - 1, + [SEP], zero-padded. Returns three (N, max_words) int64 tensors.
    """
    ids = np.zeros((len(prompts), max_words), dtype=np.int64)
    mask = np.zeros_like(ids)
    for i, prompt in enumerate(prompts):
        words = [CLS_TOKEN] + tokenizer.tokenize(prompt)
        words = words[:max_words - 1] + [SEP_TOKEN]
        token_ids = tokenizer.convert_tokens_to_ids(words)
        ids[i, :len(token_ids)] = token_ids
        mask[i, :len(token_ids)] = 1
    return torch.from_numpy(ids), torch.from_numpy(mask), torch.zeros_like(torch.from_numpy(ids))


# ---------------------------------------------------------------------- #
# Model loading
# ---------------------------------------------------------------------- #

def load_variant(checkpoint, visual_branches=VISUAL_BRANCHES, device="cpu", pretrained_clip_name="ViT-B/32"):
    """
    Build a CLIP4ClipCompressed with the given visual branches and load a
    trained checkpoint (a state_dict saved by main_task_retrieval.save_model)
    into it. A full three-branch checkpoint can be loaded into any ablation;
    the weights of dropped branches are simply not used.

    checkpoint=None gives the CLIP initialisation: pretrained I-frame, residual
    and text towers, but a *randomly initialised* motion-vector tower. Use it
    only to check the pipeline runs, never as a result.
    """
    state_dict = None
    if checkpoint is not None:
        state_dict = torch.load(checkpoint, map_location="cpu")
    task_config = types.SimpleNamespace(pretrained_clip_name=pretrained_clip_name,
                                        visual_branches=list(visual_branches))
    model = CLIP4ClipCompressed.from_pretrained(state_dict=state_dict, task_config=task_config)
    if torch.device(device).type == "cpu":
        model = model.float()    # CLIP keeps fp16 weights, which CPU kernels lack
    return model.to(device).eval()


# ---------------------------------------------------------------------- #
# Embeddings
# ---------------------------------------------------------------------- #

def _normalize(x):
    return x / x.norm(dim=-1, keepdim=True)


@torch.no_grad()
def encode_text(model, input_ids, attention_mask=None, token_type_ids=None):
    """(N, L) prompt tokens -> (N, D) L2-normalised text embeddings."""
    sequence_output = model.get_sequence_output(input_ids, token_type_ids, attention_mask)
    return _normalize(sequence_output.squeeze(1).float())


@torch.no_grad()
def encode_video(model, batch):
    """
    A dataloader batch -> (B, D) L2-normalised video embeddings, pooled the
    same way as CLIP4ClipCompressed._loose_similarity. A video whose every
    position is masked out (a failed decode) comes out as NaN.
    """
    visual_output, video_mask = model.get_visual_output(
        batch["iframe"], batch["iframe_mask"], batch["residuals"], batch["residuals_mask"],
        batch["mv"], batch["mv_mask"])
    visual_output = _normalize(visual_output)
    pooled = model._mean_pooling_for_similarity_visual(visual_output, video_mask)
    pooled = _normalize(pooled)
    pooled[video_mask.sum(dim=1) == 0] = float("nan")
    return pooled


def batch_to(batch, device):
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}


@torch.no_grad()
def extract_video_embeddings(models, dataloader, device="cpu", progress=None):
    """
    Run every model over the dataloader, decoding each batch only once.

    models: {variant name: model}
    Returns ({variant name: (N, D) float32 array}, {"index", "label_index",
    "decode_ok": (N,) arrays}), rows in dataset order.
    """
    embeddings = {name: [] for name in models}
    meta = {"index": [], "label_index": [], "decode_ok": []}
    batches = dataloader if progress is None else progress(dataloader)
    for batch in batches:
        for key in meta:
            meta[key].append(np.asarray(batch[key]))
        batch = batch_to(batch, device)
        for name, model in models.items():
            embeddings[name].append(encode_video(model, batch).cpu().numpy().astype(np.float32))
    embeddings = {name: np.concatenate(chunks) for name, chunks in embeddings.items()}
    meta = {key: np.concatenate(chunks) for key, chunks in meta.items()}
    return embeddings, meta


# ---------------------------------------------------------------------- #
# Metrics
# ---------------------------------------------------------------------- #

def classify(video_embeddings, text_embeddings):
    """Cosine scores, (N_videos, N_classes). Inputs must be L2-normalised."""
    return np.asarray(video_embeddings) @ np.asarray(text_embeddings).T


def true_class_ranks(scores, labels):
    """1-based rank of the true class among all classes, per video (ties count against it)."""
    true_scores = scores[np.arange(len(labels)), labels][:, None]
    return (scores >= true_scores).sum(axis=1)


def action_recognition_metrics(scores, labels, num_classes=None, topk=(1, 5, 10)):
    """
    Accuracy metrics for a (N, C) score matrix. Videos whose scores contain
    NaN (failed decodes) are excluded and counted in "n_invalid".
    """
    scores, labels = np.asarray(scores, dtype=np.float64), np.asarray(labels)
    num_classes = scores.shape[1] if num_classes is None else num_classes
    valid = np.isfinite(scores).all(axis=1)
    scores, labels = scores[valid], labels[valid]

    ranks = true_class_ranks(scores, labels)
    predictions = scores.argmax(axis=1)

    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(confusion, (labels, predictions), 1)
    support = confusion.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        per_class = np.diag(confusion) / support

    metrics = {"top{}".format(k): 100.0 * float(np.mean(ranks <= k)) for k in topk}
    metrics.update({
        "mean_class_acc": 100.0 * float(np.nanmean(per_class)),
        "median_rank": float(np.median(ranks)),
        "mean_rank": float(np.mean(ranks)),
        "n_videos": int(valid.sum()),
        "n_invalid": int((~valid).sum()),
    })
    return metrics, per_class * 100.0, confusion
