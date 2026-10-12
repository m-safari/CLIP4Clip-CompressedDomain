import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mlflow import MlflowClient


def plot_mlflow_learning_curves(
    run_id,
    metric_names=("train_loss", "val_t2v_R1", "val_v2t_R1"),
    client=None,
):
    """Plot logged metric histories against MLflow step."""
    client = client or MlflowClient()
    fig, axes = plt.subplots(
        len(metric_names), 1,
        figsize=(10, 3 * len(metric_names)),
        squeeze=False,
    )

    for ax, name in zip(axes[:, 0], metric_names):
        history = client.get_metric_history(run_id, name)

        if not history:
            ax.set_title(f"{name} — no history logged")
            continue

        history = sorted(history, key=lambda item: (item.step, item.timestamp))
        ax.plot(
            [item.step for item in history],
            [item.value for item in history],
            marker="." if len(history) < 100 else None,
        )
        ax.set(title=name, xlabel="Global step", ylabel="Value")
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig, axes


def plot_similarity_heatmap(
    video_embeddings,
    text_embeddings,
    video_ids,
    caption_video_ids,
    video_indices=None,
    max_captions=100,
):
    """
    Rows: selected videos.
    Columns: captions.
    Matching video-caption pairs are outlined.
    """
    V = np.asarray(video_embeddings, dtype=float)
    T = np.asarray(text_embeddings, dtype=float)

    V = V / np.clip(np.linalg.norm(V, axis=1, keepdims=True), 1e-12, None)
    T = T / np.clip(np.linalg.norm(T, axis=1, keepdims=True), 1e-12, None)

    video_ids = list(video_ids)
    caption_video_ids = list(caption_video_ids)

    if video_indices is None:
        video_indices = list(range(min(len(V), 20)))

    caption_indices = list(range(min(len(T), max_captions)))

    scores = V[video_indices] @ T[caption_indices].T

    fig, ax = plt.subplots(figsize=(14, 7))
    image = ax.imshow(scores, aspect="auto", interpolation="nearest")
    fig.colorbar(image, ax=ax, label="Cosine similarity")

    ax.set_xlabel("Caption index")
    ax.set_ylabel("Video index")
    ax.set_xticks(range(len(caption_indices)))
    ax.set_xticklabels(caption_indices, rotation=90, fontsize=7)
    ax.set_yticks(range(len(video_indices)))
    ax.set_yticklabels([str(video_ids[i]) for i in video_indices])

    # Outline every ground-truth video-caption match.
    for row, vi in enumerate(video_indices):
        for col, ci in enumerate(caption_indices):
            if video_ids[vi] == caption_video_ids[ci]:
                ax.add_patch(plt.Rectangle(
                    (col - 0.5, row - 0.5), 1, 1,
                    fill=False, edgecolor="white", linewidth=1.2,
                ))

    ax.set_title("Video–text cosine similarity matrix")
    fig.tight_layout()
    return fig, ax


def qualitative_video_to_text(
    video_embeddings,
    text_embeddings,
    video_ids,
    captions,
    caption_video_ids,
    video_indices,
    top_k=5,
):
    """
    Retrieve captions for selected video queries.

    Returns a DataFrame with ranked predictions and ground-truth flags.
    """
    V = np.asarray(video_embeddings, dtype=float)
    T = np.asarray(text_embeddings, dtype=float)

    V = V / np.clip(np.linalg.norm(V, axis=1, keepdims=True), 1e-12, None)
    T = T / np.clip(np.linalg.norm(T, axis=1, keepdims=True), 1e-12, None)

    video_ids = list(video_ids)
    captions = list(captions)
    caption_video_ids = list(caption_video_ids)

    # Similarity matrix: videos x captions.
    similarities = V[video_indices] @ T.T
    rows = []

    for row, vi in enumerate(video_indices):
        ranking = np.argsort(-similarities[row])
        relevant = [
            rank + 1
            for rank, ci in enumerate(ranking)
            if caption_video_ids[ci] == video_ids[vi]
        ]

        for rank, ci in enumerate(ranking[:top_k], start=1):
            rows.append({
                "video_id": video_ids[vi],
                "rank": rank,
                "caption": captions[ci],
                "similarity": float(similarities[row, ci]),
                "is_ground_truth": (
                    caption_video_ids[ci] == video_ids[vi]
                ),
                "best_ground_truth_rank": (
                    min(relevant) if relevant else None
                ),
            })

    return pd.DataFrame(rows)
