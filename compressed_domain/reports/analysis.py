"""
Analysis helpers for the MSR-VTT compressed-domain retrieval runs.

Everything here is CPU-only and works from two things main_task_retrieval.py
leaves in a run's --output_dir:

    mlflow.db                  metric histories (learning curves)
    embeddings_<split>.npz     text + per-branch video embeddings of the selected
                               model (see main_task_retrieval.save_embeddings)

No model, GPU or video decoding is needed. Used by
reports/msrvtt_retrieval_analysis.ipynb.
"""
import itertools
import json
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# compressed_domain/ on the path, for metrics.py.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from metrics import compute_metrics, branch_subset_similarity  # noqa: E402

# ---------------------------------------------------------------------- #
# Style: one palette for every figure
# ---------------------------------------------------------------------- #

SURFACE, INK, MUTED, AXIS, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#c3c2b7", "#e1e0d9"
# Categorical slots, in fixed order. A run keeps its slot in every figure.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
# A branch always has the same colour, whatever else is on the chart.
BRANCH_COLORS = {"iframe": "#2a78d6", "residual": "#eb6834", "mv": "#1baf7a"}
BRANCH_LABELS = {"iframe": "I-frame", "residual": "residual", "mv": "motion vector"}
NEUTRAL = "#898781"
SEQUENTIAL = LinearSegmentedColormap.from_list(
    "seq_blue", ["#fcfcfb", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])


def style_axes(ax, grid_axis="y"):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)
    ax.title.set_color(INK)
    if grid_axis:
        ax.grid(True, axis=grid_axis, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
    return ax


def subset_label(branches):
    return " + ".join(BRANCH_LABELS.get(b, b) for b in branches)


# ---------------------------------------------------------------------- #
# Loading
# ---------------------------------------------------------------------- #

def load_embeddings(path):
    """embeddings_<split>.npz -> dict of arrays (`branches` as a list of str)."""
    with np.load(path, allow_pickle=False) as data:
        emb = {key: data[key] for key in data.files}
    emb["branches"] = [str(b) for b in emb["branches"]]
    for key in ("video_ids", "captions", "failed_video_ids"):
        emb[key] = emb[key].astype(str)
    return emb


def load_run(output_dir, name=None, tracking_uri=None, run_id=None):
    """
    Everything known about one run, from its --output_dir.

    Returns {"name", "output_dir", "tracking_uri", "run_id", "embeddings": {split: emb}}.
    The mlflow location is read from the directory (mlflow_run.json, or the
    default mlflow.db + last_run_id.txt); pass tracking_uri / run_id to override,
    e.g. for runs tracked with --mlflow_tracking_uri. run_id is None when no
    history is found; the run is then only usable for the embedding analyses.
    """
    output_dir = os.path.abspath(output_dir)
    run = {"name": name or os.path.basename(os.path.normpath(output_dir)), "output_dir": output_dir,
           "tracking_uri": None, "run_id": None, "embeddings": {}}

    for split in ("val", "test"):
        path = os.path.join(output_dir, "embeddings_{}.npz".format(split))
        if os.path.exists(path):
            run["embeddings"][split] = load_embeddings(path)

    # main_task_retrieval.py records where each output_dir's run is tracked.
    info_file = os.path.join(output_dir, "mlflow_run.json")
    db = os.path.join(output_dir, "mlflow.db")
    id_file = os.path.join(output_dir, "last_run_id.txt")
    if tracking_uri or run_id:
        run["tracking_uri"], run["run_id"] = tracking_uri, run_id
    if os.path.exists(info_file) and not (run["tracking_uri"] and run["run_id"]):
        with open(info_file) as f:
            info = json.load(f)
        run["tracking_uri"] = run["tracking_uri"] or info.get("tracking_uri")
        run["run_id"] = run["run_id"] or info.get("run_id")
    if run["tracking_uri"] is None and os.path.exists(db):
        run["tracking_uri"] = "sqlite:///" + db
    if run["run_id"] is None and os.path.exists(id_file):
        with open(id_file) as f:
            run["run_id"] = f.read().strip()
    if run["tracking_uri"] is None:
        run["run_id"] = None
    return run


def metric_history(run, key):
    """(steps, values) of one mlflow metric, one point per step (the latest logged)."""
    if not run.get("run_id"):
        return np.array([]), np.array([])
    from mlflow import MlflowClient
    client = MlflowClient(tracking_uri=run["tracking_uri"])
    latest = {}
    for item in sorted(client.get_metric_history(run["run_id"], key), key=lambda m: (m.step, m.timestamp)):
        latest[item.step] = item.value
    steps = np.array(sorted(latest))
    return steps, np.array([latest[s] for s in steps])


def run_params(run):
    if not run.get("run_id"):
        return {}
    from mlflow import MlflowClient
    return dict(MlflowClient(tracking_uri=run["tracking_uri"]).get_run(run["run_id"]).data.params)


# ---------------------------------------------------------------------- #
# Retrieval metrics from embeddings
# ---------------------------------------------------------------------- #

def branch_indices(emb, branches=None):
    if branches is None:
        return None
    missing = [b for b in branches if b not in emb["branches"]]
    if missing:
        raise ValueError("branches {} are not in this model ({})".format(missing, emb["branches"]))
    return [emb["branches"].index(b) for b in branches]


def similarity(emb, branches=None):
    """(N captions, N videos) cosine similarity, using only `branches` (None = all built)."""
    return branch_subset_similarity(emb["text_embeddings"], emb["branch_embeddings"], emb["branch_mask"],
                                    branch_indices(emb, branches))


def ranks(emb, branches=None, direction="t2v"):
    """1-based rank of the ground truth for every query (ties count against the model)."""
    sim = similarity(emb, branches)
    if direction == "v2t":
        sim = sim.T
    return (sim >= np.diag(sim)[:, None]).sum(axis=1)


def retrieval_metrics(emb, branches=None):
    """Both directions, as one flat dict."""
    sim = similarity(emb, branches)
    out = {"queries": sim.shape[0]}
    for direction, matrix in (("t2v", sim), ("v2t", sim.T)):
        m = compute_metrics(matrix)
        out.update({direction + "_R@1": m["R1"], direction + "_R@5": m["R5"], direction + "_R@10": m["R10"],
                    direction + "_MedR": float(m["MR"]), direction + "_MeanR": float(m["MeanR"])})
    return out


def results_table(runs, split="val"):
    """One row per run: the selected model with all its branches."""
    rows = []
    for run in runs:
        emb = run["embeddings"].get(split)
        if emb is None:
            continue
        rows.append({"run": run["name"], "branches": subset_label(emb["branches"]), **retrieval_metrics(emb),
                     "failed videos": len(emb["failed_video_ids"])})
    return pd.DataFrame(rows).set_index("run") if rows else pd.DataFrame()


def all_branch_subsets(branches):
    """Every non-empty subset, single branches first, in the model's branch order."""
    return [list(c) for r in range(1, len(branches) + 1) for c in itertools.combinations(branches, r)]


def branch_ablation_table(emb):
    """
    Inference-time ablation of ONE trained model: retrieval using every subset of
    its branches. Nothing is retrained; the other branches are simply left out
    of the video embedding.
    """
    rows = []
    for subset in all_branch_subsets(emb["branches"]):
        rows.append({"branches used": subset_label(subset), "n": len(subset), "_subset": tuple(subset),
                     **retrieval_metrics(emb, subset)})
    return pd.DataFrame(rows).set_index("branches used")


def complementarity_table(emb, base=("iframe",), k=10, direction="t2v"):
    """
    Does adding branches to `base` help queries that `base` gets wrong?

    For every superset of `base`: queries whose ground truth enters the top-k
    ("gained"), leaves it ("lost"), and the net change.
    """
    base = [b for b in emb["branches"] if b in base]
    base_hit = ranks(emb, base, direction) <= k
    rows = []
    for subset in all_branch_subsets(emb["branches"]):
        if set(subset) <= set(base) or not set(base) <= set(subset):
            continue
        hit = ranks(emb, subset, direction) <= k
        rows.append({"branches used": subset_label(subset),
                     "in top-{} with {}".format(k, subset_label(base)): int(base_hit.sum()),
                     "gained": int((hit & ~base_hit).sum()), "lost": int((~hit & base_hit).sum()),
                     "net": int(hit.sum()) - int(base_hit.sum())})
    return pd.DataFrame(rows).set_index("branches used") if rows else pd.DataFrame()


def branch_alignment_table(emb):
    """
    How well each branch's embedding lines up with the text, by itself.

    matched:    mean cosine between a caption and ITS video's branch embedding
    mismatched: mean cosine between a caption and the other videos'
    gap:        matched - mismatched. A branch that carries no text-relevant
                information has a gap near 0.
    """
    rows = []
    for branch in emb["branches"]:
        sim = similarity(emb, [branch])
        n = sim.shape[0]
        matched = float(np.diag(sim).mean())
        mismatched = float((sim.sum() - np.trace(sim)) / max(n * (n - 1), 1))
        rows.append({"branch": BRANCH_LABELS.get(branch, branch), "matched": matched, "mismatched": mismatched,
                     "gap": matched - mismatched})
    return pd.DataFrame(rows).set_index("branch")


# ---------------------------------------------------------------------- #
# Figures
# ---------------------------------------------------------------------- #

def plot_learning_curves(runs, metrics, titles=None, ncols=2, xlabel="optimizer step"):
    """
    Small multiples: one panel per metric, one line per run (a run keeps its
    colour in every panel). All metrics share the x-axis: optimizer steps since
    the start of the run, continuous across resumed sessions.
    """
    runs = [r for r in runs if r.get("run_id")]
    nrows = int(np.ceil(len(metrics) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.6 * ncols, 3.1 * nrows), squeeze=False, facecolor=SURFACE)
    for ax, key in zip(axes.flat, metrics):
        style_axes(ax)
        for i, run in enumerate(runs):
            steps, values = metric_history(run, key)
            if len(steps) == 0:
                continue
            ax.plot(steps, values, color=SERIES[i % len(SERIES)], linewidth=2, label=run["name"],
                    marker="o" if len(steps) <= 20 else None, markersize=5)
        ax.set_title((titles or {}).get(key, key), loc="left", fontsize=11)
        ax.set_xlabel(xlabel)
        if not ax.lines:
            ax.text(0.5, 0.5, "not logged", transform=ax.transAxes, ha="center", va="center", color=MUTED)
    for ax in axes.flat[len(metrics):]:
        ax.set_visible(False)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    if len(labels) > 1:
        fig.legend(handles, labels, loc="upper center", ncol=min(len(labels), 4), frameon=False,
                   labelcolor=MUTED, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout()
    return fig, axes


def plot_branch_curves(run, metric="val_t2v_R1", title=None):
    """One run: validation retrieval of the full model and of each branch alone, over training."""
    fig, ax = plt.subplots(figsize=(8, 3.6), facecolor=SURFACE)
    style_axes(ax)
    steps, values = metric_history(run, metric)
    if len(steps):
        ax.plot(steps, values, color=INK, linewidth=2, marker="o", markersize=5, label="all branches")
    for branch, color in BRANCH_COLORS.items():
        steps, values = metric_history(run, "{}_{}".format(metric, branch))
        if len(steps):
            ax.plot(steps, values, color=color, linewidth=2, marker="o", markersize=5,
                    label=BRANCH_LABELS[branch] + " alone")
    ax.set_xlabel("optimizer step")
    ax.set_ylabel("R@1 (%)")
    ax.set_ylim(bottom=0)
    ax.set_title(title or "{}: validation text-to-video R@1 during training".format(run["name"]),
                 loc="left", fontsize=11)
    if ax.lines:
        ax.legend(frameon=False, labelcolor=MUTED, loc="center left", bbox_to_anchor=(1.0, 0.5))
    fig.tight_layout()
    return fig, ax


def plot_branch_ablation(table, metric="t2v_R@1", title=None, chance=None):
    """
    Horizontal bars, one per branch subset. A single-branch bar wears that
    branch's colour; combinations are neutral, so colour never implies a branch
    that is only part of the mix.
    """
    labels = list(table.index)
    values = table[metric].to_numpy(dtype=float)
    colors = [BRANCH_COLORS.get(s[0], NEUTRAL) if len(s) == 1 else NEUTRAL for s in table["_subset"]]
    if len(labels):
        colors[-1] = INK if len(table["_subset"].iloc[-1]) > 1 else colors[-1]      # the full model

    fig, ax = plt.subplots(figsize=(8, 0.5 * len(labels) + 1.4), facecolor=SURFACE)
    style_axes(ax, grid_axis="x")
    y = np.arange(len(labels))[::-1]
    bars = ax.barh(y, values, color=colors, height=0.62)
    ax.bar_label(bars, fmt="%.1f", padding=4, color=MUTED, fontsize=9)
    ax.set_yticks(y, labels, color=INK)
    ax.set_xlim(0, max(5.0, float(np.nanmax(values)) * 1.15) if len(values) else 5.0)
    ax.set_xlabel(metric.replace("_", " ") + (" (%)" if "R@" in metric else ""))
    if chance is not None:
        ax.axvline(chance, color=MUTED, linewidth=1, linestyle="--")
        ax.set_ylim(-1.0, len(labels) - 0.4)
        ax.text(chance, -0.72, " chance", color=MUTED, fontsize=8, va="center")
    ax.set_title(title or "Retrieval using a subset of the trained model's branches", loc="left", fontsize=11)
    fig.tight_layout()
    return fig, ax


def plot_runs_comparison(table, metrics=("t2v_R@1", "t2v_R@5", "t2v_R@10"), title=None):
    """Grouped bars: one group per metric, one bar per run (run colours as in the learning curves)."""
    fig, ax = plt.subplots(figsize=(max(7, 1.1 * len(table) * len(metrics) + 2), 4), facecolor=SURFACE)
    style_axes(ax)
    x = np.arange(len(metrics))
    width = 0.8 / max(len(table), 1)
    for i, (name, row) in enumerate(table.iterrows()):
        offset = (i - (len(table) - 1) / 2) * width
        bars = ax.bar(x + offset, [row[m] for m in metrics], width * 0.9, color=SERIES[i % len(SERIES)], label=name)
        ax.bar_label(bars, fmt="%.1f", padding=2, color=MUTED, fontsize=8)
    ax.set_xticks(x, [m.replace("_", " ") for m in metrics], color=INK)
    ax.set_ylabel("recall (%)")
    ax.set_ylim(0, 100)
    ax.set_title(title or "Selected model of each run", loc="left", fontsize=11)
    if len(table) > 1:
        ax.legend(frameon=False, labelcolor=MUTED, loc="center left", bbox_to_anchor=(1.0, 0.5))
    fig.tight_layout()
    return fig, ax


def plot_recall_curve(emb, subsets=None, direction="t2v", max_k=100, title=None):
    """Recall@K for K = 1..max_k: the share of queries whose ground truth is in the top K."""
    subsets = subsets or ([[b] for b in emb["branches"]] + ([emb["branches"]] if len(emb["branches"]) > 1 else []))
    fig, ax = plt.subplots(figsize=(8, 3.8), facecolor=SURFACE)
    style_axes(ax)
    ks = np.arange(1, max_k + 1)
    for subset in subsets:
        r = ranks(emb, subset, direction)
        color = BRANCH_COLORS.get(subset[0], NEUTRAL) if len(subset) == 1 else INK
        ax.plot(ks, [(r <= k).mean() * 100 for k in ks], color=color, linewidth=2,
                label=subset_label(subset) if len(subset) == 1 else "all: " + subset_label(subset))
    n = len(emb["video_ids"])
    ax.plot(ks, ks / n * 100, color=MUTED, linewidth=1, linestyle="--", label="chance")
    ax.set_xscale("log")
    ax.set_xticks([1, 5, 10, 50, 100][: 5 if max_k >= 100 else 3], [1, 5, 10, 50, 100][: 5 if max_k >= 100 else 3])
    ax.set_xlabel("K")
    ax.set_ylabel("recall@K (%)")
    ax.set_ylim(0, 100)
    ax.set_title(title or "Recall@K, {} ({} queries)".format(
        "text-to-video" if direction == "t2v" else "video-to-text", n), loc="left", fontsize=11)
    ax.legend(frameon=False, labelcolor=MUTED, loc="center left", bbox_to_anchor=(1.0, 0.5))
    fig.tight_layout()
    return fig, ax


def plot_rank_scatter(emb, branches_x, branches_y, direction="t2v", title=None):
    """
    Per-query rank with one set of branches against another (log axes). Points
    below the diagonal are queries the y-axis variant ranks better.
    """
    rx, ry = ranks(emb, branches_x, direction), ranks(emb, branches_y, direction)
    fig, ax = plt.subplots(figsize=(5.2, 5.2), facecolor=SURFACE)
    style_axes(ax, grid_axis="both")
    ax.scatter(rx, ry, s=22, color=SERIES[0], alpha=0.45, linewidths=0)
    top = max(rx.max(), ry.max())
    ax.plot([1, top], [1, top], color=MUTED, linewidth=1, linestyle="--")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("rank with " + subset_label(branches_x))
    ax.set_ylabel("rank with " + subset_label(branches_y))
    better, worse = int((ry < rx).sum()), int((ry > rx).sum())
    ax.set_title(title or "{} better for {} queries, worse for {}, equal for {}".format(
        subset_label(branches_y), better, worse, len(rx) - better - worse), loc="left", fontsize=10)
    fig.tight_layout()
    return fig, ax


def plot_similarity_heatmap(emb, n=30, branches=None, start=0, title=None):
    """
    Cosine similarity between n captions (rows) and their n videos (columns).
    The ground truth is the diagonal, so a model that works shows a dark
    diagonal on a light background.
    """
    sim = similarity(emb, branches)[start:start + n, start:start + n]
    fig, ax = plt.subplots(figsize=(7.5, 6.5), facecolor=SURFACE)
    image = ax.imshow(sim, cmap=SEQUENTIAL, aspect="equal", interpolation="nearest")
    cbar = fig.colorbar(image, ax=ax, shrink=0.8)
    cbar.set_label("cosine similarity", color=MUTED)
    cbar.ax.tick_params(colors=MUTED, labelsize=8)
    ticks = np.arange(0, sim.shape[0], max(1, sim.shape[0] // 10))
    ax.set_xticks(ticks, [emb["video_ids"][start + t] for t in ticks], rotation=90, fontsize=7, color=MUTED)
    ax.set_yticks(ticks, [emb["video_ids"][start + t] for t in ticks], fontsize=7, color=MUTED)
    ax.set_xlabel("video", color=MUTED)
    ax.set_ylabel("caption of video", color=MUTED)
    ax.set_title(title or "Caption-video similarity, {} ({} pairs)".format(
        subset_label(branches or emb["branches"]), sim.shape[0]), loc="left", fontsize=11, color=INK)
    for side in ax.spines.values():
        side.set_color(AXIS)
    fig.tight_layout()
    return fig, ax


# ---------------------------------------------------------------------- #
# Qualitative examples
# ---------------------------------------------------------------------- #

def qualitative_text_to_video(emb, query_indices, top_k=5, branches=None):
    """For each caption query: the top-k retrieved videos and where the right one ranks."""
    sim = similarity(emb, branches)
    r = ranks(emb, branches, "t2v")
    rows = []
    for q in query_indices:
        order = np.argsort(-sim[q], kind="stable")[:top_k]
        for position, v in enumerate(order, start=1):
            rows.append({"query caption": emb["captions"][q], "ground truth": emb["video_ids"][q],
                         "ground-truth rank": int(r[q]), "position": position,
                         "retrieved video": emb["video_ids"][v],
                         "retrieved video's caption": emb["captions"][v],
                         "similarity": round(float(sim[q, v]), 3), "correct": bool(v == q)})
    return pd.DataFrame(rows)


def qualitative_video_to_text(emb, query_indices, top_k=5, branches=None):
    """For each video query: the top-k retrieved captions and where the right one ranks."""
    sim = similarity(emb, branches).T
    r = ranks(emb, branches, "v2t")
    rows = []
    for q in query_indices:
        order = np.argsort(-sim[q], kind="stable")[:top_k]
        for position, c in enumerate(order, start=1):
            rows.append({"query video": emb["video_ids"][q], "ground-truth caption": emb["captions"][q],
                         "ground-truth rank": int(r[q]), "position": position,
                         "retrieved caption": emb["captions"][c],
                         "similarity": round(float(sim[q, c]), 3), "correct": bool(c == q)})
    return pd.DataFrame(rows)


def pick_examples(emb, branches=None, direction="t2v", n=3, seed=0):
    """Query indices for a balanced qualitative sample: n successes (rank 1) and n failures (rank > 10)."""
    r = ranks(emb, branches, direction)
    rng = np.random.default_rng(seed)
    good, bad = np.flatnonzero(r == 1), np.flatnonzero(r > 10)
    return {"success": sorted(rng.choice(good, size=min(n, len(good)), replace=False).tolist()),
            "failure": sorted(rng.choice(bad, size=min(n, len(bad)), replace=False).tolist())}


def load_iframe_thumbnail(features_path, video_id, gop_fraction=0.5):
    """
    One decoded I-frame (RGB uint8) from the middle of a video, for showing
    retrieved videos. Needs the coviar extension and the re-encoded videos;
    returns None when either is unavailable.
    """
    try:
        import coviar
    except ImportError:
        return None
    path = os.path.join(features_path, "{}.mp4".format(video_id))
    if not os.path.exists(path):
        return None
    try:
        num_gops = int(np.ceil(coviar.get_num_frames(path) / 12))
        if num_gops <= 0:
            return None
        frame = coviar.load(path, min(int(num_gops * gop_fraction), num_gops - 1), 0, 0, True)
        return None if frame is None else np.ascontiguousarray(frame[..., ::-1])       # BGR -> RGB
    except Exception:
        return None


def show_retrieved_videos(emb, query_index, features_path, top_k=5, branches=None):
    """A row of I-frame thumbnails: the top-k videos retrieved for one caption. None without coviar."""
    sim = similarity(emb, branches)
    order = np.argsort(-sim[query_index], kind="stable")[:top_k]
    frames = [load_iframe_thumbnail(features_path, emb["video_ids"][v]) for v in order]
    if all(f is None for f in frames):
        return None
    fig, axes = plt.subplots(1, top_k, figsize=(2.6 * top_k, 2.6), facecolor=SURFACE, squeeze=False)
    for ax, v, frame in zip(axes[0], order, frames):
        ax.set_xticks([])
        ax.set_yticks([])
        if frame is not None:
            ax.imshow(frame)
        correct = v == query_index
        for side in ax.spines.values():
            side.set_color(INK if correct else AXIS)
            side.set_linewidth(2.5 if correct else 1)
        ax.set_title("{}{}".format(emb["video_ids"][v], "  (correct)" if correct else ""), fontsize=8,
                     color=INK if correct else MUTED)
    fig.suptitle('"{}"'.format(emb["captions"][query_index]), fontsize=10, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------- #
# Kept for older notebooks
# ---------------------------------------------------------------------- #

def plot_mlflow_learning_curves(run_id, metric_names=("train_loss", "val_t2v_R1", "val_v2t_R1"), client=None,
                                tracking_uri=None):
    """Plot logged metric histories of one mlflow run against the optimizer step."""
    run = {"name": run_id, "run_id": run_id,
           "tracking_uri": tracking_uri or (client.tracking_uri if client is not None else None)}
    return plot_learning_curves([run], list(metric_names), ncols=1)
