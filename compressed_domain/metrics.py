from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import numpy as np
import torch

def compute_metrics(x):
    # x[i, j]: similarity of query i to candidate j; candidate i is query i's
    # ground truth. `ind` is the 0-based rank of the ground truth: the number of
    # other candidates scoring at least as high, so ties count against the model
    # (a collapsed embedding space ranks last, not first) and there is always
    # exactly one rank per query.
    x = np.asarray(x)
    if x.ndim != 2 or x.shape[0] != x.shape[1]:
        raise ValueError("compute_metrics needs a square similarity matrix, got {}".format(x.shape))
    if not np.isfinite(x).all():
        raise ValueError("similarity matrix has {} non-finite values".format(int((~np.isfinite(x)).sum())))
    d = np.diag(x)[:, np.newaxis]
    ind = (x >= d).sum(axis=1) - 1
    metrics = {}
    metrics['R1'] = float(np.sum(ind == 0)) * 100 / len(ind)
    metrics['R5'] = float(np.sum(ind < 5)) * 100 / len(ind)
    metrics['R10'] = float(np.sum(ind < 10)) * 100 / len(ind)
    metrics['MR'] = np.median(ind) + 1
    metrics["MedianR"] = metrics['MR']
    metrics["MeanR"] = np.mean(ind) + 1
    metrics["cols"] = [int(i) for i in list(ind)]
    return metrics

def compute_metrics_multi(x, gt_index):
    """
    Text-to-video metrics when a video has several captions (MSVD).

    x[i, j]: similarity of caption i to video j; gt_index[i] is caption i's
    video. Same rank and tie rule as compute_metrics, which this equals when
    x is square and gt_index is arange(N).
    """
    x = np.asarray(x)
    gt_index = np.asarray(gt_index, dtype=np.int64)
    if x.ndim != 2 or gt_index.shape != (x.shape[0],):
        raise ValueError("compute_metrics_multi needs (N_text, N_video) similarities and N_text ground-truth "
                         "indices, got {} and {}".format(x.shape, gt_index.shape))
    if len(gt_index) and (gt_index.min() < 0 or gt_index.max() >= x.shape[1]):
        raise ValueError("ground-truth video index out of range [0, {})".format(x.shape[1]))
    if not np.isfinite(x).all():
        raise ValueError("similarity matrix has {} non-finite values".format(int((~np.isfinite(x)).sum())))
    d = x[np.arange(len(gt_index)), gt_index][:, np.newaxis]
    ind = (x >= d).sum(axis=1) - 1
    metrics = {}
    metrics['R1'] = float(np.sum(ind == 0)) * 100 / len(ind)
    metrics['R5'] = float(np.sum(ind < 5)) * 100 / len(ind)
    metrics['R10'] = float(np.sum(ind < 10)) * 100 / len(ind)
    metrics['MR'] = np.median(ind) + 1
    metrics["MedianR"] = metrics['MR']
    metrics["MeanR"] = np.mean(ind) + 1
    metrics["cols"] = [int(i) for i in list(ind)]
    return metrics


def video_to_text_similarity(x, gt_index):
    """
    (N_video, N_video) video-to-text similarities from (N_text, N_video) ones,
    for videos with several captions: entry [v, u] is video v's best score
    over the captions of video u, as in CLIP4Clip's multi-sentence evaluation.
    compute_metrics on the result gives the video-to-text metrics.
    """
    x = np.asarray(x)
    gt_index = np.asarray(gt_index, dtype=np.int64)
    out = np.full((x.shape[1], x.shape[1]), -np.inf, dtype=x.dtype)
    for u in range(x.shape[1]):
        rows = x[gt_index == u]
        if len(rows):
            out[:, u] = rows.max(axis=0)
    return out


def print_computed_metrics(metrics):
    r1 = metrics['R1']
    r5 = metrics['R5']
    r10 = metrics['R10']
    mr = metrics['MR']
    print('R@1: {:.4f} - R@5: {:.4f} - R@10: {:.4f} - Median R: {}'.format(r1, r5, r10, mr))

# below two functions directly come from: https://github.com/Deferf/Experiments
def tensor_text_to_video_metrics(sim_tensor, top_k = [1,5,10]):
    if not torch.is_tensor(sim_tensor):
      sim_tensor = torch.tensor(sim_tensor)

    # Permute sim_tensor so it represents a sequence of text-video similarity matrices.
    # Then obtain the double argsort to position the rank on the diagonal
    stacked_sim_matrices = sim_tensor.permute(1, 0, 2)
    first_argsort = torch.argsort(stacked_sim_matrices, dim = -1, descending= True)
    second_argsort = torch.argsort(first_argsort, dim = -1, descending= False)

    # Extracts ranks i.e diagonals
    ranks = torch.flatten(torch.diagonal(second_argsort, dim1 = 1, dim2 = 2))

    # Now we need to extract valid ranks, as some belong to inf padding values
    permuted_original_data = torch.flatten(torch.diagonal(sim_tensor, dim1 = 0, dim2 = 2))
    mask = ~ torch.logical_or(torch.isinf(permuted_original_data), torch.isnan(permuted_original_data))
    valid_ranks = ranks[mask]
    # A quick dimension check validates our results, there may be other correctness tests pending
    # Such as dot product localization, but that is for other time.
    #assert int(valid_ranks.shape[0]) ==  sum([len(text_dict[k]) for k in text_dict])
    if not torch.is_tensor(valid_ranks):
      valid_ranks = torch.tensor(valid_ranks)
    results = {f"R{k}": float(torch.sum(valid_ranks < k) * 100 / len(valid_ranks)) for k in top_k}
    results["MedianR"] = float(torch.median(valid_ranks + 1))
    results["MeanR"] = float(np.mean(valid_ranks.numpy() + 1))
    results["Std_Rank"] = float(np.std(valid_ranks.numpy() + 1))
    results['MR'] = results["MedianR"]
    return results

def tensor_video_to_text_sim(sim_tensor):
    if not torch.is_tensor(sim_tensor):
      sim_tensor = torch.tensor(sim_tensor)
    # Code to avoid nans
    sim_tensor[sim_tensor != sim_tensor] = float('-inf')
    # Forms a similarity matrix for use with rank at k
    values, _ = torch.max(sim_tensor, dim=1, keepdim=True)
    return torch.squeeze(values).T


# ---------------------------------------------------------------------- #
# Branch-subset retrieval from saved per-branch embeddings (NumPy only)
# ---------------------------------------------------------------------- #

def _unit(x, eps=1e-6):
    x = np.asarray(x, dtype=np.float64)
    return x / np.clip(np.linalg.norm(x, axis=-1, keepdims=True), eps, None)


def pool_branches(branch_embeddings, branch_mask, keep=None):
    """
    Video embeddings from any subset of the visual branches.

    branch_embeddings: (N, M, D) one embedding per video and built branch, as
                       returned by CLIP4ClipCompressed.get_visual_output with
                       visual_pooling == "modality".
    branch_mask:       (N, M) 1 where the video has that branch.
    keep:              indices of the branches to use (None = all).

    Mirrors CLIP4ClipCompressed.pool_video: L2-normalise each branch embedding,
    average the kept ones that exist, L2-normalise. A video with none of the
    kept branches gets a zero vector (similarity 0 to every caption).
    Returns (N, D).
    """
    emb = _unit(branch_embeddings)
    mask = np.asarray(branch_mask, dtype=np.float64)
    if keep is not None:
        keep = list(keep)
        emb, mask = emb[:, keep], mask[:, keep]
    pooled = (emb * mask[..., None]).sum(axis=1) / np.clip(mask.sum(axis=1, keepdims=True), 1.0, None)
    return _unit(pooled)


def branch_subset_similarity(text_embeddings, branch_embeddings, branch_mask, keep=None):
    """(N_text, N_video) cosine similarity using only the branches in `keep`."""
    return _unit(text_embeddings) @ pool_branches(branch_embeddings, branch_mask, keep).T
