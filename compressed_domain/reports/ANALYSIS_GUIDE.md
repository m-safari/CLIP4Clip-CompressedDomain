# Analysis guide: what is logged, what each analysis means, how to read it

Companion to `reports/msrvtt_retrieval_analysis.ipynb`. The functions the notebook calls are in `reports/analysis.py`.

## 1. Where the numbers come from

The notebook needs no GPU and no model. It reads two sources from each run's `--output_dir`.

### 1.1 MLflow metric history (learning curves)

The x-axis of every metric is the **global step**: optimizer steps since the start of the run, continuous across resumed sessions.

Logged every `--n_display` steps during training:

| Metric | Meaning |
|---|---|
| `train_loss` | Contrastive loss, averaged over the steps since the previous log line |
| `lr_min`, `lr_max` | Smallest and largest learning rate in use (CLIP-initialised weights vs. from-scratch weights) |
| `grad_norm` | Gradient norm of all trainable weights, before clipping |
| `grad_norm_iframe`, `_residual`, `_mv` | The same, per encoder |
| `adam_v_zero_frac` | Share of Adam second-moment entries that are exactly zero |
| `amp_scale` | GradScaler's current scale |
| `gpu_peak_gb` | Peak GPU memory in this epoch |
| `logit_scale` | CLIP temperature (similarities are multiplied by it before the loss) |

Logged once per epoch:

| Metric | Meaning |
|---|---|
| `train_epoch_loss` | Mean training loss of the epoch |
| `train_num_failed_videos` | Training videos excluded because they could not be decoded |
| `val_t2v_R1`, `_R5`, `_R10`, `_MedianR`, `_MeanR` | Text-to-video retrieval on validation |
| `val_v2t_*` | Video-to-text retrieval on validation |
| `val_t2v_R1_<branch>`, `_R10_<branch>`, `_MedianR_<branch>` | Text-to-video using one branch's embedding alone |
| `val_num_candidates`, `val_num_failed_videos` | Pool size and excluded videos |

`--do_eval` logs the same retrieval metrics once, with the prefix `test_` (test set) or `evalval_` (validation).

### 1.2 Embedding files

`embeddings_val.npz` and `embeddings_test.npz` hold, for the selected model:

| Array | Shape | Content |
|---|---|---|
| `text_embeddings` | N × D | One normalised embedding per caption |
| `branch_embeddings` | N × M × D | One video embedding per built branch |
| `branch_mask` | N × M | 1 where the video has that branch |
| `branches` | M | Branch names |
| `video_ids`, `captions` | N | Row *i*'s caption is the query for row *i*'s video |
| `failed_video_ids` | — | Videos excluded from the pool |

Because each branch has its own embedding, the video embedding for **any subset of branches** is computed afterwards (normalise each, average the chosen ones, normalise). All tables and figures from section 3 onward are computed this way.

## 2. The metrics

- **R@K (recall at K):** percentage of queries whose correct item is among the top K results. Higher is better. With 1000 candidates, chance is K/10 percent (R@1 = 0.1%).
- **Median rank:** the rank of the correct item for the middle query. Lower is better; 1 is perfect, chance is about 500 with 1000 candidates.
- **Mean rank:** the average rank. It is dominated by the worst queries, so it can stay high while R@K is good.
- **Text-to-video (t2v):** a caption retrieves videos. **Video-to-text (v2t):** a video retrieves captions.
- Ties count against the model, so a collapsed embedding space scores at the bottom, not the top.

Reference points for MSR-VTT with 1000 candidates: untrained CLIP on plain frames gives R@1 around 30 and median rank around 4; the original CLIP4Clip with RGB frames reports about 43.

## 3. Notebook sections

### Section 1: Setup of each run

A table of the arguments each run was trained with. Rows that are equal across runs are what the comparison holds fixed. Rows that differ are what is being compared. Check this before reading any comparison: two runs that differ in more than the intended setting cannot be attributed to that setting.

### Section 2: Learning curves

| Panel | Healthy | Warning sign |
|---|---|---|
| Training loss | Falls, then flattens | Flat from the start (not learning); near zero very early (batch too small, task trivial) |
| Learning rate | Linear warmup, then smooth cosine decay to zero at the last step | Jumps between sessions (schedule not preserved on resume) |
| Validation R@K | Rises, may plateau | Falls while training loss keeps falling (overfitting) |
| Validation median rank | Falls | Rises over epochs |

**Per-branch curves** (one run): validation R@1 of the full model and of each branch alone. A branch whose line stays near 0.1% never produces a text-aligned embedding. If the "all branches" line coincides with the I-frame line, the other branches are not adding anything.

**Diagnostics:**
- `grad_norm_<branch>`: a tower with a much smaller gradient than the others is receiving little training signal.
- `adam_v_zero_frac`: should be about 0. A large value means the optimizer is not behaving as Adam.
- `gpu_peak_gb`: headroom for a larger batch or more frames.

### Section 3: Main results

One row per run: the selected (best validation) model with all its branches, both directions. This is the table to quote. Use validation while choosing between variants and the test split for final numbers only.

### Section 4: Contribution of each branch inside the trained model

**Inference-time ablation.** One trained model is scored with each of the 7 subsets of its branches; nothing is retrained.

How to read the table and bar chart:
- A single branch near chance carries no text-relevant information *in this model*.
- "All branches" equal to the best single branch: the others add nothing.
- "All branches" below the best single branch: the others add noise.
- "All branches" above every single branch: the branches are complementary.

**Limit of this analysis:** it shows what the jointly trained model relies on. A branch can look useless only because another branch dominated training. It does not show what a branch can do when trained alone; that is section 5.

**Recall@K curve.** Recall for every K from 1 to 100, with the chance line. A branch whose curve lies on the chance line is uninformative at every depth, not just at the top.

**Alignment table.**
- *matched*: mean cosine similarity between a caption and its own video's branch embedding.
- *mismatched*: the same against other videos.
- *gap* = matched − mismatched. Retrieval depends on the gap, not on the absolute values. A gap near 0 means the branch does not separate right from wrong videos.

**Complementarity.** Starting from the I-frame branch alone, the number of queries that enter ("gained") or leave ("lost") the top K when other branches are added. A positive net with many gained queries means the other branches recover cases the I-frame misses. In the scatter plot, each point is a query; points below the diagonal are ranked better by the full model than by the I-frame alone.

### Section 5: Training-time ablation

Each run was **trained** with its own set of branches under the same data, schedule and budget. This is the fair test of what a combination can reach.

Reading the comparison:
- *I-frame only* vs. *zero-shot I-frame*: what fine-tuning adds to the pretrained I-frame tower.
- *Full* vs. *I-frame only*: what residuals and motion vectors add on top.
- *Residual + motion vector*: how far retrieval gets with no I-frame at all. This is the number most specific to the compressed-domain question.
- *Full (fair)* vs. *Full*: whether the per-branch loss, CLIP initialisation of the motion-vector encoder and higher learning rates make the weak branches useful.

Differences of one or two R@1 points between single runs are within what a different seed can produce. Treat them as inconclusive unless repeated.

### Section 6: Similarity structure

A heatmap of cosine similarity between 30 captions (rows) and their videos (columns). Correct pairs are on the diagonal.
- Dark diagonal on a light background: the model separates matching pairs.
- Uniform colour, or horizontal/vertical stripes: embeddings are not discriminative. Stripes mean some captions or videos are similar to everything.
- Comparing the I-frame-only heatmap with the non-I-frame one shows directly which branches produce the diagonal.

### Section 7: Qualitative examples

Randomly drawn successes (correct item ranked first) and failures (correct item below rank 10), in both directions, with the top retrieved items. Use the failures to see *how* the model is wrong: retrieved items that are semantically close to the query (near-duplicates, same activity) are a milder failure than unrelated ones. MSR-VTT has many near-duplicate captions, so some "failures" are reasonable answers.

### Section 8: Excluded videos

Videos that could not be decoded. They are removed from the candidate pool with their captions. If the list is not empty, the pool is smaller than 1000 and results are not strictly comparable with runs that have a different list.

## 4. Two kinds of ablation, side by side

| | Inference-time (section 4) | Training-time (section 5) |
|---|---|---|
| What changes | Branches used to build the video embedding | Branches the model is built and trained with |
| Retraining | No | Yes, one run per variant |
| Answers | What does this trained model rely on? | What can this set of branches achieve? |
| Main caveat | Reflects how training went, not a branch's potential | Costs a session per variant; single runs have seed noise |

Report both. Agreement between them is strong evidence; disagreement (a branch useless in section 4 but useful in section 5) means the joint training under-used that branch.

## 5. Checklist before presenting a number

- The runs being compared share splits, epochs, `--max_train_samples`, frame budget and seed (section 1 of the notebook).
- The number comes from the intended split (validation for choices, test for the final table).
- The excluded-video list is empty or identical across the compared runs.
- The learning-rate curve reaches zero at the end, i.e. the run actually finished.

## 6. Addendum: per-epoch retrieval figure and data-loading metrics

**Retrieval after each epoch** (notebook section 2, `analysis.plot_epoch_metrics`). For each run, six panels: recall (R@1, R@5, R@10 on one percentage axis), median rank and mean rank, for text-to-video and video-to-text. The table below the figure holds the same numbers.

- Recall rising and ranks falling: the model is still improving; more epochs may help.
- Recall flat while training loss keeps falling: the model is fitting the training set without generalising.
- Median rank improving while mean rank does not: typical queries get better, but a tail of very hard queries stays far down the ranking.
- These curves use the full model (all built branches). When one branch is weak, the full-model curve can sit below the best single branch; compare with the per-branch curves in the same section.

**Data-loading metrics** (logged with the training metrics):

| Metric | Meaning |
|---|---|
| `time_per_step` | Wall-clock seconds per optimizer step |
| `data_wait_per_step` | Seconds per step the training loop spent waiting for decoded batches |

If `data_wait_per_step` is most of `time_per_step`, video decoding on the CPU is the bottleneck and the GPU is idle part of the time. `--cache_dir` (decode each video once, then read it back) and more `--num_thread_reader` workers address that; a larger batch or more samples per GOP make it worse.
