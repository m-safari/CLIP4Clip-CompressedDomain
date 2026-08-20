# Measurements

## What was measured, and what was not

Two very different things are recorded here. The storage and retrieval numbers
are measurements of the real shipped database. The inference timing is a
**proxy**, and is labelled as one below rather than presented as a dataset
result.

| Result | Status |
|---|---|
| Storage capacity, packing sizes | Measured, exact byte counts |
| Retrieval quality per packing | Measured on 1,000 real MSR-VTT vectors |
| Encoder compute cost | Proxy: 16 copies of one 8-frame clip, on CPU |
| Real MSR-VTT decode cost | **Not measured** — no raw videos were processed |
| GPU throughput | **Not measured** — no CUDA device was available |

## Storage (measured, 2026-08-20)

Exact file sizes of the 1,000-item database in each packing:

| Packing | embeddings.npy | scales.npy | Database total |
|---|---:|---:|---:|
| float32 | 2,048,128 | — | 2,175,559 |
| float16 | 1,024,128 | — | 1,151,729 |
| int8 | 512,128 | 4,128 | 643,854 |
| binary | 64,128 | — | 191,728 |

Per-item metadata is 127.4 bytes (`items.jsonl` with captions, `completed.npy`,
`manifest.json`). Capacity against a 2 GiB budget is in
[`EMBEDDING_DB.md`](EMBEDDING_DB.md); the headline is 987,153 float32 items,
so 2,000 items uses 0.2% of the budget.

## Retrieval (measured, 2026-08-20)

Text-to-video on the official 1,000-video JSFusion test split, scored against
the matched published text vectors. No model is loaded for this.

| Packing | R@1 | R@5 | R@10 | MedianR | MeanR |
|---|---:|---:|---:|---:|---:|
| float32 | 43.40 | 70.00 | 80.90 | 2.0 | 15.67 |
| float16 | 43.20 | 70.00 | 81.00 | 2.0 | 15.67 |
| int8 | 42.80 | 69.80 | 80.60 | 2.0 | 15.42 |
| binary | 29.30 | 53.40 | 65.10 | 4.0 | 40.79 |

The float32 row matches the published CLIP4Clip meanP result for MSR-VTT-9k,
which is the check that the database and the metric code are both correct.

## Encoder compute proxy (2026-08-19)

Measured on this machine's CPU (`torch 2.7.1+cpu`, no CUDA device) using the
model repository's published example clip. To get a steady reading, 16 copies of
that one 8-frame clip were decoded as a batch and inferred as 128 frames.

**Read this as the cost of the vision transformer, not as a dataset
measurement.** Repeating one file means the decode path ran warm-cache against a
single small video, so real MSR-VTT decoding — seeking into 10,000 distinct
files of varying length and codec — is not represented at all.

| Measurement | Result |
|---|---:|
| Model load | 15.03 s |
| Processing 16 videos / 128 frames | 10.76 s |
| Vision inference only | 9.22 s |
| Inference throughput | 13.88 frames/s |
| End-to-end throughput | 1.49 videos/s |

Extrapolating inference alone at the configured 12 sampled frames per video
gives roughly 14 minutes of CPU compute per 1,000 videos and 29 minutes per
2,000, plus the one-time 605 MB checkpoint download. Decode time is additional
and unquantified.

No GPU run was performed. A real `build` writes `benchmark.json` with measured
decode, inference, and wall time plus 1,000/2,000-item projections for whatever
machine it ran on; that file, not this proxy, is the number to trust when a
Paperspace GPU is available.
