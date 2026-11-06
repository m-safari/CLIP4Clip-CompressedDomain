# Gated-fusion model: zero-shot evaluation on MSVD and ActivityNet

Model: best checkpoint of the gated MSR-VTT run (epoch 2, `pytorch_model.best`, MSR-VTT val R@1 33.9),
MLflow run `6a8a27d6e89c46b3afde40dfbda63fea`, HF repo `amv2002/clip4clip-msrvtt-gated`.
Trained on MSR-VTT only; no fine-tuning on MSVD or ActivityNet.
Logs and embeddings: `eval_results/` in the same HF repo. Run on molab, 2026-11-05.

## MSVD test (670 videos, 27,763 captions; every caption is a query)
Settings: `--max_words 32 --max_frames 12` (GOPs), residual 8 / MV 2 per GOP.

| | R@1 | R@5 | R@10 | Median R |
|---|---|---|---|---|
| **Combined, text-to-video** | **38.1** | 67.2 | 77.5 | 2 |
| I-frame branch alone (T2V) | 41.5 | 69.6 | 78.8 | 2 |
| Residual branch alone (T2V) | 19.0 | 42.7 | 53.9 | 8 |
| Motion-vector branch alone (T2V) | 0.8 | 3.3 | 5.7 | 182 |
| **Combined, video-to-text** | **57.5** | 84.6 | 90.0 | 1 |
| I-frame branch alone (V2T) | 64.3 | 89.0 | 94.2 | 1 |
| Residual branch alone (V2T) | 23.0 | 47.3 | 57.3 | 6 |
| Motion-vector branch alone (V2T) | 0.3 | 2.8 | 5.5 | 198 |

Gate weights (mean over videos): I-frame 0.43, residual 0.46, MV 0.11.

## ActivityNet Captions val_1 (4,917 videos; paragraph-to-video)
Settings: `--max_words 64 --max_frames 64` (GOPs), residual 8 / MV 2 per GOP.

| | R@1 | R@5 | R@10 | Median R |
|---|---|---|---|---|
| **Combined, text-to-video** | **23.6** | 50.8 | 64.0 | 5 |
| I-frame branch alone (T2V) | 25.9 | 53.4 | 67.1 | 5 |
| Residual branch alone (T2V) | 12.1 | 29.5 | 41.7 | 16 |
| Motion-vector branch alone (T2V) | 0.1 | 0.4 | 0.9 | 1128 |
| **Combined, video-to-text** | **23.6** | 49.4 | 63.4 | 6 |
| I-frame branch alone (V2T) | 24.9 | 52.7 | 65.2 | 5 |
| Residual branch alone (V2T) | 9.2 | 24.5 | 35.0 | 23 |
| Motion-vector branch alone (V2T) | 0.1 | 0.3 | 0.8 | 1355 |

Gate weights (mean over videos): I-frame 0.43, residual 0.47, MV 0.10.

## Takeaway
On both datasets the I-frame branch alone beats the fused model (MSVD +3.4, ActivityNet +2.3 R@1),
the same pattern as on MSR-VTT. Video-to-text shows the same gap (MSVD +6.8, ActivityNet +1.3 R@1).
Per-branch video-to-text was computed from the saved embeddings; the combined rows recomputed
the same way match the logged values exactly. The motion-vector branch is at chance level.
