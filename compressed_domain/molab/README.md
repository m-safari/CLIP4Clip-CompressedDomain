# Running on molab

Scripts used to train and evaluate on [molab](https://molab.marimo.io) (free RTX PRO 6000, 96 GB).
A molab machine is ephemeral: it can reset at any time and comes back empty, so every
script restores what it needs and resumes. Checkpoints are mirrored to private
Hugging Face repos after every epoch.

Each script expects a Hugging Face write token in `/home/marimo/.hf_token` (chmod 600).
No token is stored in this folder.

| File | What it does |
|---|---|
| `molab_bootstrap_msrvtt.sh` | One-shot setup: apt packages, clone `dev`, pip, FFmpeg 4.4.5 + CoViAR, MSR-VTT download, parallel re-encode, 8k/1k/JSFUSION splits. Prints `BOOTSTRAP_DONE`. |
| `run_msrvtt_gated.sh` | Resumable MSR-VTT training with `--fusion gated`: restores `output_dir` from HF if missing, trains with `--resume_model auto`, mirrors to HF while it runs. |
| `hf_sync.py` | `push <dir> <repo> [--watch PID]` mirrors `output_dir` to a private HF repo once files are stable; `restore <dir> <repo>` downloads it back. |
| `eval_all.sh` | Zero-shot eval of the best gated checkpoint on MSVD test and ActivityNet val_1. Every step leaves a marker in `/home/marimo/eval`, so a rerun skips finished steps. |
| `anet_stream.py` | Extracts only the ActivityNet val videos from `friedrichor/ActivityNet_Captions`' 42 GB split tar, deleting each part after reading it. |
| `push_eval.py` | Uploads eval logs and embeddings to the gated run's HF repo under `eval_results/`. |
| `main_run_settings.md` | All settings of the mean-fusion main run and its results. |
| `gated_eval_msvd_activitynet.md` | MSVD and ActivityNet results of the gated model. |

## Usage

```bash
bash molab_bootstrap_msrvtt.sh > /tmp/boot.log 2>&1
# <output_dir> <hf_repo> <epochs> <epochs_per_session> <max_train_samples> <num_thread_reader>
bash run_msrvtt_gated.sh /home/marimo/ckpts/msrvtt_gated <user>/clip4clip-msrvtt-gated 10 6 100000 16
# THREADS / ANET_BSV override the eval data workers and the ActivityNet batch size (defaults 16 / 2)
bash eval_all.sh > /home/marimo/eval.log 2>&1
```

`eval_all.sh` downloads the checkpoint from `amv2002/clip4clip-msrvtt-gated`; change the
repo and run id in step 2 to evaluate another run. ActivityNet with 64 GOPs needs
`--batch_size_val 2`: at 8, the queued batches of 16 workers took over 100 GB of RAM.

## Results (text-to-video R@1)

| | Combined | I-frame alone | Residual alone | Motion vector alone |
|---|---|---|---|---|
| MSR-VTT val, mean fusion, best (epoch 1 of 8) | 33.9 | 31.5 | 22.6 | 0.8 |
| MSR-VTT val, gated fusion, best (epoch 2 of 9) | 33.9 | 34.2 | 22.8 | 0.9 |
| MSVD test, gated best, zero-shot | 38.1 | 41.5 | 19.0 | 0.8 |
| ActivityNet val_1, gated best, zero-shot | 23.6 | 25.9 | 12.1 | 0.1 |

Gated run per epoch (MSR-VTT val):

| Epoch | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|
| Combined R@1 | 33.3 | **33.9** | 31.8 | 31.7 | 28.3 | 27.7 | 26.6 | 27.1 | 26.3 |
| I-frame alone | 32.2 | 34.2 | 34.5 | 35.4 | 35.3 | 35.2 | 35.1 | 35.1 | 35.1 |
| Gate weight I-frame / residual / MV | .43/.44/.14 | .42/.47/.11 | .35/.50/.15 | .32/.51/.17 | .26/.53/.21 | .23/.53/.24 | .18/.55/.27 | .16/.53/.31 | .14/.54/.32 |

The gate is trained on the training loss, so it moves weight to the residual and motion-vector
branches, which overfit, and away from the I-frames, which generalise. After epoch 2 the fused
model falls behind the I-frame branch alone. Ideas not tried yet: train the gate on validation
data with the encoders frozen; initialise it in favour of the I-frames with a low gate lr; or
drop the motion-vector branch.
