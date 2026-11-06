# MSR-VTT training on molab: settings

Main run of CLIP4Clip-CompressedDomain (`dev` branch) on molab, MLflow run `b246f53e5c654ea7af9aec06e261793e`.
Checkpoints are mirrored after every epoch to the private Hugging Face repo `amv2002/clip4clip-msrvtt-main`.

## Command

```bash
cd compressed_domain
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python main_task_retrieval.py --do_train --resume_model auto \
  --epochs 16 --epochs_per_session 2 --max_train_samples 100000 --num_thread_reader 16 \
  --batch_size 32 --batch_size_val 16 --n_display 50 \
  --datatype msrvtt --expand_msrvtt_sentences \
  --train_csv $M/raw_data/MSRVTT_train.8k.csv --val_csv $M/raw_data/MSRVTT_val.1k.csv \
  --test_csv $M/raw_data/MSRVTT_JSFUSION_test.csv --data_path $M/raw_data/MSRVTT_data.json \
  --features_path $M/mpeg4_videos --output_dir /home/marimo/ckpts/msrvtt_main \
  --max_frames 4 --residual_per_gop 8 --mv_per_gop 2 --max_words 32 --feature_framerate 1 --slice_framepos 2 \
  --lr 1e-4 --coef_lr 1e-3 --residual_coef_lr 0.1 --mv_coef_lr 1.0 --modality_loss_weight 0.5 \
  --freeze_layer_num 0 --freeze_text 1 --seed 42 \
  --loose_type --linear_patch 2d --sim_header meanP --pretrained_clip_name ViT-B/32
```

`$M` is `/home/marimo/data/msrvtt`.

## Settings passed explicitly

### Run control
| Setting | Value | Note |
|---|---|---|
| `--do_train` | on | |
| `--resume_model` | `auto` | Continues from the last checkpoint in `output_dir` |
| `--epochs` | 16 | |
| `--epochs_per_session` | 2 | Each molab session trains 2 epochs, then stops at an epoch boundary |
| `--max_train_samples` | 100000 | Seeded random subset each epoch: 3,125 steps per epoch, 50,000 in total |
| `--seed` | 42 | |
| `--n_display` | 50 | Log every 50 steps |

### Batching and data loading
| Setting | Value | Note |
|---|---|---|
| `--batch_size` | 32 | |
| `--batch_size_val` | 16 | |
| `--num_thread_reader` | 16 | molab shows 20 CPUs |

### Data (MSR-VTT)
| Setting | Value | Note |
|---|---|---|
| `--datatype` | `msrvtt` | |
| `--expand_msrvtt_sentences` | on | |
| `--train_csv` | `MSRVTT_train.8k.csv` | The 9k train split minus the val videos |
| `--val_csv` | `MSRVTT_val.1k.csv` | 1k videos carved out of train.9k by `make_msrvtt_val_split` |
| `--test_csv` | `MSRVTT_JSFUSION_test.csv` | 1k test, untouched |
| `--data_path` | `MSRVTT_data.json` | |
| `--features_path` | `mpeg4_videos` | Re-encoded with reencode.sh's arguments (see below) |

### Frame sampling (compressed domain)
| Setting | Value |
|---|---|
| `--max_frames` | 4 |
| `--feature_framerate` | 1 |
| `--slice_framepos` | 2 |
| `--residual_per_gop` | 8 |
| `--mv_per_gop` | 2 |
| `--max_words` | 32 |

### Learning rates and loss
| Setting | Value |
|---|---|
| `--lr` | 1e-4 |
| `--coef_lr` | 1e-3 |
| `--residual_coef_lr` | 0.1 |
| `--mv_coef_lr` | 1.0 |
| `--modality_loss_weight` | 0.5 |

### Model
| Setting | Value |
|---|---|
| `--pretrained_clip_name` | ViT-B/32 |
| `--linear_patch` | 2d |
| `--sim_header` | meanP |
| `--loose_type` | on |
| `--freeze_layer_num` | 0 |
| `--freeze_text` | 1 |

## Repo defaults left unchanged
| Setting | Value |
|---|---|
| `--warmup_proportion` | 0.1 |
| `--gradient_accumulation_steps` | 1 |
| `--margin` | 0.1 |
| `--hard_negative_rate` | 0.5 |
| `--negative_weighting` | 1 |
| `--eval_frame_order` | 0 |
| Mixed precision (AMP) | on |

## Environment
| Item | Value |
|---|---|
| GPU | NVIDIA RTX PRO 6000 Blackwell (96 GB), molab |
| Python / torch | 3.13 / 2.11.0+cu130 (molab's own build; the repo's cu126 pin is dropped) |
| CoViAR | chaoyuaw/pytorch-coviar, built against FFmpeg 4.4.5 (shared libraries). GCC 14 errors relaxed with `-Wno-error=incompatible-pointer-types -Wno-error=return-mismatch -Wno-error=int-conversion -Wno-error=implicit-function-declaration` |
| CUDA allocator | `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` |
| Video re-encode | `ffmpeg -i <in> -vf scale=340:256,setsar=1:1 -q:v 1 -c:v mpeg4 -f rawvideo <out>.mp4` (reencode.sh's arguments, run 16 at a time) |
| Dataset source | Hugging Face dataset `friedrichor/MSR-VTT` (`MSRVTT_Videos.zip`, `raw_data/*`) |
| Code | `dev` branch of CLIP4Clip-CompressedDomain |
| Checkpoints | MLflow (`mlflow.db`, `mlartifacts/`) in `output_dir`, mirrored to `amv2002/clip4clip-msrvtt-main` |

## Resume rule
`--epochs`, `--batch_size`, `--max_train_samples` and the training CSV must stay the same in every session. Otherwise the script refuses to resume. `--epochs_per_session` may change.

## Results so far (validation, text-to-video R@1)
| Epoch | Train loss | Combined | I-frame | Residual | Motion vector |
|---|---|---|---|---|---|
| 1 | 1.68 | **33.9** (best) | 31.5 | 22.6 | 0.8 |
| 2 | 1.33 | 33.1 | 33.8 | 20.5 | 0.9 |
| 3 | 1.12 | 31.7 | 34.3 | 19.8 | 0.4 |
| 4 | 0.93 | 31.3 | 35.1 | 20.5 | 0.2 |
| 5 | 0.76 | 30.0 | **35.4** | 19.9 | 0.3 |

The checkpoint sanity test that came before the main run had passed: 2,000 samples, `epochs 2`, `epochs_per_session 1`, with the second session restored from Hugging Face and resumed.
