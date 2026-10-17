# ActivityNet Captions in the compressed domain

Run these commands from `compressed_domain/`.

This implements **full-video / paragraph retrieval**, following this repository's
RGB ActivityNet loader: join each video's `sentences` in annotation order and
sample compressed representations across the full video. Event timestamps do
not crop the video. I-frames, residuals and motion vectors share sampled GOPs
and retain separate sequence lengths and masks.

## Prepare the data

Install your usual training environment and FFmpeg. For downloading YouTube
source videos, also install the optional downloader:

```bash
pip install yt-dlp
python -m dataloaders.prepare_activitynet_dataset \
  --download-path /storage/data/activitynet --num-val 1000 --seed 42
```

Annotations come from the [official ActivityNet Captions release](https://cs.stanford.edu/people/ranjaykrishna/densevid/).
Videos are requested from their YouTube IDs. Unavailable videos are recorded in
`failures.json`; only successfully prepared videos enter the split CSVs. Keep
`preparation_report.json` with results: download availability affects pool size.
The preparation script does not download the entire dataset during tests.

To reuse videos you already have, including nested train/validation directories:

```bash
python -m dataloaders.prepare_activitynet_dataset \
  --download-path /storage/data/activitynet \
  --videos-dir /storage/data/activitynet_raw \
  --annotations-dir /storage/data/activitynet_annotations \
  --num-val 1000 --seed 42
```

`--annotations-dir` must hold official `train.json` and `val_1.json`; omit it to
download captions. Video stems may include or omit `v_`. Duplicate stems across
directories are rejected rather than choosing a file silently. Local-source
mode does not attempt downloads for missing videos.

Outputs:

- `mpeg4_videos/v_<id>.mp4`: MPEG-4 Part 2 elementary streams for CoViAR, fixed GOP size 12, no B-frames. The `.mp4` suffix follows the existing pipeline's convention.
- `splits/ActivityNet_train.csv`, `ActivityNet_val.csv`, `ActivityNet_test.csv`: `video_id,sentence` rows.
- `splits/ActivityNet_data.json`: merged official train and `val_1` annotations, shared by all three loaders.
- `splits/preparation_report.json`: seed, split counts and missing source IDs.
- `failures.json`: download/re-encoding failures from the latest preparation attempt.

The validation set is a seeded holdout from available **training** videos;
`val_1.json` supplies the held-out retrieval test set. This is an explicit
retrieval split convention, not the hidden official ActivityNet challenge test
set. `val_2.json` is not treated as an independent test split: its annotations
can describe the same videos as `val_1`. Different available source videos can
produce different splits even with the same seed. Reuse the generated CSVs for
comparable runs.

Re-running reuses downloaded annotations and nonempty encoded outputs, and
retries failed sources. Reuse the same preparation arguments for stable splits.
For a small preparation check, add `--max-videos-per-split 5 --num-val 1`
(at least two available training videos and one held-out video are required).

## Train

Your existing training convention works with `--datatype activitynet`:

```bash
python -m main_task_retrieval --do_train --datatype activitynet \
  --epochs 6 --epochs_per_session 1 --batch_size 16 --n_display 50 \
  --train_csv /storage/data/activitynet/splits/ActivityNet_train.csv \
  --val_csv /storage/data/activitynet/splits/ActivityNet_val.csv \
  --test_csv /storage/data/activitynet/splits/ActivityNet_test.csv \
  --data_path /storage/data/activitynet/splits/ActivityNet_data.json \
  --features_path /storage/data/activitynet/mpeg4_videos \
  --lr 1e-4 --max_words 64 --max_frames 4 --batch_size_val 16 \
  --max_train_samples 32000 --feature_framerate 1 --coef_lr 1e-3 \
  --freeze_layer_num 0 --freeze_text 1 --slice_framepos 2 \
  --pretrained_clip_name ViT-B/32 --grad_checkpoint --num_thread_reader 6 \
  --resume_model auto --residual_per_gop 6 --mv_per_gop 2 \
  --residual_arch vit --residual_coef_lr 0.1 --mv_arch cnn \
  --modality_loss_weight 0.5 \
  --output_dir /storage/ckpts/activitynet_small_arch
```

Do not pass `--expand_msrvtt_sentences`: ActivityNet has one paragraph per video.
`--max_train_samples` is capped at the number of training videos, so 32000 does
not repeat a smaller split. `--max_words 64` gives paragraphs more context than
32; both are supported, up to CLIP's 77-token context including special tokens.
`--max_frames` remains the number of sampled GOPs, not decoded RGB frames;
`--feature_framerate` is a legacy argument and does not control GOP sampling.
All existing architecture, loss, optimizer, precision, MLflow and resume options
continue to apply. Omitting `--datatype` retains MSR-VTT behavior.

The ActivityNet loader can also read an official annotation JSON directly, or a
directory containing `train.json` and `val_1.json`. For a validation holdout drawn
from training, use the generated merged JSON as above so validation IDs exist
in the selected annotations. CSV IDs use the official `v_` prefix. A CSV fixes
row order and membership; paragraph text is always read from annotations.

## Evaluate the selected checkpoint

Use a separate evaluation invocation, with the same architecture and sampling
settings as training. Training selects the best checkpoint on validation and
does not automatically evaluate `--test_csv`.

```bash
python -m main_task_retrieval --do_eval --datatype activitynet --eval_on test \
  --test_csv /storage/data/activitynet/splits/ActivityNet_test.csv \
  --data_path /storage/data/activitynet/splits/ActivityNet_data.json \
  --features_path /storage/data/activitynet/mpeg4_videos \
  --max_words 64 --max_frames 4 --batch_size_val 16 \
  --residual_per_gop 6 --mv_per_gop 2 --slice_framepos 2 \
  --residual_arch vit --mv_arch cnn --pretrained_clip_name ViT-B/32 \
  --num_thread_reader 6 --resume_model auto \
  --output_dir /storage/ckpts/activitynet_small_arch
```

Missing or failed decodes yield zero masks. Existing training/evaluation code
excludes them, counts failures, reports both retrieval directions and exports
`embeddings_test.npz` with paragraph text, video IDs and per-branch embeddings.
The existing MSR-VTT analysis helpers can read this same embedding format.

## Offline checks

```bash
python3 -m unittest discover -s tests -p test_activitynet.py -v
```

Preparation checks use synthetic annotations, mocked downloads and an FFmpeg
synthetic video when FFmpeg is available. Tensor loading, failed decode masking,
batch trimming and train/validation/test loader integration use mocked decoding
and require torch, torchvision and pandas. No checkpoint or dataset download is
needed. These checks do not establish real CoViAR decoding or training accuracy.
