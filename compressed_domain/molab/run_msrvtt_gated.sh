#!/bin/bash
# Resumable MSR-VTT training on molab.
#   run_msrvtt.sh <output_dir> <hf_repo> <epochs> <epochs_per_session> <max_train_samples> <num_thread_reader>
# Restores <output_dir> from <hf_repo> if it is missing, trains with
# --resume_model auto (so the same command continues the run), and mirrors the
# folder to <hf_repo> while it runs. Keep epochs / batch size / max_train_samples
# identical across sessions of one run: the script refuses to resume otherwise.
OUT=$1; REPO=$2; EPOCHS=$3; PER_SESSION=$4; SAMPLES=$5; THREADS=$6
M=/home/marimo/data/msrvtt
[ -d "$OUT" ] || python /home/marimo/hf_sync.py restore "$OUT" "$REPO" || mkdir -p "$OUT"
cd /home/marimo/repo/compressed_domain
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python main_task_retrieval.py --do_train --resume_model auto \
  --epochs $EPOCHS --epochs_per_session $PER_SESSION --max_train_samples $SAMPLES --num_thread_reader $THREADS \
  --batch_size 32 --batch_size_val 16 --n_display 50 \
  --datatype msrvtt --expand_msrvtt_sentences \
  --train_csv $M/raw_data/MSRVTT_train.8k.csv --val_csv $M/raw_data/MSRVTT_val.1k.csv \
  --test_csv $M/raw_data/MSRVTT_JSFUSION_test.csv --data_path $M/raw_data/MSRVTT_data.json \
  --features_path $M/mpeg4_videos --output_dir "$OUT" \
  --max_frames 4 --residual_per_gop 8 --mv_per_gop 2 --max_words 32 --feature_framerate 1 --slice_framepos 2 \
  --lr 1e-4 --coef_lr 1e-3 --residual_coef_lr 0.1 --mv_coef_lr 1.0 --modality_loss_weight 0.5 \
  --freeze_layer_num 0 --freeze_text 1 --seed 42 \
  --loose_type --linear_patch 2d --sim_header meanP --fusion gated --pretrained_clip_name ViT-B/32 > /tmp/train.log 2>&1 &
TRAIN=$!
python /home/marimo/hf_sync.py push "$OUT" "$REPO" --watch $TRAIN > /tmp/sync.log 2>&1
wait $TRAIN; echo EXIT=$? >> /tmp/train.log
