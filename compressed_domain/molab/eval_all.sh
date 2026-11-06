#!/bin/bash
# Zero-shot evaluation of the gated MSR-VTT best checkpoint (epoch 2) on MSVD (test) and ActivityNet (val_1).
# Every step leaves a marker in /home/marimo/eval, so a second run skips what is done.
set -x
cd /home/marimo
E=/home/marimo/eval; mkdir -p $E
step() { [ -f $E/$1.done ]; }
mark() { touch $E/$1.done; echo "STEP_DONE $1 $(date -u +%H:%M:%S)"; }

# 1. Environment (same as the training bootstrap, minus MSR-VTT).
if ! step env; then
  sed -n '1,/coviar OK/p' /home/marimo/boot_env.sh > /tmp/env.sh
  bash /tmp/env.sh || { echo ENV_FAILED; exit 1; }
  python /home/marimo/apply_patch.py || exit 1
  mark env
fi
C=/home/marimo/repo/compressed_domain

# 2. Best checkpoint (epoch 2) of the gated run.
BEST=$E/best_state_dict.pth
if ! step ckpt; then
  python - <<'PY' || exit 1
import os, shutil
from huggingface_hub import hf_hub_download
p = hf_hub_download("amv2002/clip4clip-msrvtt-gated",
    "mlartifacts/6a8a27d6e89c46b3afde40dfbda63fea/artifacts/pytorch_model.best/state_dict.pth",
    token=open("/home/marimo/.hf_token").read().strip(), local_dir="/home/marimo/eval/_hf")
shutil.move(p, "/home/marimo/eval/best_state_dict.pth")
PY
  mark ckpt
fi

encode() {  # encode <raw dir> <out dir>: reencode.sh's ffmpeg arguments, 18 at a time; out name = raw stem
  mkdir -p "$2"
  find "$1" -type f \( -iname '*.mp4' -o -iname '*.avi' -o -iname '*.mkv' -o -iname '*.webm' \) | xargs -P 18 -I{} sh -c \
    'f="{}"; b=$(basename "$f"); out="'$2'/${b%.*}.mp4"; [ -s "$out" ] || ffmpeg -loglevel quiet -y -i "$f" -vf scale=340:256,setsar=1:1 -q:v 1 -c:v mpeg4 -f rawvideo "$out"'
}

# 3. MSVD: splits + captions from CLIP4Clip's msvd_data.zip, videos from friedrichor/MSVD.
V=/home/marimo/data/msvd
if ! step msvd_data; then
  cd $C && python - <<PY || exit 1
from dataloaders.prepare_msvd_dataset import download, extract_data, extract_videos, download_videos_archive, DATA_URL
from pathlib import Path
d = Path("$V"); d.mkdir(parents=True, exist_ok=True)
extract_data(download(DATA_URL, d), d / "msvd_data")
raw = extract_videos(download_videos_archive(d), d / "videos")
open("$E/msvd_raw_dir", "w").write(str(raw))
PY
  rm -f $V/MSVD_Videos.zip
  encode "$(cat $E/msvd_raw_dir)" $V/mpeg4_videos
  echo "msvd re-encoded: $(ls $V/mpeg4_videos | wc -l)"
  mark msvd_data
fi

run_eval() {  # run_eval <name> <datatype> <data_path> <features_path> <max_words> <max_frames> <batch_size_val>
  step eval_$1 && return
  cd $C && PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python main_task_retrieval.py --do_eval \
    --init_model $BEST --output_dir $E/out_$1 --num_thread_reader ${THREADS:-16} \
    --datatype $2 --data_path $3 --features_path $4 \
    --max_words $5 --max_frames $6 --batch_size_val $7 \
    --residual_per_gop 8 --mv_per_gop 2 --feature_framerate 1 --slice_framepos 2 \
    --loose_type --linear_patch 2d --sim_header meanP --fusion gated --pretrained_clip_name ViT-B/32 \
    > $E/eval_$1.log 2>&1
  local rc=$?; echo "EXIT=$rc" >> $E/eval_$1.log
  [ $rc -eq 0 ] && mark eval_$1
  python /home/marimo/push_eval.py || true
}

# 4. MSVD test (670 videos), CLIP4Clip's MSVD setting: 32 words, 12 GOPs.
run_eval msvd msvd $V/msvd_data $V/mpeg4_videos 32 12 16

# 5. ActivityNet val_1: captions via the repo, val videos streamed out of the mirror's split tar.
A=/home/marimo/data/activitynet
if ! step anet_data; then
  cd $C && python - <<PY || exit 1
from dataloaders.prepare_activitynet_dataset import download_captions
from pathlib import Path
Path("$A").mkdir(parents=True, exist_ok=True)
download_captions(Path("$A") / "activitynet_data")
PY
  python /home/marimo/anet_stream.py $A/activitynet_data/val_ids.json $A/videos || exit 1
  encode $A/videos $A/mpeg4_videos
  echo "activitynet re-encoded: $(ls $A/mpeg4_videos | wc -l)"
  # Writes missing_videos_val.txt; everything found is already re-encoded, so nothing is encoded again.
  cd $C && python dataloaders/prepare_activitynet_dataset.py --download-path $A --videos-dir $A/videos || exit 1
  mark anet_data
fi

# 6. ActivityNet val_1 paragraph-to-video, CLIP4Clip's setting: 64 words, 64 GOPs.
run_eval activity activity $A/activitynet_data $A/mpeg4_videos 64 64 ${ANET_BSV:-2}
echo ALL_DONE
