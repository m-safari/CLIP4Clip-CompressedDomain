#!/bin/bash
# One-shot molab setup for CLIP4Clip-CompressedDomain (dev) + MSR-VTT.
# Same environment steps as molab_bootstrap.sh (verified 2026-10-31); the data
# part downloads MSR-VTT from friedrichor/MSR-VTT, re-encodes it in parallel with
# reencode.sh's exact ffmpeg arguments, and builds the 8k/1k/JSFUSION splits.
set -ex
cd /home/marimo

apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq make nasm build-essential

[ -d repo ] || git clone -q -b dev https://github.com/m-safari/CLIP4Clip-CompressedDomain.git repo
git -C repo log --oneline -1
# Keep molab's own torch build (supports Blackwell); drop the cu126 pin.
grep -vE "^(--extra-index-url|torch==)" repo/compressed_domain/env/requirements.txt > /tmp/req.txt
pip install -q --root-user-action=ignore -r /tmp/req.txt setuptools

# CoViAR needs the pre-5.0 FFmpeg API: build FFmpeg 4.4 as shared libs.
if ! (cd /tmp && python -c "import coviar" 2>/dev/null); then
  [ -d coviar ] || git clone -q https://github.com/chaoyuaw/pytorch-coviar coviar
  D=/home/marimo/coviar/data_loader
  # ffmpeg.org sometimes resets the connection; retry, then fall back to the GitHub mirror.
  cd /tmp && rm -rf ffmpeg-4.4.5
  if curl -sSLf --retry 5 --retry-all-errors -o ff.tar.xz https://ffmpeg.org/releases/ffmpeg-4.4.5.tar.xz; then tar xf ff.tar.xz
  else curl -sSLf --retry 5 -o ff.tar.gz https://github.com/FFmpeg/FFmpeg/archive/refs/tags/n4.4.5.tar.gz && tar xzf ff.tar.gz && mv FFmpeg-n4.4.5 ffmpeg-4.4.5; fi
  cd ffmpeg-4.4.5
  ./configure --prefix=$D/ffmpeg --enable-pic --enable-shared --disable-static --disable-programs --disable-doc > /tmp/ffcfg.log
  make -j8 > /tmp/ffmake.log && make install > /dev/null
  echo $D/ffmpeg/lib > /etc/ld.so.conf.d/coviar-ffmpeg.conf && ldconfig
  # GCC 14 turns these old-code warnings into errors; relax them back.
  cd $D
  sed -i "s#'-L./ffmpeg/lib/'#'-L$D/ffmpeg/lib/'#" setup.py
  CFLAGS="-Wno-error=incompatible-pointer-types -Wno-error=return-mismatch -Wno-error=int-conversion -Wno-error=implicit-function-declaration" \
    pip install -q --root-user-action=ignore --no-build-isolation .
fi
cd /tmp && python -c "import coviar; print('coviar OK')"

# MSR-VTT download (videos zip + raw_data captions/splits).
M=/home/marimo/data/msrvtt
if [ ! -f $M/raw_data/MSRVTT_data.json ]; then
  python - <<PY
from huggingface_hub import snapshot_download
snapshot_download(repo_id="friedrichor/MSR-VTT", repo_type="dataset", local_dir="$M",
                  allow_patterns=["MSRVTT_Videos.zip", "raw_data/*"])
PY
  cd $M && python -c "import zipfile; zipfile.ZipFile('MSRVTT_Videos.zip').extractall('.')" && rm MSRVTT_Videos.zip
fi
RAW=$(dirname "$(find $M -name 'video*.mp4' -print -quit)")
echo "raw videos: $RAW ($(ls $RAW | wc -l) files)"

# Re-encode to CoViAR's MPEG-4 layout: reencode.sh's arguments, run in parallel.
mkdir -p $M/mpeg4_videos
ls $RAW | grep -E '\.(mp4|avi)$' | xargs -P 16 -I{} sh -c \
  'out='$M'/mpeg4_videos/$(basename {} | sed "s/\.[^.]*$//").mp4; [ -s "$out" ] || ffmpeg -loglevel quiet -y -i '$RAW'/{} -vf scale=340:256,setsar=1:1 -q:v 1 -c:v mpeg4 -f rawvideo "$out"'
echo "re-encoded: $(ls $M/mpeg4_videos | wc -l)"

# 8k train / 1k val carved from train.9k by video; JSFUSION test untouched.
cd /home/marimo/repo/compressed_domain
python -m dataloaders.make_msrvtt_val_split --train_csv $M/raw_data/MSRVTT_train.9k.csv \
  --data_path $M/raw_data/MSRVTT_data.json --test_csv $M/raw_data/MSRVTT_JSFUSION_test.csv
ls -la $M/raw_data
echo BOOTSTRAP_DONE
