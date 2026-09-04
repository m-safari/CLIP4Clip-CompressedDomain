#!/usr/bin/env bash
if [ "$#" -lt 2 ] || [ "$#" -gt 3 ]; then
    echo "Usage: ./reencode.sh [input dir] [output dir] [max videos (optional)]"
    exit 1
fi

indir=$1
outdir=$2
limit=$3   # empty means no limit

if [[ ! -d "${outdir}" ]]; then
  echo "${outdir} doesn't exist. Creating it."
  mkdir -p "${outdir}"
fi

count=0
for inname in "${indir}"/*.avi "${indir}"/*.mp4
do
        [ -e "$inname" ] || continue   # skip if glob didn't match anything

        if [[ -n "$limit" && "$count" -ge "$limit" ]]; then
                break
        fi

        outname="${outdir}/${inname##*/}"
        outname="${outname%.*}.mp4"

        # avoid re-encoding an mp4 onto itself
        if [[ "$inname" == "$outname" ]]; then
                continue
        fi

        ffmpeg -loglevel quiet -y -i "${inname}" -vf scale=340:256,setsar=1:1 -q:v 1 -c:v mpeg4 -f rawvideo "${outname}"

        count=$((count + 1))

        if (( count % 100 == 0 )); then
                echo "Processed ${count} videos..."
        fi
done

echo "Converted ${count} video(s)."
