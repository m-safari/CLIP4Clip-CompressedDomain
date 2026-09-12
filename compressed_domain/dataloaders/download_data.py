import os
import kagglehub
import shutil
from pathlib import Path
import json
import csv
from zipfile import ZipFile
from urllib.request import urlretrieve
from huggingface_hub import snapshot_download

def download_via_hf():
    snapshot_download(
    repo_id="friedrichor/MSR-VTT",
    repo_type="dataset",
    local_dir="/tmp/msrvtt"
    )
    
    zip_output = "/tmp/msrvtt/MSRVTT_Videos.zip"
    videos_output = "/tmp/msrvtt/MSRVTT_Videos"
    
    with ZipFile(zip_output, "r") as z:
        z.extractall(videos_output)
    z.close()
    os.remove(zip_output)
    

def download_official_videos():
    url = 'https://www.robots.ox.ac.uk/~maxbain/frozen-in-time/data/MSRVTT.zip'
    zip_output = '/tmp/msrvtt/videos_official.zip'
    extraction_path = "/tmp/msrvtt"
    urlretrieve(url, zip_output)
    with ZipFile(zip_output, 'r') as zObject:
        zObject.extractall(path=extraction_path)
    zObject.close()
    os.rename(extraction_path+'/msrvtt_data', extraction_path+'/videos_official')
    os.remove(zip_output)


def download_official_captions():
    url =  'https://github.com/ArrowLuo/CLIP4Clip/releases/download/v0.0/msrvtt_data.zip'
    zip_output = '/tmp/msrvtt/captions_official.zip'
    extraction_path = "/tmp/msrvtt"
    urlretrieve(url, zip_output)
    with ZipFile(zip_output, 'r') as zObject:
        zObject.extractall(path=extraction_path)
    zObject.close()
    os.rename(extraction_path+'/msrvtt_data', extraction_path+'/captions_official')
    os.remove(zip_output)

def download_dataset():
    base = Path("/tmp/msrvtt")
    video_dir = base / "videos"
    caption_dir = base / "captions"

    video_dir.mkdir(parents=True, exist_ok=True)
    caption_dir.mkdir(parents=True, exist_ok=True)

    video_path = kagglehub.dataset_download(
        "vishnutheepb/msrvtt",
        output_dir=str(video_dir)
    )

    caption_path = kagglehub.dataset_download(
        "vishnutheepb/msrvttdatainfo",
        output_dir=str(caption_dir)
    )

    print("Videos:", video_path)
    print("Captions:", caption_path)
    
    
def caption_json_to_csv(input_path, output_path):

    with open(input_path) as jf:
        d = json.load(jf)

    Ed = d['videos']
    df = open(output_path, 'w')
    cw = csv.writer(df)
    c = 0
    for emp in Ed:
        if c == 0:
        # Writing headers of CSV file
            h = emp.keys()
            cw.writerow(h)
            c += 1

    # Writing data of CSV file
        cw.writerow(emp.values())
    df.close()