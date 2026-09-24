import os
import subprocess
import kagglehub
import shutil
from pathlib import Path
import json
import csv
from zipfile import ZipFile
from urllib.request import urlretrieve
from huggingface_hub import snapshot_download

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DOWNLOAD_PATH = '/tmp/msrvtt'

def download_msrvtt_via_hf(download_path= DEFAULT_DOWNLOAD_PATH):
    snapshot_download(
    repo_id="friedrichor/MSR-VTT",
    repo_type="dataset",
    local_dir=download_path
    )
    
    zip_output = os.path.join(download_path, "MSRVTT_Videos.zip")
    videos_output = download_path
    
    with ZipFile(zip_output, "r") as z:
        z.extractall(videos_output)
    z.close()
    os.remove(zip_output)
    
    captions_path = os.path.join(download_path, "captions")
    os.rename(os.path.join(download_path, "raw_data"), captions_path)
    video_path = os.path.join(download_path, "video")
    
    return video_path, captions_path
    
    
def download_msrvtt_via_kagglehub(download_path= DEFAULT_DOWNLOAD_PATH):
    base = Path(download_path)
    video_dir = base
    videos_path = kagglehub.dataset_download(
        "vishnutheepb/msrvtt",
        output_dir=str(video_dir)
    )
    
    videos_path = os.path.join(video_dir, "raw_videos")
    os.rename(os.path.join(video_dir, "TrainValVideo"), videos_path)
    
    return videos_path
 

def reencode(src_path, dst_path, encoder_path=None):
    if encoder_path is None:
        encoder_path = SCRIPT_DIR / "reencode.sh"

    subprocess.run(
        [str(encoder_path), str(src_path), str(dst_path)],
        check=True
    )
    
    return dst_path
    
    
def prepare_msrvtt_dataset(download_path=DEFAULT_DOWNLOAD_PATH):
    try:
        if download_path is not None:
            videos_path, captions_path = download_msrvtt_via_hf(download_path)
        else:
            videos_path, captions_path = download_msrvtt_via_hf()

    except Exception as e:
        print(f"Error while downloading MSRVTT dataset: {e}")
        return

    else:
        print(f"Raw videos downloaded at: {videos_path}")
        print(f"Captions downloaded at: {captions_path}")

    videos_path = Path(videos_path)
    reencoded_path = videos_path.parent / "mpeg4_videos"
    reencoded_path.mkdir(exist_ok=True)

    print(f"Reencoding raw videos into mpeg4...")
    reencode(videos_path, reencoded_path)

    print(f"Mpeg4 reencoded videos path: {reencoded_path}")
    
    
    
######################## VERY SLOW SERVER! ##############################
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