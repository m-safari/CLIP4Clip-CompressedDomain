"""
Extract only the ActivityNet val videos out of friedrichor/ActivityNet_Captions' split tar,
without keeping the whole 42 GB on disk: each part is downloaded (the next one in the
background), read once as part of a single tar stream, then deleted.

    python anet_stream.py <val_ids.json> <out_dir>

Videos are written as <out_dir>/v_<id>.<ext>, the name the dataloader looks for.
"""
import io, json, os, sys, tarfile, threading
from pathlib import Path
from huggingface_hub import hf_hub_download

REPO = "friedrichor/ActivityNet_Captions"
PARTS = ["ActivityNet_Videos.tar.part-{:03d}".format(i) for i in range(8)]
EXTS = (".mp4", ".mkv", ".webm", ".avi")
ids_path, out = Path(sys.argv[1]), Path(sys.argv[2])
want = set(json.load(open(ids_path)))
out.mkdir(parents=True, exist_ok=True)
dl_dir = out.parent / "_parts"


def fetch(name, box):
    try:
        box["path"] = hf_hub_download(REPO, name, repo_type="dataset", local_dir=str(dl_dir))
    except Exception as e:  # surfaced when the part is needed
        box["error"] = e


class Parts(io.RawIOBase):
    def __init__(self):
        self.i, self.f, self.cur = 0, None, None
        self.next = self._start(0)

    def _start(self, i):
        if i >= len(PARTS):
            return None
        box = {}
        t = threading.Thread(target=fetch, args=(PARTS[i], box), daemon=True)
        t.start()
        return (t, box)

    def readable(self):
        return True

    def readinto(self, b):
        while True:
            if self.f is None:
                if self.next is None:
                    return 0
                t, box = self.next
                t.join()
                if "error" in box:
                    raise box["error"]
                self.cur = box["path"]
                print("reading", PARTS[self.i], flush=True)
                self.f = open(self.cur, "rb")
                self.i += 1
                self.next = self._start(self.i)
            n = self.f.readinto(b)
            if n:
                return n
            self.f.close(); self.f = None
            os.remove(self.cur)


kept = 0
with tarfile.open(fileobj=io.BufferedReader(Parts(), 1 << 22), mode="r|") as tar:
    for m in tar:
        if not m.isfile():
            continue
        stem, ext = os.path.splitext(os.path.basename(m.name))
        vid = stem if stem.startswith("v_") else "v_" + stem
        if ext.lower() not in EXTS or vid not in want:
            continue
        target = out / (vid + ext.lower())
        if target.exists() and target.stat().st_size == m.size:
            kept += 1
            continue
        src = tar.extractfile(m)
        with open(str(target) + ".tmp", "wb") as f:
            while True:
                chunk = src.read(1 << 22)
                if not chunk:
                    break
                f.write(chunk)
        os.replace(str(target) + ".tmp", target)
        kept += 1
        if kept % 250 == 0:
            print("kept", kept, flush=True)
print("ANET_STREAM_DONE kept", kept, "of", len(want), flush=True)
