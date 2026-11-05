"""Mirror a training --output_dir to a private Hugging Face model repo.

    python hf_sync.py push    <output_dir> <repo_id> [--watch PID]
    python hf_sync.py restore <output_dir> <repo_id>

push uploads the folder whenever its files change and have been stable for a
minute (so a checkpoint is never uploaded half-written); with --watch it keeps
going until that process exits and then does one final upload. restore pulls
the folder back so `--resume_model auto` continues the run.
"""
import os
import sys
import time

from huggingface_hub import HfApi, snapshot_download

TOKEN = open(os.path.expanduser("~/.hf_token")).read().strip()
STABLE_SECONDS = 60
POLL_SECONDS = 30


def signature(folder):
    # Only the checkpoint artifacts decide when to upload: the log and mlflow.db
    # are rewritten every few steps, so they would never look stable mid-epoch.
    sig = []
    for root, _, files in os.walk(os.path.join(folder, "mlartifacts")):
        for name in files:
            path = os.path.join(root, name)
            try:
                st = os.stat(path)
            except FileNotFoundError:
                continue
            sig.append((path, st.st_size, st.st_mtime))
    return sorted(sig)


def alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def push(folder, repo_id, watch_pid=None):
    api = HfApi(token=TOKEN)
    api.create_repo(repo_id, private=True, exist_ok=True)
    uploaded, last_seen, stable_since = None, None, None
    while True:
        running = watch_pid is not None and alive(watch_pid)
        sig = signature(folder)
        if sig != last_seen:
            last_seen, stable_since = sig, time.time()
        stable = time.time() - stable_since >= STABLE_SECONDS or not running
        if sig and sig != uploaded and stable:
            api.upload_folder(folder_path=folder, repo_id=repo_id, ignore_patterns=[".cache/*"],
                              commit_message="sync {}".format(time.strftime("%Y-%m-%d %H:%M:%S")))
            uploaded = sig
            print("uploaded", time.strftime("%H:%M:%S"), flush=True)
        if watch_pid is None or not running:
            if watch_pid is not None:
                print("SYNC_DONE", flush=True)
            return
        time.sleep(POLL_SECONDS)


def restore(folder, repo_id):
    snapshot_download(repo_id=repo_id, local_dir=folder, token=TOKEN)
    print("restored", repo_id, "->", folder, flush=True)


if __name__ == "__main__":
    mode, folder, repo_id = sys.argv[1:4]
    if mode == "push":
        pid = int(sys.argv[sys.argv.index("--watch") + 1]) if "--watch" in sys.argv else None
        push(folder, repo_id, pid)
    else:
        restore(folder, repo_id)
