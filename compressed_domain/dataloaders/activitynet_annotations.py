"""ActivityNet Captions paragraph annotations (no ML dependencies)."""
import json
from pathlib import Path


def load_paragraphs(path):
    """Read official train.json / val_1.json, joining sentences in annotation order."""
    with open(path) as handle:
        annotations = json.load(handle)
    paragraphs = {}
    for video_id, item in annotations.items():
        sentences = item.get("sentences")
        if not isinstance(sentences, list) or not sentences or any(
                not isinstance(s, str) or not s.strip() for s in sentences):
            raise ValueError(f"{path}: {video_id} needs a nonempty list of sentences")
        paragraphs[video_id] = " ".join(s.strip() for s in sentences)
    return paragraphs


def index_videos(root):
    """Index nested video directories; accept IDs with or without ActivityNet's v_ prefix."""
    paths = {}
    for path in sorted(Path(root).rglob("*")):
        if not path.is_file() or path.suffix.lower() not in (".mp4", ".avi", ".mkv", ".webm", ".mov"):
            continue
        video_id = path.stem if path.stem.startswith("v_") else "v_" + path.stem
        if video_id in paths:
            raise ValueError(f"Ambiguous video {video_id}: {paths[video_id]} and {path}")
        paths[video_id] = path
    return paths
