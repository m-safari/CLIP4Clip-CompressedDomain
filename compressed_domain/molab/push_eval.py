"""Copy the eval logs (and saved embeddings) to the gated run's HF repo, under eval_results/."""
import glob, os
from huggingface_hub import HfApi

E = "/home/marimo/eval"
api = HfApi(token=open("/home/marimo/.hf_token").read().strip())
files = glob.glob(E + "/eval_*.log") + glob.glob(E + "/out_*/**/embeddings_*.npz", recursive=True) \
    + glob.glob(E + "/out_*/log.txt")
for f in files:
    rel = os.path.relpath(f, E)
    api.upload_file(path_or_fileobj=f, path_in_repo="eval_results/" + rel,
                    repo_id="amv2002/clip4clip-msrvtt-gated", commit_message="eval: " + rel)
    print("pushed", rel)
