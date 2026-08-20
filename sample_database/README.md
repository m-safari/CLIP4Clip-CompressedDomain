# Published MSR-VTT 1k vectors

The source data for the shipped database: real MSR-VTT embeddings published
with the same CLIP4Clip model repository this project references.

- `embeddings.npy`: 1,000 x 512 normalized float32 visual vectors (1.95 MiB).
- `MSRVTT_JSFUSION_test.csv`: the official 1,000-row test split. Row order maps
  directly to the embedding rows, and its `sentence` column supplies captions.

These are a vector matrix and a CSV, not yet a database. Turn them into one:

```bash
python -m embedding_db import-vectors --vectors sample_database/embeddings.npy --split-csv sample_database/MSRVTT_JSFUSION_test.csv --output-dir msrvtt_1k_db --model Searchium-ai/clip4clip-webvid150k
```

The published feature filename is
`MSRVTT_test_visual_vectors_Cl4Cl_msrvtt9k.npy`, so these vectors come from the
MSR-VTT-9k-finetuned CLIP4Clip checkpoint. They were downloaded, not produced by
this repository's `build` command, whose default checkpoint is
`Searchium-ai/clip4clip-webvid150k`. Scored against the matched published text
vectors they reach R@1 43.40, the expected figure for that checkpoint.

Load the matrix directly:

```python
import numpy as np

vectors = np.load("sample_database/embeddings.npy", mmap_mode="r")
assert vectors.shape == (1000, 512)
```

SHA-256 of `embeddings.npy`:
`3234cf34e1d63e2008d3c9ff2a1e7da909a46adf9080f7ea3f99b5f524cff784`.
