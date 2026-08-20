# MSR-VTT embedding database

A small, file-backed database of one normalized 512-dimensional CLIP4Clip
vector per MSR-VTT video, with a capacity planner for fitting it into a storage
budget and a packing study for buying more headroom.

The shipped database is real: 1,000 MSR-VTT test videos, published CLIP4Clip
vectors, measured at **R@1 = 43.40** on text-to-video retrieval.

## The short answer

For a **2 GiB budget**, 512-dimensional vectors, measured overhead included:

| Packing | Bytes/item | Items in 2 GiB | 2,000 items costs | % of budget |
|---|---:|---:|---:|---:|
| float32 | 2,175 | 987,176 | 4.15 MiB | 0.20% |
| float16 | 1,151 | 1,865,140 | 2.20 MiB | 0.11% |
| int8 | 643 | 3,337,820 | 1.23 MiB | 0.06% |
| binary | 191 | 11,221,103 | 0.37 MiB | 0.02% |

Bytes/item includes the **measured** 127.4 bytes of per-item metadata (row
mapping, captions, valid-row mask, manifest) taken from the real 1,000-item
database, not an assumption. That figure tracks the actual caption lengths and
manifest, so it moves by a few bytes between databases; re-run the command
against your own to get its number. Vector payload alone is 2,048 / 1,024 / 516 / 64
bytes.

Two conclusions follow:

- The 1,000–2,000 item target is not a storage problem. 2,000 float32 vectors
  occupy **0.2%** of a 2 GiB budget. The entire 10,000-video MSR-VTT dataset is
  20 MiB of vectors — 2 GiB holds about **100 copies of it**.
- The budget only starts to bind around **one million items**. If the corpus is
  ever meant to reach that scale, float16 doubles the ceiling for a measured
  0.12-point R@1 cost, which is the best trade in the table.

Reproduce the numbers:

```bash
python -m embedding_db estimate --budget 2GiB --db msrvtt_1k_db
```

Drop `--db` to size the vector payload alone, and use `--dimension` for other
embedding widths. Budgets accept `2GB` (decimal) or `2GiB` (binary).

## Where to store it

Plain files on **Paperspace persistent `/storage`**. At a few megabytes the
database is smaller than a single MSR-VTT video, so nothing about it justifies
a hosted vector database, and there is no reason to risk the ephemeral
`/tmp`-style storage when the persistent volume costs nothing to use here.

Exact NumPy cosine search stays appropriate far past this scale: a full scan of
1,000 float32 vectors is a 1,000x512 matrix multiply. FAISS or an ANN index
only starts to matter in the hundreds of thousands to millions of rows — which,
per the table above, is also where the 2 GiB budget starts to bind. Below that,
an index would add operational weight and lose exactness for no gain.

## Layout on disk

| File | Contents |
|---|---|
| `embeddings.npy` | `(N, D)` vectors in the manifest's packing |
| `scales.npy` | per-row float32 scales; int8 packing only |
| `completed.npy` | valid-row mask, also used for safe resume |
| `items.jsonl` | row to video id, path, and caption |
| `manifest.json` | packing, dimensions, provenance, build config |
| `benchmark.json` | timings, written by `build` |
| `failures.jsonl` | unreadable videos, written by `build` |

Everything is portable and inspectable with `numpy.load`; there is no server
and no binary index format.

## The packing trade-off, measured

Measured on the real 1,000-item database. Neighbour agreement is the share of
each row's float32 top-10 neighbourhood that survives the packing; the retrieval
columns are text-to-video against the matched published text vectors.

| Packing | Vector file | Neighbour agreement@10 | R@1 | R@5 | R@10 | MedianR |
|---|---:|---:|---:|---:|---:|---:|
| float32 | 1.95 MiB | 1.000 | 43.40 | 70.00 | 80.90 | 2 |
| float16 | 1000 KiB | 0.999 | 43.20 | 70.00 | 81.00 | 2 |
| int8 | 504 KiB | 0.978 | 42.80 | 69.80 | 80.60 | 2 |
| binary | 62.6 KiB | 0.599 | 29.30 | 53.40 | 65.10 | 4 |

- **float16 is free.** Half the size, 0.2 points of R@1, and R@10 actually ties.
- **int8 is nearly free.** A quarter of the size for 0.6 points of R@1.
- **binary is not a drop-in.** 32x smaller but 14 points of R@1. It earns its
  place only as a cheap first-stage filter that a float32 rescoring pass then
  reranks — not as the only copy of the data.

A detail worth knowing at the small end: in the binary database, `items.jsonl`
(125 KB of captions) is **twice** the size of the vectors (64 KB). Past int8,
compressing vectors further stops helping, because the metadata is the file.

```bash
python -m embedding_db compare --db msrvtt_1k_db --top-k 10
```

```bash
python -m embedding_db pack --db msrvtt_1k_db --output-dir msrvtt_1k_db_float16 --packing float16
```

## Building the shipped database

`sample_database/` holds the published CLIP4Clip vectors for the official
1,000-video JSFusion test split, plus the split CSV. `import-vectors` turns that
matrix into a working database — the row mapping, captions, valid-row mask, and
manifest — without loading a model:

```bash
python -m embedding_db import-vectors --vectors sample_database/embeddings.npy --split-csv sample_database/MSRVTT_JSFUSION_test.csv --output-dir msrvtt_1k_db --model Searchium-ai/clip4clip-webvid150k
```

Pass `--videos-dir` to resolve item paths against real video files; without it,
paths hold the bare video id and the manifest records `paths_resolved: false`.

## Validation

The evaluation needs the matched text half of the published feature pair, a
2 MB download that is not tracked in this repository:

```bash
curl -L -o features/MSRVTT_test_textual_vectors_Cl4Cl_msrvtt9k.npy --create-dirs https://huggingface.co/Searchium-ai/clip4clip-webvid150k/resolve/main/Notebooks/features/MSRVTT_test_textual_vectors_Cl4Cl_msrvtt9k.npy
```

```bash
python -m embedding_db evaluate --db msrvtt_1k_db --text-vectors features/MSRVTT_test_textual_vectors_Cl4Cl_msrvtt9k.npy
```

The text vectors are the matched half of the same published feature pair, so
this is one matrix multiply and loads no model. The result, **R@1 43.40 /
R@5 70.00 / R@10 80.90**, matches the published CLIP4Clip meanP figure for
MSR-VTT-9k, which confirms both the stored vectors and the metric code.

```bash
python -m unittest discover -s tests -v
```

The tests download nothing and load no checkpoint.

## Building from raw video

To embed videos yourself rather than importing published vectors:

```bash
python -m embedding_db build --videos-dir /datasets/MSRVTT/MSRVTT_Videos --split-csv /datasets/MSRVTT/MSRVTT_train.9k.csv --output-dir /storage/msrvtt-2k-embeddings --limit 2000 --device cuda --precision float16 --frame-rate 1 --max-frames 12 --video-batch-size 16 --frame-batch-size 128 --decode-workers 4
```

Videos are the expensive part, not vectors: the raw MSR-VTT set is a 2.19 GB
download ([friedrichor/MSR-VTT](https://huggingface.co/datasets/friedrichor/MSR-VTT)
mirrors all 10,000 clips), and the checkpoint is a further 605 MB. Neither is
copied into the database.

An interrupted job resumes with the same arguments plus `--resume`; use
`--overwrite` to intentionally replace an output. Follows meanP pooling:
normalize every frame vector, average, normalize the result.

See [`BENCHMARK_RESULTS.md`](BENCHMARK_RESULTS.md) for what has and has not been
timed.

## Text search

```bash
python -m embedding_db search --db msrvtt_1k_db --query "a person is playing guitar" --top-k 10
```

This is the only command that loads a model, because the query text has to be
encoded. It works against any packing, and it loads the cached safetensors
weights, so no second copy of the checkpoint is downloaded.

One caveat with the imported database: its visual vectors come from the
MSR-VTT-9k checkpoint, while `--model` defaults to the `webvid150k` text
encoder. Mixing the two branches still returns sensible neighbours but is not
the matched pair, which is why `evaluate` uses the published msrvtt9k text
vectors instead. Pass `--model` explicitly if you have a matched text encoder,
or build the database with `build` so both branches come from one checkpoint.
