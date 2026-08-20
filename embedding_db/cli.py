"""Command line interface."""

from __future__ import annotations

import argparse
import json
import sys

from .core import DEFAULT_MODEL_ID, BuildConfig, EmbeddingDatabase, build_database, estimate_database_size
from .evaluate import evaluate_database
from .pack import PACKINGS
from .storage import (
    capacity_report,
    compare_packings,
    human_bytes,
    import_vectors,
    measure_metadata_bytes_per_item,
    pack_database,
    parse_byte_budget,
)


def _print(value) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build and query a small MSR-VTT video embedding database")
    commands = parser.add_subparsers(dest="command", required=True)

    estimate = commands.add_parser("estimate", help="Size vectors, or fit them to a storage budget")
    estimate.add_argument("--counts", nargs="+", type=int, default=[1000, 2000])
    estimate.add_argument("--dimension", type=int, default=512)
    estimate.add_argument("--dtype", choices=("float16", "float32"), default="float32")
    estimate.add_argument("--budget", help="Storage budget such as 2GB or 2GiB; switches to a capacity table")
    estimate.add_argument("--db", help="Charge each item the metadata overhead measured from this database")

    build = commands.add_parser("build", help="Build embeddings.npy and benchmark.json")
    build.add_argument("--videos-dir", required=True)
    build.add_argument("--output-dir", required=True)
    build.add_argument("--split-csv")
    build.add_argument("--id-column", default="video_id")
    build.add_argument("--limit", type=int, default=2000)
    build.add_argument("--model", default=DEFAULT_MODEL_ID)
    build.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    build.add_argument("--precision", choices=("auto", "float16", "float32"), default="auto")
    build.add_argument("--storage-dtype", choices=("float16", "float32"), default="float32")
    build.add_argument("--frame-rate", type=float, default=1.0)
    build.add_argument("--max-frames", type=int, default=12)
    build.add_argument("--frame-batch-size", type=int, default=128)
    build.add_argument("--video-batch-size", type=int, default=16)
    build.add_argument("--decode-workers", type=int, default=4)
    build.add_argument("--overwrite", action="store_true")
    build.add_argument("--resume", action="store_true")
    build.add_argument("--strict", action="store_true", help="Fail if a CSV video is missing")

    importer = commands.add_parser(
        "import-vectors", help="Turn an existing vector matrix plus a split CSV into a database"
    )
    importer.add_argument("--vectors", required=True, help="Path to an [N, D] .npy matrix")
    importer.add_argument("--output-dir", required=True)
    importer.add_argument("--split-csv", help="Row order must match the matrix")
    importer.add_argument("--id-column", default="video_id")
    importer.add_argument("--caption-column", default="sentence")
    importer.add_argument("--videos-dir", help="Optional; resolves item paths to real files")
    importer.add_argument("--model", default="", help="Checkpoint that produced the vectors")
    importer.add_argument("--source", default="", help="Provenance note stored in the manifest")
    importer.add_argument("--normalize", action="store_true", help="L2-normalize rows that are not already unit length")
    importer.add_argument("--overwrite", action="store_true")

    pack = commands.add_parser("pack", help="Rewrite a database in a cheaper vector layout")
    pack.add_argument("--db", required=True)
    pack.add_argument("--output-dir", required=True)
    pack.add_argument("--packing", choices=PACKINGS, required=True)
    pack.add_argument("--overwrite", action="store_true")

    compare = commands.add_parser("compare", help="Score every packing against the stored float32 vectors")
    compare.add_argument("--db", required=True)
    compare.add_argument("--top-k", type=int, default=10)

    evaluate = commands.add_parser("evaluate", help="Text-to-video retrieval metrics from stored vectors")
    evaluate.add_argument("--db", required=True)
    evaluate.add_argument("--text-vectors", required=True, help="[N, D] .npy aligned row-for-row with the database")
    evaluate.add_argument("--recall-at", nargs="+", type=int, default=[1, 5, 10])

    search = commands.add_parser("search", help="Text-to-video cosine search")
    search.add_argument("--db", required=True)
    search.add_argument("--query", required=True)
    search.add_argument("--top-k", type=int, default=10)
    search.add_argument("--device", default="auto")
    search.add_argument("--model", help="Defaults to the model recorded in manifest.json")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "estimate":
        if args.budget:
            overhead = 0.0
            if args.db:
                overhead = measure_metadata_bytes_per_item(args.db)["bytes_per_item"]
            _print(capacity_report(parse_byte_budget(args.budget), args.dimension, overhead))
            return 0
        rows = []
        for count in args.counts:
            row = estimate_database_size(count, args.dimension, args.dtype)
            row["human_npy_size"] = human_bytes(int(row["estimated_npy_bytes"]))
            rows.append(row)
        _print(rows)
        return 0

    if args.command == "build":
        if args.overwrite and args.resume:
            raise SystemExit("--overwrite and --resume are mutually exclusive")
        config = BuildConfig(
            model_id=args.model,
            device=args.device,
            precision=args.precision,
            storage_dtype=args.storage_dtype,
            frame_rate=args.frame_rate,
            max_frames=args.max_frames,
            frame_batch_size=args.frame_batch_size,
            video_batch_size=args.video_batch_size,
            decode_workers=args.decode_workers,
        )
        report = build_database(
            args.videos_dir,
            args.output_dir,
            config,
            args.split_csv,
            args.id_column,
            args.limit,
            args.overwrite,
            args.resume,
            args.strict,
            lambda status: print(
                f"processed {status['attempted_this_run']}/{status['pending_this_run']} "
                f"(completed={status['completed_total']}, failed={status['failed_this_run']})",
                file=sys.stderr,
                flush=True,
            ),
        )
        _print(report)
        return 0

    if args.command == "import-vectors":
        _print(
            import_vectors(
                args.vectors,
                args.output_dir,
                args.split_csv,
                args.id_column,
                args.caption_column,
                args.videos_dir,
                args.model,
                args.source,
                args.normalize,
                args.overwrite,
            )
        )
        return 0

    if args.command == "pack":
        _print(pack_database(args.db, args.output_dir, args.packing, args.overwrite))
        return 0

    if args.command == "compare":
        _print(compare_packings(args.db, args.top_k))
        return 0

    if args.command == "evaluate":
        _print(evaluate_database(args.db, args.text_vectors, tuple(args.recall_at)))
        return 0

    from .model import encode_text_query

    database = EmbeddingDatabase(args.db)
    _, _, _, manifest = database.load()
    model_id = args.model or manifest["config"]["model_id"]
    query = encode_text_query(model_id, args.query, args.device)
    _print(database.search_vector(query, args.top_k))
    return 0
