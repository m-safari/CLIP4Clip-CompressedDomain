"""Command line interface."""

from __future__ import annotations

import argparse
import json
import sys

from .config import (
    DEFAULT_ENCODER,
    DEFAULT_MODEL_ID,
    DEFAULT_POOLING,
    EncoderConfig,
    encoder_config_from_dict,
    load_config_file,
)
from .core import (
    BuildConfig,
    EmbeddingDatabase,
    build_database,
    encoder_config_from_manifest,
    estimate_database_size,
    release_memmaps,
)
from .encoders import available_encoders, create_encoder
from .evaluate import evaluate_database
from .model import available_poolings
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


def add_model_arguments(parser: argparse.ArgumentParser) -> None:
    """Model-facing flags. They default to None so that an explicit flag is
    distinguishable from a default, which is what lets a config file sit
    between the defaults and the command line."""
    group = parser.add_argument_group("model")
    group.add_argument("--config", help="JSON (or YAML, with PyYAML) file of encoder/build settings")
    group.add_argument("--encoder", help=f"Encoder backend; available: {', '.join(available_encoders())}")
    group.add_argument("--model", help="Hub id or local path of the checkpoint")
    group.add_argument("--revision", help="Checkpoint revision, when the backend supports one")
    group.add_argument("--cache-dir", help="Where downloaded checkpoints are cached")
    group.add_argument("--local-files-only", action="store_true", default=None, help="Never reach the network")
    group.add_argument("--device", help="auto, cpu, cuda, or cuda:N")
    group.add_argument("--precision", choices=("auto", "float16", "float32"))
    group.add_argument("--pooling", help=f"Frame pooling; available: {', '.join(available_poolings())}")
    group.add_argument("--image-size", type=int, help="Override the checkpoint's frame size")
    group.add_argument("--image-mean", nargs="+", type=float, help="Override normalization mean")
    group.add_argument("--image-std", nargs="+", type=float, help="Override normalization std")


def resolve_configs(args: argparse.Namespace, base: EncoderConfig | None = None) -> tuple[EncoderConfig, dict]:
    """Apply precedence: defaults < config file < explicit command line."""
    file_data = load_config_file(args.config) if getattr(args, "config", None) else {}
    encoder = encoder_config_from_dict(file_data.get("encoder", {}), base)

    overrides = {
        "name": args.encoder,
        "model": args.model,
        "revision": args.revision,
        "cache_dir": args.cache_dir,
        "local_files_only": args.local_files_only,
        "device": args.device,
        "precision": args.precision,
        "pooling": args.pooling,
        "image_size": args.image_size,
        "image_mean": args.image_mean,
        "image_std": args.image_std,
    }
    encoder = encoder_config_from_dict({k: v for k, v in overrides.items() if v is not None}, encoder)
    return encoder, dict(file_data.get("build", {}))


def build_settings(args: argparse.Namespace, file_build: dict) -> dict:
    """Non-model build settings, with the same defaults < file < CLI order."""
    settings = {
        "storage_dtype": "float32",
        "frame_rate": 1.0,
        "max_frames": 12,
        "frame_batch_size": 128,
        "video_batch_size": 16,
        "decode_workers": 4,
    }
    unknown = sorted(set(file_build) - set(settings))
    if unknown:
        raise SystemExit(f"Unknown build settings {unknown}; valid settings are {sorted(settings)}")
    settings.update(file_build)
    for key in settings:
        value = getattr(args, key, None)
        if value is not None:
            settings[key] = value
    return settings


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
    build.add_argument("--storage-dtype", choices=("float16", "float32"))
    build.add_argument("--frame-rate", type=float)
    build.add_argument("--max-frames", type=int)
    build.add_argument("--frame-batch-size", type=int)
    build.add_argument("--video-batch-size", type=int)
    build.add_argument("--decode-workers", type=int)
    build.add_argument("--overwrite", action="store_true")
    build.add_argument("--resume", action="store_true")
    build.add_argument("--strict", action="store_true", help="Fail if a CSV video is missing")
    add_model_arguments(build)

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
    add_model_arguments(search)

    encoders = commands.add_parser("encoders", help="List the registered encoders and poolings")
    encoders.add_argument("--json", action="store_true", help="Machine-readable output")
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
        encoder_config, file_build = resolve_configs(args)
        config = BuildConfig(encoder=encoder_config, **build_settings(args, file_build))
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

    if args.command == "encoders":
        listing = {"encoders": available_encoders(), "poolings": available_poolings()}
        if args.json:
            _print(listing)
        else:
            print("encoders:", ", ".join(listing["encoders"]))
            print("poolings:", ", ".join(listing["poolings"]))
        return 0

    database = EmbeddingDatabase(args.db)
    vectors, completed, _, manifest = database.load()
    release_memmaps(vectors, completed)
    # Query a database with the model that built it, unless told otherwise.
    encoder_config, _ = resolve_configs(args, encoder_config_from_manifest(manifest))
    encoder = create_encoder(encoder_config)
    query = encoder.encode_text([args.query])[0]
    _print(database.search_vector(query, args.top_k))
    return 0
