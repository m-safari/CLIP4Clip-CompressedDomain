import csv
import json
import shutil
import unittest
import uuid
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from embedding_db.core import (
    BuildConfig,
    EmbeddingDatabase,
    estimate_database_size,
    npy_bytes,
    release_memmaps,
)
from embedding_db.evaluate import ranking_metrics
from embedding_db.model import mean_pool_frame_embeddings
from embedding_db.pack import (
    PACKINGS,
    bytes_per_vector,
    pack_vectors,
    score_matrix,
    score_query,
    unpack_vectors,
)
from embedding_db.storage import (
    capacity_report,
    compare_packings,
    import_vectors,
    pack_database,
    parse_byte_budget,
)
from embedding_db.video import VideoItem, discover_videos, preprocess_bgr_frame, sampled_frame_indices


TEST_TEMP_ROOT = Path(__file__).resolve().parent


@contextmanager
def workspace_tempdir():
    # tempfile.TemporaryDirectory applies an ACL that is not writable in the
    # managed Windows test sandbox, so use a normal workspace directory.
    path = TEST_TEMP_ROOT / f"work-{uuid.uuid4().hex}"
    path.mkdir()
    try:
        yield str(path)
    finally:
        shutil.rmtree(path)


def write_split_csv(path, rows):
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["video_id", "sentence"])
        writer.writeheader()
        writer.writerows(rows)


def load_released(db):
    """Read a database and hand its mappings straight back.

    EmbeddingDatabase.load returns memmaps on purpose; a test that leaves them
    open would keep the files locked on Windows and break cleanup.
    """
    vectors, completed, items, manifest = db.load()
    values, mask = np.array(vectors), np.array(completed)
    release_memmaps(vectors, completed)
    return values, mask, items, manifest


def unit_vectors(rows, dimension, seed=0):
    values = np.random.default_rng(seed).normal(size=(rows, dimension)).astype(np.float32)
    return values / np.linalg.norm(values, axis=1, keepdims=True)


class VideoTests(unittest.TestCase):
    def test_uniform_sampling_is_capped(self):
        indices = sampled_frame_indices(frame_count=300, fps=30.0, frame_rate=1.0, max_frames=5)
        np.testing.assert_array_equal(indices, np.array([0, 60, 120, 180, 270]))

    def test_preprocessing_shape_and_dtype(self):
        bgr = np.zeros((100, 200, 3), dtype=np.uint8)
        result = preprocess_bgr_frame(bgr)
        self.assertEqual(result.shape, (3, 224, 224))
        self.assertEqual(result.dtype, np.float32)
        self.assertTrue(np.isfinite(result).all())

    def test_csv_discovery_deduplicates_and_reports_missing(self):
        with workspace_tempdir() as folder:
            root = Path(folder)
            (root / "video1.mp4").touch()
            (root / "video2.webm").touch()
            csv_path = root / "split.csv"
            write_split_csv(csv_path, [
                {"video_id": "video2", "sentence": "a"},
                {"video_id": "video2", "sentence": "b"},
                {"video_id": "missing", "sentence": "c"},
                {"video_id": "video1", "sentence": "d"},
            ])
            items, missing = discover_videos(root, csv_path)
            self.assertEqual([item.video_id for item in items], ["video2", "video1"])
            self.assertEqual(missing, ["missing"])

    def test_captions_are_carried_off_the_split_csv(self):
        with workspace_tempdir() as folder:
            root = Path(folder)
            (root / "video1.mp4").touch()
            csv_path = root / "split.csv"
            write_split_csv(csv_path, [{"video_id": "video1", "sentence": "a dog runs"}])
            items, _ = discover_videos(root, csv_path)
            self.assertEqual(items[0].caption, "a dog runs")

    def test_two_csv_ids_resolving_to_one_file_yield_one_row(self):
        # Regression: distinct ids that resolve to the same file used to produce
        # two rows sharing a path, and one of them could never be completed.
        with workspace_tempdir() as folder:
            root = Path(folder)
            (root / "video1.mp4").touch()
            csv_path = root / "split.csv"
            write_split_csv(csv_path, [
                {"video_id": "video1", "sentence": "bare id"},
                {"video_id": "video1.mp4", "sentence": "id with suffix"},
            ])
            items, missing = discover_videos(root, csv_path)
            self.assertEqual(missing, [])
            self.assertEqual(len(items), 1)
            self.assertEqual(items[0].caption, "bare id")
            self.assertEqual(len({str(item.path) for item in items}), len(items))


class EmbeddingTests(unittest.TestCase):
    def test_mean_pool_matches_clip4clip_contract(self):
        frames = np.array([[3.0, 0.0], [0.0, 4.0]], dtype=np.float32)
        result = mean_pool_frame_embeddings(frames)
        expected = np.array([1.0, 1.0], dtype=np.float32) / np.sqrt(2.0)
        np.testing.assert_allclose(result, expected, rtol=1e-6)

    def test_exact_npy_size_estimate(self):
        with workspace_tempdir() as folder:
            path = Path(folder) / "vectors.npy"
            array = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=(2000, 512))
            array.flush()
            del array
            estimate = estimate_database_size(2000)
            self.assertEqual(path.stat().st_size, estimate["estimated_npy_bytes"])

    def test_database_round_trip_and_search(self):
        with workspace_tempdir() as folder:
            root = Path(folder)
            videos = root / "videos"
            videos.mkdir()
            paths = []
            for name in ("a", "b"):
                path = videos / f"{name}.mp4"
                path.touch()
                paths.append(path.resolve())
            items = [VideoItem("a", paths[0]), VideoItem("b", paths[1])]
            db = EmbeddingDatabase(root / "db")
            vectors, completed = db.initialize(items, BuildConfig(), 2)
            vectors[:] = [[1.0, 0.0], [0.0, 1.0]]
            completed[:] = [True, True]
            vectors.flush()
            completed.flush()
            hits = db.search_vector(np.array([0.9, 0.1], dtype=np.float32), top_k=1)
            self.assertEqual(hits[0]["video_id"], "a")

            resumed_vectors, resumed_completed = db.initialize(items, BuildConfig(), 2, resume=True)
            self.assertEqual(resumed_vectors.shape, (2, 2))
            self.assertTrue(resumed_completed.all())
            for mapping in (vectors, completed, resumed_vectors, resumed_completed):
                mapping._mmap.close()

    def test_database_refuses_unrelated_nonempty_directory(self):
        with workspace_tempdir() as folder:
            root = Path(folder)
            (root / "keep.txt").write_text("user data", encoding="utf-8")
            db = EmbeddingDatabase(root)
            with self.assertRaises(FileExistsError):
                db.initialize([], BuildConfig(), 512)
            self.assertEqual((root / "keep.txt").read_text(encoding="utf-8"), "user data")


class ByteBudgetTests(unittest.TestCase):
    def test_units_are_decimal_or_binary(self):
        self.assertEqual(parse_byte_budget("2GB"), 2_000_000_000)
        self.assertEqual(parse_byte_budget("2GiB"), 2 * 2**30)
        self.assertEqual(parse_byte_budget("512mib"), 512 * 2**20)
        self.assertEqual(parse_byte_budget("4096"), 4096)
        self.assertEqual(parse_byte_budget(4096), 4096)

    def test_rejects_nonsense(self):
        for value in ("", "2 parsecs", "0"):
            with self.assertRaises(ValueError):
                parse_byte_budget(value)

    def test_capacity_is_the_inverse_of_the_per_item_cost(self):
        budget = 2 * 2**30
        report = capacity_report(budget, dimension=512)
        by_packing = {row["packing"]: row for row in report["packings"]}
        self.assertEqual(by_packing["float32"]["max_items"], 1_048_576)
        self.assertEqual(by_packing["float16"]["max_items"], 2_097_152)
        self.assertEqual(by_packing["binary"]["max_items"], 33_554_432)
        for row in report["packings"]:
            # Capacity must fit the budget, and one more item must not.
            self.assertLessEqual(row["max_items"] * row["bytes_per_item"], budget)
            self.assertGreater((row["max_items"] + 1) * row["bytes_per_item"], budget)

    def test_metadata_overhead_shrinks_capacity(self):
        plain = capacity_report(2**30, 512)["packings"][0]["max_items"]
        charged = capacity_report(2**30, 512, metadata_bytes_per_item=128.0)["packings"][0]["max_items"]
        self.assertLess(charged, plain)


class NpySizeTests(unittest.TestCase):
    def test_predicted_size_matches_what_numpy_writes(self):
        with workspace_tempdir() as folder:
            for shape, dtype in (((1000, 512), "float32"), ((2000, 512), "float16"), ((7, 3), "int8")):
                path = Path(folder) / f"{shape[0]}-{dtype}.npy"
                np.save(path, np.zeros(shape, dtype=dtype))
                self.assertEqual(path.stat().st_size, npy_bytes(shape[0], shape[1], dtype))


class PackingTests(unittest.TestCase):
    def test_bytes_per_vector_matches_the_packed_array(self):
        vectors = unit_vectors(64, 32)
        for packing in PACKINGS:
            packed = pack_vectors(vectors, packing)
            stored = sum(array.nbytes for array in packed.values())
            self.assertEqual(stored, len(vectors) * bytes_per_vector(vectors.shape[1], packing))

    def test_lossless_and_lossy_round_trips(self):
        vectors = unit_vectors(64, 32)
        np.testing.assert_array_equal(unpack_vectors(pack_vectors(vectors, "float32"), "float32", 32), vectors)
        restored = unpack_vectors(pack_vectors(vectors, "int8"), "int8", 32)
        cosine = (restored * vectors).sum(axis=1) / np.linalg.norm(restored, axis=1)
        self.assertGreater(cosine.min(), 0.99)
        signs = unpack_vectors(pack_vectors(vectors, "binary"), "binary", 32)
        np.testing.assert_array_equal(np.sign(signs), np.where(vectors > 0, 1.0, -1.0))

    def test_score_matrix_agrees_with_single_queries(self):
        vectors = unit_vectors(64, 32)
        for packing in PACKINGS:
            packed = pack_vectors(vectors, packing)
            matrix = score_matrix(packed, packing, vectors[:4], 32)
            for row in range(4):
                np.testing.assert_allclose(
                    matrix[row], score_query(packed, packing, vectors[row], 32), rtol=1e-5, atol=1e-3
                )

    def test_unknown_packing_is_rejected(self):
        with self.assertRaises(ValueError):
            bytes_per_vector(512, "float8")


class MetricsTests(unittest.TestCase):
    def test_perfect_ranking(self):
        metrics = ranking_metrics(np.eye(10, dtype=np.float32))
        self.assertEqual(metrics["R@1"], 100.0)
        self.assertEqual(metrics["MedianR"], 1.0)
        self.assertEqual(metrics["MeanR"], 1.0)

    def test_known_offset_ranking(self):
        # Every query puts its correct row second, so R@1 is 0 and R@5 is 100.
        scores = np.zeros((4, 4), dtype=np.float32)
        for i in range(4):
            scores[i, i] = 0.5
            scores[i, (i + 1) % 4] = 0.9
        metrics = ranking_metrics(scores)
        self.assertEqual(metrics["R@1"], 0.0)
        self.assertEqual(metrics["R@5"], 100.0)
        self.assertEqual(metrics["MedianR"], 2.0)

    def test_non_square_scores_are_rejected(self):
        with self.assertRaises(ValueError):
            ranking_metrics(np.zeros((3, 4), dtype=np.float32))


class ImportAndPackTests(unittest.TestCase):
    def _write_source(self, root, rows=8, dimension=16):
        values = unit_vectors(rows, dimension, seed=1)
        vectors_path = root / "vectors.npy"
        np.save(vectors_path, values)
        csv_path = root / "split.csv"
        write_split_csv(
            csv_path,
            [{"video_id": f"video{index}", "sentence": f"caption {index}"} for index in range(rows)],
        )
        return values, vectors_path, csv_path

    def test_import_produces_a_loadable_database(self):
        with workspace_tempdir() as folder:
            root = Path(folder)
            values, vectors_path, csv_path = self._write_source(root)
            report = import_vectors(vectors_path, root / "db", csv_path)
            self.assertEqual(report["count"], len(values))
            self.assertTrue(report["source_normalized"])

            db = EmbeddingDatabase(root / "db")
            stored, completed, items, manifest = load_released(db)
            self.assertTrue(np.asarray(completed).all())
            self.assertEqual(manifest["packing"], "float32")
            self.assertEqual(items[3].caption, "caption 3")
            np.testing.assert_allclose(np.asarray(stored), values, rtol=1e-6)
            self.assertEqual(db.search_vector(values[5], top_k=1)[0]["video_id"], "video5")

    def test_import_rejects_a_csv_of_the_wrong_length(self):
        with workspace_tempdir() as folder:
            root = Path(folder)
            _, vectors_path, _ = self._write_source(root, rows=8)
            short = root / "short.csv"
            write_split_csv(short, [{"video_id": "video0", "sentence": "only one"}])
            with self.assertRaises(ValueError):
                import_vectors(vectors_path, root / "db", short)

    def test_import_renormalizes_on_request(self):
        with workspace_tempdir() as folder:
            root = Path(folder)
            values = unit_vectors(6, 16, seed=3) * 7.5
            vectors_path = root / "scaled.npy"
            np.save(vectors_path, values)
            report = import_vectors(vectors_path, root / "db", normalize=True)
            self.assertFalse(report["source_normalized"])
            stored, _, _, _ = load_released(EmbeddingDatabase(root / "db"))
            np.testing.assert_allclose(np.linalg.norm(np.asarray(stored), axis=1), 1.0, rtol=1e-5)

    def test_packed_databases_shrink_and_still_search(self):
        # Realistic width: at toy dimensions the fixed NPY headers dominate and
        # int8 (two files) can out-weigh float16 (one file).
        with workspace_tempdir() as folder:
            root = Path(folder)
            values, vectors_path, csv_path = self._write_source(root, rows=32, dimension=512)
            import_vectors(vectors_path, root / "db", csv_path)
            sizes = {}
            for packing in PACKINGS:
                report = pack_database(root / "db", root / f"db-{packing}", packing)
                self.assertEqual(report["bytes_per_item"], bytes_per_vector(512, packing))
                sizes[packing] = report["vector_bytes"]
                hits = EmbeddingDatabase(root / f"db-{packing}").search_vector(values[5], top_k=1)
                self.assertEqual(hits[0]["video_id"], "video5")
            self.assertGreater(sizes["float32"], sizes["float16"])
            self.assertGreater(sizes["float16"], sizes["int8"])
            self.assertGreater(sizes["int8"], sizes["binary"])

    def test_packing_preserves_captions(self):
        with workspace_tempdir() as folder:
            root = Path(folder)
            _, vectors_path, csv_path = self._write_source(root)
            import_vectors(vectors_path, root / "db", csv_path)
            pack_database(root / "db", root / "db-int8", "int8")
            _, _, items, _ = load_released(EmbeddingDatabase(root / "db-int8"))
            self.assertEqual(items[2].caption, "caption 2")

    def test_packing_refuses_to_overwrite_its_own_source(self):
        with workspace_tempdir() as folder:
            root = Path(folder)
            _, vectors_path, csv_path = self._write_source(root)
            import_vectors(vectors_path, root / "db", csv_path)
            with self.assertRaises(ValueError):
                pack_database(root / "db", root / "db", "int8")

    def test_compare_ranks_packings_by_fidelity(self):
        with workspace_tempdir() as folder:
            root = Path(folder)
            _, vectors_path, csv_path = self._write_source(root, rows=40, dimension=64)
            import_vectors(vectors_path, root / "db", csv_path)
            report = compare_packings(root / "db", top_k=5)
            agreement = {row["packing"]: row["neighbour_agreement_at_5"] for row in report["packings"]}
            self.assertEqual(agreement["float32"], 1.0)
            self.assertGreaterEqual(agreement["float16"], agreement["int8"])
            self.assertGreater(agreement["int8"], agreement["binary"])
            self.assertIsNone({row["packing"]: row for row in report["packings"]}["binary"]["mean_cosine_to_float32"])

    def test_compare_output_is_strict_json(self):
        with workspace_tempdir() as folder:
            root = Path(folder)
            _, vectors_path, csv_path = self._write_source(root, rows=20, dimension=32)
            import_vectors(vectors_path, root / "db", csv_path)
            encoded = json.dumps(compare_packings(root / "db", top_k=5), allow_nan=False)
            self.assertIn("binary", encoded)


if __name__ == "__main__":
    unittest.main()
