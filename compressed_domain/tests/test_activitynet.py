"""ActivityNet preparation tests run without ML dependencies or downloads.

Loader checks use synthetic decoded tensors when torch/torchvision/pandas are
installed; CoViAR is mocked. Run: python3 -m unittest discover -s tests -p test_activitynet.py -v
"""
import csv
import importlib.util
import json
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dataloaders import prepare_activitynet_dataset as prep
from dataloaders.activitynet_annotations import index_videos, load_paragraphs

HAVE_ML = all(importlib.util.find_spec(m) is not None for m in ("torch", "torchvision", "pandas"))


class PreparationTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.annotations = self.root / "annotations"
        self.annotations.mkdir()
        self.train = {f"v_train{i}": {"sentences": ["First event.", "Second event."],
                                       "timestamps": [[0, 1], [1, 2]], "duration": 2} for i in range(5)}
        self.test = {"v_test": {"sentences": ["Held-out paragraph."], "duration": 3}}
        for filename, data in (("train.json", self.train), ("val_1.json", self.test)):
            (self.annotations / filename).write_text(json.dumps(data))
        self.videos = self.root / "videos"
        self.videos.mkdir()
        for video_id in (*self.train, *self.test):
            (self.videos / (video_id + ".mp4")).write_bytes(b"video")

    def test_paragraphs_and_validation(self):
        paragraphs = load_paragraphs(self.annotations / "train.json")
        self.assertEqual(paragraphs["v_train0"], "First event. Second event.")
        path = self.annotations / "bad.json"
        for sentences in ([], "caption", [None], [" "]):
            path.write_text(json.dumps({"v_a": {"sentences": sentences}}))
            with self.assertRaises(ValueError):
                load_paragraphs(path)

    def test_nested_names_and_ambiguous_paths(self):
        nested = self.videos / "nested"
        nested.mkdir()
        (nested / "bare_id.mkv").touch()
        self.assertEqual(index_videos(self.videos)["v_bare_id"], nested / "bare_id.mkv")
        (nested / "train0.mp4").touch()
        with self.assertRaisesRegex(ValueError, "Ambiguous"):
            index_videos(self.videos)

    def test_disjoint_reproducible_splits_and_metadata(self):
        output = self.root / "splits"
        splits = prep.write_splits(self.annotations, self.videos, output, num_val=2)
        self.assertEqual({s: len(ids) for s, ids in splits.items()}, {"train": 3, "val": 2, "test": 1})
        self.assertFalse(set(splits["train"]) & set(splits["val"]))
        self.assertFalse(set(splits["test"]) & (set(splits["train"]) | set(splits["val"])))
        self.assertEqual(splits, prep.write_splits(self.annotations, self.videos, output, num_val=2))
        merged = load_paragraphs(output / "ActivityNet_data.json")
        for split, ids in splits.items():
            with open(output / f"ActivityNet_{split}.csv") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual([r["video_id"] for r in rows], ids)
            self.assertEqual([r["sentence"] for r in rows], [merged[v] for v in ids])

    def test_missing_videos_are_reported_and_omitted(self):
        (self.videos / "v_train0.mp4").unlink()
        out = self.root / "splits"
        splits = prep.write_splits(self.annotations, self.videos, out, num_val=1)
        self.assertNotIn("v_train0", splits["train"] + splits["val"])
        report = json.loads((out / "preparation_report.json").read_text())
        self.assertEqual(report["missing_train"], ["v_train0"])
        with self.assertRaises(ValueError):
            prep.write_splits(self.annotations, self.videos, out, num_val=4)

    def test_overlapping_annotations_rejected(self):
        (self.annotations / "val_1.json").write_text(json.dumps(self.train))
        with self.assertRaisesRegex(ValueError, "overlap"):
            prep.write_splits(self.annotations, self.videos, self.root / "splits", num_val=1)

    def test_fixed_gop_encoding_atomic_and_resumable(self):
        destination = self.root / "encoded" / "v_train0.mp4"
        def encode(command, check):
            self.assertTrue(check)
            self.assertEqual(command[command.index("-g") + 1], "12")
            self.assertEqual(command[command.index("-bf") + 1], "0")
            self.assertEqual(command[command.index("-f") + 1], "m4v")
            Path(command[-1]).write_bytes(b"encoded")
        with mock.patch.object(prep.subprocess, "run", side_effect=encode) as run:
            prep.reencode_video(self.videos / "v_train0.mp4", destination)
            prep.reencode_video(self.videos / "v_train0.mp4", destination)
            self.assertEqual(run.call_count, 1)
        self.assertFalse(destination.with_suffix(".partial").exists())
        destination.unlink()
        with mock.patch.object(prep.subprocess, "run", side_effect=subprocess.CalledProcessError(1, "ffmpeg")):
            with self.assertRaises(subprocess.CalledProcessError):
                prep.reencode_video(self.videos / "v_train0.mp4", destination)
        self.assertFalse(destination.exists())

    def test_local_preparation_pipeline(self):
        def encode(source, destination):
            Path(destination).write_bytes(b"encoded")
        with mock.patch.object(prep, "reencode_video", side_effect=encode), \
                mock.patch.object(prep, "download_video") as download:
            videos, splits = prep.prepare_activitynet_dataset(
                self.root / "prepared", self.videos, self.annotations, num_val=1)
        download.assert_not_called()
        self.assertEqual(len(list(videos.glob("*.mp4"))), 6)
        self.assertTrue((splits / "ActivityNet_data.json").exists())

    @unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg required")
    def test_real_encoder_has_fixed_gops_and_no_b_frames(self):
        source = self.root / "source.mp4"
        subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-f", "lavfi",
                        "-i", "testsrc2=size=64x48:rate=12", "-t", "3", "-c:v", "mpeg4",
                        str(source)], check=True)
        destination = prep.reencode_video(source, self.root / "encoded.mp4")
        result = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                                 "-show_frames", "-show_streams", "-of", "json", str(destination)],
                                check=True, capture_output=True, text=True)
        info = json.loads(result.stdout)
        self.assertEqual(info["streams"][0]["codec_name"], "mpeg4")
        frames = info["frames"]
        self.assertEqual(len(frames), 36)
        self.assertEqual([i for i, f in enumerate(frames) if f["key_frame"]], [0, 12, 24])
        self.assertNotIn("B", [f["pict_type"] for f in frames])

    def test_annotation_zip_download(self):
        from zipfile import ZipFile
        out = self.root / "downloaded"
        def download(url, path):
            with ZipFile(path, "w") as archive:
                archive.writestr("captions/train.json", json.dumps(self.train))
                archive.writestr("captions/val_1.json", json.dumps(self.test))
                archive.writestr("../unexpected", "ignored")
        with mock.patch.object(prep, "urlretrieve", side_effect=download) as fetch:
            prep.download_annotations(out)
            prep.download_annotations(out)
        self.assertEqual(fetch.call_count, 1)
        self.assertFalse((self.root / "unexpected").exists())
        self.assertEqual(load_paragraphs(out / "train.json")["v_train0"], "First event. Second event.")


@unittest.skipUnless(HAVE_ML, "torch, torchvision and pandas required for tensor loader checks")
class LoaderTest(unittest.TestCase):
    setUp = PreparationTest.setUp

    def test_batch_and_failed_decode(self):
        import torch
        with mock.patch.dict(sys.modules, {"coviar": types.ModuleType("coviar")}):
            from dataloaders.dataloader_activitynet_compressed import ActivityNet_Compressed_DataLoader
            from dataloaders.data_dataloaders import trim_padding_collate
            tokenizer = mock.Mock()
            tokenizer.tokenize.side_effect = lambda text: text.split()
            tokenizer.convert_tokens_to_ids.side_effect = lambda words: list(range(1, len(words) + 1))
            split = self.root / "train.csv"
            split.write_text("video_id\nv_train0\nv_train1\n")
            dataset = ActivityNet_Compressed_DataLoader(
                split, self.annotations, self.videos, tokenizer, max_words=6,
                max_gops=2, residual_sampling_rate=3, mv_sampling_rate=2, image_resolution=8)
            decoded = {"iframe": torch.ones(2, 3, 8, 8), "residuals": torch.ones(3, 3, 8, 8),
                       "mv": torch.ones(1, 2, 8, 8)}
            with mock.patch.object(dataset.extractor, "video_to_compressed_tensors", return_value=decoded):
                item = dataset[0]
            self.assertEqual(len(item), 9)
            self.assertEqual([int(item[i].sum()) for i in (4, 6, 8)], [2, 3, 1])
            self.assertEqual(item[5].shape, (1, 6, 3, 8, 8))
            tokenizer.tokenize.assert_called_with("First event. Second event.")
            with mock.patch.object(dataset.extractor, "video_to_compressed_tensors", side_effect=RuntimeError("decode")):
                failed = dataset[1]
            self.assertEqual([int(failed[i].sum()) for i in (4, 6, 8)], [0, 0, 0])
            batch = trim_padding_collate([item, failed])
            self.assertEqual([batch[i].shape[2] for i in (3, 5, 7)], [2, 3, 1])
            self.assertEqual(dataset.csv.sentence.tolist(), ["First event. Second event."] * 2)
            from dataloaders.data_dataloaders import DATALOADER_DICT
            args = types.SimpleNamespace(
                train_csv=split, val_csv=split, test_csv=split,
                data_path=self.annotations / "train.json", features_path=self.videos,
                max_words=6, max_frames=2, slice_framepos=2, residual_per_gop=3, mv_per_gop=2,
                max_train_samples=2, seed=42, batch_size=2, batch_size_val=1, num_thread_reader=0)
            for subset in ("train", "val", "test"):
                loader, count = DATALOADER_DICT["activitynet"][subset](args, tokenizer)
                self.assertEqual(count, 2)
                with mock.patch.object(loader.dataset.extractor, "video_to_compressed_tensors", return_value=decoded):
                    batch = next(iter(loader))
                self.assertEqual(len(batch), 9)
                self.assertEqual(batch[0].shape[0], 2 if subset == "train" else 1)



if __name__ == "__main__":
    unittest.main()
