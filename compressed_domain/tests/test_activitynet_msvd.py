"""
Tests for MSVD and ActivityNet retrieval: their compressed-domain datasets,
their prepare modules, multi-sentence metrics and evaluation, and the
upstream-compatible command line.

    cd compressed_domain && python -m unittest discover -s tests -v

CoViAR and mlflow are stubbed; nothing is downloaded or decoded.
"""
import io
import json
import logging
import os
import pickle
import shutil
import sys
import tarfile
import tempfile
import types
import unittest
import zipfile
from pathlib import Path
from unittest import mock

from _helpers import CONTEXT_LENGTH, IMAGE_RESOLUTION, VOCAB_SIZE, build_model

import numpy as np
import torch

from metrics import compute_metrics, compute_metrics_multi, video_to_text_similarity

_stubbed = [name for name in ("mlflow", "mlflow.pytorch", "coviar") if name not in sys.modules]
for _name in _stubbed:
    sys.modules[_name] = types.ModuleType(_name)
if "mlflow" in _stubbed:
    sys.modules["mlflow"].__path__ = []
try:
    import main_task_retrieval as main
    from dataloaders.data_dataloaders import DATALOADER_DICT, trim_padding_collate
    from dataloaders.dataloader_activitynet_compressed import ActivityNet_Compressed_DataLoader
    from dataloaders.dataloader_msrvtt_compressed import MSRVTT_Compressed_DataLoader
    from dataloaders.dataloader_msvd_compressed import MSVD_Compressed_DataLoader
finally:
    for _name in _stubbed:
        del sys.modules[_name]

from dataloaders import prepare_activitynet_dataset as prep_anet  # noqa: E402
from dataloaders import prepare_msvd_dataset as prep_msvd  # noqa: E402

MAX_GOPS = 2


class FakeTokenizer:
    """Whitespace tokens with stable ids below the end-of-text id, which is VOCAB_SIZE - 1."""

    def tokenize(self, text):
        return text.lower().split()

    def convert_tokens_to_ids(self, tokens):
        special = {"<|startoftext|>": VOCAB_SIZE - 2, "<|endoftext|>": VOCAB_SIZE - 1}
        return [special.get(t, 1 + sum(map(ord, t)) % (VOCAB_SIZE - 3)) for t in tokens]


def fake_video(self, choice_video_ids):
    """Stands in for _get_compressed_video: frames whose value encodes the video, no decoding."""
    res = self.compressedVideoExtractor.image_resolution
    out = []
    for length, channels in ((self.max_iframe_length, 3), (self.max_residual_length, 3), (self.max_mv_length, 2)):
        frames = torch.zeros(len(choice_video_ids), length, channels, res, res)
        for i, video_id in enumerate(choice_video_ids):
            frames[i] = (sum(map(ord, str(video_id))) % 7) / 7.0
        out.append(frames)
        out.append(torch.ones(len(choice_video_ids), length, dtype=torch.long))
    return tuple(out)


def touch(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")


class _TmpDirTest(unittest.TestCase):

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp)


# ---------------------------------------------------------------------- #
# Datasets
# ---------------------------------------------------------------------- #

def write_msvd(root, videos_on_disk):
    data, videos = root / "msvd_data", root / "mpeg4_videos"
    data.mkdir(parents=True)
    splits = {"train": ["a", "b"], "val": ["c"], "test": ["d", "e", "f"]}
    for subset, ids in splits.items():
        (data / "{}_list.txt".format(subset)).write_text("".join(i + "\n" for i in ids))
    captions = {v: [["a", "video", "of", v, str(k)] for k in range(n)]
                for v, n in zip("abcdef", (2, 1, 2, 3, 1, 2))}
    with open(data / "raw-captions.pkl", "wb") as f:
        pickle.dump(captions, f)
    for v in videos_on_disk:
        touch(videos / "sub" / (v + ".mp4"))
    touch(videos / "a.avi")          # an original left next to the re-encoded files is not a video
    return data, videos


def write_activitynet(root, videos_on_disk):
    data, videos = root / "activitynet_data", root / "mpeg4_videos"
    data.mkdir(parents=True)
    captions = {
        "train": {"v_t1": ["A man runs.", " He stops. "], "v_t2": ["A dog."]},
        "val": {"v_x1": ["One."], "v_x2": ["Two.", "Three."], "v_x3": ["Four."]},
    }
    for subset, caption_file in (("train", "train.json"), ("val", "val_1.json")):
        items = {v: {"duration": 10.0, "timestamps": [[0, 1]] * len(s), "sentences": s}
                 for v, s in captions[subset].items()}
        (data / caption_file).write_text(json.dumps(items))
    (data / "train_ids.json").write_text(json.dumps(["v_t2", "v_t1", "v_t1"]))
    (data / "val_ids.json").write_text(json.dumps(["v_x1", "v_x2", "v_x3", "v_nocaption"]))
    for name in videos_on_disk:
        touch(videos / (name + ".mp4"))
    return data, videos


def dataset_kwargs():
    return dict(tokenizer=FakeTokenizer(), max_words=CONTEXT_LENGTH, max_gops=MAX_GOPS,
                image_resolution=IMAGE_RESOLUTION)


@mock.patch.object(MSRVTT_Compressed_DataLoader, "_get_compressed_video", fake_video)
class MSVDDatasetTest(_TmpDirTest):

    def setUp(self):
        super().setUp()
        self.data, self.videos = write_msvd(self.tmp, videos_on_disk="abcdf")   # e is missing

    def dataset(self, subset):
        return MSVD_Compressed_DataLoader(subset, str(self.data), str(self.videos), **dataset_kwargs())

    def test_train_has_one_item_per_caption(self):
        ds = self.dataset("train")
        self.assertFalse(ds.multi_sentence_per_video)
        self.assertEqual(len(ds), 3)                                 # a: 2 captions, b: 1
        item = ds[2]
        self.assertEqual(len(item), 9)
        self.assertEqual(item[0].shape, (1, CONTEXT_LENGTH))
        self.assertEqual(item[3].shape, (1, MAX_GOPS, 3, IMAGE_RESOLUTION, IMAGE_RESOLUTION))

    def test_eval_has_one_item_per_video_and_every_caption_as_a_query(self):
        ds = self.dataset("test")
        self.assertTrue(ds.multi_sentence_per_video)
        self.assertEqual(list(ds.video_ids), ["d", "f"])
        self.assertEqual(ds.missing_video_ids, ["e"])
        self.assertEqual(len(ds), 2)
        self.assertEqual(list(ds.caption_video_index), [0, 0, 0, 1, 1])
        self.assertEqual(list(ds.captions[:2]), ["a video of d 0", "a video of d 1"])

        ids, mask, segment = ds.caption_tokens()
        self.assertEqual(ids.shape, (5, CONTEXT_LENGTH))
        self.assertEqual(mask.shape, segment.shape)
        self.assertEqual(int(mask[0].sum()), 2 + 5)                  # start, 5 words, end
        self.assertEqual(int(ids[0, 6]), VOCAB_SIZE - 1)

    def test_videos_are_found_recursively_and_only_as_mp4(self):
        ds = self.dataset("train")
        self.assertTrue(ds._video_path("a").endswith(os.path.join("sub", "a.mp4")))

    def test_unknown_subset_rejected(self):
        with self.assertRaises(ValueError):
            self.dataset("dev")

    def test_registered(self):
        self.assertEqual(set(DATALOADER_DICT["msvd"]), {"train", "val", "test"})


@mock.patch.object(MSRVTT_Compressed_DataLoader, "_get_compressed_video", fake_video)
class ActivityNetDatasetTest(_TmpDirTest):

    def setUp(self):
        super().setUp()
        # v_x2 is stored without its "v_" prefix; v_x3 is missing.
        self.data, self.videos = write_activitynet(self.tmp, videos_on_disk=["v_t1", "t2", "v_x1", "x2"])

    def dataset(self, subset):
        return ActivityNet_Compressed_DataLoader(subset, str(self.data), str(self.videos), **dataset_kwargs())

    def test_one_paragraph_per_video(self):
        ds = self.dataset("val")
        self.assertFalse(ds.multi_sentence_per_video)
        self.assertEqual(list(ds.video_ids), ["v_x1", "v_x2"])
        self.assertEqual(list(ds.captions), ["One.", "Two. Three."])
        self.assertEqual(ds.missing_video_ids, ["v_x3"])            # no caption -> not listed at all
        self.assertTrue(ds._video_path("v_x2").endswith("x2.mp4"))
        self.assertEqual(len(ds[1]), 9)

    def test_ids_file_order_without_duplicates(self):
        ds = self.dataset("train")
        self.assertEqual(list(ds.video_ids), ["v_t2", "v_t1"])
        self.assertEqual(list(ds.captions), ["A dog.", "A man runs. He stops."])

    def test_no_test_split(self):
        self.assertIsNone(DATALOADER_DICT["activity"]["test"])
        with self.assertRaises(ValueError):
            self.dataset("test")


# ---------------------------------------------------------------------- #
# Metrics
# ---------------------------------------------------------------------- #

class MultiSentenceMetricsTest(unittest.TestCase):

    def test_reduces_to_compute_metrics_when_square(self):
        x = np.random.default_rng(0).normal(size=(6, 6))
        multi, square = compute_metrics_multi(x, np.arange(6)), compute_metrics(x)
        for key in ("R1", "R5", "R10", "MR", "MeanR", "cols"):
            self.assertEqual(multi[key], square[key])

    def test_known_ranks(self):
        # 4 captions of 2 videos; gt: 0, 0, 1, 1
        x = np.array([[0.9, 0.1], [0.2, 0.8], [0.3, 0.7], [0.5, 0.5]])
        m = compute_metrics_multi(x, [0, 0, 1, 1])
        self.assertEqual(m["cols"], [0, 1, 0, 1])                   # a tie ranks against the model
        self.assertEqual(m["R1"], 50.0)

    def test_bad_input_rejected(self):
        with self.assertRaises(ValueError):
            compute_metrics_multi(np.zeros((3, 2)), [0, 1])
        with self.assertRaises(ValueError):
            compute_metrics_multi(np.zeros((2, 2)), [0, 2])

    def test_video_to_text_takes_each_videos_best_caption(self):
        x = np.array([[0.9, 0.1], [0.2, 0.8], [0.3, 0.7]])
        v2t = video_to_text_similarity(x, [0, 0, 1])
        np.testing.assert_allclose(v2t, [[0.9, 0.3], [0.8, 0.7]])
        self.assertEqual(compute_metrics(v2t)["cols"], [0, 1])


# ---------------------------------------------------------------------- #
# Evaluation
# ---------------------------------------------------------------------- #

class _FakeRetrievalDataset(torch.utils.data.Dataset):
    """Videos in the dataloader's item layout; video `failed` has no frames at all."""

    def __init__(self, n_videos, captions_per_video, failed=None, multi_sentence=True, seed=0):
        g = torch.Generator().manual_seed(seed)
        self.multi_sentence_per_video = multi_sentence
        self.frames = [torch.randn(1, 2, c, IMAGE_RESOLUTION, IMAGE_RESOLUTION, generator=g) for c in (3, 3, 2)
                       for _ in range(n_videos)]
        self.n_videos, self.failed = n_videos, failed
        self.caption_video_index = np.repeat(np.arange(n_videos), captions_per_video)
        n_captions = len(self.caption_video_index) if multi_sentence else n_videos
        self.ids = torch.randint(1, VOCAB_SIZE - 3, (n_captions, CONTEXT_LENGTH), generator=g)
        self.ids[:, 5] = VOCAB_SIZE - 1
        self.mask = torch.zeros_like(self.ids)
        self.mask[:, :6] = 1
        self.video_ids = np.asarray(["vid{}".format(i) for i in range(n_videos)])
        self.captions = np.asarray(["caption {}".format(i) for i in range(n_captions)])

    def caption_tokens(self):
        return self.ids.numpy(), self.mask.numpy(), np.zeros_like(self.mask.numpy())

    def __len__(self):
        return self.n_videos

    def __getitem__(self, idx):
        row = 0 if self.multi_sentence_per_video else idx
        item = [self.ids[row:row + 1], self.mask[row:row + 1], torch.zeros_like(self.mask[row:row + 1])]
        for m in range(3):
            frames = self.frames[m * self.n_videos + idx]
            mask = torch.ones(1, 2, dtype=torch.long)
            if idx == self.failed:
                frames, mask = torch.zeros_like(frames), torch.zeros_like(mask)
            item += [frames, mask]
        return tuple(item)


class MultiSentenceEvalTest(unittest.TestCase):

    def setUp(self):
        self.model = build_model()
        self.model.eval()
        self.args = types.SimpleNamespace(precision="amp", batch_size_val=4)
        patches = [mock.patch.object(main, "mlflow", mock.Mock(), create=True),
                   mock.patch.object(main, "logger", logging.getLogger("test"), create=True)]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)

    def evaluate(self, dataset):
        loader = torch.utils.data.DataLoader(dataset, batch_size=3, collate_fn=trim_padding_collate)
        return main.eval_epoch(self.args, self.model, loader, torch.device("cpu"))

    def expected(self, dataset, videos, captions, gt):
        with torch.no_grad():
            text = self.model.get_sequence_output(dataset.ids[captions], None, dataset.mask[captions])
            visual, video_mask = self.model.get_visual_output(
                *[torch.cat([dataset[i][3 + k] for i in videos]) for k in range(6)])
            sim = self.model.get_similarity_logits(text, visual, None, video_mask).numpy()
        return compute_metrics_multi(sim, gt)["R1"]

    def test_every_caption_is_a_query_against_its_video(self):
        dataset = _FakeRetrievalDataset(n_videos=5, captions_per_video=3)
        r1, emb = self.evaluate(dataset)
        self.assertEqual(emb["text_embeddings"].shape[0], 15)
        self.assertEqual(emb["branch_embeddings"].shape[0], 5)
        self.assertEqual(list(emb["caption_video_index"]), list(dataset.caption_video_index))
        self.assertEqual(len(emb["captions"]), 15)
        self.assertAlmostEqual(r1, self.expected(dataset, range(5), np.arange(15), dataset.caption_video_index))

    def test_failed_video_drops_its_captions(self):
        dataset = _FakeRetrievalDataset(n_videos=5, captions_per_video=2, failed=1)
        r1, emb = self.evaluate(dataset)
        self.assertEqual(list(emb["failed_video_ids"]), ["vid1"])
        self.assertEqual(list(emb["video_ids"]), ["vid0", "vid2", "vid3", "vid4"])
        self.assertEqual(list(emb["caption_video_index"]), [0, 0, 1, 1, 2, 2, 3, 3])
        self.assertEqual(list(emb["captions"]), ["caption {}".format(i) for i in (0, 1, 4, 5, 6, 7, 8, 9)])
        self.assertAlmostEqual(r1, self.expected(dataset, [0, 2, 3, 4], np.array([0, 1, 4, 5, 6, 7, 8, 9]),
                                                 [0, 0, 1, 1, 2, 2, 3, 3]))

    def test_one_caption_per_video_path_is_unchanged(self):
        dataset = _FakeRetrievalDataset(n_videos=4, captions_per_video=1, multi_sentence=False)
        _, emb = self.evaluate(dataset)
        self.assertNotIn("caption_video_index", emb)
        self.assertEqual(emb["text_embeddings"].shape[0], 4)
        self.assertEqual(list(emb["captions"]), ["caption {}".format(i) for i in range(4)])


# ---------------------------------------------------------------------- #
# Command line
# ---------------------------------------------------------------------- #

# The ActivityNet command of the CLIP4Clip README, after main_task_retrieval.py.
UPSTREAM_ACTIVITY_COMMAND = (
    "--do_train --num_thread_reader=2 --epochs=5 --batch_size=128 --n_display=50 "
    "--data_path data --features_path data/Activity_Videos --output_dir ckpts/ckpt_activity_retrieval_looseType "
    "--lr 1e-4 --max_words 64 --max_frames 64 --batch_size_val 16 --datatype activity --feature_framerate 1 "
    "--coef_lr 1e-3 --freeze_layer_num 0 --slice_framepos 2 --loose_type --linear_patch 2d --sim_header meanP "
    "--pretrained_clip_name ViT-B/32"
).split()


class CommandLineTest(unittest.TestCase):

    def parse(self, argv):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("WORLD_SIZE", None)
            return main.get_args(argv=argv)

    def test_upstream_activitynet_command_parses(self):
        args = self.parse(UPSTREAM_ACTIVITY_COMMAND + ["--local_rank=0"])
        self.assertEqual((args.datatype, args.max_words, args.max_frames), ("activity", 64, 64))
        self.assertTrue(args.loose_type)

    def test_unimplemented_model_choices_rejected(self):
        for flags in (["--sim_header", "seqTransf"], ["--linear_patch", "3d"], ["--datatype", "lsmdc"]):
            with mock.patch("sys.stderr"), self.assertRaises(SystemExit):
                self.parse(["--do_eval", "--output_dir", "out"] + flags)

    def test_more_than_one_process_rejected(self):
        with mock.patch.dict(os.environ, {"WORLD_SIZE": "8"}), self.assertRaises(ValueError):
            main.get_args(argv=UPSTREAM_ACTIVITY_COMMAND)

    def test_default_eval_split(self):
        def split(*flags):
            return main.resolve_eval_split(self.parse(["--do_eval", "--output_dir", "out"] + list(flags)))

        self.assertEqual(split(), "val")
        self.assertEqual(split("--test_csv", "t.csv"), "test")
        self.assertEqual(split("--datatype", "msvd"), "test")
        self.assertEqual(split("--datatype", "msvd", "--eval_on", "val"), "val")
        self.assertEqual(split("--datatype", "activity"), "val")
        with self.assertRaises(ValueError):
            split("--datatype", "activity", "--eval_on", "test")
        with self.assertRaises(ValueError):
            split("--eval_on", "test")


# ---------------------------------------------------------------------- #
# Prepare modules
# ---------------------------------------------------------------------- #

class PrepareMSVDTest(_TmpDirTest):

    def test_extract_data_finds_the_split_files_inside_the_zip(self):
        archive = self.tmp / "msvd_data.zip"
        with zipfile.ZipFile(archive, "w") as z:
            for name in prep_msvd.DATA_FILES:
                z.writestr("msvd_data/" + name, "x")
        data_dir = prep_msvd.extract_data(archive, self.tmp / "out" / "msvd_data")
        self.assertEqual(sorted(p.name for p in data_dir.iterdir()), sorted(prep_msvd.DATA_FILES))
        self.assertFalse((self.tmp / "out" / "_msvd_data_zip").exists())

    def test_videos_dir_is_the_one_holding_the_videos(self):
        for v in ("a", "b"):
            touch(self.tmp / "MSVD_Videos" / "YouTubeClips" / (v + ".avi"))
        touch(self.tmp / "MSVD_Videos" / "readme.avi")
        self.assertEqual(prep_msvd.find_videos_dir(self.tmp), self.tmp / "MSVD_Videos" / "YouTubeClips")
        with self.assertRaises(RuntimeError):
            prep_msvd.find_videos_dir(self.tmp / "nothing")


class PrepareActivityNetTest(_TmpDirTest):

    def test_subset_ids_keep_order_and_need_a_caption(self):
        data, _ = write_activitynet(self.tmp, videos_on_disk=[])
        self.assertEqual(prep_anet.subset_video_ids(data, "val"), ["v_x1", "v_x2", "v_x3"])
        self.assertEqual(prep_anet.subset_video_ids(data, "train"), ["v_t2", "v_t1"])

    def test_raw_videos_matched_with_or_without_prefix_and_staged_under_the_full_id(self):
        raw = self.tmp / "raw"
        touch(raw / "v_x1.mkv")
        touch(raw / "deep" / "x2.webm")
        found, missing = prep_anet.match_raw_videos(["v_x1", "v_x2", "v_x3"], prep_anet.index_raw_videos(raw))
        self.assertEqual(set(found), {"v_x1", "v_x2"})
        self.assertEqual(missing, ["v_x3"])
        staging = prep_anet.stage_videos(found, self.tmp / "staging")
        self.assertEqual(sorted(p.name for p in staging.iterdir()), ["v_x1.mkv", "v_x2.webm"])

    def test_split_tar_parts_are_read_as_one_stream(self):
        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode="w") as tar:
            for name in ("Activity_Videos/v_a.mp4", "Activity_Videos/v_b.mp4"):
                payload = name.encode() * 500
                info = tarfile.TarInfo(name)
                info.size = len(payload)
                tar.addfile(info, io.BytesIO(payload))
        blob, parts = buffer.getvalue(), []
        for i, start in enumerate(range(0, len(blob), 3000)):
            parts.append(self.tmp / "part-{:03d}".format(i))
            parts[-1].write_bytes(blob[start:start + 3000])
        self.assertGreater(len(parts), 2)

        stream = io.BufferedReader(prep_anet._ConcatenatedFiles(parts), 4096)
        with tarfile.open(fileobj=stream, mode="r|") as tar:
            tar.extractall(self.tmp / "out")
        self.assertEqual((self.tmp / "out" / "Activity_Videos" / "v_b.mp4").read_bytes(),
                         b"Activity_Videos/v_b.mp4" * 500)

    def test_exactly_one_video_source(self):
        with self.assertRaises(ValueError):
            prep_anet.prepare_activitynet_dataset(None, self.tmp)
        with self.assertRaises(ValueError):
            prep_anet.prepare_activitynet_dataset(self.tmp, self.tmp, download_raw_videos=True)


if __name__ == "__main__":
    unittest.main()
