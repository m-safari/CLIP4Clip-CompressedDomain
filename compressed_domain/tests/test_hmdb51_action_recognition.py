"""
Tests for the HMDB51 zero-shot action-recognition pipeline: dataset
preparation helpers, the compressed dataloader and its padding-trimming
collate, prompt tokenisation, embeddings and metrics.

The end-to-end test decodes real MPEG-4 videos (synthesised with ffmpeg and
re-encoded by reencode.sh) through CoViAR, so it needs the coviar extension
and the ffmpeg CLI; it is skipped when either is missing. Nothing here uses
HMDB51 itself or a trained checkpoint.
"""
import csv
import importlib.util
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from _helpers import IMAGE_RESOLUTION, build_model, fake_batch

import numpy as np
import torch

import action_recognition as ar
from dataloaders import prepare_hmdb51_dataset as prep

HAVE_COVIAR = importlib.util.find_spec("coviar") is not None
HAVE_FFMPEG = shutil.which("ffmpeg") is not None

if not HAVE_COVIAR:
    # Only the collate function is needed without CoViAR; the module imports
    # coviar at load time, so give it a placeholder.
    sys.modules.setdefault("coviar", types.ModuleType("coviar"))
from dataloaders.dataloader_hmdb51_compressed import (  # noqa: E402
    MODALITIES, dataloader_hmdb51, trim_padding_collate)


def _item(index, lengths, max_len=6, seed=0):
    """A dataset item padded to max_len, with `lengths` real frames per modality."""
    g = torch.Generator().manual_seed(seed + index)
    item = {"index": index, "label_index": index % 2, "decode_ok": True}
    for (frames_key, mask_key), length, channels in zip(MODALITIES, lengths, (3, 3, 2)):
        frames = torch.zeros(max_len, channels, IMAGE_RESOLUTION, IMAGE_RESOLUTION)
        frames[:length] = torch.randn(length, channels, IMAGE_RESOLUTION, IMAGE_RESOLUTION, generator=g)
        mask = torch.zeros(max_len, dtype=torch.long)
        mask[:length] = 1
        item[frames_key], item[mask_key] = frames, mask
    return item


class PrepareHMDB51Test(unittest.TestCase):

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp)

    def test_class_list(self):
        self.assertEqual(len(prep.HMDB51_CLASSES), 51)
        self.assertEqual(list(prep.HMDB51_CLASSES), sorted(prep.HMDB51_CLASSES))
        self.assertEqual(set(ar.CLASS_PHRASES), set(prep.HMDB51_CLASSES))

    def test_extractor_preference(self):
        def which(*available):
            return lambda name: "/usr/bin/" + name if name in available else None

        cases = [
            (("unrar", "7z", "bsdtar"), "unrar"),
            (("7z", "bsdtar"), "7z"),
            (("7za",), "7za"),
            (("bsdtar",), "bsdtar"),
        ]
        for available, expected in cases:
            with self.subTest(available=available), mock.patch("shutil.which", which(*available)):
                self.assertEqual(prep.rar_extract_command("a.rar", "out")[0], expected)
        with mock.patch("shutil.which", which()), self.assertRaises(RuntimeError):
            prep.rar_extract_command("a.rar", "out")

    def test_read_split_file(self):
        path = self.tmp / "wave_test_split1.txt"
        path.write_text("v_a.avi 1 \nv_b.avi 2 \n\nv_c.avi 0 \n")
        self.assertEqual(prep.read_split_file(path), {"v_a": 1, "v_b": 2, "v_c": 0})
        path.write_text("v_a.avi 7\n")
        with self.assertRaises(ValueError):
            prep.read_split_file(path)

    def test_build_index(self):
        splits_dir, mpeg4_dir = self.tmp / "splits", self.tmp / "mpeg4"
        splits_dir.mkdir()
        for c in prep.HMDB51_CLASSES:
            for s in (1, 2, 3):
                lines = ""
                if c == "wave":
                    lines = "w1.avi {}\nw2.avi {}\n".format(s % 3, 2)
                (splits_dir / "{}_test_split{}.txt".format(c, s)).write_text(lines)
        for name in ("wave/w1", "wave/w2", "wave/w_extra", "brush_hair/b1"):
            (mpeg4_dir / name).parent.mkdir(parents=True, exist_ok=True)
            (mpeg4_dir / (name + ".mp4")).touch()

        index_path, n = prep.build_index(mpeg4_dir, splits_dir, self.tmp / "index.csv")
        self.assertEqual(n, 4)
        with open(index_path) as f:
            rows = {r["video_id"]: r for r in csv.DictReader(f)}
        self.assertEqual(rows["wave/w1"]["label_index"], str(prep.HMDB51_CLASSES.index("wave")))
        self.assertEqual([rows["wave/w1"]["split%d" % s] for s in (1, 2, 3)], ["1", "2", "0"])
        self.assertEqual([rows["wave/w2"]["split%d" % s] for s in (1, 2, 3)], ["2", "2", "2"])
        self.assertEqual([rows["wave/w_extra"]["split%d" % s] for s in (1, 2, 3)], ["0", "0", "0"])
        self.assertEqual(rows["brush_hair/b1"]["label"], "brush_hair")


class PromptTest(unittest.TestCase):

    def test_prompt_sets(self):
        self.assertEqual(ar.class_prompts("label", ["brush_hair"]), ["brush hair"])
        self.assertEqual(ar.class_prompts("phrase", ["brush_hair"]), ["brushing hair"])
        self.assertEqual(ar.class_prompts("a person is {phrase}", ["brush_hair"]), ["a person is brushing hair"])
        for name in ar.PROMPT_SETS:
            self.assertEqual(len(ar.class_prompts(name)), 51)

    def test_tokenization_matches_msrvtt_loader(self):
        from modules.tokenization_clip import SimpleTokenizer
        from dataloaders.dataloader_msrvtt_compressed import MSRVTT_Compressed_DataLoader

        tokenizer = SimpleTokenizer()
        loader = MSRVTT_Compressed_DataLoader.__new__(MSRVTT_Compressed_DataLoader)
        loader.tokenizer, loader.max_words = tokenizer, 8
        loader.SPECIAL_TOKEN = {"CLS_TOKEN": ar.CLS_TOKEN, "SEP_TOKEN": ar.SEP_TOKEN}

        prompts = ["a person is brushing hair", "a person is doing a back handspring very slowly"]
        ids, mask, segment = ar.tokenize_prompts(prompts, tokenizer, max_words=8)
        for i, prompt in enumerate(prompts):
            text, text_mask, text_segment, _ = loader._get_text("v", prompt)
            np.testing.assert_array_equal(ids[i].numpy(), text[0])
            np.testing.assert_array_equal(mask[i].numpy(), text_mask[0])
            np.testing.assert_array_equal(segment[i].numpy(), text_segment[0])
        # The second prompt is truncated and still ends in end-of-text.
        self.assertEqual(int(mask[1].sum()), 8)
        self.assertEqual(int(ids[1, -1]), tokenizer.encoder[ar.SEP_TOKEN])


class EmbeddingTest(unittest.TestCase):

    def test_scores_match_model_similarity(self):
        for branches in (None, ["residual", "mv"], ["iframe"]):
            with self.subTest(visual_branches=branches):
                model = build_model(branches).eval()
                batch = fake_batch()
                text = ar.encode_text(model, batch["input_ids"])
                video = ar.encode_video(model, batch)
                with torch.no_grad():
                    seq, vis, vmask = model.get_sequence_visual_output(
                        batch["input_ids"], batch["token_type_ids"], batch["attention_mask"],
                        batch["iframe"], batch["iframe_mask"], batch["residuals"], batch["residuals_mask"],
                        batch["mv"], batch["mv_mask"])
                    expected = model.get_similarity_logits(seq, vis, batch["attention_mask"], vmask)
                torch.testing.assert_close(model.clip.logit_scale.exp() * text @ video.T, expected)
                torch.testing.assert_close(video.norm(dim=-1), torch.ones(len(video)))

    def test_trimmed_collate_gives_identical_embeddings(self):
        items = [_item(0, (2, 3, 1)), _item(1, (1, 4, 2)), _item(2, (3, 2, 2))]
        full = torch.utils.data.default_collate(items)
        trimmed = trim_padding_collate(items)
        self.assertEqual(trimmed["iframe"].shape[1], 3)
        self.assertEqual(trimmed["residuals"].shape[1], 4)
        self.assertEqual(trimmed["mv"].shape[1], 2)
        model = build_model().eval()
        torch.testing.assert_close(ar.encode_video(model, trimmed), ar.encode_video(model, full))

    def test_all_padding_batch_keeps_one_column(self):
        batch = trim_padding_collate([_item(0, (0, 0, 0)), _item(1, (0, 2, 0))])
        self.assertEqual(batch["iframe"].shape[1], 1)
        self.assertEqual(batch["residuals"].shape[1], 2)

    def test_failed_decode_is_nan(self):
        model = build_model(["iframe", "mv"]).eval()
        batch = trim_padding_collate([_item(0, (2, 2, 2)), _item(1, (0, 3, 0))])
        video = ar.encode_video(model, batch)
        self.assertTrue(torch.isfinite(video[0]).all())
        self.assertTrue(torch.isnan(video[1]).all())


class MetricsTest(unittest.TestCase):

    def test_metrics(self):
        scores = np.array([
            [0.9, 0.1, 0.0],    # true 0, rank 1
            [0.8, 0.5, 0.1],    # true 1, rank 2
            [0.2, 0.3, 0.1],    # true 2, rank 3
            [0.1, 0.9, 0.0],    # true 1, rank 1
            [np.nan, 0, 0],     # failed decode
        ])
        labels = np.array([0, 1, 2, 1, 0])
        metrics, per_class, confusion = ar.action_recognition_metrics(scores, labels, topk=(1, 2))
        self.assertAlmostEqual(metrics["top1"], 50.0)
        self.assertAlmostEqual(metrics["top2"], 75.0)
        self.assertAlmostEqual(metrics["median_rank"], 1.5)
        self.assertAlmostEqual(metrics["mean_rank"], 7 / 4)
        self.assertEqual((metrics["n_videos"], metrics["n_invalid"]), (4, 1))
        np.testing.assert_allclose(per_class, [100.0, 50.0, 0.0])
        self.assertAlmostEqual(metrics["mean_class_acc"], 50.0)
        np.testing.assert_array_equal(confusion, [[1, 0, 0], [1, 1, 0], [0, 1, 0]])

    def test_ties_count_against_true_class(self):
        ranks = ar.true_class_ranks(np.array([[0.5, 0.5, 0.5], [0.4, 0.5, 0.6]]), np.array([2, 2]))
        self.assertEqual(ranks.tolist(), [3, 1])

    def test_class_without_videos_is_ignored_in_mean_class_acc(self):
        metrics, per_class, _ = ar.action_recognition_metrics(np.eye(3)[:2], np.array([0, 1]))
        self.assertTrue(np.isnan(per_class[2]))
        self.assertAlmostEqual(metrics["mean_class_acc"], 100.0)


@unittest.skipUnless(HAVE_COVIAR and HAVE_FFMPEG, "needs the coviar extension and the ffmpeg CLI")
class EndToEndTest(unittest.TestCase):
    """Synthetic videos -> reencode.sh -> index -> CoViAR dataloader -> embeddings -> metrics."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp())
        raw, mpeg4 = cls.tmp / "raw", cls.tmp / "mpeg4"
        sources = {"wave": ["testsrc", "smptebars"], "walk": ["rgbtestsrc", "testsrc2"]}
        for class_name, patterns in sources.items():
            (raw / class_name).mkdir(parents=True)
            for i, pattern in enumerate(patterns):
                subprocess.run(
                    ["ffmpeg", "-loglevel", "error", "-y", "-f", "lavfi",
                     "-i", "{}=duration={}:size=320x240:rate=25".format(pattern, 2 + i),
                     str(raw / class_name / "{}_{}.avi".format(class_name, i))], check=True)
            subprocess.run(["bash", str(Path(prep.SCRIPT_DIR) / "reencode.sh"), str(raw / class_name),
                            str(mpeg4 / class_name)], check=True, stdout=subprocess.DEVNULL)

        splits = cls.tmp / "splits"
        splits.mkdir()
        for c in prep.HMDB51_CLASSES:
            for s in (1, 2, 3):
                lines = "{0}_0.avi 1\n{0}_1.avi 2\n".format(c) if c in sources else ""
                (splits / "{}_test_split{}.txt".format(c, s)).write_text(lines)
        cls.index, _ = prep.build_index(mpeg4, splits, cls.tmp / "index.csv")
        # A row whose video does not exist, to exercise the failed-decode path.
        with open(cls.index, "a", newline="") as f:
            csv.writer(f).writerow(["wave/missing", "wave", prep.HMDB51_CLASSES.index("wave"), 0, 0, 0])
        cls.mpeg4 = mpeg4

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp)

    def loader(self, **kwargs):
        return dataloader_hmdb51(self.index, str(self.mpeg4), batch_size=3, image_resolution=IMAGE_RESOLUTION,
                                 max_iframe_length=8, max_residual_length=12, max_mv_length=12, **kwargs)

    def test_items(self):
        loader, n = self.loader()
        self.assertEqual(n, 5)
        item = loader.dataset[0]
        self.assertTrue(item["decode_ok"])
        self.assertEqual(tuple(item["iframe"].shape), (8, 3, IMAGE_RESOLUTION, IMAGE_RESOLUTION))
        self.assertEqual(tuple(item["mv"].shape), (12, 2, IMAGE_RESOLUTION, IMAGE_RESOLUTION))
        self.assertGreater(int(item["iframe_mask"].sum()), 0)
        self.assertGreater(int(item["mv_mask"].sum()), 0)
        self.assertFalse(loader.dataset[4]["decode_ok"])

    def test_official_split_subset(self):
        loader, n = self.loader(split=1, subset="test")
        self.assertEqual(n, 2)
        self.assertEqual(sorted(loader.dataset.index["video_id"]), ["walk/walk_1", "wave/wave_1"])

    def test_embeddings_and_metrics(self):
        models = {"full": build_model(), "no_iframe": build_model(["residual", "mv"])}
        loader, n = self.loader()
        embeddings, meta = ar.extract_video_embeddings(models, loader)
        self.assertEqual(meta["decode_ok"].tolist(), [True, True, True, True, False])
        np.testing.assert_array_equal(meta["index"], np.arange(5))
        for name, emb in embeddings.items():
            self.assertEqual(emb.shape, (5, 32))
            self.assertTrue(np.isfinite(emb[:4]).all(), name)
            self.assertTrue(np.isnan(emb[4]).all(), name)

        # Embeddings do not depend on how videos are grouped into batches.
        single, _ = ar.extract_video_embeddings(
            models, torch.utils.data.DataLoader(loader.dataset, batch_size=1, collate_fn=trim_padding_collate))
        for name in models:
            np.testing.assert_allclose(single[name][:4], embeddings[name][:4], rtol=1e-4, atol=1e-5)

        text = torch.nn.functional.normalize(torch.randn(51, 32), dim=-1).numpy()
        metrics, _, confusion = ar.action_recognition_metrics(ar.classify(embeddings["full"], text),
                                                              meta["label_index"])
        self.assertEqual((metrics["n_videos"], metrics["n_invalid"]), (4, 1))
        self.assertEqual(int(confusion.sum()), 4)


if __name__ == "__main__":
    unittest.main()
