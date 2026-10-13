"""
Tests for the training-side fixes: preprocessing ranges and channel order,
shared GOP sampling, per-modality pooling, valid-frame-only encoding, optimizer
groups, text freezing, retrieval metrics, the epoch sampler and batch trimming.

    cd compressed_domain && python -m unittest discover -s tests -v

CoViAR and mlflow are stubbed; nothing is downloaded or decoded.
"""
import os
import sys
import types
import unittest
from unittest import mock

from _helpers import IMAGE_RESOLUTION, build_model, fake_batch

import numpy as np
import torch

from metrics import compute_metrics

_stubbed = [name for name in ("mlflow", "mlflow.pytorch", "coviar") if name not in sys.modules]
for _name in _stubbed:
    sys.modules[_name] = types.ModuleType(_name)
if "mlflow" in _stubbed:
    sys.modules["mlflow"].__path__ = []
try:
    import main_task_retrieval as main
    from dataloaders import compressedvideo_util
    from dataloaders.compressedvideo_util import CLIP_MEAN, CLIP_STD, CompressedVideoExtractor
    from dataloaders.data_dataloaders import EpochSubsetSampler, trim_padding_collate
finally:
    for _name in _stubbed:
        del sys.modules[_name]

VISUAL_ARGS = ("iframe", "iframe_mask", "residuals", "residuals_mask", "mv", "mv_mask")


def _visual(batch):
    return [batch[k] for k in VISUAL_ARGS]


class PreprocessingTest(unittest.TestCase):

    def setUp(self):
        self.extractor = CompressedVideoExtractor(image_resolution=32)

    def test_iframe_bgr_is_converted_to_rgb(self):
        bgr = np.zeros((64, 64, 3), dtype=np.uint8)
        bgr[..., 0] = 255                                   # pure blue in BGR
        out = self.extractor._preprocess_iframe(bgr)
        expected = [(v - m) / s for v, m, s in zip((0., 0., 1.), CLIP_MEAN, CLIP_STD)]   # RGB = (0, 0, 1)
        for c in range(3):
            self.assertAlmostEqual(float(out[c].mean()), expected[c], places=4)

    def test_residual_is_rgb_and_clip_scaled(self):
        bgr = np.zeros((64, 64, 3), dtype=np.int32)
        bgr[..., 0], bgr[..., 2] = 255, -255                # blue +255, red -255
        out = self.extractor._preprocess_residual(bgr)
        self.assertEqual(tuple(out.shape), (3, 32, 32))
        self.assertAlmostEqual(float(out[0].mean()), -1 / CLIP_STD[0], places=3)   # red first
        self.assertAlmostEqual(float(out[1].mean()), 0., places=5)
        self.assertAlmostEqual(float(out[2].mean()), 1 / CLIP_STD[2], places=3)
        self.assertLess(float(out.abs().max()), 4.)         # same order of magnitude as I-frames

    def test_mv_is_clamped_to_unit_range(self):
        rng = np.random.default_rng(0)
        mv = rng.integers(-255, 256, size=(64, 64, 2)).astype(np.int32)
        out = self.extractor._preprocess_mv(mv)
        self.assertEqual(tuple(out.shape), (2, 32, 32))
        self.assertLessEqual(float(out.max()), 1.)
        self.assertGreaterEqual(float(out.min()), -1.)
        small = np.full((64, 64, 2), 10, dtype=np.int32)
        self.assertAlmostEqual(float(self.extractor._preprocess_mv(small).mean()), 0.5, places=4)

    def test_normalisation_can_be_switched_off(self):
        raw = CompressedVideoExtractor(image_resolution=32, normalize=False, bgr_to_rgb=False)
        mv = np.full((64, 64, 2), 100, dtype=np.int32)
        self.assertAlmostEqual(float(raw._preprocess_mv(mv).mean()), 100., places=3)

    def test_gop_selection(self):
        self.assertEqual(CompressedVideoExtractor(max_gops=None)._select_gops(5), [0, 1, 2, 3, 4])
        self.assertEqual(CompressedVideoExtractor(max_gops=12)._select_gops(5), [0, 1, 2, 3, 4])
        self.assertEqual(CompressedVideoExtractor(max_gops=3, gop_slice=0)._select_gops(10), [0, 1, 2])
        self.assertEqual(CompressedVideoExtractor(max_gops=3, gop_slice=1)._select_gops(10), [7, 8, 9])
        self.assertEqual(CompressedVideoExtractor(max_gops=3, gop_slice=2)._select_gops(10), [0, 4, 9])

    def test_delta_positions(self):
        e = self.extractor
        self.assertEqual(e._sample_delta_positions(12, 3), [1, 6, 11])
        self.assertEqual(e._sample_delta_positions(12, 1), [6])      # middle of the GOP, not the first P-frame
        self.assertEqual(e._sample_delta_positions(1, 3), [])
        self.assertEqual(e._sample_delta_positions(2, 1), [1])

    def test_all_modalities_come_from_the_same_gops(self):
        calls = []

        def load(path, gop, pos, representation, accumulate):
            calls.append((representation, gop, pos))
            channels = 2 if representation == CompressedVideoExtractor.MV else 3
            dtype = np.uint8 if representation == CompressedVideoExtractor.IFRAME else np.int32
            return np.zeros((48, 64, channels), dtype=dtype)

        fake = types.SimpleNamespace(get_num_frames=lambda path: 12 * 30, load=load)
        extractor = CompressedVideoExtractor(image_resolution=32, max_gops=4, gop_slice=2,
                                             residual_sampling_rate=1, mv_sampling_rate=1)
        with mock.patch.object(compressedvideo_util, "coviar", fake):
            out = extractor.video_to_compressed_tensors("x.mp4")
        self.assertEqual(tuple(out["iframe"].shape), (4, 3, 32, 32))
        self.assertEqual(tuple(out["residuals"].shape), (4, 3, 32, 32))
        self.assertEqual(tuple(out["mv"].shape), (4, 2, 32, 32))
        gops = {r: [g for rr, g, _ in calls if rr == r] for r in (0, 1, 2)}
        self.assertEqual(gops[0], [0, 9, 19, 29])
        self.assertEqual(gops[1], gops[0])
        self.assertEqual(gops[2], gops[0])


class DecodedVideoCacheTest(unittest.TestCase):

    def setUp(self):
        import shutil
        import tempfile
        from dataloaders.dataloader_msrvtt_compressed import MSRVTT_Compressed_DataLoader
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp)
        self.calls = []

        def load(path, gop, pos, representation, accumulate):
            self.calls.append((representation, gop, pos))
            rng = np.random.default_rng(gop * 31 + pos * 7 + representation)
            if representation == CompressedVideoExtractor.IFRAME:
                return rng.integers(0, 256, (48, 64, 3)).astype(np.uint8)
            if representation == CompressedVideoExtractor.MV:
                return rng.integers(-60, 61, (48, 64, 2)).astype(np.int32)
            return rng.integers(-255, 256, (48, 64, 3)).astype(np.int32)

        self.frames = 12 * 5
        fake = types.SimpleNamespace(get_num_frames=lambda path: self.frames, load=load)
        patcher = mock.patch.object(compressedvideo_util, "coviar", fake)
        patcher.start()
        self.addCleanup(patcher.stop)

        self.dataset = MSRVTT_Compressed_DataLoader.__new__(MSRVTT_Compressed_DataLoader)
        self.dataset.compressedVideoExtractor = CompressedVideoExtractor(
            image_resolution=32, max_gops=3, residual_sampling_rate=1, mv_sampling_rate=1)
        self.dataset.cache_dir = os.path.join(self.tmp, "cache")
        self.dataset.cache_min_free_gb = 0.0

    def test_pack_unpack_round_trip(self):
        extractor = self.dataset.compressedVideoExtractor
        data = extractor.video_to_compressed_tensors("x.mp4")
        restored = extractor.unpack(extractor.pack(data))
        for key in ("iframe", "residuals", "mv"):
            self.assertEqual(restored[key].shape, data[key].shape)
            self.assertEqual(restored[key].dtype, torch.float32)
        torch.testing.assert_close(restored["iframe"], data["iframe"], rtol=0, atol=1e-5)       # exact pixels
        self.assertLess(float((restored["residuals"] - data["residuals"]).abs().max()), 0.5 / (255 * 0.26) + 1e-6)
        self.assertLess(float((restored["mv"] - data["mv"]).abs().max()), 0.5 / 127 + 1e-6)

    def test_second_read_comes_from_the_cache(self):
        first = self.dataset._extract("x.mp4", "video7")
        decoded = len(self.calls)
        self.assertEqual(decoded, 9)
        second = self.dataset._extract("x.mp4", "video7")
        self.assertEqual(len(self.calls), decoded)                    # CoViAR not called again
        torch.testing.assert_close(second["iframe"], first["iframe"], rtol=0, atol=1e-5)
        torch.testing.assert_close(second["mv"], first["mv"], rtol=0, atol=0.5 / 127 + 1e-6)
        files = [f for _, _, names in os.walk(self.dataset.cache_dir) for f in names]
        self.assertEqual(files, ["video7.npz"])

    def test_cache_is_keyed_by_the_sampling_settings(self):
        self.dataset._extract("x.mp4", "video7")
        decoded = len(self.calls)
        self.dataset.compressedVideoExtractor = CompressedVideoExtractor(
            image_resolution=32, max_gops=2, residual_sampling_rate=1, mv_sampling_rate=1)
        out = self.dataset._extract("x.mp4", "video7")
        self.assertGreater(len(self.calls), decoded)                  # different settings: decoded afresh
        self.assertEqual(out["iframe"].shape[0], 2)

    def test_nothing_is_written_when_the_disk_is_nearly_full_or_decoding_failed(self):
        self.dataset.cache_min_free_gb = 1e9
        self.dataset._extract("x.mp4", "video7")
        self.frames = 0
        self.dataset.cache_min_free_gb = 0.0
        empty = self.dataset._extract("broken.mp4", "video8")
        self.assertEqual(empty["iframe"].shape[0], 0)
        self.assertEqual([f for _, _, names in os.walk(self.dataset.cache_dir) for f in names], [])

    def test_corrupt_cache_file_is_replaced(self):
        self.dataset._extract("x.mp4", "video7")
        path = [os.path.join(r, f) for r, _, names in os.walk(self.dataset.cache_dir) for f in names][0]
        with open(path, "wb") as f:
            f.write(b"not an npz")
        decoded = len(self.calls)
        out = self.dataset._extract("x.mp4", "video7")
        self.assertGreater(len(self.calls), decoded)
        self.assertEqual(out["iframe"].shape[0], 3)
        self.dataset._extract("x.mp4", "video7")                      # and the rewritten file is valid
        self.assertEqual(len(self.calls), decoded + 9)

    def test_no_cache_dir_means_no_files(self):
        self.dataset.cache_dir = None
        self.dataset._extract("x.mp4", "video7")
        self.dataset._extract("x.mp4", "video7")
        self.assertEqual(len(self.calls), 18)


class PoolingTest(unittest.TestCase):

    def test_modality_pooling_weighs_modalities_equally(self):
        # Repeating one modality's frames must not change the video embedding.
        model = build_model().eval()
        batch = fake_batch()
        more = dict(batch, mv=torch.cat([batch["mv"]] * 3, dim=1), mv_mask=torch.cat([batch["mv_mask"]] * 3, dim=1))
        with torch.no_grad():
            a = model.pool_video(*model.get_visual_output(*_visual(batch)))
            b = model.pool_video(*model.get_visual_output(*_visual(more)))
        torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-5)
        torch.testing.assert_close(a.norm(dim=-1), torch.ones(3))

    def test_padded_frame_content_is_irrelevant(self):
        for pooling in ("modality", "frame"):
            with self.subTest(pooling=pooling):
                model = build_model(visual_pooling=pooling).eval()
                batch = fake_batch()
                noisy = dict(batch)
                for frames_key, mask_key in zip(VISUAL_ARGS[0::2], VISUAL_ARGS[1::2]):
                    frames = batch[frames_key].clone()
                    frames[batch[mask_key] == 0] = 1e4
                    noisy[frames_key] = frames
                with torch.no_grad():
                    a = model.pool_video(*model.get_visual_output(*_visual(batch)))
                    b = model.pool_video(*model.get_visual_output(*_visual(noisy)))
                self.assertTrue(torch.isfinite(a).all())
                self.assertTrue(torch.equal(a, b))

    def test_missing_modality_is_skipped_not_nan(self):
        model = build_model().eval()
        batch = fake_batch()
        batch["mv_mask"] = batch["mv_mask"].clone()
        batch["mv_mask"][1] = 0                              # video 1 has no motion vectors
        with torch.no_grad():
            visual, mask = model.get_visual_output(*_visual(batch))
            pooled = model.pool_video(visual, mask)
        self.assertEqual(mask[1].tolist(), [1, 1, 0])
        self.assertTrue(torch.isfinite(pooled).all())

    def test_valid_video_rows(self):
        model = build_model(["residual", "mv"])
        batch = fake_batch()
        masks = {k: batch[k].clone() for k in ("iframe_mask", "residuals_mask", "mv_mask")}
        masks["residuals_mask"][0] = 0
        masks["mv_mask"][0] = 0                              # video 0: I-frames only, but that branch is not built
        valid = model.valid_video_rows(masks["iframe_mask"], masks["residuals_mask"], masks["mv_mask"])
        self.assertEqual(valid.tolist(), [False, True, True])
        full = build_model()
        self.assertTrue(full.valid_video_rows(masks["iframe_mask"], masks["residuals_mask"],
                                              masks["mv_mask"]).all())

    def test_grad_checkpointing_gives_the_same_loss_and_gradients(self):
        batch = fake_batch()
        plain, ckpt = build_model(), build_model(grad_checkpoint=True)
        self.assertTrue(ckpt.mv_encoder.transformer.grad_checkpointing)
        grads = []
        for model in (plain, ckpt):
            model.train()
            loss = model(**batch)
            loss.backward()
            grads.append((loss.detach(), model.mv_encoder.conv1.weight.grad.clone()))
        torch.testing.assert_close(grads[0][0], grads[1][0])
        torch.testing.assert_close(grads[0][1], grads[1][1], rtol=1e-4, atol=1e-6)


class OptimizerAndFreezingTest(unittest.TestCase):

    def setUp(self):
        self.logger = mock.patch.object(main, "logger", mock.MagicMock(), create=True)
        self.logger.start()
        self.addCleanup(self.logger.stop)

    def args(self, **kwargs):
        defaults = dict(freeze_text=1, freeze_layer_num=0, lr=1e-4, coef_lr=1e-3, warmup_proportion=0.1)
        defaults.update(kwargs)
        return types.SimpleNamespace(**defaults)

    def test_text_branch_is_frozen_by_default_and_everything_else_trains(self):
        model = build_model()
        main.freeze_parameters(self.args(), model)
        for name, p in model.named_parameters():
            is_text = name.startswith(tuple("clip." + t for t in main.TEXT_BRANCH_PREFIXES))
            self.assertEqual(p.requires_grad, not is_text, name)
        self.assertTrue(model.clip.logit_scale.requires_grad)
        model.train()
        model(**fake_batch()).backward()
        self.assertIsNone(model.clip.token_embedding.weight.grad)
        self.assertIsNone(model.clip.text_projection.grad)
        for encoder in (model.clip.visual, model.residual_encoder, model.mv_encoder):
            self.assertIsNotNone(encoder.conv1.weight.grad)

    def test_text_branch_can_be_unfrozen(self):
        model = build_model()
        main.freeze_parameters(self.args(freeze_text=0), model)
        self.assertTrue(all(p.requires_grad for p in model.parameters()))

    def test_freeze_layer_num_still_applies_to_the_iframe_tower(self):
        model = build_model()
        main.freeze_parameters(self.args(freeze_layer_num=1), model)
        frozen = [n for n, p in model.clip.visual.named_parameters() if not p.requires_grad]
        self.assertTrue(frozen and all(n.startswith("transformer.resblocks.0.") for n in frozen))
        self.assertTrue(all(p.requires_grad for p in model.residual_encoder.parameters()))

    def test_residual_encoder_gets_the_pretrained_learning_rate(self):
        model = build_model()
        args = self.args()
        main.freeze_parameters(args, model)
        optimizer, _ = main.prep_optimizer(args, model, 100, coef_lr=args.coef_lr)
        lr_of = {id(p): g["lr"] for g in optimizer.param_groups for p in g["params"]}
        for name, p in model.named_parameters():
            if not p.requires_grad:
                self.assertNotIn(id(p), lr_of, name)         # frozen: not in the optimizer at all
            elif name.startswith("mv_encoder."):
                self.assertAlmostEqual(lr_of[id(p)], 1e-4, msg=name)
            else:
                self.assertAlmostEqual(lr_of[id(p)], 1e-7, msg=name)

    def test_one_step_moves_every_trainable_tower_and_not_the_text_tower(self):
        model = build_model()
        args = self.args(coef_lr=1.)
        main.freeze_parameters(args, model)
        optimizer, _ = main.prep_optimizer(args, model, 10, coef_lr=1.)
        before = {n: p.detach().clone() for n, p in model.named_parameters()}
        model.train()
        for _ in range(3):                                   # step 0 has LR 0 (warmup)
            optimizer.zero_grad()
            model(**fake_batch()).backward()
            lrs = optimizer.get_lr()
            optimizer.step()
        self.assertTrue(lrs and max(lrs) > 0)
        after = dict(model.named_parameters())
        for name in ("clip.visual.conv1.weight", "residual_encoder.conv1.weight", "mv_encoder.conv1.weight"):
            self.assertFalse(torch.equal(before[name], after[name]), name)
        for name in ("clip.token_embedding.weight", "clip.text_projection", "clip.ln_final.weight"):
            self.assertTrue(torch.equal(before[name], after[name]), name)
        self.assertEqual(main.optimizer_diagnostics(optimizer) < 0.5, True)


class MetricsTest(unittest.TestCase):

    def test_perfect_retrieval(self):
        m = compute_metrics(np.eye(50) * 10)
        self.assertEqual((m["R1"], m["MR"], m["MeanR"]), (100.0, 1.0, 1.0))

    def test_worst_retrieval(self):
        m = compute_metrics(-np.eye(50) * 10)
        self.assertEqual((m["R1"], m["R10"], m["MR"]), (0.0, 0.0, 50.0))

    def test_random_is_near_half_the_pool(self):
        m = compute_metrics(np.random.default_rng(0).standard_normal((1000, 1000)))
        self.assertTrue(400 < m["MR"] < 600, m["MR"])
        self.assertLess(m["R1"], 1.0)
        self.assertEqual(len(m["cols"]), 1000)

    def test_known_ranks(self):
        x = np.array([[0.9, 0.1, 0.0], [0.8, 0.5, 0.9], [0.3, 0.2, 0.1]])
        self.assertEqual(compute_metrics(x)["cols"], [0, 2, 2])

    def test_collapsed_embeddings_rank_last_not_first(self):
        m = compute_metrics(np.ones((20, 20)))
        self.assertEqual((m["R1"], m["MR"]), (0.0, 20.0))

    def test_bad_input_is_rejected(self):
        with self.assertRaises(ValueError):
            compute_metrics(np.ones((3, 4)))
        x = np.eye(4)
        x[1, 2] = np.nan
        with self.assertRaises(ValueError):
            compute_metrics(x)


class SamplerAndCollateTest(unittest.TestCase):

    def test_epoch_order_is_reproducible_and_differs_between_epochs(self):
        a, b = EpochSubsetSampler(100, 30, seed=42), EpochSubsetSampler(100, 30, seed=42)
        a.set_epoch(3), b.set_epoch(3)
        self.assertEqual(list(a), list(b))
        self.assertEqual(len(set(a)), 30)
        b.set_epoch(4)
        self.assertNotEqual(list(a), list(b))
        full = EpochSubsetSampler(100, 0, seed=1)
        self.assertEqual(sorted(full), list(range(100)))

    def test_trim_padding_collate(self):
        res = IMAGE_RESOLUTION

        def item(lengths, max_len=6):
            out = [torch.zeros(1, 8, dtype=torch.long)] * 3
            for length, channels in zip(lengths, (3, 3, 2)):
                mask = torch.zeros(1, max_len, dtype=torch.long)
                mask[0, :length] = 1
                out += [torch.randn(1, max_len, channels, res, res), mask]
            return tuple(out)

        batch = trim_padding_collate([item((2, 3, 1)), item((1, 4, 0))])
        self.assertEqual(tuple(batch[3].shape), (2, 1, 2, 3, res, res))
        self.assertEqual(tuple(batch[5].shape), (2, 1, 4, 3, res, res))
        self.assertEqual(tuple(batch[7].shape), (2, 1, 1, 2, res, res))
        self.assertEqual(tuple(batch[8].shape), (2, 1, 1))
        empty = trim_padding_collate([item((0, 0, 0))])
        self.assertEqual(batch[4].sum().item(), 3)
        self.assertEqual(tuple(empty[3].shape), (1, 1, 1, 3, res, res))


if __name__ == "__main__":
    unittest.main()
