"""
Tests for the branch-subset metrics, the embedding export they rely on, the
auxiliary per-branch loss, the motion-vector CLIP initialisation, the optimizer
learning-rate classes and the report helpers in reports/analysis.py.

    cd compressed_domain && python -m unittest discover -s tests -v
"""
import os
import sys
import tempfile
import types
import unittest
from unittest import mock

from _helpers import COMPRESSED_DOMAIN, build_model, fake_batch

import numpy as np
import torch

from metrics import branch_subset_similarity, compute_metrics, pool_branches

_stubbed = [name for name in ("mlflow", "mlflow.pytorch", "coviar") if name not in sys.modules]
for _name in _stubbed:
    sys.modules[_name] = types.ModuleType(_name)
if "mlflow" in _stubbed:
    sys.modules["mlflow"].__path__ = []
try:
    import main_task_retrieval as main
finally:
    for _name in _stubbed:
        del sys.modules[_name]

try:
    import matplotlib
    matplotlib.use("Agg")
    sys.path.insert(0, os.path.join(COMPRESSED_DOMAIN, "reports"))
    import analysis as an
    HAVE_PLOTTING = True
except ImportError:                                           # matplotlib / pandas not installed
    HAVE_PLOTTING = False

VISUAL_ARGS = ("iframe", "iframe_mask", "residuals", "residuals_mask", "mv", "mv_mask")


def _encode(model, batch):
    with torch.no_grad():
        return model.get_sequence_visual_output(
            batch["input_ids"], batch["token_type_ids"], batch["attention_mask"], *[batch[k] for k in VISUAL_ARGS])


def synthetic_embeddings(n=40, d=16, seed=0, strong=("iframe",)):
    """Captions aligned with the `strong` branches; the other branches are noise."""
    rng = np.random.default_rng(seed)
    text = rng.standard_normal((n, d))
    branches = ["iframe", "residual", "mv"]
    emb = np.stack([text + 0.3 * rng.standard_normal((n, d)) if b in strong else rng.standard_normal((n, d))
                    for b in branches], axis=1)
    return {"text_embeddings": (text / np.linalg.norm(text, axis=1, keepdims=True)).astype(np.float32),
            "branch_embeddings": emb.astype(np.float32), "branch_mask": np.ones((n, 3), dtype=np.int8),
            "branches": np.asarray(branches), "video_ids": np.asarray(["video%d" % i for i in range(n)]),
            "captions": np.asarray(["caption %d" % i for i in range(n)]),
            "failed_video_ids": np.asarray(["video999"]), "logit_scale": np.float32(100.),
            "visual_pooling": np.asarray("modality"), "step": np.int64(7)}


class BranchSubsetTest(unittest.TestCase):

    def test_all_branches_matches_the_model_similarity(self):
        model = build_model().eval()
        batch = fake_batch()
        batch["mv_mask"] = batch["mv_mask"].clone()
        batch["mv_mask"][1] = 0                                # one video without motion vectors
        seq, vis, mask = _encode(model, batch)
        with torch.no_grad():
            expected = model.get_similarity_logits(seq, vis, batch["attention_mask"], mask)
        text = torch.nn.functional.normalize(seq.squeeze(1), dim=-1).numpy()
        sim = branch_subset_similarity(text, vis.numpy(), mask.numpy())
        np.testing.assert_allclose(model.clip.logit_scale.exp().item() * sim, expected.numpy(), rtol=1e-4, atol=1e-4)

    def test_subset_matches_a_model_built_with_only_those_branches(self):
        full = build_model().eval()
        batch = fake_batch()
        seq, vis, mask = _encode(full, batch)
        text = torch.nn.functional.normalize(seq.squeeze(1), dim=-1).numpy()
        for subset in (["iframe"], ["residual", "mv"], ["iframe", "mv"]):
            with self.subTest(subset=subset):
                # Same weights, fewer towers: its similarity is the ablation's ground truth.
                ablated = build_model(subset, state_dict=dict(full.state_dict())).eval()
                seq_a, vis_a, mask_a = _encode(ablated, batch)
                with torch.no_grad():
                    expected = ablated.get_similarity_logits(seq_a, vis_a, batch["attention_mask"], mask_a)
                keep = [list(full.visual_branches).index(b) for b in subset]
                sim = branch_subset_similarity(text, vis.numpy(), mask.numpy(), keep)
                np.testing.assert_allclose(full.clip.logit_scale.exp().item() * sim, expected.numpy(),
                                           rtol=1e-4, atol=1e-4)

    def test_video_without_any_kept_branch_gets_zero_similarity(self):
        emb = np.random.default_rng(0).standard_normal((3, 2, 8))
        mask = np.array([[1, 1], [0, 1], [1, 0]])
        pooled = pool_branches(emb, mask, keep=[0])
        self.assertTrue(np.allclose(pooled[1], 0))
        np.testing.assert_allclose(np.linalg.norm(pooled[[0, 2]], axis=1), 1.0, rtol=1e-6)
        self.assertTrue(np.isfinite(compute_metrics(branch_subset_similarity(emb[:, 0], emb, mask, [0]))["MR"]))


class ModalityLossTest(unittest.TestCase):

    def _loss(self, model, batch):
        model.train()
        return model(**batch)

    def test_zero_weight_is_the_plain_loss(self):
        batch = fake_batch()
        torch.testing.assert_close(self._loss(build_model(), batch),
                                   self._loss(build_model(modality_loss_weight=0.), batch))

    def test_weighted_loss_is_main_plus_mean_of_branch_losses(self):
        batch = fake_batch()
        base, aux = build_model(), build_model(modality_loss_weight=0.5)
        main_loss = self._loss(base, batch)
        seq, vis, mask = aux.get_sequence_visual_output(
            batch["input_ids"], batch["token_type_ids"], batch["attention_mask"], *[batch[k] for k in VISUAL_ARGS])
        text = torch.nn.functional.normalize(seq.squeeze(1), dim=-1)
        per_branch = []
        for m in range(3):
            sim = aux.clip.logit_scale.exp() * text @ torch.nn.functional.normalize(vis[:, m], dim=-1).T
            per_branch.append((aux.loss_fct(sim) + aux.loss_fct(sim.T)) / 2)
        expected = main_loss + 0.5 * torch.stack(per_branch).mean()
        torch.testing.assert_close(self._loss(aux, batch), expected, rtol=1e-4, atol=1e-5)

    def test_every_tower_gets_gradients_and_missing_branches_are_skipped(self):
        batch = fake_batch()
        batch["mv_mask"] = torch.zeros_like(batch["mv_mask"])
        batch["mv_mask"][0, 0] = 1                              # only one video has motion vectors
        model = build_model(modality_loss_weight=1.0)
        loss = self._loss(model, batch)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        for encoder in (model.clip.visual, model.residual_encoder):
            self.assertGreater(float(encoder.conv1.weight.grad.abs().sum()), 0)

    def test_off_for_single_branch_and_frame_pooling(self):
        batch = fake_batch()
        for kwargs in (dict(visual_branches=["iframe"]), dict(visual_pooling="frame")):
            with self.subTest(**kwargs):
                torch.testing.assert_close(self._loss(build_model(**kwargs), batch),
                                           self._loss(build_model(modality_loss_weight=1.0, **kwargs), batch))


class MotionVectorInitTest(unittest.TestCase):

    def test_clip_init_copies_the_visual_tower(self):
        model = build_model(mv_init="clip")
        sd = model.state_dict()
        for key, value in sd.items():
            if not key.startswith("mv_encoder."):
                continue
            twin = sd["clip.visual." + key[len("mv_encoder."):]]
            if key == "mv_encoder.conv1.weight":
                self.assertEqual(tuple(value.shape[:2]), (twin.shape[0], 2))
                expected = twin.mean(dim=1, keepdim=True).repeat(1, 2, 1, 1)
                torch.testing.assert_close(value, expected, rtol=1e-3, atol=1e-3)     # fp16 round trip
            else:
                self.assertTrue(torch.equal(value, twin), key)

    def test_random_init_is_the_default_and_differs_from_clip(self):
        sd = build_model().state_dict()
        self.assertFalse(torch.equal(sd["mv_encoder.proj"], sd["clip.visual.proj"]))

    def test_checkpoint_weights_win_over_clip_init(self):
        source = build_model(seed=5)
        checkpoint = {k: (v + 0.25).half().float() if v.is_floating_point() else v.clone()
                      for k, v in source.state_dict().items()}
        model = build_model(mv_init="clip", state_dict=dict(checkpoint))
        self.assertTrue(torch.equal(model.state_dict()["mv_encoder.proj"], checkpoint["mv_encoder.proj"]))

    def test_unknown_value_rejected(self):
        with self.assertRaises(ValueError):
            build_model(mv_init="imagenet")


class LearningRateClassesTest(unittest.TestCase):

    def setUp(self):
        patcher = mock.patch.object(main, "logger", mock.MagicMock(), create=True)
        patcher.start()
        self.addCleanup(patcher.stop)

    def optimizer(self, **kwargs):
        args = types.SimpleNamespace(freeze_text=1, freeze_layer_num=0, lr=1e-4, coef_lr=1e-3,
                                     warmup_proportion=0.1, **kwargs)
        model = build_model()
        main.freeze_parameters(args, model)
        optimizer, _ = main.prep_optimizer(args, model, 100, coef_lr=args.coef_lr)
        lr_of = {}
        for group in optimizer.param_groups:
            for p in group["params"]:
                lr_of[id(p)] = group["lr"]
        by_tower = {}
        for name, p in model.named_parameters():
            if p.requires_grad:
                tower = "mv" if name.startswith("mv_encoder.") else \
                    "residual" if name.startswith("residual_encoder.") else "clip"
                by_tower.setdefault(tower, set()).add(round(lr_of[id(p)], 12))
        return optimizer, by_tower

    def test_default_layout_is_unchanged(self):
        optimizer, lrs = self.optimizer()
        self.assertEqual(len(optimizer.param_groups), 4)       # resumable from earlier checkpoints
        self.assertEqual(lrs, {"clip": {1e-7}, "residual": {1e-7}, "mv": {1e-4}})
        explicit, _ = self.optimizer(residual_coef_lr=1e-3, mv_coef_lr=1.)
        self.assertEqual([len(g["params"]) for g in explicit.param_groups],
                         [len(g["params"]) for g in optimizer.param_groups])

    def test_separate_multipliers(self):
        optimizer, lrs = self.optimizer(residual_coef_lr=0.1, mv_coef_lr=0.5)
        self.assertEqual(len(optimizer.param_groups), 6)
        self.assertEqual(lrs, {"clip": {1e-7}, "residual": {1e-5}, "mv": {5e-5}})

    def test_tower_grad_norms(self):
        model = build_model()
        main.freeze_parameters(types.SimpleNamespace(freeze_text=1, freeze_layer_num=0), model)
        model.train()
        model(**fake_batch()).backward()
        norms = main.tower_grad_norms(model)
        self.assertEqual(sorted(norms), ["iframe", "mv", "residual"])
        self.assertTrue(all(v > 0 for v in norms.values()))


@unittest.skipUnless(HAVE_PLOTTING, "needs matplotlib and pandas")
class ReportHelpersTest(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(lambda: __import__("shutil").rmtree(self.tmp))
        np.savez_compressed(os.path.join(self.tmp, "embeddings_val.npz"), **synthetic_embeddings())
        self.run_ = an.load_run(self.tmp, name="synthetic")
        self.emb = self.run_["embeddings"]["val"]
        self.addCleanup(lambda: __import__("matplotlib.pyplot").pyplot.close("all"))

    def test_load_run_without_mlflow(self):
        self.assertEqual(self.run_["name"], "synthetic")
        self.assertIsNone(self.run_["run_id"])
        self.assertEqual(self.emb["branches"], ["iframe", "residual", "mv"])
        self.assertEqual(list(self.emb["failed_video_ids"]), ["video999"])
        self.assertEqual(len(an.metric_history(self.run_, "train_loss")[0]), 0)

    def test_ablation_table_finds_the_informative_branch(self):
        table = an.branch_ablation_table(self.emb)
        self.assertEqual(len(table), 7)
        self.assertGreater(table.loc["I-frame", "t2v_R@1"], 90)
        self.assertLess(table.loc["residual", "t2v_R@1"], 20)
        self.assertLess(table.loc["motion vector", "t2v_R@1"], 20)
        self.assertEqual(table["_subset"].iloc[-1], ("iframe", "residual", "mv"))
        alignment = an.branch_alignment_table(self.emb)
        self.assertGreater(alignment.loc["I-frame", "gap"], 0.5)
        self.assertLess(abs(alignment.loc["residual", "gap"]), 0.1)

    def test_ranks_agree_with_compute_metrics(self):
        for direction in ("t2v", "v2t"):
            sim = an.similarity(self.emb)
            m = compute_metrics(sim if direction == "t2v" else sim.T)
            self.assertEqual((an.ranks(self.emb, direction=direction) - 1).tolist(), m["cols"])

    def test_results_and_complementarity_tables(self):
        results = an.results_table([self.run_], "val")
        self.assertEqual(results.loc["synthetic", "queries"], 40)
        self.assertEqual(results.loc["synthetic", "failed videos"], 1)
        self.assertEqual(len(an.results_table([self.run_], "test")), 0)
        comp = an.complementarity_table(self.emb, base=("iframe",), k=1)
        self.assertEqual(len(comp), 3)
        self.assertTrue((comp["net"] == comp["gained"] - comp["lost"]).all())

    def test_qualitative_tables(self):
        picks = an.pick_examples(self.emb, branches=["residual"], n=2)
        self.assertEqual(len(picks["failure"]), 2)
        t2v = an.qualitative_text_to_video(self.emb, [0, 1], top_k=3, branches=["iframe"])
        self.assertEqual(len(t2v), 6)
        self.assertTrue(t2v[t2v["position"] == 1]["correct"].all())
        v2t = an.qualitative_video_to_text(self.emb, [0], top_k=4)
        self.assertEqual(list(v2t["position"]), [1, 2, 3, 4])
        self.assertIsNone(an.show_retrieved_videos(self.emb, 0, "/nonexistent"))

    def test_figures_render(self):
        table = an.branch_ablation_table(self.emb)
        an.plot_branch_ablation(table, chance=2.5)
        an.plot_runs_comparison(an.results_table([self.run_], "val"))
        an.plot_recall_curve(self.emb)
        an.plot_rank_scatter(self.emb, ["iframe"], ["iframe", "residual", "mv"])
        an.plot_similarity_heatmap(self.emb, n=10, branches=["iframe"])
        an.plot_learning_curves([self.run_], ["train_loss"])
        an.plot_branch_curves(self.run_)


if __name__ == "__main__":
    unittest.main()
