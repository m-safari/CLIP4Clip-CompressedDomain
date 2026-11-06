"""
Tests for the learned fusion of the modality embeddings (--fusion weighted / gated).

    cd compressed_domain && python -m unittest discover -s tests -v
"""
import sys
import types
import unittest
from unittest import mock

from _helpers import build_model, fake_batch

import torch

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

LEARNED = ("weighted", "gated")


def _visual(model, batch):
    return model.get_visual_output(batch["iframe"], batch["iframe_mask"], batch["residuals"],
                                   batch["residuals_mask"], batch["mv"], batch["mv_mask"])


def _mean_pool(visual, mask):
    unit = visual / visual.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    pooled = (unit * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp_min(1)
    return pooled / pooled.norm(dim=-1, keepdim=True)


class FusionTest(unittest.TestCase):

    def test_default_is_mean_with_no_fusion_parameters(self):
        model = build_model()
        self.assertEqual(model.fusion, "mean")
        self.assertFalse(any(n.startswith("fusion_") for n, _ in model.named_parameters()))

    def test_learned_fusion_starts_as_the_mean(self):
        batch = fake_batch()
        for fusion in LEARNED:
            with self.subTest(fusion=fusion), torch.no_grad():
                model = build_model(fusion=fusion).eval()
                visual, mask = _visual(model, batch)
                weights = model.fusion_weights(visual, mask)
                self.assertTrue(torch.allclose(weights, torch.full_like(weights, 1 / 3)))
                self.assertTrue(torch.allclose(model.pool_video(visual, mask), _mean_pool(visual, mask), atol=1e-6))

    def test_missing_branch_gets_zero_weight(self):
        batch = fake_batch()
        batch["mv_mask"][0] = 0
        for fusion in ("mean",) + LEARNED:
            with self.subTest(fusion=fusion), torch.no_grad():
                model = build_model(fusion=fusion).eval()
                visual, mask = _visual(model, batch)
                weights = model.fusion_weights(visual, mask)
                self.assertEqual(float(weights[0, 2]), 0.)
                self.assertTrue(torch.allclose(weights.sum(dim=-1), torch.ones(3)))
                self.assertTrue(torch.isfinite(model.pool_video(visual, mask)).all())

    def test_gate_receives_gradients_and_logs_weights(self):
        for fusion, name in (("weighted", "fusion_logits"), ("gated", "fusion_gate.2.weight")):
            with self.subTest(fusion=fusion):
                model = build_model(fusion=fusion).train()
                model(**fake_batch()).backward()
                grad = dict(model.named_parameters())[name].grad
                self.assertIsNotNone(grad)
                self.assertGreater(float(grad.abs().sum()), 0.)
                self.assertEqual(tuple(model.last_fusion_weights.shape), (3,))

    def test_gate_can_turn_a_noisy_branch_down(self):
        # Weights that favour the I-frames: the fused embedding moves towards the I-frame one.
        model = build_model(fusion="weighted").eval()
        with torch.no_grad():
            model.fusion_logits.copy_(torch.tensor([5., 0., -5.]))
            visual, mask = _visual(model, fake_batch())
            weights = model.fusion_weights(visual, mask)
            self.assertGreater(float(weights[:, 0].min()), 0.95)
            iframe = visual[:, 0] / visual[:, 0].norm(dim=-1, keepdim=True)
            fused = model.pool_video(visual, mask)
            self.assertGreater(float((fused * iframe).sum(dim=-1).min()), 0.95)

    def test_needs_modality_pooling(self):
        with self.assertRaises(ValueError):
            build_model(fusion="gated", visual_pooling="frame")
        with self.assertRaises(ValueError):
            build_model(fusion="attention")


class FusionOptimizerTest(unittest.TestCase):

    def setUp(self):
        self.logger = mock.patch.object(main, "logger", mock.MagicMock(), create=True)
        self.logger.start()
        self.addCleanup(self.logger.stop)

    def args(self, **kwargs):
        defaults = dict(freeze_text=1, freeze_layer_num=0, lr=1e-4, coef_lr=1e-3, warmup_proportion=0.1,
                        mv_coef_lr=0.5)
        defaults.update(kwargs)
        return types.SimpleNamespace(**defaults)

    def test_fusion_parameters_train_at_the_base_lr(self):
        for fusion in LEARNED:
            with self.subTest(fusion=fusion):
                model = build_model(fusion=fusion)
                args = self.args()
                main.freeze_parameters(args, model)
                optimizer, _ = main.prep_optimizer(args, model, 100, coef_lr=args.coef_lr)
                lr_of = {id(p): g.get("lr", args.lr) for g in optimizer.param_groups for p in g["params"]}
                decay_of = {id(p): g["weight_decay"] for g in optimizer.param_groups for p in g["params"]}
                for name, p in model.named_parameters():
                    if name.startswith("fusion_"):
                        self.assertAlmostEqual(lr_of[id(p)], 1e-4, msg=name)
                    elif name.startswith("mv_encoder."):
                        self.assertAlmostEqual(lr_of[id(p)], 5e-5, msg=name)
                if fusion == "weighted":
                    self.assertEqual(decay_of[id(model.fusion_logits)], 0.)

    def test_mean_fusion_keeps_the_group_layout(self):
        model = build_model()
        args = self.args()
        main.freeze_parameters(args, model)
        optimizer, _ = main.prep_optimizer(args, model, 100, coef_lr=args.coef_lr)
        self.assertEqual(len(optimizer.param_groups), 4)

    def test_command_line_flag(self):
        base = ["--do_eval", "--output_dir", "out"]
        self.assertEqual(main.get_args(argv=base).fusion, "mean")
        self.assertEqual(main.get_args(argv=base + ["--fusion", "gated"]).fusion, "gated")
        with mock.patch("sys.stderr"), self.assertRaises(SystemExit):
            main.get_args(argv=base + ["--fusion", "attention"])


if __name__ == "__main__":
    unittest.main()
