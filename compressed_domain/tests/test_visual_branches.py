"""
Tests for CLIP4ClipCompressed's selectable visual branches (the ablation switch).

Run from compressed_domain/:
    python -m unittest discover -s tests -v
"""
import itertools
import sys
import types
import unittest
from unittest import mock

from _helpers import build_model, fake_batch, task_config

import torch

from modules.modeling import VISUAL_BRANCHES, resolve_visual_branches

ALL_SUBSETS = [subset for r in (1, 2, 3) for subset in itertools.combinations(VISUAL_BRANCHES, r)]
ENCODER_PREFIX = {"iframe": "clip.visual.", "residual": "residual_encoder.", "mv": "mv_encoder."}


def _perturbed(state_dict, offset):
    # Shifted so a load is distinguishable from the CLIP init, and rounded through
    # fp16 because the model holds most weights in half precision until .float().
    return {k: (v + offset).half().float() if v.is_floating_point() else v.clone()
            for k, v in state_dict.items()}


def _loss(model, batch):
    model.train()
    return model(**batch)


class ResolveVisualBranchesTest(unittest.TestCase):

    def test_absent_or_none_means_all_three(self):
        self.assertEqual(resolve_visual_branches(task_config()), VISUAL_BRANCHES)
        self.assertEqual(resolve_visual_branches(types.SimpleNamespace(visual_branches=None)), VISUAL_BRANCHES)
        self.assertEqual(resolve_visual_branches(None), VISUAL_BRANCHES)

    def test_returned_in_canonical_order(self):
        self.assertEqual(resolve_visual_branches(task_config(["mv", "iframe"])), ("iframe", "mv"))

    def test_single_string_accepted(self):
        self.assertEqual(resolve_visual_branches(task_config("residual")), ("residual",))

    def test_invalid_sets_rejected(self):
        for bad in ([], ["rgb"], ["iframe", "audio"], ["mv", "mv"]):
            with self.subTest(visual_branches=bad), self.assertRaises(ValueError):
                resolve_visual_branches(task_config(bad))

    def test_invalid_set_rejected_at_model_build(self):
        with self.assertRaises(ValueError):
            build_model([])


class DefaultModelTest(unittest.TestCase):

    def test_default_builds_all_three_towers(self):
        model = build_model()
        self.assertEqual(model.visual_branches, VISUAL_BRANCHES)
        self.assertIsNotNone(model.clip.visual)
        self.assertIsNotNone(model.residual_encoder)
        self.assertIsNotNone(model.mv_encoder)
        names = [n for n, _ in model.named_parameters()]
        for prefix in ENCODER_PREFIX.values():
            self.assertTrue(any(n.startswith(prefix) for n in names), prefix)

    def test_default_equals_explicit_all_three(self):
        default, explicit = build_model(), build_model(list(VISUAL_BRANCHES))
        self.assertEqual(default.state_dict().keys(), explicit.state_dict().keys())
        for key, value in default.state_dict().items():
            self.assertTrue(torch.equal(value, explicit.state_dict()[key]), key)
        batch = fake_batch()
        torch.testing.assert_close(_loss(default, batch), _loss(explicit, batch))

    def test_residual_tower_warm_started_from_clip_visual(self):
        model = build_model()
        sd = model.state_dict()
        for key, value in sd.items():
            if key.startswith("residual_encoder."):
                twin = "clip.visual." + key[len("residual_encoder."):]
                self.assertTrue(torch.equal(value, sd[twin]), key)


class EverySubsetTest(unittest.TestCase):

    def test_every_subset_trains(self):
        batch = fake_batch()
        for subset in ALL_SUBSETS:
            with self.subTest(visual_branches=subset):
                model = build_model(list(subset))
                self.assertEqual(model.visual_branches, subset)

                names = [n for n, _ in model.named_parameters()]
                for branch, prefix in ENCODER_PREFIX.items():
                    built = any(n.startswith(prefix) for n in names)
                    self.assertEqual(built, branch in subset, prefix)

                loss = _loss(model, batch)
                self.assertTrue(torch.isfinite(loss), loss)
                loss.backward()

                for n, p in model.named_parameters():
                    if any(n.startswith(ENCODER_PREFIX[b]) for b in subset) and n.endswith("conv1.weight"):
                        self.assertIsNotNone(p.grad, n)
                        self.assertTrue(torch.isfinite(p.grad).all(), n)
                # The text tower is always trained.
                self.assertIsNotNone(model.clip.token_embedding.weight.grad)

    def test_visual_sequence_length_matches_built_branches(self):
        batch = fake_batch(n_iframe=2, n_residual=3, n_mv=4)
        lengths = {"iframe": 2, "residual": 3, "mv": 4}
        for subset in ALL_SUBSETS:
            with self.subTest(visual_branches=subset):
                model = build_model(list(subset)).eval()
                with torch.no_grad():
                    visual, mask = model.get_visual_output(
                        batch["iframe"], batch["iframe_mask"], batch["residuals"], batch["residuals_mask"],
                        batch["mv"], batch["mv_mask"])
                expected = sum(lengths[b] for b in subset)
                self.assertEqual(tuple(visual.shape[:2]), (3, expected))
                self.assertEqual(tuple(mask.shape), (3, expected))
                masks = {"iframe": batch["iframe_mask"], "residual": batch["residuals_mask"],
                         "mv": batch["mv_mask"]}
                self.assertTrue(torch.equal(mask, torch.cat([masks[b] for b in subset], dim=1)))

    def test_dropped_branch_frames_are_ignored(self):
        model = build_model(["iframe", "residual"]).eval()
        batch = fake_batch()
        other = dict(batch, mv=torch.full_like(batch["mv"], 1e6), mv_mask=torch.zeros_like(batch["mv_mask"]))
        with torch.no_grad():
            a = model.get_visual_output(batch["iframe"], batch["iframe_mask"], batch["residuals"],
                                        batch["residuals_mask"], batch["mv"], batch["mv_mask"])
            b = model.get_visual_output(other["iframe"], other["iframe_mask"], other["residuals"],
                                        other["residuals_mask"], other["mv"], other["mv_mask"])
        self.assertTrue(torch.equal(a[0], b[0]))
        self.assertTrue(torch.equal(a[1], b[1]))


class NoIFrameTest(unittest.TestCase):

    def setUp(self):
        self.model = build_model(["residual", "mv"])

    def test_clip_visual_is_dropped(self):
        self.assertIsNone(self.model.clip.visual)
        names = [n for n, _ in self.model.named_parameters()] + list(self.model.state_dict().keys())
        self.assertFalse([n for n in names if n.startswith("clip.visual.")])

    def test_text_still_encodes(self):
        batch = fake_batch()
        self.assertEqual(self.model.clip.dtype, torch.float32)
        self.model.eval()
        with torch.no_grad():
            text = self.model.get_sequence_output(batch["input_ids"], batch["token_type_ids"],
                                                  batch["attention_mask"])
        self.assertEqual(tuple(text.shape), (3, 1, 32))
        self.assertTrue(torch.isfinite(text).all())

    def test_text_encoder_matches_full_model(self):
        # Same CLIP init and same seed: the text tower must not depend on which
        # visual branches exist.
        full = build_model()
        batch = fake_batch()
        full.eval(), self.model.eval()
        with torch.no_grad():
            a = full.get_sequence_output(batch["input_ids"])
            b = self.model.get_sequence_output(batch["input_ids"])
        torch.testing.assert_close(a, b)

    def test_dtype_falls_back_to_text_projection(self):
        self.assertIs(self.model.clip.dtype, self.model.clip.text_projection.dtype)
        self.model.half()
        self.assertEqual(self.model.clip.dtype, torch.float16)


class CheckpointIntoAblatedModelTest(unittest.TestCase):

    def test_full_checkpoint_loads_into_every_ablation(self):
        # A "trained" full checkpoint: perturb every weight so loading it is
        # distinguishable from the CLIP initialisation.
        full = build_model(seed=1)
        checkpoint = _perturbed(full.state_dict(), 0.123)

        for subset in ALL_SUBSETS:
            with self.subTest(visual_branches=subset):
                ablated = build_model(list(subset), state_dict=dict(checkpoint), seed=2)
                ablated_sd = ablated.state_dict()
                self.assertTrue(set(ablated_sd) <= set(checkpoint))
                for key, value in ablated_sd.items():
                    self.assertTrue(torch.equal(value, checkpoint[key]), key)
                dropped = [p for b, p in ENCODER_PREFIX.items() if b not in subset]
                self.assertFalse([k for k in ablated_sd if any(k.startswith(p) for p in dropped)])

    def test_ablated_checkpoint_round_trips(self):
        source = build_model(["residual", "mv"], seed=3)
        checkpoint = _perturbed(source.state_dict(), 0.5)
        reloaded = build_model(["residual", "mv"], state_dict=dict(checkpoint), seed=4)
        for key, value in reloaded.state_dict().items():
            self.assertTrue(torch.equal(value, checkpoint[key]), key)


class CommandLineTest(unittest.TestCase):
    """--visual_branches parsing in main_task_retrieval (mlflow / coviar stubbed out)."""

    @classmethod
    def setUpClass(cls):
        # Only the stubs, and the modules that bound them, are dropped afterwards:
        # restoring all of sys.modules would also unload torch internals imported
        # lazily meanwhile, which torch cannot re-register.
        stubbed = [name for name in ("mlflow", "coviar") if name not in sys.modules]
        for name in stubbed:
            sys.modules[name] = types.ModuleType(name)
        before = set(sys.modules)
        try:
            import main_task_retrieval
            cls.get_args = staticmethod(main_task_retrieval.get_args)
        finally:
            if stubbed:
                for name in set(sys.modules) - before:
                    if name == "main_task_retrieval" or name.startswith("dataloaders"):
                        del sys.modules[name]
                for name in stubbed:
                    del sys.modules[name]

    def parse(self, *extra):
        argv = ["main_task_retrieval.py", "--do_eval", "--output_dir", "out", *extra]
        with mock.patch.object(sys, "argv", argv):
            return self.get_args()

    def test_default_is_all_three(self):
        args = self.parse()
        self.assertEqual(resolve_visual_branches(args), VISUAL_BRANCHES)

    def test_subset(self):
        args = self.parse("--visual_branches", "mv", "residual")
        self.assertEqual(resolve_visual_branches(args), ("residual", "mv"))

    def test_unknown_branch_rejected(self):
        with mock.patch("sys.stderr"), self.assertRaises(SystemExit):
            self.parse("--visual_branches", "rgb")

    def test_empty_rejected(self):
        with mock.patch("sys.stderr"), self.assertRaises(SystemExit):
            self.parse("--visual_branches")


if __name__ == "__main__":
    unittest.main()
