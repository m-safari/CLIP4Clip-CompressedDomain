"""Tests for the pluggable model layer: registry, config precedence, spec-driven
preprocessing, manifest back-compat, and an end-to-end build with no model."""

import json
import shutil
import unittest
import uuid
from contextlib import contextmanager
from pathlib import Path
from unittest import mock

import numpy as np

from embedding_db.config import (
    CLIP_IMAGE_MEAN,
    CLIP_IMAGE_STD,
    DEFAULT_MODEL_ID,
    EncoderConfig,
    PreprocessSpec,
    encoder_config_from_dict,
    load_config_file,
)
from embedding_db.core import (
    BuildConfig,
    EmbeddingDatabase,
    build_database,
    encoder_config_from_manifest,
    release_memmaps,
)
from embedding_db.encoders import (
    ENCODERS,
    available_encoders,
    create_encoder,
    register_encoder,
    resolve_precision,
)
from embedding_db.model import available_poolings, get_pooling, register_pooling
from embedding_db.video import DecodeResult, preprocess_bgr_frame


TEST_TEMP_ROOT = Path(__file__).resolve().parent


@contextmanager
def workspace_tempdir():
    path = TEST_TEMP_ROOT / f"work-{uuid.uuid4().hex}"
    path.mkdir()
    try:
        yield str(path)
    finally:
        shutil.rmtree(path)


@contextmanager
def temporary_encoder(name, factory):
    """Register an encoder for the duration of one test."""
    ENCODERS[name] = factory
    try:
        yield name
    finally:
        ENCODERS.pop(name, None)


class StubEncoder:
    """A frame encoder with no torch, no checkpoint and no network.

    Emits a deterministic vector per frame so a built database can be checked
    exactly. This is what the registry buys: the whole build pipeline becomes
    testable without a model.
    """

    encoder_name = "stub"

    def __init__(self, config):
        self.config = config
        self.embedding_dim = 4
        self.preprocess = config.apply_overrides(PreprocessSpec(image_size=8))
        self.batches = []

    def embed_frames(self, frames, batch_size=128):
        if frames.ndim != 4:
            raise ValueError("frames must have shape [N, C, H, W]")
        self.batches.append(len(frames))
        # One vector per frame, keyed on the frame mean so that distinct frames
        # produce distinct embeddings.
        seeds = frames.reshape(len(frames), -1).mean(axis=1)
        out = np.zeros((len(frames), self.embedding_dim), dtype=np.float32)
        out[:, 0] = 1.0
        out[:, 1] = seeds
        return out

    def synchronize(self):
        pass

    def encode_text(self, queries):
        out = np.zeros((len(queries), self.embedding_dim), dtype=np.float32)
        out[:, 0] = 1.0
        return out

    def describe(self):
        return {"encoder": self.encoder_name, "model": self.config.model, "device": "cpu"}


def constant_decode(value=1.0, frames_per_video=2):
    """Stand in for OpenCV decoding, honouring whatever spec it is handed."""

    def decode(item, frame_rate, max_frames, spec):
        frames = np.full(
            (frames_per_video, spec.channels, spec.image_size, spec.image_size),
            float(value),
            dtype=np.float32,
        )
        return DecodeResult(frames, 0.01)

    return decode


class RegistryTests(unittest.TestCase):
    def test_builtin_encoder_and_pooling_are_registered(self):
        self.assertIn("hf-clip", available_encoders())
        self.assertIn("meanp", available_poolings())

    def test_create_encoder_dispatches_by_name(self):
        with temporary_encoder("stub", StubEncoder):
            encoder = create_encoder(EncoderConfig(name="stub"))
            self.assertIsInstance(encoder, StubEncoder)
            self.assertEqual(encoder.embedding_dim, 4)

    def test_unknown_names_are_rejected_with_the_available_list(self):
        with self.assertRaises(ValueError) as caught:
            create_encoder(EncoderConfig(name="no-such-encoder"))
        self.assertIn("hf-clip", str(caught.exception))
        with self.assertRaises(ValueError):
            get_pooling("no-such-pooling")

    def test_registering_a_duplicate_name_is_refused(self):
        with temporary_encoder("stub", StubEncoder):
            with self.assertRaises(ValueError):
                register_encoder("stub")(type("Other", (), {}))

    def test_pooling_registry_round_trip(self):
        self.assertIs(get_pooling("MeanP"), get_pooling("meanp"))

        @register_pooling("firstframe-test")
        def first_frame(frame_embeddings):
            return frame_embeddings[0]

        try:
            self.assertIs(get_pooling("firstframe-test"), first_frame)
        finally:
            from embedding_db.model import POOLINGS

            POOLINGS.pop("firstframe-test", None)


class PreprocessSpecTests(unittest.TestCase):
    def test_defaults_match_the_published_clip_constants(self):
        spec = PreprocessSpec()
        self.assertEqual(spec.image_size, 224)
        self.assertEqual(spec.mean, CLIP_IMAGE_MEAN)
        self.assertEqual(spec.std, CLIP_IMAGE_STD)

    def test_spec_drives_frame_shape_and_normalization(self):
        bgr = np.full((60, 90, 3), 255, dtype=np.uint8)
        spec = PreprocessSpec(image_size=32, mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0))
        result = preprocess_bgr_frame(bgr, spec)
        self.assertEqual(result.shape, (3, 32, 32))
        # mean 0 / std 1 leaves white at exactly 1.0.
        np.testing.assert_allclose(result, 1.0, rtol=1e-6)

    def test_stretch_mode_uses_the_whole_frame(self):
        bgr = np.zeros((40, 120, 3), dtype=np.uint8)
        result = preprocess_bgr_frame(bgr, PreprocessSpec(image_size=16, resize_mode="stretch"))
        self.assertEqual(result.shape, (3, 16, 16))

    def test_invalid_specs_are_rejected(self):
        with self.assertRaises(ValueError):
            PreprocessSpec(image_size=0)
        with self.assertRaises(ValueError):
            PreprocessSpec(mean=(0.1, 0.2))
        with self.assertRaises(ValueError):
            PreprocessSpec(std=(0.0, 0.5, 0.5))
        with self.assertRaises(ValueError):
            PreprocessSpec(resize_mode="nearest-neighbour-magic")

    def test_config_overrides_layer_onto_the_model_spec(self):
        config = EncoderConfig(image_size=112, image_mean=(0.1, 0.2, 0.3))
        merged = config.apply_overrides(PreprocessSpec())
        self.assertEqual(merged.image_size, 112)
        self.assertEqual(merged.mean, (0.1, 0.2, 0.3))
        # Untouched fields keep whatever the model reported.
        self.assertEqual(merged.std, CLIP_IMAGE_STD)


class ConfigPrecedenceTests(unittest.TestCase):
    def test_defaults_are_the_shipped_checkpoint(self):
        self.assertEqual(EncoderConfig().model, DEFAULT_MODEL_ID)
        self.assertEqual(EncoderConfig().name, "hf-clip")

    def test_later_layers_win_and_omitted_keys_survive(self):
        from_file = encoder_config_from_dict({"model": "org/from-file", "pooling": "meanp"})
        self.assertEqual(from_file.model, "org/from-file")

        from_cli = encoder_config_from_dict({"model": "org/from-cli"}, from_file)
        self.assertEqual(from_cli.model, "org/from-cli")
        self.assertEqual(from_cli.pooling, "meanp")

    def test_unknown_settings_fail_loudly(self):
        with self.assertRaises(ValueError) as caught:
            encoder_config_from_dict({"modle": "typo"})
        self.assertIn("modle", str(caught.exception))

    def test_json_config_round_trip(self):
        with workspace_tempdir() as folder:
            path = Path(folder) / "settings.json"
            path.write_text(
                json.dumps({"encoder": {"model": "org/x"}, "build": {"max_frames": 4}}),
                encoding="utf-8",
            )
            data = load_config_file(path)
            self.assertEqual(data["encoder"]["model"], "org/x")
            self.assertEqual(data["build"]["max_frames"], 4)

    def test_unknown_top_level_sections_fail_loudly(self):
        with workspace_tempdir() as folder:
            path = Path(folder) / "settings.json"
            path.write_text(json.dumps({"encodr": {}}), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_config_file(path)

    def test_missing_config_file_is_reported(self):
        with self.assertRaises(FileNotFoundError):
            load_config_file(TEST_TEMP_ROOT / "definitely-not-here.json")

    def test_yaml_explains_itself_when_pyyaml_is_absent(self):
        with workspace_tempdir() as folder:
            path = Path(folder) / "settings.yaml"
            path.write_text("encoder:\n  model: org/x\n", encoding="utf-8")
            try:
                import yaml  # noqa: F401
            except ImportError:
                with self.assertRaises(ValueError) as caught:
                    load_config_file(path)
                self.assertIn("PyYAML", str(caught.exception))
            else:
                self.assertEqual(load_config_file(path)["encoder"]["model"], "org/x")

    def test_shipped_example_configs_load(self):
        for name in ("hf-clip4clip.json", "local-checkpoint.json"):
            data = load_config_file(TEST_TEMP_ROOT.parent / "configs" / name)
            encoder_config_from_dict(data.get("encoder", {}))

    def test_precision_guard_still_applies(self):
        self.assertEqual(resolve_precision("auto", "cpu"), "float32")
        self.assertEqual(resolve_precision("auto", "cuda"), "float16")
        with self.assertRaises(ValueError):
            resolve_precision("float16", "cpu")


class ManifestCompatibilityTests(unittest.TestCase):
    def test_new_nested_manifest_round_trips(self):
        manifest = {"config": {"encoder": {"model": "org/new", "pooling": "meanp"}}}
        self.assertEqual(encoder_config_from_manifest(manifest).model, "org/new")

    def test_legacy_flat_manifest_still_loads(self):
        # Databases built before the settings were nested must keep working.
        manifest = {
            "config": {
                "model_id": "org/legacy",
                "device": "cpu",
                "precision": "float32",
                "image_size": 224,
                "frame_rate": 1.0,
            }
        }
        config = encoder_config_from_manifest(manifest)
        self.assertEqual(config.model, "org/legacy")
        self.assertEqual(config.device, "cpu")
        self.assertEqual(config.image_size, 224)

    def test_empty_manifest_falls_back_to_defaults(self):
        self.assertEqual(encoder_config_from_manifest({}).model, DEFAULT_MODEL_ID)


class BuildWithoutAModelTests(unittest.TestCase):
    """The build pipeline, exercised end to end with a stub backend."""

    def _videos(self, root, count):
        videos = root / "videos"
        videos.mkdir()
        for index in range(count):
            (videos / f"video{index}.mp4").touch()
        return videos

    def test_build_produces_a_searchable_database(self):
        with workspace_tempdir() as folder:
            root = Path(folder)
            videos = self._videos(root, 3)
            seen_specs = []

            def decode(item, frame_rate, max_frames, spec):
                seen_specs.append(spec)
                index = float(item.video_id.replace("video", ""))
                frames = np.full(
                    (3, spec.channels, spec.image_size, spec.image_size), index, dtype=np.float32
                )
                return DecodeResult(frames, 0.01)

            config = BuildConfig(encoder=EncoderConfig(name="stub", model="stub://test"))
            with temporary_encoder("stub", StubEncoder), mock.patch(
                "embedding_db.core.decode_video", decode
            ):
                report = build_database(videos, root / "db", config)

            self.assertEqual(report["completed_count"], 3)
            self.assertEqual(report["failed_this_run"], 0)
            self.assertEqual(report["frames_this_run"], 9)
            self.assertEqual(report["system"]["encoder"], "stub")

            # The spec the decoder received came from the encoder, not a global.
            self.assertTrue(seen_specs)
            self.assertEqual(seen_specs[0].image_size, 8)

            db = EmbeddingDatabase(root / "db")
            vectors, completed, items, manifest = db.load()
            values = np.array(vectors)
            release_memmaps(vectors, completed)

            self.assertEqual(manifest["embedding_dim"], 4)
            self.assertEqual(manifest["status"], "complete")
            self.assertEqual(manifest["config"]["encoder"]["name"], "stub")
            self.assertEqual(len(items), 3)
            # meanP normalizes, so every stored row is a unit vector.
            np.testing.assert_allclose(np.linalg.norm(values, axis=1), 1.0, rtol=1e-5)

    def test_build_records_a_failure_without_aborting(self):
        with workspace_tempdir() as folder:
            root = Path(folder)
            videos = self._videos(root, 3)

            def decode(item, frame_rate, max_frames, spec):
                if item.video_id == "video1":
                    return DecodeResult(None, 0.01, "simulated decode failure")
                return constant_decode()(item, frame_rate, max_frames, spec)

            with temporary_encoder("stub", StubEncoder), mock.patch(
                "embedding_db.core.decode_video", decode
            ):
                report = build_database(videos, root / "db", BuildConfig(encoder=EncoderConfig(name="stub")))

            self.assertEqual(report["completed_count"], 2)
            self.assertEqual(report["failed_this_run"], 1)
            failures = (root / "db" / "failures.jsonl").read_text(encoding="utf-8")
            self.assertIn("simulated decode failure", failures)

    def test_resume_rejects_a_changed_model(self):
        with workspace_tempdir() as folder:
            root = Path(folder)
            videos = self._videos(root, 1)
            with temporary_encoder("stub", StubEncoder), mock.patch(
                "embedding_db.core.decode_video", constant_decode()
            ):
                build_database(videos, root / "db", BuildConfig(encoder=EncoderConfig(name="stub", model="a")))
                # A different checkpoint would mix incomparable vectors into one
                # matrix, so resuming onto it has to fail.
                with self.assertRaises(ValueError):
                    build_database(
                        videos,
                        root / "db",
                        BuildConfig(encoder=EncoderConfig(name="stub", model="b")),
                        resume=True,
                    )

    def test_resume_rejects_changed_preprocessing(self):
        with workspace_tempdir() as folder:
            root = Path(folder)
            videos = self._videos(root, 1)
            with temporary_encoder("stub", StubEncoder), mock.patch(
                "embedding_db.core.decode_video", constant_decode()
            ):
                build_database(videos, root / "db", BuildConfig(encoder=EncoderConfig(name="stub")))
                with self.assertRaises(ValueError):
                    build_database(
                        videos,
                        root / "db",
                        BuildConfig(encoder=EncoderConfig(name="stub", image_size=16)),
                        resume=True,
                    )

    def test_resume_ignores_where_the_build_ran(self):
        with workspace_tempdir() as folder:
            root = Path(folder)
            videos = self._videos(root, 1)
            with temporary_encoder("stub", StubEncoder), mock.patch(
                "embedding_db.core.decode_video", constant_decode()
            ):
                build_database(
                    videos, root / "db", BuildConfig(encoder=EncoderConfig(name="stub", device="cpu"))
                )
                report = build_database(
                    videos,
                    root / "db",
                    BuildConfig(encoder=EncoderConfig(name="stub", device="cuda:0")),
                    resume=True,
                )
            self.assertEqual(report["completed_count"], 1)


if __name__ == "__main__":
    unittest.main()
