"""Offline checks for data isolation, target alignment, and saved training state."""
from pathlib import Path
import tempfile
import unittest

import pytest

torch = pytest.importorskip("torch")

from captionbench.custom.models import ModelConfig
from captionbench.custom.experiment import (CaptionDataset, build_vocab, collate_captions,
                               dataset_signature, fit_condition, load_run, predict_records,
                               read_flickr8k, score_predictions, validate_records,
                               save_feature_cache, load_feature_cache)


def toy_records():
    return [
        {"image_id": "train_a", "split": "train", "image_path": "unused-a",
         "references": ["red ball near dog"] * 5, "sha256": "a" * 64},
        {"image_id": "train_b", "split": "train", "image_path": "unused-b",
         "references": ["blue ball near cat"] * 5, "sha256": "b" * 64},
        {"image_id": "val_a", "split": "val", "image_path": "unused-c",
         "references": ["novel validation word"] * 5, "sha256": "c" * 64},
    ]


class DataIntegrityTests(unittest.TestCase):
    def test_train_only_vocab_and_explicit_eos(self):
        records = toy_records()
        vocab = build_vocab(records[:2], min_freq=2)
        self.assertNotIn("novel", vocab.word_to_id)
        with self.assertRaises(ValueError):
            build_vocab(records)
        encoded = vocab.encode("red ball near dog blue cat", max_length=5)
        self.assertEqual(encoded[0], vocab.bos_id)
        self.assertEqual(encoded[-1], vocab.eos_id)
        self.assertEqual(len(encoded), 5)
        self.assertEqual(vocab.encode("novel"), [vocab.bos_id, vocab.unk_id, vocab.eos_id])
        self.assertEqual(vocab.decode([vocab.bos_id, vocab.word_to_id["red"], vocab.eos_id,
                                      vocab.word_to_id["ball"]]), ["red"])

    def test_teacher_forcing_target_alignment_and_padding(self):
        records = toy_records()
        vocab = build_vocab(records[:2])
        features = {record["image_id"]: torch.zeros(8, 2, 2) for record in records}
        dataset = CaptionDataset(features, records[:1], vocab, max_length=6)
        _, tokens, _ = collate_captions([dataset[0], (features["train_a"],
                        torch.tensor([vocab.bos_id, vocab.eos_id]), "short")])
        inputs, targets = tokens[:, :-1], tokens[:, 1:]
        self.assertEqual(int(inputs[0, 0]), vocab.bos_id)
        self.assertEqual(int(targets[0, 0]), vocab.word_to_id["red"])
        self.assertEqual(int(targets[0, -1]), vocab.eos_id)
        self.assertEqual(int(targets[1, 0]), vocab.eos_id)
        self.assertTrue(torch.all(targets[1, 1:] == vocab.pad_id))

    def test_duplicate_hash_and_missing_refs_fail(self):
        records = toy_records()
        records[-1]["sha256"] = records[0]["sha256"]
        with self.assertRaisesRegex(ValueError, "Image-byte leakage"):
            validate_records(records, verify_files=False)
        records = toy_records()
        records[-1]["references"].pop()
        with self.assertRaisesRegex(ValueError, "5 nonempty references"):
            validate_records(records, verify_files=False)

    def test_original_loader_hashes_bytes_and_rejects_leak(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            names = ["a.jpg", "b.jpg", "c.jpg"]
            for index, name in enumerate(names):
                (root / name).write_bytes(f"distinct image {index}".encode())
            (root / "Flickr8k.token.txt").write_text("\n".join(
                f"{name}#{reference}\tcaption number {reference}" for name in names for reference in range(5)),
                encoding="utf-8")
            for suffix, name in zip(("train", "dev", "test"), names):
                (root / f"Flickr_8k.{suffix}Images.txt").write_text(name + "\n", encoding="utf-8")
            records = read_flickr8k(root)
            self.assertEqual([record["split"] for record in records], ["train", "val", "test"])
            self.assertEqual(len(records.source_provenance), 4)
            self.assertEqual(dataset_signature(records), dataset_signature(list(reversed(records))))
            (root / "c.jpg").write_bytes((root / "a.jpg").read_bytes())
            with self.assertRaisesRegex(ValueError, "Image-byte leakage"):
                read_flickr8k(root)

    def test_original_loader_does_not_drop_missing_reference(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "Flickr8k.token.txt").write_text("a.jpg#0\tone caption\n", encoding="utf-8")
            for suffix in ("train", "dev", "test"):
                (root / f"Flickr_8k.{suffix}Images.txt").write_text("a.jpg\n", encoding="utf-8")
            (root / "a.jpg").write_bytes(b"bytes")
            with self.assertRaisesRegex(ValueError, "reference indices"):
                read_flickr8k(root)

    def test_feature_cache_rejects_changed_data_and_encoder(self):
        records = toy_records()
        features = {record["image_id"]: torch.zeros(8, 2, 2) for record in records}
        source = {"state_sha256": "test frozen encoder", "transform": "test transform"}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "features.pt"
            metadata = save_feature_cache(path, features, records, source)
            loaded, loaded_metadata = load_feature_cache(path, records, expected_source=source)
            self.assertEqual(set(loaded), set(features))
            self.assertIn("cache_file_sha256", metadata)
            self.assertEqual(loaded_metadata["cache_file_sha256"], metadata["cache_file_sha256"])
            with self.assertRaises(FileExistsError):
                save_feature_cache(path, features, records, source)
            with self.assertRaisesRegex(ValueError, "encoder provenance"):
                load_feature_cache(path, records, {"state_sha256": "changed"})
            records[-1]["references"][0] = "changed validation reference"
            with self.assertRaisesRegex(ValueError, "dataset signature"):
                load_feature_cache(path, records)


class TrainingChecks(unittest.TestCase):
    def test_offline_fit_reloads_best_and_never_overwrites_run(self):
        torch.set_num_threads(1)
        records = toy_records()
        vocab = build_vocab(records[:2])
        generator = torch.Generator().manual_seed(3)
        features = {record["image_id"]: torch.randn(8, 2, 2, generator=generator) for record in records}
        config = ModelConfig(feature_channels=8, d_model=16, n_heads=2, n_layers=1,
                             ff_dim=32, dropout=0.0, max_length=10)
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "toy-run"
            model, result = fit_condition(features, records[:2], records[2:], vocab, run_dir,
                        variant="hybrid", model_config=config, epochs=2, batch_size=5,
                        learning_rate=0.001, max_length=8)
            loaded, loaded_vocab, checkpoint = load_run(run_dir)
            self.assertEqual(result["best_validation_loss"], min(row["validation_token_loss"] for row in result["epochs"]))
            self.assertEqual(checkpoint["best_epoch"], result["best_epoch"])
            for key, value in model.state_dict().items():
                self.assertTrue(torch.equal(value.cpu(), loaded.state_dict()[key].cpu()))
            self.assertEqual(loaded_vocab.to_dict(), vocab.to_dict())
            self.assertEqual(result["metadata"]["train_image_ids"], ["train_a", "train_b"])
            predictions = predict_records(loaded, features, records[2:], vocab, max_new_tokens=5)
            self.assertEqual(set(predictions), {"val_a"})
            self.assertIsInstance(predictions["val_a"], list)
            with self.assertRaises(FileExistsError):
                fit_condition(features, records[:2], records[2:], vocab, run_dir, model_config=config, epochs=1, max_length=8)

    def test_test_records_are_rejected_for_training(self):
        records = toy_records()
        records[-1]["split"] = "test"
        vocab = build_vocab(records[:2])
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "keep test held out"):
                fit_condition({}, records[:2], records[2:], vocab, Path(directory) / "should-not-exist", epochs=1)
            self.assertFalse((Path(directory) / "should-not-exist").exists())

    def test_multireference_scoring_uses_matching_image(self):
        try:
            import nltk  # noqa: F401
        except ImportError:
            self.skipTest("NLTK scoring dependency is not installed")
        records = toy_records()[:2]
        predictions = {"train_a": ["red", "ball", "near", "dog"],
                       "train_b": ["blue", "ball", "near", "cat"]}
        result = score_predictions(predictions, records)
        self.assertAlmostEqual(result["bleu4"], 100.0)
        with self.assertRaisesRegex(ValueError, "exactly one caption"):
            score_predictions({"train_a": predictions["train_a"]}, records)


if __name__ == "__main__":
    unittest.main(verbosity=2)
