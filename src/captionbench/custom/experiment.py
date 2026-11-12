"""Data, training, and measurement helpers for custom CaptionBench experiments.

Backbone downloads happen only when build_frozen_encoder(pretrained=True) is
called. Tests and the synthetic smoke path do not need image data or a network.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict
from pathlib import Path
import hashlib
import json
import random
import re
import time

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from .models import Captioner, ModelConfig


VARIANTS = ("plain", "cnn_only", "fixed_kernel", "hybrid", "hybrid_coverage", "relation_only", "hybrid_relation")


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_write(path, value):
    """Replace one JSON file after the complete new document is on disk."""
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _find_unique(root, filename):
    matches = sorted(path for path in root.rglob(filename) if path.is_file())
    if len(matches) != 1:
        raise ValueError(f"Expected one {filename} under {root}; found {len(matches)}")
    return matches[0]


class CaptionRecords(list):
    """A normal list with hashes of the original annotation and split files."""
    def __init__(self, values=(), source_provenance=None):
        super().__init__(values)
        self.source_provenance = source_provenance or {}


def validate_records(records, required_references=5, verify_files=True):
    """Reject missing references, repeated IDs, and image-byte leakage across splits."""
    seen_ids, seen_hashes = {}, {}
    for record in records:
        image_id, split = record["image_id"], record["split"]
        if split not in {"train", "val", "test"}:
            raise ValueError(f"Unknown split {split!r} for {image_id}")
        if image_id in seen_ids:
            raise ValueError(f"Duplicate image ID {image_id}: {seen_ids[image_id]} and {split}")
        seen_ids[image_id] = split
        references = record.get("references", [])
        if len(references) != required_references or any(not text.strip() for text in references):
            raise ValueError(f"{image_id} requires {required_references} nonempty references")
        if verify_files:
            path = Path(record["image_path"])
            if not path.is_file():
                raise FileNotFoundError(path)
            actual_hash = _sha256(path)
            if record.get("sha256") and record["sha256"] != actual_hash:
                raise ValueError(f"Image bytes changed for {image_id}")
            record["sha256"] = actual_hash
        digest = record.get("sha256")
        if not digest:
            raise ValueError(f"Missing image SHA256 for {image_id}")
        prior = seen_hashes.get(digest)
        if prior is not None and prior[0] != split:
            raise ValueError(f"Image-byte leakage: {prior[1]} ({prior[0]}) and {image_id} ({split})")
        seen_hashes[digest] = (split, image_id)
    return records


def read_flickr8k(root):
    """Read original Flickr8k token text and train/dev/test image lists.

    Put Flickr8k.token.txt, Flickr_8k.trainImages.txt,
    Flickr_8k.devImages.txt, Flickr_8k.testImages.txt, and image files anywhere
    below root. Duplicate filenames and incomplete annotations are errors.
    Five references with indices #0 through #4 are required for each listed image.
    """
    root = Path(root).expanduser().resolve()
    token_path = _find_unique(root, "Flickr8k.token.txt")
    split_files = {
        "train": _find_unique(root, "Flickr_8k.trainImages.txt"),
        "val": _find_unique(root, "Flickr_8k.devImages.txt"),
        "test": _find_unique(root, "Flickr_8k.testImages.txt"),
    }
    captions = {}
    for line_number, line in enumerate(token_path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        try:
            key, caption = line.split("\t", 1)
            image_id, index_text = key.rsplit("#", 1)
            index = int(index_text)
        except ValueError as exc:
            raise ValueError(f"Malformed token line {line_number}") from exc
        bucket = captions.setdefault(image_id, {})
        if index in bucket:
            raise ValueError(f"Repeated reference {key}")
        bucket[index] = caption.strip()

    selected = []
    for split, path in split_files.items():
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            if line.strip():
                selected.append((split, line.strip()))
    selected_ids = {image_id for _, image_id in selected}
    image_paths = {}
    for path in root.rglob("*"):
        if path.is_file() and path.name in selected_ids:
            if path.name in image_paths:
                raise ValueError(f"Ambiguous image filename: {path.name}")
            image_paths[path.name] = path
    records = []
    for split, image_id in selected:
        references = captions.get(image_id, {})
        if set(references) != set(range(5)):
            raise ValueError(f"{image_id} requires reference indices #0 through #4; got {sorted(references)}")
        if image_id not in image_paths:
            raise FileNotFoundError(f"Image {image_id} is listed but absent below {root}")
        path = image_paths[image_id]
        records.append({"split": split, "image_id": image_id, "image_path": str(path),
                        "references": [references[i] for i in range(5)], "sha256": _sha256(path)})
    for split in split_files:
        if not any(record["split"] == split for record in records):
            raise ValueError(f"Empty {split} image list")
    validate_records(records, verify_files=False)
    provenance = {path.name: {"path": str(path), "sha256": _sha256(path)}
                  for path in [token_path, *split_files.values()]}
    return CaptionRecords(records, provenance)


def tokenize(text):
    """One declared word tokenizer shared by vocabulary, references, and BLEU."""
    return re.findall(r"[a-z0-9]+(?:'[a-z0-9]+)?", text.lower())


class Vocabulary:
    specials = ("<pad>", "<bos>", "<eos>", "<unk>")

    def __init__(self, words, min_freq=2):
        self.words = list(words)
        if self.words[:4] != list(self.specials) or len(self.words) != len(set(self.words)):
            raise ValueError("Vocabulary must start with four unique special tokens")
        self.word_to_id = {word: index for index, word in enumerate(self.words)}
        self.pad_id, self.bos_id, self.eos_id, self.unk_id = range(4)
        self.min_freq = min_freq

    def __len__(self):
        return len(self.words)

    def encode(self, text, max_length=30):
        if max_length < 2:
            raise ValueError("max_length must leave room for BOS and EOS")
        body = [self.word_to_id.get(word, self.unk_id) for word in tokenize(text)]
        return [self.bos_id, *body[:max_length - 2], self.eos_id]

    def decode(self, ids):
        result = []
        for index in ids:
            index = int(index)
            if index == self.eos_id:
                break
            if index not in (self.pad_id, self.bos_id):
                result.append(self.words[index] if 0 <= index < len(self) else "<unk>")
        return result

    def to_dict(self):
        return {"words": self.words, "min_freq": self.min_freq, "tokenizer": "lowercase_ascii_words_apostrophes_v1"}

    @classmethod
    def from_dict(cls, data):
        return cls(data["words"], min_freq=data["min_freq"])


def build_vocab(train_records, min_freq=2, max_size=None):
    records = list(train_records)
    if not records or any(record["split"] != "train" for record in records):
        raise ValueError("Build vocabulary from nonempty training records only")
    if min_freq < 1 or (max_size is not None and max_size < 4):
        raise ValueError("Invalid vocabulary limits")
    counts = Counter(word for record in records for caption in record["references"] for word in tokenize(caption))
    words = sorted((word for word, count in counts.items() if count >= min_freq), key=lambda word: (-counts[word], word))
    if max_size is not None:
        words = words[:max_size - 4]
    return Vocabulary([*Vocabulary.specials, *words], min_freq=min_freq)


def dataset_signature(records):
    canonical = [{key: record[key] for key in ("image_id", "split", "sha256", "references")}
                 for record in sorted(records, key=lambda item: (item["split"], item["image_id"]))]
    return hashlib.sha256(json.dumps(canonical, sort_keys=True, ensure_ascii=True).encode()).hexdigest()


def dataset_manifest(records):
    return {"signature": dataset_signature(records), "images": len(records),
            "counts": {split: sum(record["split"] == split for record in records) for split in ("train", "val", "test")},
            "source_files": getattr(records, "source_provenance", {})}


def build_frozen_encoder(pretrained=True):
    """Return spatial ResNet18 and its declared deterministic image transform.

    pretrained=True may download ImageNet weights. pretrained=False uses random
    weights and is suitable only for wiring checks, not a pretrained comparison.
    """
    from torchvision.models import resnet18, ResNet18_Weights
    weights = ResNet18_Weights.DEFAULT if pretrained else None
    backbone = resnet18(weights=weights)
    encoder = nn.Sequential(*list(backbone.children())[:-2])
    encoder.requires_grad_(False).eval()
    transform = ResNet18_Weights.DEFAULT.transforms()
    return encoder, transform


def extract_features(records, encoder, transform, device="cpu", batch_size=32):
    """Extract one frozen spatial tensor per image; labels do not enter the encoder."""
    from PIL import Image
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    ids = [record["image_id"] for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError("Repeated image ID during feature extraction")
    encoder = encoder.to(device).requires_grad_(False).eval()
    features = {}
    with torch.inference_mode():
        for start in range(0, len(records), batch_size):
            batch = records[start:start + batch_size]
            images = []
            for record in batch:
                with Image.open(record["image_path"]) as image:
                    images.append(transform(image.convert("RGB")))
            encoded = encoder(torch.stack(images).to(device)).detach().cpu()
            if encoded.ndim != 4:
                raise ValueError("Spatial encoder must return [batch,channels,height,width]")
            for record, tensor in zip(batch, encoded):
                features[record["image_id"]] = tensor.contiguous()
    return features


def feature_source_manifest(encoder, transform, pretrained=True):
    """Identify the exact frozen state and image transform used by a cache."""
    from importlib.metadata import version
    digest = hashlib.sha256()
    for name, value in sorted(encoder.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(f"{name}:{tensor.dtype}:{tuple(tensor.shape)}".encode())
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return {"encoder": "torchvision.resnet18.children[:-2]",
            "weights": "ResNet18_Weights.IMAGENET1K_V1" if pretrained else "random initialization",
            "state_sha256": digest.hexdigest(), "transform": repr(transform),
            "torch_version": str(torch.__version__), "torchvision_version": version("torchvision")}


def save_feature_cache(path, features, records, feature_source):
    """Write a new spatial cache with dataset and frozen-encoder provenance."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"Cache already exists: {path}")
    expected = {record["image_id"] for record in records}
    if set(features) != expected:
        raise ValueError("Feature cache must cover exactly the manifest image IDs")
    shape = None
    for image_id, value in features.items():
        if value.ndim != 3 or not torch.isfinite(value).all():
            raise ValueError(f"Invalid spatial features for {image_id}")
        if shape is None:
            shape = tuple(value.shape)
        elif tuple(value.shape) != shape:
            raise ValueError("Feature maps must use one common shape")
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {"dataset": dataset_manifest(records), "feature_source": feature_source,
                "shape": list(shape) if shape else None}
    temporary = path.with_name(path.name + ".tmp")
    torch.save({"features": {key: value.detach().cpu() for key, value in features.items()},
                "metadata": metadata}, temporary)
    temporary.replace(path)
    metadata["cache_file_sha256"] = _sha256(path)
    _json_write(path.with_suffix(path.suffix + ".json"), metadata)
    return metadata


def load_feature_cache(path, records, expected_source=None):
    """Refuse a cache from different images, splits, references, or encoder state."""
    path = Path(path)
    sidecar = path.with_suffix(path.suffix + ".json")
    if not sidecar.is_file():
        raise ValueError("Feature cache is missing its provenance sidecar")
    saved = json.loads(sidecar.read_text(encoding="utf-8"))
    cache_hash = _sha256(path)
    if saved.get("cache_file_sha256") != cache_hash:
        raise ValueError("Feature cache bytes do not match its sidecar hash")
    cache = torch.load(path, map_location="cpu", weights_only=True)
    metadata = cache["metadata"]
    if metadata["dataset"]["signature"] != dataset_signature(records):
        raise ValueError("Feature cache dataset signature does not match current records")
    if expected_source is not None and metadata["feature_source"] != expected_source:
        raise ValueError("Feature cache encoder provenance does not match requested source")
    if set(cache["features"]) != {record["image_id"] for record in records}:
        raise ValueError("Feature cache image IDs do not match records")
    metadata = {**metadata, "cache_file_sha256": cache_hash}
    return cache["features"], metadata


class CaptionDataset(Dataset):
    def __init__(self, features, records, vocab, max_length=30):
        self.features, self.vocab, self.max_length = features, vocab, max_length
        self.items = [(record["image_id"], caption) for record in records for caption in record["references"]]
        for image_id, _ in self.items:
            if image_id not in features:
                raise KeyError(f"Missing cached features for {image_id}")
            if features[image_id].ndim != 3:
                raise ValueError(f"Feature {image_id} must have [channels,height,width] shape")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        image_id, caption = self.items[index]
        return self.features[image_id], torch.tensor(self.vocab.encode(caption, self.max_length), dtype=torch.long), image_id


def collate_captions(batch):
    features, tokens, ids = zip(*batch)
    # Vocabulary reserves padding ID zero. Right padding preserves causal positions.
    return torch.stack(features), pad_sequence(tokens, batch_first=True, padding_value=0), list(ids)


def make_loader(features, records, vocab, max_length=30, batch_size=32, shuffle=False, seed=17):
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(CaptionDataset(features, records, vocab, max_length), batch_size=batch_size,
                      shuffle=shuffle, collate_fn=collate_captions, generator=generator, num_workers=0)


def seed_everything(seed):
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def _epoch(model, loader, device, pad_id, optimizer=None, gradient_clip=1.0):
    training = optimizer is not None
    model.train(training)
    total_loss, total_tokens = 0.0, 0
    criterion = nn.CrossEntropyLoss(ignore_index=pad_id, reduction="sum")
    with torch.set_grad_enabled(training):
        for features, tokens, _ in loader:
            features, tokens = features.to(device), tokens.to(device)
            logits = model(features, tokens[:, :-1])
            targets = tokens[:, 1:]
            token_count = int(targets.ne(pad_id).sum())
            loss_sum = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
            loss = loss_sum / max(token_count, 1)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite caption loss")
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
                optimizer.step()
            total_loss += float(loss_sum.detach())
            total_tokens += token_count
    if total_tokens == 0:
        raise ValueError("No target tokens in epoch")
    return total_loss / total_tokens


def fit_condition(features, train_records, val_records, vocab, run_dir, variant="plain",
                  model_config=None, seed=17, epochs=12, batch_size=32, learning_rate=3e-4,
                  weight_decay=0.01, gradient_clip=1.0, max_length=30, device="cpu"):
    """Fit exactly one run and reload its lowest validation-token-loss checkpoint.

    The run directory must not exist. Test records are rejected here. Validation
    references select an epoch; only training references produce gradients.
    Returns (best_model, result), where result contains metadata and epoch logs.
    """
    if variant not in VARIANTS:
        raise ValueError(f"Unknown variant: {variant}")
    if epochs < 1 or batch_size < 1 or learning_rate <= 0 or gradient_clip <= 0:
        raise ValueError("Training limits must be positive")
    train_records, val_records = list(train_records), list(val_records)
    if not train_records or not val_records:
        raise ValueError("Both training and validation images are required")
    if any(record["split"] != "train" for record in train_records) or any(record["split"] != "val" for record in val_records):
        raise ValueError("fit_condition accepts training and validation records only; keep test held out")
    validate_records([*train_records, *val_records], verify_files=False)
    feature_channels = features[train_records[0]["image_id"]].shape[0]
    model_config = model_config or ModelConfig(feature_channels=feature_channels)
    if model_config.feature_channels != feature_channels:
        raise ValueError("Feature channel count disagrees with model config")
    if not 2 <= max_length <= model_config.max_length:
        raise ValueError("Caption max_length must fit model positional capacity")
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=False)
    seed_everything(seed)
    model = Captioner(vocab_size=len(vocab), pad_id=vocab.pad_id, config=model_config, variant=variant).to(device)
    train_loader = make_loader(features, train_records, vocab, max_length, batch_size, True, seed)
    val_loader = make_loader(features, val_records, vocab, max_length, batch_size, False, seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    metadata = {"variant": variant, "seed": seed, "model_config": asdict(model_config), "vocabulary": vocab.to_dict(),
                "dataset_signature": dataset_signature([*train_records, *val_records]),
                "train_image_ids": sorted(record["image_id"] for record in train_records),
                "validation_image_ids": sorted(record["image_id"] for record in val_records),
                "epochs_requested": epochs, "batch_size": batch_size, "learning_rate": learning_rate,
                "weight_decay": weight_decay, "gradient_clip": gradient_clip, "max_length": max_length,
                "selection_metric": "minimum validation next-token cross entropy",
                "decode": "greedy; no reference captions supplied to generate", "torch_version": str(torch.__version__),
                "device": str(device), "deterministic_algorithms": "enabled with warn_only=True",
                "trainable_parameters": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)}
    _json_write(run_dir / "metadata.json", metadata)
    history, best_loss, best_epoch = [], float("inf"), None
    try:
        for epoch in range(1, epochs + 1):
            started = time.perf_counter()
            train_loss = _epoch(model, train_loader, device, vocab.pad_id, optimizer, gradient_clip)
            validation_loss = _epoch(model, val_loader, device, vocab.pad_id)
            history.append({"epoch": epoch, "train_token_loss": train_loss, "validation_token_loss": validation_loss,
                            "elapsed_seconds": time.perf_counter() - started})
            if validation_loss < best_loss:
                best_loss, best_epoch = validation_loss, epoch
                checkpoint = {"state_dict": {name: value.detach().cpu().clone() for name, value in model.state_dict().items()},
                              "metadata": metadata, "best_epoch": best_epoch, "best_validation_loss": best_loss}
                temporary = run_dir / "best.pt.tmp"
                torch.save(checkpoint, temporary)
                temporary.replace(run_dir / "best.pt")
            _json_write(run_dir / "history.json", {"epochs": history, "best_epoch": best_epoch,
                                                   "best_validation_loss": best_loss, "status": "running"})
    except Exception as exc:
        _json_write(run_dir / "history.json", {"epochs": history, "best_epoch": best_epoch,
                    "best_validation_loss": best_loss if best_epoch else None, "status": "failed", "error": repr(exc)})
        raise
    model, _, _ = load_run(run_dir, device)
    result = {"metadata": metadata, "epochs": history, "best_epoch": best_epoch,
              "best_validation_loss": best_loss, "status": "complete", "run_dir": str(run_dir.resolve())}
    _json_write(run_dir / "history.json", result)
    return model, result


def load_run(run_dir, device="cpu"):
    checkpoint = torch.load(Path(run_dir) / "best.pt", map_location="cpu", weights_only=True)
    metadata = checkpoint["metadata"]
    model = Captioner(vocab_size=len(metadata["vocabulary"]["words"]), pad_id=0,
                      config=ModelConfig(**metadata["model_config"]), variant=metadata["variant"])
    model.load_state_dict(checkpoint["state_dict"])
    return model.to(device).eval(), Vocabulary.from_dict(metadata["vocabulary"]), checkpoint


def predict_records(model, features, records, vocab, device="cpu", batch_size=32, max_new_tokens=24):
    """Generate one caption per image using features, BOS, and previous predictions."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    model.to(device).eval()
    predictions = {}
    with torch.inference_mode():
        for start in range(0, len(records), batch_size):
            batch = records[start:start + batch_size]
            ids = [record["image_id"] for record in batch]
            generated = model.generate(torch.stack([features[image_id] for image_id in ids]).to(device),
                                       vocab.bos_id, vocab.eos_id, max_new_tokens=max_new_tokens)
            for image_id, tokens in zip(ids, generated.cpu().tolist()):
                predictions[image_id] = vocab.decode(tokens)
    return predictions


def _bleu_inputs(predictions, records):
    expected = {record["image_id"] for record in records}
    if len(expected) != len(records) or set(predictions) != expected:
        raise ValueError("Predictions must contain exactly one caption per evaluation image")
    references = [[tokenize(caption) for caption in record["references"]] for record in records]
    hypotheses = [predictions[record["image_id"]] for record in records]
    if any(isinstance(caption, str) for caption in hypotheses):
        raise TypeError("Predictions must be word-token lists, not unprocessed strings")
    return references, hypotheses


def score_predictions(predictions, records):
    """Corpus multi-reference BLEU4, both unsmoothed and method-1 smoothed, 0-100."""
    from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
    references, hypotheses = _bleu_inputs(predictions, records)
    if not references:
        raise ValueError("Cannot score zero images")
    return {"bleu4": 100.0 * corpus_bleu(references, hypotheses, weights=(0.25,) * 4),
            "bleu4_smoothed": 100.0 * corpus_bleu(references, hypotheses, weights=(0.25,) * 4,
                                                 smoothing_function=SmoothingFunction().method1),
            "scale": "0-100", "images": len(records), "references_per_image": [len(group) for group in references],
            "mean_generated_words": sum(map(len, hypotheses)) / len(hypotheses)}


def paired_bootstrap_bleu(predictions_a, predictions_b, records, samples=1000, seed=17):
    """Paired image resampling; a positive interval favors A over B (BLEU points)."""
    from nltk.translate.bleu_score import corpus_bleu
    references, hypotheses_a = _bleu_inputs(predictions_a, records)
    _, hypotheses_b = _bleu_inputs(predictions_b, records)
    if not references or samples < 1:
        raise ValueError("Bootstrap needs images and a positive sample count")
    rng, differences, count = random.Random(seed), [], len(references)
    def score(refs, hyps):
        return 100.0 * corpus_bleu(refs, hyps, weights=(0.25,) * 4)
    for _ in range(samples):
        indices = [rng.randrange(count) for _ in range(count)]
        refs = [references[index] for index in indices]
        differences.append(score(refs, [hypotheses_a[index] for index in indices]) -
                           score(refs, [hypotheses_b[index] for index in indices]))
    differences.sort()
    return {"difference_a_minus_b": score(references, hypotheses_a) - score(references, hypotheses_b),
            "ci95_percentile": [differences[int(0.025 * (samples - 1))], differences[int(0.975 * (samples - 1))]],
            "samples": samples, "seed": seed, "unit": "image", "metric": "unsmoothed corpus BLEU4 points"}
