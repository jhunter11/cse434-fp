"""Reference-based evaluation with explicit scorer conventions and coverage checks."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import re
import statistics
from collections import defaultdict
from collections.abc import Iterable, Mapping
from typing import Any


CONDITION_FIELDS = ("model_id", "revision", "precision", "generation", "device", "device_name",
                    "latency_scope", "torch_version", "transformers_version")


class EvaluationError(ValueError):
    """The input cannot support a valid model comparison."""


class MetricSetupError(RuntimeError):
    """A requested metric is missing its implementation or resources."""


def validate_inputs(
    manifest: Iterable[Mapping[str, Any]],
    predictions: Iterable[Mapping[str, Any]],
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, dict[str, Any]]]]:
    """Require five references, test data, and the same images for each model."""
    images: dict[str, dict[str, Any]] = {}
    for row in manifest:
        if not isinstance(row, Mapping):
            raise EvaluationError("Each manifest row must be an object.")
        image_id = _nonempty_string(row.get("image_id"), "manifest image_id")
        if image_id in images:
            raise EvaluationError(f"Duplicate manifest image_id: {image_id}")
        if row.get("split") != "test":
            raise EvaluationError(f"Evaluation requires split=test: {image_id}")
        references = row.get("references")
        if not isinstance(references, (list, tuple)) or len(references) != 5:
            raise EvaluationError(f"Exactly five references are required: {image_id}")
        for reference in references:
            _nonempty_string(reference, f"reference for {image_id}")
        image_path = _nonempty_string(row.get("image_path"), f"image_path for {image_id}")
        images[image_id] = {**row, "image_path": image_path, "references": list(references)}
    if not images:
        raise EvaluationError("The evaluation manifest is empty.")

    by_model: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in predictions:
        if not isinstance(row, Mapping):
            raise EvaluationError("Each prediction row must be an object.")
        image_id = _nonempty_string(row.get("image_id"), "prediction image_id")
        model = _nonempty_string(row.get("model"), "prediction model")
        caption = _nonempty_string(row.get("caption"), f"caption for {model}/{image_id}")
        if image_id not in images:
            raise EvaluationError(f"Prediction has an unknown image_id: {image_id}")
        if image_id in by_model[model]:
            raise EvaluationError(f"Duplicate prediction for {model}/{image_id}")
        latency = row.get("latency_seconds")
        if latency is not None and (
            isinstance(latency, bool)
            or not isinstance(latency, (int, float))
            or not math.isfinite(latency)
            or latency < 0
        ):
            raise EvaluationError(f"latency_seconds must be finite and nonnegative: {model}/{image_id}")
        by_model[model][image_id] = {**row, "caption": caption}
    if not by_model:
        raise EvaluationError("No predictions were provided.")
    expected = set(images)
    for model, rows in by_model.items():
        missing = sorted(expected - set(rows))
        if missing:
            raise EvaluationError(f"Image coverage mismatch for {model}; missing: {', '.join(missing)}")
        model_condition(rows, model)
    image_content_hashes(images, by_model)
    return images, dict(by_model)


def image_content_hashes(
    images: Mapping[str, Mapping[str, Any]],
    by_model: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> dict[str, dict[str, Any]]:
    """Bind comparisons to common image bytes when input digests are supplied."""
    resolved = {}
    for image_id, image in images.items():
        manifest_supplied = "image_sha256" in image
        manifest_hash = image.get("image_sha256")
        rows = [predictions[image_id] for predictions in by_model.values()]
        supplied = ["image_sha256" in row for row in rows]
        for value, present in [(manifest_hash, manifest_supplied),
                               *((row.get("image_sha256"), present) for row, present in zip(rows, supplied, strict=True))]:
            if present and (not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value)):
                raise EvaluationError(f"image_sha256 must be a lowercase 64-character hex digest: {image_id}")
        if manifest_supplied and not all(supplied):
            raise EvaluationError(f"A manifest image hash requires matching prediction hashes: {image_id}")
        if any(supplied) and not all(supplied):
            raise EvaluationError(f"Mixed supplied/missing image hashes across models: {image_id}")
        prediction_hashes = {row["image_sha256"] for row, present in zip(rows, supplied, strict=True) if present}
        if len(prediction_hashes) > 1:
            raise EvaluationError(f"Image content hash mismatch across models: {image_id}")
        if manifest_supplied and prediction_hashes != {manifest_hash}:
            raise EvaluationError(f"Prediction image content hash does not match the manifest: {image_id}")
        resolved[image_id] = {
            "image_sha256": manifest_hash if manifest_supplied else next(iter(prediction_hashes), None),
            "source": "manifest_and_predictions" if manifest_supplied else "predictions_only" if prediction_hashes else "unknown",
        }
    return resolved


def model_condition(rows: Mapping[str, Mapping[str, Any]], model: str) -> dict[str, Any]:
    """Reject pooled experiment conditions, including partially missing metadata."""
    first = next(iter(rows.values()))
    supplied = [field for field in CONDITION_FIELDS if field in first]
    missing = [field for field in CONDITION_FIELDS if field not in first]
    expected = {field: first[field] for field in supplied}
    try:
        signature = json.dumps(expected, sort_keys=True, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise EvaluationError(f"Model condition metadata must be finite JSON values: {model}") from exc
    for row in rows.values():
        if any((field in row) != (field in first) for field in CONDITION_FIELDS):
            raise EvaluationError(f"Mixed supplied/missing model condition metadata: {model}")
        try:
            observed = json.dumps({field: row[field] for field in supplied}, sort_keys=True, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise EvaluationError(f"Model condition metadata must be finite JSON values: {model}") from exc
        if observed != signature:
            changed = [field for field in supplied if row[field] != first[field]]
            raise EvaluationError(f"Mixed model conditions for {model}: {', '.join(changed)}")
    return {"values": expected, "missing_fields": missing,
            "status": "complete" if not missing else "partial" if supplied else "unknown"}


def _nonempty_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise EvaluationError(f"{label} must be a nonempty string.")
    return value


def _load_bleu() -> tuple[Any, Any, Any, Any]:
    try:
        from nltk.tokenize import wordpunct_tokenize
        from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu, sentence_bleu
    except ImportError as exc:
        raise MetricSetupError("BLEU requires NLTK. Install the project dependencies first.") from exc
    return corpus_bleu, sentence_bleu, SmoothingFunction, wordpunct_tokenize


def _load_meteor() -> Any:
    try:
        from nltk.corpus import wordnet
        from nltk.translate.meteor_score import meteor_score

        # METEOR can avoid synonym matching for easy examples. Check the resource
        # before scoring so a partial run never appears to have measured METEOR.
        wordnet.ensure_loaded()
    except (ImportError, LookupError) as exc:
        raise MetricSetupError(
            "Requested NLTK METEOR requires WordNet. Run `python -m nltk.downloader wordnet` "
            "in the evaluation environment, then retry. No fallback METEOR score was computed."
        ) from exc
    return meteor_score


def evaluate(
    manifest: Iterable[Mapping[str, Any]],
    predictions: Iterable[Mapping[str, Any]],
    include_meteor: bool = False,
) -> dict[str, Any]:
    """Compute BLEU-4 and optional NLTK METEOR; never substitute proxy metrics.

    Corpus BLEU uses unsmoothed modified n-gram counts across the full corpus.
    Sentence BLEU uses Chen-Cherry method 1, epsilon 0.1. Both retain four
    equal weights even for short captions and use all five references.
    """
    images, by_model = validate_inputs(manifest, predictions)
    corpus_bleu, sentence_bleu, smoothing_class, tokenize = _load_bleu()
    meteor_score = _load_meteor() if include_meteor else None
    image_ids = sorted(images)
    content_hashes = image_content_hashes(images, by_model)
    refs = [[tokenize(reference.lower()) for reference in images[i]["references"]] for i in image_ids]
    reference_hashes = {
        image_id: hashlib.sha256(json.dumps(sorted(references), ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest()
        for image_id, references in zip(image_ids, refs, strict=True)
    }
    weights = (0.25, 0.25, 0.25, 0.25)
    smoothing = smoothing_class(epsilon=0.1).method1
    per_model: dict[str, Any] = {}
    per_image: list[dict[str, Any]] = []

    for model in sorted(by_model):
        hypotheses = [tokenize(by_model[model][i]["caption"].lower()) for i in image_ids]
        sentences: list[float] = []
        meteors: list[float] = []
        latencies: list[float] = []
        for image_id, references, hypothesis in zip(image_ids, refs, hypotheses, strict=True):
            prediction = by_model[model][image_id]
            bleu = float(sentence_bleu(references, hypothesis, weights=weights,
                                       smoothing_function=smoothing, auto_reweigh=False))
            meteor = float(meteor_score(references, hypothesis)) if meteor_score else None
            sentences.append(bleu)
            if meteor is not None:
                meteors.append(meteor)
            latency = prediction.get("latency_seconds")
            if latency is not None:
                latencies.append(float(latency))
            per_image.append({"image_id": image_id, "model": model, "caption": prediction["caption"],
                              "image_sha256": content_hashes[image_id]["image_sha256"],
                              "references_sha256": reference_hashes[image_id],
                              "bleu4_sentence": bleu, "meteor": meteor, "latency_seconds": latency})
        per_model[model] = {
            "image_count": len(image_ids),
            "condition": model_condition(by_model[model], model),
            "metrics": {
                "bleu4_corpus": float(corpus_bleu(refs, hypotheses, weights=weights,
                                                  smoothing_function=None, auto_reweigh=False)),
                "bleu4_sentence_mean": statistics.fmean(sentences),
                "meteor_mean": statistics.fmean(meteors) if meteors else None,
                "cider": None,
                "bertscore_f1": None,
                "clipscore": None,
            },
            "mean_latency_seconds": statistics.fmean(latencies) if latencies else None,
            "latency_observation_count": len(latencies),
        }

    unmeasured = {
        "cider": "Not implemented. Add and validate an official COCO-caption CIDEr scorer before reporting this metric.",
        "bertscore_f1": "Not implemented. Select and pin the BERTScore checkpoint, package, and multi-reference convention first.",
        "clipscore": "Not implemented. Select and pin a CLIPScore implementation, image preprocessing, and checkpoint first.",
    }
    if not include_meteor:
        unmeasured["meteor_mean"] = "Not requested. Use --include-meteor after installing the NLTK WordNet resource."
    return {
        "schema": "captionbench-evaluation/v1",
        "per_model": per_model,
        "per_image": per_image,
        "unmeasured_metrics": unmeasured,
        "provenance": {
            "nltk_version": importlib.metadata.version("nltk"),
            "split": "test",
            "image_count": len(image_ids),
            "image_ids": image_ids,
            "image_ids_sha256": hashlib.sha256("\n".join(image_ids).encode()).hexdigest(),
            "image_content_hashes": content_hashes,
            "image_content_hash_limit": "Input digests are validated and compared here; the manifest loader and inference runner verify file bytes. Legacy inputs without digests remain unknown.",
            "normalized_references_sha256": hashlib.sha256(json.dumps(reference_hashes, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest(),
            "reference_hash_convention": "SHA256 of JSON sorted lowercase WordPunct token lists per image; aggregate SHA256 of the sorted image-id to digest mapping; UTF-8 compact JSON",
            "model_conditions": {model: per_model[model]["condition"] for model in sorted(per_model)},
            "references_per_image": 5,
            "tokenizer": "nltk.tokenize.wordpunct_tokenize",
            "normalization": "str.lower; punctuation retained; no downloads for BLEU",
            "bleu4_weights": list(weights),
            "bleu4_corpus": "nltk.translate.bleu_score.corpus_bleu; no smoothing; auto_reweigh=False; scale 0-1",
            "bleu4_sentence": "nltk.translate.bleu_score.sentence_bleu; method1 epsilon=0.1; auto_reweigh=False; scale 0-1",
            "meteor": "nltk.translate.meteor_score.meteor_score; best of five references; default alpha=0.9 beta=3 gamma=0.5" if include_meteor else None,
            "comparability": "These tokenizer and scorer settings differ from COCO PTB-tokenized leaderboard protocols.",
            "sentence_score_use": "Use per-caption sentence scores for human correlations; corpus BLEU is not a per-caption score.",
        },
    }
