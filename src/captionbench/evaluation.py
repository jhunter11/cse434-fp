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
    split: str = "test",
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, dict[str, Any]]]]:
    """Require five references, one declared split, and the same images for each model."""
    if split not in {"val", "test"}:
        raise EvaluationError("Evaluation split must be val or test.")
    images: dict[str, dict[str, Any]] = {}
    for row in manifest:
        if not isinstance(row, Mapping):
            raise EvaluationError("Each manifest row must be an object.")
        image_id = _nonempty_string(row.get("image_id"), "manifest image_id")
        if image_id in images:
            raise EvaluationError(f"Duplicate manifest image_id: {image_id}")
        if row.get("split") != split:
            raise EvaluationError(f"Evaluation requires split={split}: {image_id}")
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


CIDER_N = 4
CIDER_SIGMA = 6.0


def _ngram_counts(tokens: list[str]) -> dict[tuple[str, ...], int]:
    counts: dict[tuple[str, ...], int] = defaultdict(int)
    for n in range(1, CIDER_N + 1):
        for start in range(len(tokens) - n + 1):
            counts[tuple(tokens[start:start + n])] += 1
    return counts


def cider_d(references: list[list[list[str]]], hypotheses: list[list[str]]) -> list[float]:
    """Score each caption with CIDEr-D (Vedantam et al., 2015), scale 0-10.

    The function follows the COCO caption reference code: n-grams from 1 to 4,
    TF-IDF weights with document frequency over the evaluated reference images,
    clipped candidate counts, a Gaussian length penalty with sigma 6, and a
    factor of 10. It takes pre-tokenized text and applies no stemming.
    Scores depend on the evaluated image set because that set fixes the
    document frequencies. Compare scores only within one image set.
    """
    if len(references) != len(hypotheses) or not references:
        raise EvaluationError("CIDEr-D needs one reference list per caption.")
    reference_counts = [[_ngram_counts(reference) for reference in group] for group in references]
    document_frequency: dict[tuple[str, ...], int] = defaultdict(int)
    for group in reference_counts:
        for ngram in {ngram for counts in group for ngram in counts}:
            document_frequency[ngram] += 1
    log_documents = math.log(float(len(references)))

    def vector(counts):
        weights = [defaultdict(float) for _ in range(CIDER_N)]
        norms = [0.0] * CIDER_N
        length = 0
        for ngram, frequency in counts.items():
            order = len(ngram) - 1
            idf = log_documents - math.log(max(1.0, float(document_frequency.get(ngram, 0))))
            weights[order][ngram] = float(frequency) * idf
            norms[order] += weights[order][ngram] ** 2
            if order == 1:
                length += frequency
        return weights, [math.sqrt(value) for value in norms], length

    scores = []
    for group, hypothesis in zip(reference_counts, hypotheses, strict=True):
        candidate, candidate_norm, candidate_length = vector(_ngram_counts(hypothesis))
        total = [0.0] * CIDER_N
        for counts in group:
            reference, reference_norm, reference_length = vector(counts)
            delta = float(candidate_length - reference_length)
            penalty = math.exp(-(delta ** 2) / (2 * CIDER_SIGMA ** 2))
            for order in range(CIDER_N):
                overlap = sum(min(weight, reference[order][ngram]) * reference[order][ngram]
                              for ngram, weight in candidate[order].items() if ngram in reference[order])
                if candidate_norm[order] and reference_norm[order]:
                    overlap /= candidate_norm[order] * reference_norm[order]
                total[order] += overlap * penalty
        scores.append(10.0 * statistics.fmean(total) / len(group))
    return scores


def evaluate(
    manifest: Iterable[Mapping[str, Any]],
    predictions: Iterable[Mapping[str, Any]],
    include_meteor: bool = False,
    split: str = "test",
    include_clipscore: bool = False,
    clipscore_device: str = "auto",
) -> dict[str, Any]:
    """Compute BLEU-4, CIDEr-D, optional NLTK METEOR, and optional CLIPScore.

    The function never substitutes a proxy for a named metric.

    Corpus BLEU uses unsmoothed modified n-gram counts across the full corpus.
    Sentence BLEU uses Chen-Cherry method 1, epsilon 0.1. Both retain four
    equal weights even for short captions and use all five references.
    """
    images, by_model = validate_inputs(manifest, predictions, split=split)
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
    clip_scores, clip_provenance = None, None
    if include_clipscore:
        from .clipscore import clipscore
        clip_scores, clip_provenance = clipscore(
            {i: images[i]["image_path"] for i in image_ids},
            {model: {i: by_model[model][i]["caption"] for i in image_ids} for model in by_model},
            device=clipscore_device)
    per_model: dict[str, Any] = {}
    per_image: list[dict[str, Any]] = []

    for model in sorted(by_model):
        hypotheses = [tokenize(by_model[model][i]["caption"].lower()) for i in image_ids]
        sentences: list[float] = []
        meteors: list[float] = []
        latencies: list[float] = []
        ciders = cider_d(refs, hypotheses)
        clips = [clip_scores[model][i] for i in image_ids] if clip_scores else []
        for index, (image_id, references, hypothesis) in enumerate(
                zip(image_ids, refs, hypotheses, strict=True)):
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
                              "bleu4_sentence": bleu, "meteor": meteor, "cider_d": ciders[index],
                              "clipscore": clips[index] if clips else None,
                              "caption_tokens": len(hypothesis), "latency_seconds": latency})
        per_model[model] = {
            "image_count": len(image_ids),
            "condition": model_condition(by_model[model], model),
            "metrics": {
                "bleu4_corpus": float(corpus_bleu(refs, hypotheses, weights=weights,
                                                  smoothing_function=None, auto_reweigh=False)),
                "bleu4_sentence_mean": statistics.fmean(sentences),
                "meteor_mean": statistics.fmean(meteors) if meteors else None,
                "cider_d": statistics.fmean(ciders),
                "bertscore_f1": None,
                "clipscore": statistics.fmean(clips) if clips else None,
                "caption_tokens_mean": statistics.fmean(len(h) for h in hypotheses),
            },
            "mean_latency_seconds": statistics.fmean(latencies) if latencies else None,
            "latency_observation_count": len(latencies),
        }

    unmeasured = {
        "bertscore_f1": "Not implemented. Select and pin the BERTScore checkpoint, package, and multi-reference convention first.",
    }
    if not include_clipscore:
        unmeasured["clipscore"] = "Not requested. Use --include-clipscore with the models extra installed."
    if not include_meteor:
        unmeasured["meteor_mean"] = "Not requested. Use --include-meteor after installing the NLTK WordNet resource."
    return {
        "schema": "captionbench-evaluation/v1",
        "per_model": per_model,
        "per_image": per_image,
        "unmeasured_metrics": unmeasured,
        "provenance": {
            "nltk_version": importlib.metadata.version("nltk"),
            "split": split,
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
            "cider_d": "captionbench.evaluation.cider_d; n=1-4; sigma=6; document frequency over the evaluated reference images; scale 0-10; WordPunct tokens; no PTB tokenizer or stemming",
            "clipscore": clip_provenance,
            "comparability": "These tokenizer and scorer settings differ from COCO PTB-tokenized leaderboard protocols.",
            "sentence_score_use": "Use per-caption sentence scores for human correlations; corpus BLEU is not a per-caption score.",
        },
    }
