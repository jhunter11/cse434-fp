"""Blinded rating forms, strict rating validation, and descriptive correlations."""

from __future__ import annotations

import csv
import hashlib
import hmac
import importlib.metadata
import json
import math
import random
import re
import secrets
import statistics
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from .evaluation import EvaluationError, image_content_hashes, validate_inputs


FORM_FIELDS = ("assignment_id", "image_id", "image_path", "caption", "rater_id", "accuracy", "detail")
DIMENSIONS = ("accuracy", "detail")
AUTOMATIC_METRICS = ("bleu4_sentence", "meteor")


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def create_rating_packet(
    manifest: Iterable[Mapping[str, Any]],
    predictions: Iterable[Mapping[str, Any]],
    sample_size: int = 100,
    seed: int = 42,
    blinding_key: str | None = None,
) -> dict[str, Any]:
    """Select common test images and hide model names using a private HMAC key.

    Image selection is reproducible with seed 42. Assignment IDs and order are
    reproducible only with the same private blinding key. Generating a new key
    prevents readers of this public code from reconstructing model identities.
    """
    images, by_model = validate_inputs(manifest, predictions)
    content_hashes = image_content_hashes(images, by_model)
    if isinstance(sample_size, bool) or not isinstance(sample_size, int) or sample_size < 1:
        raise EvaluationError("sample_size must be a positive integer.")
    if sample_size > len(images):
        raise EvaluationError(f"Requested {sample_size} images but only {len(images)} test images are available.")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise EvaluationError("seed must be an integer.")
    key = blinding_key if blinding_key is not None else secrets.token_hex(32)
    if not isinstance(key, str) or len(key) < 16:
        raise EvaluationError("The private blinding_key must contain at least 16 characters.")
    selected = sorted(random.Random(seed).sample(sorted(images), sample_size))

    def blind(label: str, image_id: str, model: str) -> str:
        message = json.dumps([label, seed, image_id, model], ensure_ascii=False).encode("utf-8")
        return hmac.new(key.encode("utf-8"), message, hashlib.sha256).hexdigest()

    public_rows = []
    assignments = []
    for image_id in selected:
        for model in sorted(by_model, key=lambda m: blind("order", image_id, m)):
            assignment_id = blind("assignment", image_id, model)[:24]
            caption = by_model[model][image_id]["caption"]
            public_rows.append({"assignment_id": assignment_id, "image_id": image_id,
                                "image_path": images[image_id]["image_path"], "caption": caption,
                                "rater_id": "", "accuracy": "", "detail": ""})
            assignment = {"assignment_id": assignment_id, "image_id": image_id,
                          "model": model, "caption_sha256": _digest(caption)}
            if content_hashes[image_id]["image_sha256"] is not None:
                assignment["image_sha256"] = content_hashes[image_id]["image_sha256"]
            assignments.append(assignment)
    if len({row["assignment_id"] for row in assignments}) != len(assignments):
        raise EvaluationError("Assignment ID collision; regenerate the private blinding key.")
    return {
        "public_rows": public_rows,
        "private_key": {"schema": "captionbench-private-rating-key/v1", "blinding_key": key,
                        "assignments": assignments},
        "selection": {
            "schema": "captionbench-rating-selection/v1", "seed": seed,
            "sample_size": sample_size, "image_ids": selected,
            "image_ids_sha256": _digest("\n".join(selected)),
            "model_count": len(by_model), "assignment_count": len(assignments),
            "rubric": {
                "accuracy": "1=mostly incorrect; 2=major factual errors; 3=partly correct; 4=mostly correct; 5=fully supported by the image",
                "detail": "1=little correct visible specificity; 2=limited correct detail; 3=adequate correct detail; 4=rich correct visible detail; 5=precise and comprehensive correct visible detail. Unsupported details and verbosity earn no credit.",
            },
            "instructions": "Rate the caption against the image. Use integer scores 1-5. Each rater uses a unique pseudonymous rater_id. Do not consult references or model names. Count only correct visible details; unsupported specificity and verbosity earn no detail credit.",
            "sharing": "Share only ratings.csv, selection.json, and the selected image files with raters. Keep private-model-key.json, predictions, evaluation results, and the comparison gallery private until ratings are complete; they reveal model identities or references.",
        },
    }


def _safe_csv(value: Any) -> Any:
    # CSV quoting alone does not stop spreadsheet formula evaluation.
    if isinstance(value, str) and value.lstrip().startswith(("=", "+", "-", "@")):
        return "'" + value
    if isinstance(value, str) and value.startswith(("\t", "\r", "\n")):
        return "'" + value
    return value


def write_rating_packet(packet: Mapping[str, Any], output_dir: str | Path) -> dict[str, str]:
    """Write the public CSV and a separate private model key without overwriting."""
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    paths = {"ratings": destination / "ratings.csv", "private_key": destination / "private-model-key.json",
             "selection": destination / "selection.json"}
    if any(path.exists() for path in paths.values()):
        raise EvaluationError("Rating packet files already exist; use a new output directory.")
    with paths["ratings"].open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FORM_FIELDS)
        writer.writeheader()
        writer.writerows({key: _safe_csv(row[key]) for key in FORM_FIELDS} for row in packet["public_rows"])
    for name in ("private_key", "selection"):
        with paths[name].open("x", encoding="utf-8") as handle:
            json.dump(packet[name], handle, indent=2, ensure_ascii=False, allow_nan=False)
            handle.write("\n")
    return {name: str(path.resolve()) for name, path in paths.items()}


def read_ratings_csv(path: str | Path) -> list[dict[str, str]]:
    """Read a completed CSV; blank ratings remain errors during analysis."""
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or len(set(reader.fieldnames)) != len(reader.fieldnames):
            raise EvaluationError("Ratings CSV must have unique column names.")
        required = {"assignment_id", "rater_id", "accuracy", "detail"}
        if not required.issubset(reader.fieldnames):
            raise EvaluationError(f"Ratings CSV is missing columns: {', '.join(sorted(required - set(reader.fieldnames)))}")
        rows = list(reader)
        if any(None in row or any(value is None for value in row.values()) for row in rows):
            raise EvaluationError("Ratings CSV has a row with the wrong number of fields.")
        return rows


def _rating(value: Any, field: str) -> int:
    if isinstance(value, bool):
        raise EvaluationError(f"{field} must be an integer from 1 through 5.")
    if isinstance(value, int) and 1 <= value <= 5:
        return value
    if isinstance(value, str) and value.strip() in {"1", "2", "3", "4", "5"}:
        return int(value.strip())
    raise EvaluationError(f"{field} must be an integer from 1 through 5.")


def _rho(x: Sequence[float], y: Sequence[float]) -> float | None:
    if len(x) < 3 or len(set(x)) < 2 or len(set(y)) < 2:
        return None
    from scipy.stats import spearmanr

    value = float(spearmanr(x, y).statistic)
    return value if math.isfinite(value) else None


def spearman_summary(
    pairs: Sequence[tuple[str, float, float]],
    bootstrap_samples: int = 1000,
    seed: int = 42,
) -> dict[str, Any]:
    """Describe rank association, resampling whole image clusters for the CI."""
    if isinstance(bootstrap_samples, bool) or not isinstance(bootstrap_samples, int) or bootstrap_samples < 0:
        raise EvaluationError("bootstrap_samples must be a nonnegative integer.")
    if any(not math.isfinite(x) or not math.isfinite(y) for _, x, y in pairs):
        raise EvaluationError("Spearman input scores must be finite.")
    x = [p[1] for p in pairs]
    y = [p[2] for p in pairs]
    grouped: dict[str, list[tuple[str, float, float]]] = defaultdict(list)
    for pair in pairs:
        grouped[pair[0]].append(pair)
    status = "measured"
    if len(pairs) < 3:
        status = "insufficient_pairs"
    elif len(set(x)) < 2 or len(set(y)) < 2:
        status = "constant_input"
    rho = _rho(x, y)
    result: dict[str, Any] = {"rho": rho, "status": status, "caption_count": len(pairs),
                              "image_cluster_count": len(grouped), "ci95": None,
                              "bootstrap_valid_samples": 0, "bootstrap_requested_samples": bootstrap_samples}
    if rho is None:
        result["ci_status"] = "undefined_correlation"
        return result
    if not bootstrap_samples:
        result["ci_status"] = "not_requested"
        return result
    if len(grouped) < 3:
        result["ci_status"] = "insufficient_image_clusters"
        return result
    rng = random.Random(seed)
    clusters = sorted(grouped)
    samples = []
    for _ in range(bootstrap_samples):
        resampled = [pair for image_id in rng.choices(clusters, k=len(clusters)) for pair in grouped[image_id]]
        value = _rho([p[1] for p in resampled], [p[2] for p in resampled])
        if value is not None:
            samples.append(value)
    result["bootstrap_valid_samples"] = len(samples)
    if len(samples) < max(20, math.ceil(bootstrap_samples * 0.9)):
        result["ci_status"] = "too_many_undefined_resamples"
        return result
    samples.sort()

    def quantile(q: float) -> float:
        position = (len(samples) - 1) * q
        low = math.floor(position)
        high = math.ceil(position)
        return samples[low] + (samples[high] - samples[low]) * (position - low)

    result["ci95"] = [quantile(0.025), quantile(0.975)]
    result["ci_status"] = "image_cluster_percentile_bootstrap"
    return result


def analyze_ratings(
    rating_rows: Iterable[Mapping[str, Any]],
    private_key: Mapping[str, Any],
    evaluation_result: Mapping[str, Any],
    bootstrap_samples: int = 1000,
    seed: int = 42,
) -> dict[str, Any]:
    """Aggregate raters per caption, then correlate human and automatic scores.

    Partial forms are allowed and their coverage is reported. Correlations are
    descriptive; no independence-based p-values are returned.
    """
    if private_key.get("schema") != "captionbench-private-rating-key/v1":
        raise EvaluationError("Unrecognized private rating key schema.")
    assignments: dict[str, Any] = {}
    assignment_pairs = set()
    for row in private_key.get("assignments", []):
        if not isinstance(row, Mapping):
            raise EvaluationError("Each private key assignment must be an object.")
        assignment_id = row.get("assignment_id")
        if not isinstance(assignment_id, str) or not assignment_id:
            raise EvaluationError("The private key contains an invalid assignment_id.")
        pair = (row.get("image_id"), row.get("model"))
        if any(not isinstance(value, str) or not value.strip() for value in pair):
            raise EvaluationError("The private key contains an invalid image_id or model.")
        if not isinstance(row.get("caption_sha256"), str) or not re.fullmatch(r"[0-9a-f]{64}", row["caption_sha256"]):
            raise EvaluationError("The private key contains an invalid caption digest.")
        if "image_sha256" in row and (not isinstance(row["image_sha256"], str)
                                      or not re.fullmatch(r"[0-9a-f]{64}", row["image_sha256"])):
            raise EvaluationError("The private key contains an invalid image digest.")
        if assignment_id in assignments or pair in assignment_pairs:
            raise EvaluationError("The private key contains duplicate assignments.")
        assignments[assignment_id] = row
        assignment_pairs.add(pair)
    if not assignments:
        raise EvaluationError("The private rating key is empty.")
    model_images: dict[str, set[str]] = defaultdict(set)
    for image_id, model in assignment_pairs:
        model_images[model].add(image_id)
    if any(ids != next(iter(model_images.values())) for ids in model_images.values()):
        raise EvaluationError("Private key models must cover the same selected images.")
    metrics = {}
    for row in evaluation_result.get("per_image", []):
        if not isinstance(row, Mapping):
            raise EvaluationError("Each evaluated per-image score must be an object.")
        pair = (row.get("image_id"), row.get("model"))
        if any(not isinstance(value, str) or not value.strip() for value in pair):
            raise EvaluationError("Evaluation contains an invalid image_id or model.")
        if pair in metrics:
            raise EvaluationError("Evaluation contains duplicate per-image scores.")
        if not isinstance(row.get("caption"), str) or not row["caption"].strip():
            raise EvaluationError("Evaluation contains a missing caption.")
        for metric in AUTOMATIC_METRICS:
            value = row.get(metric)
            if value is not None and (isinstance(value, bool) or not isinstance(value, (float, int))
                                      or not math.isfinite(value) or not 0 <= value <= 1):
                raise EvaluationError("Automatic metric scores must be finite values from 0 through 1 or null.")
        metrics[pair] = row
    for assignment in assignments.values():
        pair = (assignment["image_id"], assignment["model"])
        if pair not in metrics or _digest(metrics[pair].get("caption", "")) != assignment.get("caption_sha256"):
            raise EvaluationError("Private rating key does not match the evaluated captions.")
        if "image_sha256" in assignment and assignment["image_sha256"] != metrics[pair].get("image_sha256"):
            raise EvaluationError("Private rating key does not match the evaluated image content.")

    seen = set()
    normalized = []
    for row in rating_rows:
        if not isinstance(row, Mapping):
            raise EvaluationError("Each rating row must be an object.")
        assignment_id = row.get("assignment_id")
        if not isinstance(assignment_id, str) or assignment_id not in assignments:
            raise EvaluationError(f"Unknown rating assignment_id: {assignment_id}")
        rater_id = row.get("rater_id")
        if not isinstance(rater_id, str) or not rater_id.strip():
            raise EvaluationError("Each rating requires a nonempty pseudonymous rater_id.")
        rater_id = rater_id.strip()
        identity = (rater_id, assignment_id)
        if identity in seen:
            raise EvaluationError(f"Duplicate rating by {rater_id} for {assignment_id}")
        seen.add(identity)
        assignment = assignments[assignment_id]
        if "image_id" in row and row["image_id"] not in (assignment["image_id"], _safe_csv(assignment["image_id"])):
            raise EvaluationError("A rating row has a mismatched image_id.")
        caption = row.get("caption")
        if caption is not None and not (
            isinstance(caption, str)
            and (_digest(caption) == assignment["caption_sha256"]
                 or (caption.startswith("'") and _digest(caption[1:]) == assignment["caption_sha256"]))
        ):
            raise EvaluationError("A rating row contains an altered caption.")
        normalized.append({"assignment_id": assignment_id, "image_id": assignment["image_id"],
                           "model": assignment["model"], "rater_id": rater_id,
                           **{dimension: _rating(row.get(dimension), dimension) for dimension in DIMENSIONS}})
    if not normalized:
        raise EvaluationError("No completed ratings were provided.")

    by_caption: dict[str, list[Any]] = defaultdict(list)
    by_rater: dict[str, list[Any]] = defaultdict(list)
    for row in normalized:
        by_caption[row["assignment_id"]].append(row)
        by_rater[row["rater_id"]].append(row)
    captions = []
    for assignment_id, rows in sorted(by_caption.items()):
        first = rows[0]
        captions.append({"assignment_id": assignment_id, "image_id": first["image_id"],
                         "model": first["model"], "rater_count": len(rows),
                         **{dimension: statistics.fmean(r[dimension] for r in rows) for dimension in DIMENSIONS}})

    def correlations(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        answer = {}
        for metric in AUTOMATIC_METRICS:
            answer[metric] = {}
            for dimension in DIMENSIONS:
                pairs = []
                for row in rows:
                    automatic = metrics[(row["image_id"], row["model"])].get(metric)
                    if automatic is not None:
                        if isinstance(automatic, bool) or not isinstance(automatic, (int, float)):
                            raise EvaluationError("Automatic metric scores must be numeric or null.")
                        pairs.append((row["image_id"], float(automatic), float(row[dimension])))
                answer[metric][dimension] = spearman_summary(pairs, bootstrap_samples, seed)
        return answer

    models = sorted({a["model"] for a in assignments.values()})
    per_model = {}
    for model in models:
        rows = [row for row in captions if row["model"] == model]
        expected = sum(a["model"] == model for a in assignments.values())
        per_model[model] = {"rated_caption_count": len(rows), "expected_caption_count": expected,
                            "complete": len(rows) == expected,
                            **{f"mean_{d}": statistics.fmean(r[d] for r in rows) if rows else None for d in DIMENSIONS},
                            "correlations": correlations(rows)}
    common_images = set.intersection(*[{row["image_id"] for row in captions if row["model"] == m} for m in models])
    return {
        "schema": "captionbench-human-analysis/v1", "rating_count": len(normalized),
        "rater_count": len(by_rater), "rated_caption_count": len(captions),
        "expected_caption_count": len(assignments), "complete": len(captions) == len(assignments),
        "paired_image_count": len(common_images), "per_caption": captions, "per_model": per_model,
        "aggregate_correlations": correlations(captions),
        "per_rater": {rater: {"rated_caption_count": len(rows), "expected_caption_count": len(assignments),
                               "complete": len(rows) == len(assignments), "correlations": correlations(rows)}
                      for rater, rows in sorted(by_rater.items())},
        "provenance": {
            "scipy_version": importlib.metadata.version("scipy"), "seed": seed,
            "bootstrap_samples": bootstrap_samples, "aggregation": "equal-weight mean across raters for each caption; equal-weight mean across rated captions for each model",
            "correlation": "scipy.stats.spearmanr with tied ranks; fewer than 3 captions or constant input gives null",
            "uncertainty": "95% percentile bootstrap resamples whole images, keeping model captions together; conditional on observed raters; at least 90% valid resamples and at least 20 required",
            "p_values": "Omitted. Captions share images and raters; independent-observation significance is not assumed.",
            "limits": "Partial forms can change model coverage. Pooled correlations can include between-model effects; per-model correlations are also reported. No inter-rater reliability statistic is estimated.",
        },
    }
