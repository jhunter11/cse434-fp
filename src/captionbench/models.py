"""Frozen captioners. Heavy model dependencies load only on an explicit request."""

from functools import lru_cache
import gc
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import re
import sys
from time import perf_counter

from .data import read_jsonl, sha256_file


MODEL_IDS = {
    "generic": "captionbench/generic-negative-control",
    "blip-base": "Salesforce/blip-image-captioning-base",
    "blip-large": "Salesforce/blip-image-captioning-large",
    "blip2-opt-2.7b": "Salesforce/blip2-opt-2.7b",
}
GENERATION = {"max_new_tokens": 32, "num_beams": 3, "do_sample": False}
GENERIC_CAPTION = "a photo of a person in a scene"
GENERIC_LATENCY_SCOPE = "constant caption lookup; no image processing"
VISUAL_LATENCY_SCOPE = "preprocessing + generation + decoding; excludes model load and image I/O"


def release_models():
    """Release cached models between Colab runs without importing Torch."""
    _load_model.cache_clear()
    gc.collect()
    torch = sys.modules.get("torch")
    if torch is not None and torch.cuda.is_initialized():
        torch.cuda.empty_cache()


def _device(torch, requested):
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested not in {"cpu", "cuda"}:
        raise ValueError("device must be auto, cpu, or cuda")
    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    return requested


@lru_cache(maxsize=3)
def _load_model(model_key, device, revision):
    import torch
    from transformers import (
        AutoConfig, BlipProcessor, BlipForConditionalGeneration,
        Blip2Processor, Blip2ForConditionalGeneration,
    )

    model_id = MODEL_IDS[model_key]
    config = AutoConfig.from_pretrained(model_id, revision=revision, trust_remote_code=False)
    resolved = getattr(config, "_commit_hash", None)
    if not resolved and revision and re.fullmatch(r"[0-9a-f]{40}", revision):
        resolved = revision
    if not resolved:
        raise RuntimeError("Could not resolve the model commit; pass a 40-character revision")
    is_blip2 = model_key == "blip2-opt-2.7b"
    processor_class = Blip2Processor if is_blip2 else BlipProcessor
    model_class = Blip2ForConditionalGeneration if is_blip2 else BlipForConditionalGeneration
    dtype = torch.float16 if is_blip2 and device == "cuda" else torch.float32
    processor = processor_class.from_pretrained(model_id, revision=resolved, trust_remote_code=False)
    model = model_class.from_pretrained(
        model_id, revision=resolved, config=config, dtype=dtype, trust_remote_code=False,
    ).to(device)
    model.eval()
    return processor, model, resolved, dtype


def caption_image(image_path, model_key="blip-base", device="auto", revision=None):
    """Caption a local image; latency excludes image I/O and model loading.

    A first explicit BLIP request may download public weights into the Hugging
    Face cache. The generic comparator reads no pixels and performs no network
    request. BLIP-2 and BLIP-large load only when selected by name.
    """
    if model_key not in MODEL_IDS:
        raise ValueError(f"Unknown model {model_key!r}; choose one of {list(MODEL_IDS)}")
    path = Path(image_path)
    if not path.is_file():
        raise FileNotFoundError(f"Image does not exist: {path}")
    if device not in {"auto", "cpu", "cuda"}:
        raise ValueError("device must be auto, cpu, or cuda")
    if model_key == "generic":
        if revision not in {None, "v1"}:
            raise ValueError("generic negative control only supports revision v1")
        started = perf_counter()
        caption = GENERIC_CAPTION
        return {
            "caption": caption, "model_key": model_key, "model_id": MODEL_IDS[model_key],
            "revision": "v1", "requested_revision": revision,
            "latency_seconds": perf_counter() - started, "device": "cpu",
            "device_name": "CPU", "visual_input": False,
            "baseline_type": "frozen nonvisual negative control",
            "latency_scope": GENERIC_LATENCY_SCOPE,
            "generation": {}, "precision": "not applicable",
        }

    import torch
    from PIL import Image, ImageOps

    actual_device = _device(torch, device)
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
    processor, model, resolved, dtype = _load_model(model_key, actual_device, revision)
    if actual_device == "cuda":
        torch.cuda.synchronize()
    started = perf_counter()
    inputs = processor(images=image, return_tensors="pt").to(actual_device)
    if model_key == "blip2-opt-2.7b":
        # Cast pixels only. Token IDs must remain integers.
        inputs["pixel_values"] = inputs["pixel_values"].to(dtype=dtype)
    with torch.inference_mode():
        generated = model.generate(**inputs, **GENERATION)
    caption = processor.decode(generated[0], skip_special_tokens=True).strip()
    if actual_device == "cuda":
        torch.cuda.synchronize()
    latency = perf_counter() - started
    if not caption:
        raise RuntimeError("The model produced an empty caption")
    return {
        "caption": caption, "model_key": model_key, "model_id": MODEL_IDS[model_key],
        "revision": resolved, "requested_revision": revision,
        "latency_seconds": latency, "device": actual_device,
        "device_name": torch.cuda.get_device_name(0) if actual_device == "cuda" else "CPU",
        "visual_input": True, "baseline_type": "pretrained frozen captioner",
        "latency_scope": VISUAL_LATENCY_SCOPE,
        "generation": dict(GENERATION), "precision": str(dtype).removeprefix("torch."),
        "torch_version": version("torch"), "transformers_version": version("transformers"),
    }


def generate_predictions(records, model_key="generic", split="test", limit=None,
                         device="auto", revision=None):
    """Generate one caption per selected manifest row in source order."""
    selected = _select_records(records, split, limit)
    return [_prediction_for_row(row, model_key, device, revision) for row in selected]


def _select_records(records, split, limit):
    if limit is not None and (not isinstance(limit, int) or isinstance(limit, bool) or limit <= 0):
        raise ValueError("limit must be a positive integer")
    selected = [row for row in records if row["split"] == split]
    if limit is not None:
        selected = selected[:limit]
    if not selected:
        raise ValueError(f"No images selected for split {split!r}")
    return selected


def _prediction_for_row(row, model_key, device, revision):
    image_hash = sha256_file(row["image_path"])
    if row.get("image_sha256") is not None and row["image_sha256"] != image_hash:
        raise ValueError(f"image_sha256 mismatch for {row['image_id']}")
    result = caption_image(row["image_path"], model_key=model_key, device=device, revision=revision)
    if sha256_file(row["image_path"]) != image_hash:
        raise ValueError(f"Image bytes changed during inference for {row['image_id']}")
    return {"image_id": row["image_id"], "model": model_key, **result, "image_sha256": image_hash}


def _runtime_condition(model_key, device, revision):
    if model_key == "generic":
        return {"model_id": MODEL_IDS[model_key], "revision": "v1", "precision": "not applicable",
                "generation": {}, "device": "cpu", "device_name": "CPU", "latency_scope": GENERIC_LATENCY_SCOPE}
    import torch
    actual_device = _device(torch, device)
    precision = "float16" if model_key == "blip2-opt-2.7b" and actual_device == "cuda" else "float32"
    return {"model_id": MODEL_IDS[model_key], "revision": revision, "precision": precision,
            "generation": dict(GENERATION), "device": actual_device,
            "device_name": torch.cuda.get_device_name(0) if actual_device == "cuda" else "CPU",
            "latency_scope": VISUAL_LATENCY_SCOPE,
            "torch_version": version("torch"), "transformers_version": version("transformers")}


def _validate_saved_prediction(row, selected_by_id, model_key, image_hashes):
    image_id = row.get("image_id")
    if not isinstance(image_id, str) or image_id not in selected_by_id:
        raise ValueError(f"Saved prediction has an unknown or unselected image_id: {image_id!r}")
    if row.get("model") != model_key:
        raise ValueError(f"Saved prediction model differs from {model_key}: {image_id}")
    if not isinstance(row.get("caption"), str) or not row["caption"].strip():
        raise ValueError(f"Saved prediction has an empty caption: {image_id}")
    if row.get("image_sha256") != image_hashes[image_id]:
        raise ValueError(f"Saved prediction image hash mismatch: {image_id}")
    revision = row.get("revision")
    valid_revision = revision == "v1" if model_key == "generic" else (
        isinstance(revision, str) and re.fullmatch(r"[0-9a-f]{40}", revision)
    )
    if not valid_revision:
        raise ValueError("Prediction revision must be an immutable supported revision")
    latency = row.get("latency_seconds")
    if isinstance(latency, bool) or not isinstance(latency, (int, float)) or not math.isfinite(latency) or latency < 0:
        raise ValueError(f"Saved prediction latency must be finite and nonnegative: {image_id}")


def generate_to_file(records, output_path, model_key="generic", split="test", limit=None,
                     device="auto", revision=None, resume=True):
    """Save each caption before continuing; resume only an unchanged condition.

    Use one writer per file. A damaged final JSONL record raises an error. Copy
    that file before repairing it. New hardware or model settings need a new
    output file. Flushing a mounted Drive file does not confirm cloud sync.
    """
    from .evaluation import model_condition

    if model_key not in MODEL_IDS:
        raise ValueError(f"Unknown model {model_key!r}")
    if device not in {"auto", "cpu", "cuda"}:
        raise ValueError("device must be auto, cpu, or cuda")
    selected = _select_records(records, split, limit)
    if any(not isinstance(row.get("image_id"), str) or not row["image_id"].strip() for row in selected):
        raise ValueError("Selected records need nonempty string image_id values")
    selected_by_id = {row["image_id"]: row for row in selected}
    if len(selected_by_id) != len(selected):
        raise ValueError("Selected records contain duplicate image_id values")
    output = Path(output_path)
    exists = output.exists()
    if exists and not resume:
        raise FileExistsError(f"Prediction file already exists: {output}. Use a new path or enable resume.")
    image_hashes = {row["image_id"]: sha256_file(row["image_path"]) for row in selected}
    for row in selected:
        if row.get("image_sha256") is not None and row["image_sha256"] != image_hashes[row["image_id"]]:
            raise ValueError(f"Manifest image hash mismatch: {row['image_id']}")
    saved = []
    if exists and output.stat().st_size:
        saved = read_jsonl(output)
        with output.open("rb") as source:
            source.seek(-1, os.SEEK_END)
            if source.read(1) != b"\n":
                raise ValueError("Prediction file lacks its final newline. Inspect the last record before resuming.")
    by_id = {}
    for row in saved:
        _validate_saved_prediction(row, selected_by_id, model_key, image_hashes)
        if row["image_id"] in by_id:
            raise ValueError(f"Duplicate saved prediction image_id: {row['image_id']}")
        by_id[row["image_id"]] = row
    inference_revision = revision
    expected_condition = None
    if saved:
        saved_revision = saved[0].get("revision")
        if revision is not None and revision != saved_revision:
            raise ValueError("Requested revision differs from the saved revision. Use a new output file.")
        inference_revision = saved_revision
        expected_condition = _runtime_condition(model_key, device, saved_revision)
        model_condition(dict(enumerate([expected_condition, *saved])), model_key)
    if len(by_id) == len(selected):
        return [by_id[row["image_id"]] for row in selected]
    output.parent.mkdir(parents=True, exist_ok=True)
    expected_size = output.stat().st_size if exists else 0
    with output.open("a" if exists else "x", encoding="utf-8", newline="\n") as destination:
        for row in selected:
            if row["image_id"] in by_id:
                continue
            prediction = _prediction_for_row(row, model_key, device, inference_revision)
            _validate_saved_prediction(prediction, selected_by_id, model_key, image_hashes)
            if expected_condition is None:
                expected_condition = _runtime_condition(model_key, device, prediction.get("revision"))
            model_condition({0: expected_condition, 1: prediction}, model_key)
            if output.stat().st_size != expected_size:
                raise ValueError("Prediction file changed during generation. Stop other writers and inspect the file.")
            destination.write(json.dumps(prediction, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n")
            destination.flush()
            os.fsync(destination.fileno())
            expected_size = output.stat().st_size
            by_id[row["image_id"]] = prediction
    return [by_id[row["image_id"]] for row in selected]
