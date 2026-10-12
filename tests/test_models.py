import json
import sys
from types import SimpleNamespace

import pytest
from PIL import Image


def test_generic_is_frozen_nonvisual_negative_control(tmp_path):
    from captionbench.models import caption_image
    red, blue = tmp_path / "red.png", tmp_path / "blue.png"
    Image.new("RGB", (8, 8), "red").save(red)
    Image.new("RGB", (8, 8), "blue").save(blue)
    one = caption_image(red, model_key="generic")
    two = caption_image(blue, model_key="generic")
    assert one["caption"] == two["caption"] == "a photo of a person in a scene"
    assert one["visual_input"] is False
    assert one["model_id"] == "captionbench/generic-negative-control"
    assert one["revision"] == "v1"
    assert one["latency_seconds"] >= 0


def test_invalid_model_is_rejected_without_download(tmp_path):
    from captionbench.models import caption_image
    with pytest.raises(ValueError, match="Unknown model"):
        caption_image(tmp_path / "missing.png", model_key="gpt-paid")


def test_missing_image_is_rejected_before_model_load(tmp_path):
    from captionbench.models import caption_image
    with pytest.raises(FileNotFoundError):
        caption_image(tmp_path / "missing.png", model_key="blip2-opt-2.7b")


def test_generate_predictions_filters_split_and_limit(tmp_path):
    from captionbench.models import generate_predictions
    image = tmp_path / "square.png"
    Image.new("RGB", (8, 8), "red").save(image)
    rows = [{"image_id": str(i), "image_path": str(image), "split": split}
            for i, split in enumerate(["train", "test", "test"])]
    result = generate_predictions(rows, model_key="generic", split="test", limit=1)
    assert [row["image_id"] for row in result] == ["1"]
    assert result[0]["model"] == "generic"
    assert result[0]["visual_input"] is False


def test_generate_predictions_records_actual_image_sha256(tmp_path):
    from captionbench.data import sha256_file
    from captionbench.models import generate_predictions
    image = tmp_path / "square.png"
    Image.new("RGB", (8, 8), "red").save(image)
    rows = [{"image_id": "square", "image_path": str(image), "split": "test"}]
    result = generate_predictions(rows, model_key="generic")
    assert result[0]["image_sha256"] == sha256_file(image)


def test_generate_rejects_empty_selection_and_nonpositive_limit():
    from captionbench.models import generate_predictions
    with pytest.raises(ValueError, match="No images"):
        generate_predictions([], model_key="generic")
    with pytest.raises(ValueError, match="positive"):
        generate_predictions([], model_key="generic", limit=0)


def run_records(tmp_path):
    rows = []
    for name, color in [("one.png", "red"), ("two.png", "blue")]:
        path = tmp_path / name
        Image.new("RGB", (8, 8), color).save(path)
        rows.append({"image_id": name, "image_path": str(path), "split": "test"})
    return rows


def test_incremental_generation_survives_interrupt_and_resumes_verified_rows(tmp_path, monkeypatch):
    from captionbench import models
    from captionbench.data import read_jsonl
    rows = run_records(tmp_path)
    output = tmp_path / "predictions.jsonl"
    real_caption = models.caption_image
    calls = []

    def interrupted(path, model_key="generic", device="auto", revision=None):
        calls.append((str(path), revision))
        if len(calls) == 2:
            raise KeyboardInterrupt("simulated Colab disconnect")
        return real_caption(path, model_key=model_key, device=device, revision=revision)

    monkeypatch.setattr(models, "caption_image", interrupted)
    with pytest.raises(KeyboardInterrupt):
        models.generate_to_file(rows, output)
    saved = read_jsonl(output)
    assert [row["image_id"] for row in saved] == ["one.png"]
    first_bytes = output.read_bytes()
    calls.clear()

    def resumed(path, model_key="generic", device="auto", revision=None):
        calls.append((str(path), revision))
        return real_caption(path, model_key=model_key, device=device, revision=revision)

    monkeypatch.setattr(models, "caption_image", resumed)
    result = models.generate_to_file(rows, output)
    assert [row["image_id"] for row in result] == ["one.png", "two.png"]
    assert calls == [(rows[1]["image_path"], "v1")]
    assert output.read_bytes().startswith(first_bytes)
    assert len(read_jsonl(output)) == 2


@pytest.mark.parametrize("field,value", [
    ("image_sha256", "0" * 64), ("revision", "wrong"),
    ("generation", {"num_beams": 9}), ("model_id", "wrong/model"),
    ("device_name", "different hardware"), ("caption", " "), ("model", "blip-base"),
    ("precision", "float16"), ("torch_version", "different version"), ("image_id", []),
])
def test_resume_rejects_changed_data_revision_or_model_condition(tmp_path, field, value):
    from captionbench.models import generate_to_file
    rows = run_records(tmp_path)
    output = tmp_path / "predictions.jsonl"
    saved = generate_to_file(rows, output, limit=1)
    saved[0][field] = value
    output.write_text(json.dumps(saved[0]) + "\n", encoding="utf-8")
    before = output.read_bytes()
    with pytest.raises(ValueError):
        generate_to_file(rows, output)
    assert output.read_bytes() == before


def test_resume_rejects_image_bytes_changed_after_first_run(tmp_path):
    from captionbench.models import generate_to_file
    rows = run_records(tmp_path)
    output = tmp_path / "predictions.jsonl"
    generate_to_file(rows, output, limit=1)
    Image.new("RGB", (8, 8), "green").save(rows[0]["image_path"])
    with pytest.raises(ValueError, match="hash"):
        generate_to_file(rows, output)


@pytest.mark.parametrize("corruption", ["duplicate", "unknown", "truncated"])
def test_resume_rejects_corrupt_or_unknown_saved_records(tmp_path, corruption):
    from captionbench.models import generate_to_file
    rows = run_records(tmp_path)
    output = tmp_path / "predictions.jsonl"
    saved = generate_to_file(rows, output, limit=1)
    if corruption == "duplicate":
        output.write_text((json.dumps(saved[0]) + "\n") * 2, encoding="utf-8")
    elif corruption == "unknown":
        saved[0]["image_id"] = "unknown.png"
        output.write_text(json.dumps(saved[0]) + "\n", encoding="utf-8")
    else:
        with output.open("a", encoding="utf-8") as handle:
            handle.write('{"image_id":')
    before = output.read_bytes()
    with pytest.raises(ValueError):
        generate_to_file(rows, output)
    assert output.read_bytes() == before


def test_no_resume_refuses_existing_file(tmp_path):
    from captionbench.models import generate_to_file
    rows = run_records(tmp_path)
    output = tmp_path / "predictions.jsonl"
    generate_to_file(rows, output)
    before = output.read_bytes()
    with pytest.raises(FileExistsError):
        generate_to_file(rows, output, resume=False)
    assert output.read_bytes() == before


def test_resume_pins_saved_real_model_revision_without_downloading(tmp_path, monkeypatch):
    from captionbench import models
    rows = run_records(tmp_path)
    output = tmp_path / "predictions.jsonl"
    sha = "a" * 40
    calls = []

    def condition(model_key, device, revision):
        return {"model_id": models.MODEL_IDS[model_key], "revision": revision, "precision": "float32",
                "generation": dict(models.GENERATION), "device": "cpu", "device_name": "test CPU",
                "latency_scope": models.VISUAL_LATENCY_SCOPE, "torch_version": "test", "transformers_version": "test"}

    def fake_caption(path, model_key, device, revision):
        calls.append(revision)
        return {"caption": "a colored shape", "latency_seconds": 0.1, **condition(model_key, device, sha)}

    monkeypatch.setattr(models, "_runtime_condition", condition)
    monkeypatch.setattr(models, "caption_image", fake_caption)
    models.generate_to_file(rows, output, model_key="blip-base", limit=1)
    result = models.generate_to_file(rows, output, model_key="blip-base")
    assert calls == [None, sha]
    assert len(result) == 2
    with pytest.raises(ValueError, match="revision differs"):
        models.generate_to_file(rows, output, model_key="blip-base", revision="b" * 40)
    assert calls == [None, sha]


def test_release_models_clears_cached_models_and_initialized_cuda(monkeypatch):
    from captionbench import models
    actions = []
    monkeypatch.setattr(models, "_load_model", SimpleNamespace(cache_clear=lambda: actions.append("clear")))
    fake_torch = SimpleNamespace(cuda=SimpleNamespace(
        is_initialized=lambda: True, empty_cache=lambda: actions.append("empty")))
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    models.release_models()
    assert actions == ["clear", "empty"]
