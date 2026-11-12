import importlib.util
import json
from pathlib import Path

import pytest
from PIL import Image

from captionbench.data import write_jsonl

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "grounding_gap.py"
SUBJECTS = ["a black dog runs on grass", "two children climb a wall", "a man rides a red bike",
            "a woman holds a white cup", "a boat sails near the shore", "a bird sits on a wire"]


def load_script():
    spec = importlib.util.spec_from_file_location("grounding_gap", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def workspace(tmp_path):
    rows, grounded, constant = [], [], []
    for index, subject in enumerate(SUBJECTS):
        path = tmp_path / f"{index}.png"
        Image.new("RGB", (8, 8), (index * 30, 0, 0)).save(path)
        image_id = f"{index}.png"
        rows.append({"image_id": image_id, "image_path": str(path), "split": "val",
                     "references": [f"{subject} today", subject, f"{subject} outside", subject, subject]})
        grounded.append({"image_id": image_id, "model": "grounded", "caption": subject})
        constant.append({"image_id": image_id, "model": "constant", "caption": "a black dog runs on grass"})
    write_jsonl(tmp_path / "manifest.jsonl", rows)
    write_jsonl(tmp_path / "grounded.jsonl", grounded)
    write_jsonl(tmp_path / "constant.jsonl", constant)
    (tmp_path / "ids.json").write_text(json.dumps([row["image_id"] for row in rows]), encoding="utf-8")
    return tmp_path


def run(root, *extra):
    load_script().main(["--manifest", str(root / "manifest.jsonl"), "--image-ids", str(root / "ids.json"),
                        "--predictions", str(root / "grounded.jsonl"), str(root / "constant.jsonl"),
                        "--no-clipscore", "--no-meteor", "--bootstrap-samples", "50",
                        "--output", str(root / "gap.json"), *extra])
    return json.loads((root / "gap.json").read_text(encoding="utf-8"))


def test_grounded_captions_lose_their_score_on_other_images(tmp_path, capsys):
    result = run(workspace(tmp_path))
    grounded = result["models"]["grounded"]["metrics"]
    for metric in ("bleu4_corpus", "bleu4_sentence", "cider_d"):
        assert grounded[metric]["matched"] > grounded[metric]["mismatched"]
        assert grounded[metric]["blind_share"] < 0.5
        assert grounded[metric]["gap"] == pytest.approx(grounded[metric]["matched"] - grounded[metric]["mismatched"])
    assert "meteor" not in grounded and "clipscore" not in grounded


def test_a_constant_caption_has_no_grounding_gap(tmp_path, capsys):
    result = run(workspace(tmp_path))
    constant = result["models"]["constant"]["metrics"]
    for metric in ("bleu4_sentence", "cider_d"):
        assert constant[metric]["gap"] == pytest.approx(0.0, abs=1e-12)
        assert constant[metric]["blind_share"] == pytest.approx(1.0)


def test_output_holds_no_caption_text(tmp_path, capsys):
    root = workspace(tmp_path)
    run(root)
    text = (root / "gap.json").read_text(encoding="utf-8")
    assert all(subject not in text for subject in SUBJECTS)


def test_donor_maps_replace_the_shifts_and_reject_fixed_points(tmp_path, capsys):
    root = workspace(tmp_path)
    ids = json.loads((root / "ids.json").read_text(encoding="utf-8"))
    (root / "swap.json").write_text(json.dumps(dict(zip(ids, ids[1:] + ids[:1], strict=True))), encoding="utf-8")
    result = run(root, "--donor-maps", str(root / "swap.json"))
    assert result["shifts"] is None and result["donor_maps"] == ["swap.json"]
    assert result["models"]["grounded"]["metrics"]["cider_d"]["blind_share"] < 0.5
    (root / "bad.json").write_text(json.dumps(dict(zip(ids, ids, strict=True))), encoding="utf-8")
    with pytest.raises(ValueError, match="no fixed point"):
        run(root, "--donor-maps", str(root / "bad.json"))
