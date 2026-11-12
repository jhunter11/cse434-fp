import json

import pytest
from PIL import Image

from captionbench.cli import main
from captionbench.data import write_jsonl


def workspace(tmp_path):
    rows, predictions = [], []
    for index, split in enumerate(["val", "val", "val", "test"]):
        path = tmp_path / f"{index}.png"
        Image.new("RGB", (8, 8), (index * 40, 0, 0)).save(path)
        rows.append({"image_id": f"{index}.png", "image_path": str(path), "split": split,
                     "references": [f"a red square number {index} view {n}" for n in range(5)]})
        predictions.append({"image_id": f"{index}.png", "model": "m", "caption": f"a red square number {index}"})
    write_jsonl(tmp_path / "manifest.jsonl", rows)
    write_jsonl(tmp_path / "predictions.jsonl", predictions)
    return tmp_path


def run_evaluate(root, *extra):
    main(["evaluate", "--manifest", str(root / "manifest.jsonl"), "--predictions",
          str(root / "predictions.jsonl"), "--output", str(root / "out.json"), *extra])
    return json.loads((root / "out.json").read_text(encoding="utf-8"))


def test_evaluate_scores_a_declared_validation_subset(tmp_path, capsys):
    root = workspace(tmp_path)
    (root / "ids.json").write_text(json.dumps(["0.png", "2.png"]), encoding="utf-8")
    result = run_evaluate(root, "--split", "val", "--image-ids", str(root / "ids.json"))
    assert result["provenance"]["split"] == "val"
    assert result["provenance"]["image_ids"] == ["0.png", "2.png"]
    assert result["per_model"]["m"]["image_count"] == 2


def test_evaluate_rejects_subset_ids_outside_the_split(tmp_path, capsys):
    root = workspace(tmp_path)
    (root / "ids.json").write_text(json.dumps(["3.png"]), encoding="utf-8")
    with pytest.raises(SystemExit):
        run_evaluate(root, "--split", "val", "--image-ids", str(root / "ids.json"))
    assert "outside split val" in capsys.readouterr().err


def test_evaluate_default_split_stays_test(tmp_path, capsys):
    root = workspace(tmp_path)
    with pytest.raises(SystemExit):
        run_evaluate(root)
    assert "unknown image_id" in capsys.readouterr().err
