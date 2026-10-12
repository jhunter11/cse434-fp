import json
import pytest
from PIL import Image


def record(tmp_path, image_id="one.png", split="test"):
    Image.new("RGB", (16, 16), "red").save(tmp_path / image_id)
    return {"image_id": image_id, "image_path": image_id, "split": split,
            "references": [f"a red square description {i}" for i in range(5)]}


def write_manifest(tmp_path, rows):
    path = tmp_path / "manifest.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def test_load_manifest_resolves_paths_and_preserves_five_references(tmp_path):
    from captionbench.data import load_manifest
    row = record(tmp_path)
    loaded = load_manifest(write_manifest(tmp_path, [row]))
    assert loaded[0]["image_path"] == str((tmp_path / "one.png").resolve())
    assert loaded[0]["references"] == row["references"]


def test_duplicate_ids_across_splits_are_rejected(tmp_path):
    from captionbench.data import load_manifest
    first = record(tmp_path)
    second = {**first, "split": "train"}
    with pytest.raises(ValueError, match="duplicate image_id"):
        load_manifest(write_manifest(tmp_path, [first, second]))


@pytest.mark.parametrize("references", [[], ["one"], ["one", "two", "three", "four", " "]])
def test_missing_or_empty_references_are_rejected(tmp_path, references):
    from captionbench.data import load_manifest
    row = {**record(tmp_path), "references": references}
    with pytest.raises(ValueError, match="reference"):
        load_manifest(write_manifest(tmp_path, [row]))


def test_missing_image_is_rejected(tmp_path):
    from captionbench.data import load_manifest
    row = record(tmp_path)
    (tmp_path / "one.png").unlink()
    with pytest.raises(ValueError, match="image.*exist"):
        load_manifest(write_manifest(tmp_path, [row]))


def test_relative_path_escape_rejected_but_explicit_external_path_allowed(tmp_path):
    from captionbench.data import load_manifest
    row = record(tmp_path)
    row["image_path"] = "../outside.png"
    with pytest.raises(ValueError, match="relative.*escape"):
        load_manifest(write_manifest(tmp_path, [row]), require_images=False)
    external = tmp_path / "external"
    external.mkdir()
    Image.new("RGB", (8, 8), "blue").save(external / "blue.png")
    row["image_path"] = str((external / "blue.png").resolve())
    assert load_manifest(write_manifest(tmp_path, [row]))[0]["image_path"] == row["image_path"]


def test_same_path_under_two_ids_is_rejected_as_leakage(tmp_path):
    from captionbench.data import load_manifest
    first = record(tmp_path)
    second = {**first, "image_id": "alias", "split": "train"}
    with pytest.raises(ValueError, match="duplicate image_path"):
        load_manifest(write_manifest(tmp_path, [first, second]))


def test_manifest_hash_detects_changed_image_bytes(tmp_path):
    from captionbench.data import load_manifest, sha256_file
    row = record(tmp_path)
    row["image_sha256"] = sha256_file(tmp_path / "one.png")
    manifest = write_manifest(tmp_path, [row])
    assert load_manifest(manifest)[0]["image_sha256"] == row["image_sha256"]
    Image.new("RGB", (16, 16), "blue").save(tmp_path / "one.png")
    with pytest.raises(ValueError, match="image_sha256 mismatch"):
        load_manifest(manifest)


def flickr_sources(tmp_path, legacy=False):
    images = tmp_path / "images"
    images.mkdir()
    for name in ["a.jpg", "b.jpg", "c.jpg"]:
        Image.new("RGB", (8, 8), "red").save(images / name)
    captions = tmp_path / ("tokens.txt" if legacy else "captions.txt")
    if legacy:
        text = "".join(f"{name}#{i}\ta red object number {i}\n" for name in ["a.jpg", "b.jpg", "c.jpg"] for i in range(5))
    else:
        text = "image,caption\n" + "".join(f'{name},"a red object, number {i}"\n' for name in ["a.jpg", "b.jpg", "c.jpg"] for i in range(5))
    captions.write_text(text, encoding="utf-8")
    splits = []
    for split, name in [("train", "a.jpg"), ("val", "b.jpg"), ("test", "c.jpg")]:
        path = tmp_path / f"{split}.txt"
        path.write_text(name + "\n", encoding="utf-8")
        splits.append(path)
    return images, captions, splits


@pytest.mark.parametrize("legacy", [False, True])
def test_prepare_flickr_preserves_official_assignments_and_repeatable_hash(tmp_path, legacy):
    from captionbench.data import load_manifest, prepare_flickr8k
    images, captions, splits = flickr_sources(tmp_path, legacy)
    output = tmp_path / "prepared" / "manifest.jsonl"
    report = prepare_flickr8k(images, captions, *splits, output)
    rows = load_manifest(output)
    assert {row["image_id"]: row["split"] for row in rows} == {"a.jpg": "train", "b.jpg": "val", "c.jpg": "test"}
    assert report["split_counts"] == {"train": 1, "val": 1, "test": 1}
    assert len(report["manifest_sha256"]) == 64
    assert report["manifest_sha256"] == prepare_flickr8k(images, captions, *splits, output)["manifest_sha256"]
    assert (output.with_suffix(".metadata.json")).exists()
    assert len(report["sources"]["test_split"]["sha256"]) == 64
    assert all(len(row["image_sha256"]) == 64 for row in rows)


def test_prepare_rejects_overlap_in_supplied_splits(tmp_path):
    from captionbench.data import prepare_flickr8k
    images, captions, splits = flickr_sources(tmp_path)
    splits[2].write_text("a.jpg\n", encoding="utf-8")
    with pytest.raises(ValueError, match="overlap"):
        prepare_flickr8k(images, captions, *splits, tmp_path / "manifest.jsonl")


def test_cli_generate_writes_predictions_without_models_dependency(tmp_path, capsys):
    from captionbench.cli import main
    from captionbench.data import read_jsonl
    manifest = write_manifest(tmp_path, [record(tmp_path)])
    output = tmp_path / "results" / "generic.jsonl"
    assert main(["generate", "--manifest", str(manifest), "--model", "generic", "--output", str(output)]) == 0
    rows = read_jsonl(output)
    assert rows[0]["image_id"] == "one.png"
    assert rows[0]["visual_input"] is False
    assert json.loads(capsys.readouterr().out)["prediction_count"] == 1


def test_cli_rejects_bad_manifest_with_error_exit(tmp_path):
    from captionbench.cli import main
    manifest = write_manifest(tmp_path, [{"image_id": "broken"}])
    with pytest.raises(SystemExit) as exc:
        main(["validate", "--manifest", str(manifest)])
    assert exc.value.code == 2


def test_cli_generate_resumes_by_default_and_no_resume_preserves_existing(tmp_path, capsys):
    from captionbench.cli import main
    manifest = write_manifest(tmp_path, [record(tmp_path)])
    output = tmp_path / "generic.jsonl"
    arguments = ["generate", "--manifest", str(manifest), "--output", str(output)]
    assert main(arguments) == 0
    before = output.read_bytes()
    assert main(arguments) == 0
    assert output.read_bytes() == before
    with pytest.raises(SystemExit) as exc:
        main([*arguments, "--no-resume"])
    assert exc.value.code == 2
    assert output.read_bytes() == before


def test_demo_preflights_existing_rating_packet_before_any_overwrite(tmp_path):
    from captionbench.cli import main
    output = tmp_path / "previous-run"
    ratings = output / "ratings"
    ratings.mkdir(parents=True)
    (ratings / "ratings.csv").write_text("existing human ratings", encoding="utf-8")
    predictions = output / "generic-predictions.jsonl"
    predictions.write_text("existing predictions", encoding="utf-8")
    with pytest.raises(SystemExit) as exc:
        main(["demo-pipeline", "--output", str(output)])
    assert exc.value.code == 2
    assert predictions.read_text(encoding="utf-8") == "existing predictions"
    assert (ratings / "ratings.csv").read_text(encoding="utf-8") == "existing human ratings"
    assert not (output / "synthetic-images").exists()
