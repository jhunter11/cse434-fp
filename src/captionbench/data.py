"""Local manifests and source-preserving Flickr8k preparation."""

import csv
import hashlib
import json
import os
from pathlib import Path
from collections import Counter, defaultdict


SPLITS = ("train", "val", "test")


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path):
    rows = []
    with Path(path).open(encoding="utf-8-sig") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{number}: {exc.msg}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected a JSON object at {path}:{number}")
            rows.append(row)
    if not rows:
        raise ValueError(f"No records in {path}")
    return rows


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")


def load_manifest(path, expected_references=5, require_images=True):
    """Validate records and resolve image paths relative to the manifest.

    Relative paths must remain inside the manifest's directory. Use an explicit
    absolute path for a local image directory located elsewhere.
    """
    path = Path(path).resolve()
    rows = read_jsonl(path)
    ids, image_paths = set(), set()
    validated = []
    for index, row in enumerate(rows, 1):
        label = f"record {index}"
        image_id = row.get("image_id")
        if not isinstance(image_id, str) or not image_id.strip():
            raise ValueError(f"{label}: image_id must be a nonempty string")
        if image_id in ids:
            raise ValueError(f"{label}: duplicate image_id {image_id!r}; split leakage is possible")
        ids.add(image_id)
        if row.get("split") not in SPLITS:
            raise ValueError(f"{label}: split must be train, val, or test")
        references = row.get("references")
        if not isinstance(references, list) or len(references) != expected_references:
            raise ValueError(f"{label}: exactly {expected_references} references are required")
        if any(not isinstance(ref, str) or not ref.strip() for ref in references):
            raise ValueError(f"{label}: every reference must contain text")
        value = row.get("image_path")
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{label}: image_path must be a nonempty string")
        image = Path(value).expanduser()
        if not image.is_absolute():
            image = (path.parent / image).resolve()
            if not image.is_relative_to(path.parent):
                raise ValueError(f"{label}: relative image paths cannot escape the manifest directory")
        else:
            image = image.resolve()
        normalized_path = os.path.normcase(str(image))
        if normalized_path in image_paths:
            raise ValueError(f"{label}: duplicate image_path {value!r}; split leakage is possible")
        image_paths.add(normalized_path)
        if require_images and not image.is_file():
            raise ValueError(f"{label}: image does not exist: {image}")
        expected_hash = row.get("image_sha256")
        if expected_hash is not None:
            if not isinstance(expected_hash, str) or len(expected_hash) != 64 or any(
                character not in "0123456789abcdef" for character in expected_hash
            ):
                raise ValueError(f"{label}: image_sha256 must contain 64 lowercase hexadecimal characters")
            if require_images and sha256_file(image) != expected_hash:
                raise ValueError(f"{label}: image_sha256 mismatch for {image_id}")
        validated.append({**row, "image_path": str(image), "references": [ref.strip() for ref in references]})
    return validated


def validate_manifest(path, expected_references=5, require_images=True):
    rows = load_manifest(path, expected_references, require_images)
    counts = Counter(row["split"] for row in rows)
    return {
        "record_count": len(rows), "split_counts": {split: counts[split] for split in SPLITS},
        "references_per_image": expected_references, "images_checked": require_images,
        "manifest_sha256": sha256_file(path),
    }


def _caption_map(path):
    path = Path(path)
    with path.open(encoding="utf-8-sig", newline="") as handle:
        first = handle.readline()
        handle.seek(0)
        captions = defaultdict(list)
        if first.strip().lower().replace("\"", "").startswith("image,caption"):
            reader = csv.DictReader(handle)
            for row in reader:
                name, caption = row.get("image", "").strip(), row.get("caption", "").strip()
                if not name or not caption:
                    raise ValueError("CSV caption rows need image and caption text")
                captions[name].append(caption)
        else:
            indexed = defaultdict(dict)
            for number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    token, caption = line.rstrip("\r\n").split("\t", 1)
                    name, index = token.rsplit("#", 1)
                    index = int(index)
                except ValueError as exc:
                    raise ValueError(f"Invalid legacy token row {number}; expected filename#N<TAB>caption") from exc
                if index in indexed[name]:
                    raise ValueError(f"Duplicate reference index for {name}")
                if not caption.strip():
                    raise ValueError(f"Empty reference for {name}")
                indexed[name][index] = caption.strip()
            captions.update({name: [items[key] for key in sorted(items)] for name, items in indexed.items()})
    return dict(captions)


def _split_names(path):
    names = [line.strip() for line in Path(path).read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    if not names:
        raise ValueError(f"Empty split file: {path}")
    if len(names) != len(set(names)):
        raise ValueError(f"Duplicate image IDs in split file: {path}")
    for name in names:
        if Path(name).name != name or "/" in name or "\\" in name or name in {".", ".."}:
            raise ValueError(f"Split image IDs must be filenames: {name!r}")
    return names


def prepare_flickr8k(images_dir, captions_file, train_split, val_split, test_split,
                    output_manifest, expected_references=5):
    """Prepare supplied local data using the supplied original split membership.

    No dataset is downloaded. Source hashes record which split files were used;
    a hash alone does not authenticate a mirror or grant a redistribution right.
    """
    images_dir = Path(images_dir).resolve()
    output_manifest = Path(output_manifest).resolve()
    split_files = {"train": Path(train_split), "val": Path(val_split), "test": Path(test_split)}
    captions = _caption_map(captions_file)
    assigned = {}
    for split, source in split_files.items():
        for name in _split_names(source):
            if name in assigned:
                raise ValueError(f"Split overlap for {name}: {assigned[name]} and {split}")
            assigned[name] = split
    rows = []
    for name, split in sorted(assigned.items()):
        if name not in captions or len(captions[name]) != expected_references:
            raise ValueError(f"{name} needs exactly {expected_references} references")
        image = (images_dir / name).resolve()
        if not image.is_relative_to(images_dir) or not image.is_file():
            raise ValueError(f"image does not exist inside images directory: {name}")
        rows.append({"image_id": name, "image_path": str(image), "image_sha256": sha256_file(image),
                     "split": split, "references": captions[name]})
    write_jsonl(output_manifest, rows)
    summary = validate_manifest(output_manifest, expected_references)
    source_files = {"captions": Path(captions_file), **{f"{key}_split": path for key, path in split_files.items()}}
    summary.update({
        "dataset": "Flickr8k local source files",
        "split_protocol": "supplied original train/dev/test membership; dev normalized to val",
        "source_authenticity": "user-supplied; hashes identify inputs but do not authenticate origin",
        "unused_caption_images": len(set(captions) - set(assigned)),
        "sources": {name: {"filename": source.name, "sha256": sha256_file(source)} for name, source in source_files.items()},
    })
    output_manifest.with_suffix(".metadata.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary
