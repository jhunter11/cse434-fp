"""Download Flickr8k into the runtime and write the CaptionBench manifest.

The script needs no account, no API key, and no cloud drive. It downloads the
two public archives, checks their SHA256 values, extracts them, and writes the
manifest with the original train, dev, and test image lists.

The archives come from a public mirror. A matching hash shows that the files
equal the ones this study used. It does not grant a right to redistribute the
photos. Keep the photos out of Git and out of shared outputs.

Example:
    python scripts/fetch_flickr8k.py --root data/flickr8k --manifest data/flickr8k.jsonl
"""

import argparse
import hashlib
import json
import shutil
import sys
import urllib.request
from pathlib import Path
from zipfile import ZipFile

from captionbench.data import prepare_flickr8k

MIRROR = "https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/"
ARCHIVES = {
    "Flickr8k_Dataset.zip": "61e4b111d32b24a55b69dafd91f4c3aec07391b7b9217face15dd35d517fe6de",
    "Flickr8k_text.zip": "4992ddc8110e9aa49da5bf698522b0c8f11c448814a488584ee6bf040e5137e7",
}
MARKER = ".extracted"


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _download(url, path):
    partial = path.with_suffix(path.suffix + ".part")
    with urllib.request.urlopen(url) as response, open(partial, "wb") as target:
        total, done, last = int(response.headers.get("Content-Length", 0)), 0, -1
        while block := response.read(1 << 20):
            target.write(block)
            done += len(block)
            step = done * 10 // total if total else 0
            if step != last:
                print(f"  {path.name}: {done / 2**20:.0f} MiB", flush=True)
                last = step
    partial.replace(path)


def _extract(archive, root):
    with ZipFile(archive) as zipped:
        for member in zipped.infolist():
            if not (root / member.filename).resolve().is_relative_to(root.resolve()):
                raise ValueError(f"{archive.name} contains an unsafe path: {member.filename}")
        zipped.extractall(root)


def _find(root, name):
    matches = sorted(path for path in root.rglob(name) if "__MACOSX" not in path.parts)
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one {name} below {root}, found {len(matches)}")
    return matches[0]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--root", default="data/flickr8k")
    parser.add_argument("--manifest", default="data/flickr8k.jsonl")
    parser.add_argument("--keep-archives", action="store_true", help="Keep the two zip files after extraction")
    args = parser.parse_args(argv)

    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    if not (root / MARKER).is_file():
        for name, expected in ARCHIVES.items():
            archive = root / name
            if not archive.is_file():
                print(f"Downloading {name}", flush=True)
                _download(MIRROR + name, archive)
            actual = _sha256(archive)
            if actual != expected:
                raise ValueError(f"{name} has SHA256 {actual}. The study used {expected}. Delete the file and retry.")
            _extract(archive, root)
            if not args.keep_archives:
                archive.unlink()
        shutil.rmtree(root / "__MACOSX", ignore_errors=True)
        (root / MARKER).write_text("ok\n", encoding="utf-8")
    images = _find(root, "Flicker8k_Dataset")
    summary = prepare_flickr8k(images, _find(root, "Flickr8k.token.txt"), _find(root, "Flickr_8k.trainImages.txt"),
                               _find(root, "Flickr_8k.devImages.txt"), _find(root, "Flickr_8k.testImages.txt"),
                               args.manifest)
    print(json.dumps({key: summary[key] for key in ("record_count", "split_counts") if key in summary}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
