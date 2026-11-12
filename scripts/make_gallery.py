"""Draw one page of held-out photos with every model's real caption under each.

The script picks the photos with a fixed random seed, so nobody selects them by
result. It prints each saved caption unchanged. It reads prediction files and
local photos, and it generates no new caption.

The page contains Flickr8k photos. Git ignores the PDF. Do not publish it.

Example:
    python scripts/make_gallery.py --manifest data/flickr8k.jsonl \
        --image-ids outputs/midterm/confirm-ids.json --output report/figures/gallery.pdf
"""

import argparse
import json
import random
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageOps

from captionbench.data import load_manifest, read_jsonl

MODELS = [("confirm-decoder-plain-s17.jsonl", "Ours, plain", "#0072b2"),
          ("confirm-decoder-relation-s17.jsonl", "Ours, relation", "#56b4e9"),
          ("val-blip-base.jsonl", "BLIP base", "#7a7a7a"),
          ("val-blip-large.jsonl", "BLIP large", "#444444"),
          ("confirm-blip-base-ft-s17.jsonl", "BLIP tuned", "#d55e00")]
INK = "#1f2933"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--image-ids", required=True)
    parser.add_argument("--predictions-dir", default="outputs/midterm")
    parser.add_argument("--rows", type=int, default=4)
    parser.add_argument("--columns", type=int, default=4)
    parser.add_argument("--seed", type=int, default=434)
    parser.add_argument("--wrap", type=int, default=46)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    image_ids = sorted(json.loads(Path(args.image_ids).read_text(encoding="utf-8")))
    chosen = sorted(random.Random(args.seed).sample(image_ids, args.rows * args.columns))
    by_id = {row["image_id"]: row for row in load_manifest(args.manifest)}
    captions = []
    for name, label, color in MODELS:
        if not (Path(args.predictions_dir) / name).is_file():
            continue
        rows = {row["image_id"]: row["caption"] for row in read_jsonl(Path(args.predictions_dir) / name)}
        captions.append((label, color, rows))

    figure = plt.figure(figsize=(8.5, 11))
    figure.text(0.5, 0.972, f"Real model outputs on {len(chosen)} held-out Flickr8k photos", ha="center",
                fontsize=12, fontweight="bold", color=INK)
    figure.text(0.5, 0.956, f"Photos drawn at random (seed {args.seed}) from the 200 confirmation images. "
                "Captions are unedited. No photo was chosen by its result.", ha="center", fontsize=7.5, color=INK)
    left, right, top, bottom = 0.035, 0.965, 0.945, 0.012
    cell_width, cell_height = (right - left) / args.columns, (top - bottom) / args.rows
    image_share = 0.42
    for index, image_id in enumerate(chosen):
        row, column = divmod(index, args.columns)
        x = left + column * cell_width
        y_top = top - row * cell_height
        axis = figure.add_axes([x + 0.006, y_top - image_share * cell_height, cell_width - 0.012,
                                image_share * cell_height - 0.004])
        with Image.open(by_id[image_id]["image_path"]) as source:
            photo = ImageOps.exif_transpose(source).convert("RGB")
            photo.thumbnail((420, 420))
        axis.imshow(photo)
        axis.set_axis_off()
        cursor = y_top - image_share * cell_height - 0.006
        for label, color, rows in captions:
            lines = textwrap.wrap(rows[image_id], args.wrap) or [""]
            figure.text(x + 0.008, cursor, label, fontsize=5.6, fontweight="bold", color=color, va="top")
            cursor -= 0.0078
            for line in lines:
                figure.text(x + 0.008, cursor, line, fontsize=5.8, color=INK, va="top")
                cursor -= 0.0078
            cursor -= 0.0012
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=200)
    plt.close(figure)
    print(json.dumps({"images": chosen, "output": str(Path(args.output).resolve())}, indent=1))


if __name__ == "__main__":
    main()
