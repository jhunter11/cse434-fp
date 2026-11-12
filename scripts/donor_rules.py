"""Pair every held-out image with a donor image by CLIP image similarity.

``grounding_gap.py`` scores a caption on a donor image. Its default donor is a
cyclic shift of the sorted image order, which ignores how similar two photos
are. This script writes two other donor maps for the same image set:

* nearest: the one-to-one pairing with the highest total similarity;
* farthest: the one-to-one pairing with the lowest total similarity.

Each map is a permutation with no fixed point. Every image is a donor exactly
once, so the reference corpus and the CIDEr-D document frequencies stay the same.
The assignment is optimal (Hungarian method), not greedy.

The output holds image IDs and similarity statistics. It holds no caption text.

Example:
    python scripts/donor_rules.py --manifest data/flickr8k.jsonl \
        --image-ids outputs/midterm/confirm-ids.json --output-dir outputs/midterm
"""

import argparse
import json
import statistics
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from scipy.optimize import linear_sum_assignment
from transformers import AutoConfig, CLIPModel, CLIPProcessor

from captionbench.clipscore import CLIP_MODEL_ID
from captionbench.data import load_manifest

BLOCKED = 1e6


def pairing(similarity, nearest):
    """Return the optimal one-to-one donor index for each image, with no image paired to itself."""
    cost = -similarity if nearest else similarity.copy()
    np.fill_diagonal(cost, BLOCKED)
    rows, columns = linear_sum_assignment(cost)
    if any(row == column for row, column in zip(rows, columns, strict=True)):
        raise RuntimeError("The assignment paired an image with itself")
    return [int(column) for column in columns]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--image-ids", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)

    image_ids = sorted(json.loads(Path(args.image_ids).read_text(encoding="utf-8")))
    by_id = {row["image_id"]: row for row in load_manifest(args.manifest)}
    device = "cuda" if torch.cuda.is_available() else "cpu"
    revision = AutoConfig.from_pretrained(CLIP_MODEL_ID)._commit_hash
    processor = CLIPProcessor.from_pretrained(CLIP_MODEL_ID, revision=revision)
    model = CLIPModel.from_pretrained(CLIP_MODEL_ID, revision=revision).to(device).eval()
    vectors = []
    with torch.inference_mode():
        for start in range(0, len(image_ids), 64):
            images = []
            for image_id in image_ids[start:start + 64]:
                with Image.open(by_id[image_id]["image_path"]) as source:
                    images.append(ImageOps.exif_transpose(source).convert("RGB"))
            output = model.get_image_features(**processor(images=images, return_tensors="pt").to(device))
            tensor = output if isinstance(output, torch.Tensor) else output.pooler_output
            vectors.append(torch.nn.functional.normalize(tensor.float(), dim=-1).cpu())
    similarity = (torch.cat(vectors) @ torch.cat(vectors).T).double().numpy()

    count = len(image_ids)
    off_diagonal = similarity[~np.eye(count, dtype=bool)]
    summary = {"schema": "captionbench-donor-rules/v1", "image_count": count, "clip_model": CLIP_MODEL_ID,
               "clip_revision": revision, "all_pairs_mean_similarity": float(off_diagonal.mean()), "rules": {}}
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, nearest in (("nearest", True), ("farthest", False)):
        donors = pairing(similarity, nearest)
        values = [float(similarity[index, donor]) for index, donor in enumerate(donors)]
        path = output_dir / f"donors-{name}.json"
        path.write_text(json.dumps({image_ids[i]: image_ids[d] for i, d in enumerate(donors)}, indent=1) + "\n",
                        encoding="utf-8")
        summary["rules"][name] = {"file": path.name, "mean_similarity": statistics.fmean(values),
                                  "min_similarity": min(values), "max_similarity": max(values)}
    shifts = [statistics.fmean(float(similarity[i, (i + shift) % count]) for i in range(count))
              for shift in (1, 37, 73, 109, 151)]
    summary["rules"]["cyclic_shift"] = {"mean_similarity": statistics.fmean(shifts)}
    (output_dir / "donor-rules.json").write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
