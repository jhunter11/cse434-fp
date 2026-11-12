"""Convert a frozen custom-decoder confirmation run into CaptionBench prediction files.

The custom decoder search saved one JSON prediction file per frozen checkpoint.
This script rewrites those captions in the JSONL schema that
``captionbench evaluate`` reads, so one scorer handles every model.

It also writes the confirmation image IDs and a compact training summary.
It reads saved files only. It trains nothing and generates no new caption.

Example:
    python scripts/import_decoder_run.py --workspace <search workspace> \
        --manifest data/flickr8k.jsonl --output-dir outputs/midterm \
        --summary results/midterm/decoder-training.json
"""

import argparse
import hashlib
import json
from pathlib import Path

from captionbench.data import load_manifest, write_jsonl

FAMILY_NAMES = {
    "plain_final": "decoder-plain",
    "custom_final": "decoder-relation",
    "alternative_final": "decoder-relation-wd01",
}


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--summary", required=True)
    args = parser.parse_args(argv)

    workspace, output_dir = Path(args.workspace), Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    partition = json.loads((workspace / "validation_partition.json").read_text(encoding="utf-8"))
    confirmation_ids = partition["confirmation_image_ids"]
    manifest = {row["image_id"]: row for row in load_manifest(args.manifest)}
    for image_id in confirmation_ids + partition["search_image_ids"]:
        if manifest[image_id]["split"] != "val":
            raise ValueError(f"Partition image is outside the validation split: {image_id}")
    if set(confirmation_ids) & set(partition["search_image_ids"]):
        raise ValueError("Search and confirmation images overlap")
    (output_dir / "confirm-ids.json").write_text(json.dumps(confirmation_ids, indent=1) + "\n", encoding="utf-8")
    (output_dir / "search-ids.json").write_text(json.dumps(partition["search_image_ids"], indent=1) + "\n",
                                                encoding="utf-8")

    freeze = json.loads((workspace / "confirmation" / "freeze.json").read_text(encoding="utf-8"))
    environment = json.loads((workspace / "environment.json").read_text(encoding="utf-8"))
    runs = []
    for selected in freeze["selected"]:
        run_id = selected["id"]
        family, seed = run_id.rsplit("_s", 1)
        model = f"{FAMILY_NAMES[family]}-s{seed}"
        run_dir = Path(selected["run_dir"])
        checkpoint_hash = _sha256(run_dir / "best.pt")
        if checkpoint_hash != selected["checkpoint_sha256"]:
            raise ValueError(f"Checkpoint bytes changed after the freeze: {run_id}")
        metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
        history = json.loads((run_dir / "history.json").read_text(encoding="utf-8"))
        condition = json.loads((run_dir.parent / "condition.json").read_text(encoding="utf-8"))
        saved = json.loads((workspace / "confirmation" / f"{run_id}_predictions.json").read_text(encoding="utf-8"))
        if [row["image_id"] for row in saved] != confirmation_ids:
            raise ValueError(f"Saved predictions do not match the confirmation images: {run_id}")
        rows = []
        for row in saved:
            image = manifest[row["image_id"]]
            if [reference.strip() for reference in row["references"]] != image["references"]:
                raise ValueError(f"Reference captions differ from the manifest: {row['image_id']}")
            caption = " ".join(row["prediction_tokens"]).strip()
            if not caption:
                raise ValueError(f"Empty saved caption: {run_id}/{row['image_id']}")
            rows.append({
                "image_id": row["image_id"], "model": model, "caption": caption,
                "model_id": f"captionbench/{FAMILY_NAMES[family]}", "revision": checkpoint_hash[:40],
                "precision": "float32", "device": metadata["device"],
                "generation": {"decode": "greedy", "max_new_tokens": freeze["max_new_tokens"]},
                "torch_version": metadata["torch_version"], "image_sha256": image["image_sha256"],
                "baseline_type": "decoder trained from scratch on frozen ResNet18 grid features",
            })
        write_jsonl(output_dir / f"confirm-{model}.jsonl", rows)
        variant = condition.get("variant") or condition.get("config", {}).get("variant")
        runs.append({
            "model": model, "family": FAMILY_NAMES[family], "seed": int(seed), "source_run_id": run_id,
            "variant": variant, "checkpoint_sha256": checkpoint_hash,
            "learning_rate": metadata["learning_rate"], "weight_decay": condition["config"].get("weight_decay"),
            "batch_size": metadata["batch_size"], "epochs_requested": metadata["epochs_requested"],
            "model_config": metadata["model_config"], "best_epoch": history["best_epoch"],
            "best_search_token_loss": history["best_validation_loss"],
            "epochs": [{key: epoch[key] for key in ("epoch", "train_token_loss", "validation_token_loss",
                                                    "elapsed_seconds")} for epoch in history["epochs"]],
        })
    summary = {
        "schema": "captionbench-decoder-import/v1",
        "source": "custom-search-20261022 frozen confirmation run; saved files only",
        "partition_sha256": partition["sha256"], "partition_seed": partition["seed"],
        "search_images": len(partition["search_image_ids"]), "confirmation_images": len(confirmation_ids),
        "training_environment": {"device": environment["device"], "packages": environment["packages"]},
        "runs": runs,
    }
    Path(args.summary).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary).write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({"models": [run["model"] for run in runs], "confirmation_images": len(confirmation_ids),
                      "summary": str(Path(args.summary).resolve())}, indent=2))


if __name__ == "__main__":
    main()
