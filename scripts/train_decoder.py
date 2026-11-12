"""Train one caption decoder from scratch on frozen image features and caption held-out images.

The script trains on the Flickr8k training split only. It keeps the epoch with
the lowest token loss on the search part of the validation split. It then
captions the requested held-out images with greedy decoding and writes
prediction rows that ``captionbench evaluate`` and ``grounding_gap.py`` read.

The defaults reproduce the midterm decoder (ResNet18 grid, width 128, two
layers). Use ``--encoder``, ``--width``, ``--layers``, and ``--epochs`` to test
a larger model on a larger GPU. The encoder stays frozen in every case.

The test split stays unused. The script rejects test images.

Example:
    python scripts/train_decoder.py --manifest data/flickr8k.jsonl \
        --search-ids configs/partition/search-ids.json --predict-ids configs/partition/confirm-ids.json \
        --run-dir outputs/study/decoder-plain-s17 --output outputs/study/confirm-decoder-plain-s17.jsonl \
        --model-name decoder-plain-s17
"""

import argparse
import hashlib
import json
import time
from importlib.metadata import version
from pathlib import Path

import torch
from torch import nn

from captionbench.custom.experiment import (build_frozen_encoder, build_vocab, extract_features, fit_condition,
                                            predict_records)
from captionbench.custom.models import ModelConfig
from captionbench.data import load_manifest, write_jsonl

ENCODERS = ("resnet18", "resnet50", "resnet101", "convnext_base")
VARIANTS = {"plain": "plain", "relation": "relation_only"}
LEARNING_RATES = {"plain": 9e-4, "relation": 1.2e-3}


def build_encoder(name):
    """Return a frozen spatial encoder and its image transform."""
    if name == "resnet18":
        return build_frozen_encoder(pretrained=True)
    from torchvision import models
    weights = {"resnet50": models.ResNet50_Weights.DEFAULT, "resnet101": models.ResNet101_Weights.DEFAULT,
               "convnext_base": models.ConvNeXt_Base_Weights.DEFAULT}[name]
    backbone = getattr(models, name)(weights=weights)
    encoder = backbone.features if name == "convnext_base" else nn.Sequential(*list(backbone.children())[:-2])
    encoder.requires_grad_(False).eval()
    return encoder, weights.transforms()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--search-ids", required=True, help="JSON list of validation IDs for epoch selection")
    parser.add_argument("--predict-ids", required=True, help="JSON list of validation IDs to caption")
    parser.add_argument("--run-dir", required=True, help="New folder for the checkpoint and the loss history")
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-name", required=True, help="Model label in the prediction rows")
    parser.add_argument("--variant", choices=list(VARIANTS), default="plain")
    parser.add_argument("--encoder", choices=ENCODERS, default="resnet18")
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, help="Default: 9e-4 for plain, 1.2e-3 for relation")
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--feature-cache", help="Optional .pt file that stores the frozen features between runs")
    parser.add_argument("--limit-train-images", type=int, help="Smoke runs only")
    args = parser.parse_args(argv)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    learning_rate = args.learning_rate or LEARNING_RATES[args.variant]
    records = [{**row, "sha256": row["image_sha256"]} for row in load_manifest(args.manifest)]
    by_id = {row["image_id"]: row for row in records}
    search_ids = json.loads(Path(args.search_ids).read_text(encoding="utf-8"))
    predict_ids = json.loads(Path(args.predict_ids).read_text(encoding="utf-8"))
    train_rows = [row for row in records if row["split"] == "train"]
    if args.limit_train_images:
        train_rows = train_rows[:args.limit_train_images]
    search_rows = [by_id[image_id] for image_id in search_ids]
    predict_rows = [by_id[image_id] for image_id in predict_ids]
    if any(row["split"] != "val" for row in [*search_rows, *predict_rows]):
        raise ValueError("Search and prediction images must come from the validation split")
    if set(search_ids) & set(predict_ids):
        raise ValueError("Search and prediction images must not overlap")

    needed = [*train_rows, *search_rows, *predict_rows]
    cache = Path(args.feature_cache) if args.feature_cache else None
    features = torch.load(cache, map_location="cpu") if cache and cache.is_file() else {}
    missing = [row for row in needed if row["image_id"] not in features]
    if missing:
        started = time.perf_counter()
        encoder, transform = build_encoder(args.encoder)
        features.update(extract_features(missing, encoder, transform, device=device, batch_size=64))
        del encoder
        print(f"Extracted {len(missing)} {args.encoder} feature grids in {time.perf_counter() - started:.0f}s",
              flush=True)
        if cache:
            cache.parent.mkdir(parents=True, exist_ok=True)
            torch.save(features, cache)
    channels = features[train_rows[0]["image_id"]].shape[0]

    vocab = build_vocab(train_rows, min_freq=2)
    config = ModelConfig(feature_channels=channels, d_model=args.width, n_heads=args.heads, n_layers=args.layers,
                         ff_dim=2 * args.width, dropout=args.dropout, max_length=32,
                         embedding_init_std=args.width ** -0.5)
    model, result = fit_condition(features, train_rows, search_rows, vocab, args.run_dir,
                                  variant=VARIANTS[args.variant], model_config=config, seed=args.seed,
                                  epochs=args.epochs, batch_size=args.batch_size,
                                  learning_rate=learning_rate, weight_decay=args.weight_decay,
                                  max_length=30, device=device)
    for epoch in result["epochs"]:
        print(json.dumps({key: round(value, 4) if isinstance(value, float) else value
                          for key, value in epoch.items()}), flush=True)
    started = time.perf_counter()
    # predict_records returns word-token lists. The scorer reads one caption string per row.
    captions = {image_id: " ".join(tokens).strip() for image_id, tokens in
                predict_records(model, features, predict_rows, vocab, device=device, batch_size=64).items()}
    latency = (time.perf_counter() - started) / len(predict_rows)
    empty = sorted(image_id for image_id, caption in captions.items() if not caption)
    if empty:
        raise RuntimeError(f"The decoder wrote an empty caption for {len(empty)} images, for example {empty[0]}")
    digest = hashlib.sha256((Path(args.run_dir) / "best.pt").read_bytes()).hexdigest()
    parameters = result["metadata"]["trainable_parameters"]
    model_id = (f"CaptionBench {args.variant} decoder from scratch on frozen {args.encoder} "
                f"(width {args.width}, {args.layers} layers, {parameters} parameters)")
    write_jsonl(args.output, [{
        "image_id": row["image_id"], "model": args.model_name, "caption": captions[row["image_id"]],
        "model_id": model_id, "revision": digest[:40], "latency_seconds": latency, "device": device,
        "device_name": torch.cuda.get_device_name(0) if device == "cuda" else "CPU",
        "latency_scope": "decoder only, batched; the frozen feature extraction is excluded",
        "generation": {"strategy": "greedy", "max_new_tokens": 24}, "precision": "float32",
        "torch_version": version("torch"), "image_sha256": row["image_sha256"],
        "baseline_type": "decoder trained from scratch",
    } for row in predict_rows])
    summary = {"model": args.model_name, "encoder": args.encoder, "variant": args.variant,
               "trainable_parameters": parameters, "vocabulary_size": len(vocab),
               "best_epoch": result["best_epoch"], "epochs_requested": args.epochs,
               "best_search_token_loss": result["best_validation_loss"], "checkpoint_sha256": digest,
               "output": str(Path(args.output).resolve())}
    (Path(args.run_dir) / "summary.json").write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
