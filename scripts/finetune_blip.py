"""Fine-tune BLIP base on the Flickr8k training split and caption held-out images.

The script trains on training images only. It selects the epoch with the lowest
caption loss on the search part of the validation split. It then captions the
requested held-out images with the frozen CaptionBench decoding settings and
writes prediction rows that ``captionbench evaluate`` reads.

The test split stays unused unless you pass ``--predict-split test``.

Example:
    python scripts/finetune_blip.py --manifest data/flickr8k.jsonl \
        --search-ids outputs/midterm/search-ids.json --predict-ids outputs/midterm/confirm-ids.json \
        --run-dir outputs/midterm/blip-base-ft --output outputs/midterm/confirm-blip-base-ft.jsonl
"""

import argparse
import hashlib
import json
import random
import time
from importlib.metadata import version
from pathlib import Path

import torch
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset
from transformers import AutoConfig, BlipForConditionalGeneration, BlipProcessor

from captionbench.data import load_manifest, write_jsonl
from captionbench.models import GENERATION, MODEL_IDS, VISUAL_LATENCY_SCOPE

MODEL_KEY = "blip-base"


def _open(path):
    with Image.open(path) as source:
        return ImageOps.exif_transpose(source).convert("RGB")


class Pairs(Dataset):
    def __init__(self, rows):
        self.pairs = [(row["image_path"], reference) for row in rows for reference in row["references"]]

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        path, caption = self.pairs[index]
        return _open(path), caption.lower().strip()


class Collate:
    def __init__(self, processor, max_length):
        self.processor, self.max_length = processor, max_length

    def __call__(self, batch):
        images, captions = zip(*batch, strict=True)
        encoded = self.processor(images=list(images), text=list(captions), return_tensors="pt",
                                 padding=True, truncation=True, max_length=self.max_length)
        labels = encoded["input_ids"].masked_fill(encoded["attention_mask"] == 0, -100)
        return {**encoded, "labels": labels}


def _loss(model, batch, device):
    batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
    with torch.autocast("cuda", dtype=torch.float16, enabled=device == "cuda"):
        return model(**batch).loss


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--search-ids", required=True, help="JSON list of validation IDs for epoch selection")
    parser.add_argument("--predict-ids", required=True, help="JSON list of held-out IDs to caption")
    parser.add_argument("--predict-split", choices=["val", "test"], default="val")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--max-length", type=int, default=40)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--model-name", default="blip-base-ft", help="Model label in the prediction rows")
    parser.add_argument("--base-model", choices=["blip-base", "blip-large"], default=MODEL_KEY)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit-train-images", type=int, help="Smoke runs only")
    args = parser.parse_args(argv)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    records = load_manifest(args.manifest)
    by_id = {row["image_id"]: row for row in records}
    search_ids = json.loads(Path(args.search_ids).read_text(encoding="utf-8"))
    predict_ids = json.loads(Path(args.predict_ids).read_text(encoding="utf-8"))
    train_rows = [row for row in records if row["split"] == "train"]
    if args.limit_train_images:
        train_rows = train_rows[:args.limit_train_images]
    search_rows = [by_id[image_id] for image_id in search_ids]
    predict_rows = [by_id[image_id] for image_id in predict_ids]
    if any(row["split"] != "val" for row in search_rows):
        raise ValueError("Search images must come from the validation split")
    if any(row["split"] != args.predict_split for row in predict_rows):
        raise ValueError(f"Prediction images must come from split {args.predict_split}")
    train_ids = {row["image_id"] for row in train_rows}
    if train_ids & set(search_ids) or train_ids & set(predict_ids) or set(search_ids) & set(predict_ids):
        raise ValueError("Training, search, and prediction images must not overlap")

    model_id = MODEL_IDS[args.base_model]
    from captionbench.models import resolve_revision
    base_revision = resolve_revision(model_id, config=AutoConfig.from_pretrained(model_id))
    processor = BlipProcessor.from_pretrained(model_id, revision=base_revision)
    model = BlipForConditionalGeneration.from_pretrained(model_id, revision=base_revision).to(device)
    collate = Collate(processor, args.max_length)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(Pairs(train_rows), batch_size=args.batch_size, shuffle=True, generator=generator,
                              collate_fn=collate, num_workers=args.workers, drop_last=True,
                              persistent_workers=args.workers > 0, pin_memory=device == "cuda")
    search_loader = DataLoader(Pairs(search_rows), batch_size=args.batch_size * 2, collate_fn=collate,
                               num_workers=args.workers, persistent_workers=args.workers > 0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    total_steps = args.epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.learning_rate, total_steps=total_steps,
                                                    pct_start=0.1, anneal_strategy="cos")
    scaler = torch.amp.GradScaler(enabled=device == "cuda")

    def search_loss():
        model.eval()
        total, count = 0.0, 0
        with torch.inference_mode():
            for batch in search_loader:
                size = batch["input_ids"].shape[0]
                total += float(_loss(model, batch, device)) * size
                count += size
        return total / count

    history = [{"epoch": 0, "train_loss": None, "search_loss": search_loss(), "elapsed_seconds": 0.0}]
    print(json.dumps(history[-1]), flush=True)
    best = {"epoch": 0, "search_loss": history[0]["search_loss"]}
    checkpoint = run_dir / "best.pt"
    for epoch in range(1, args.epochs + 1):
        model.train()
        started, running, steps = time.perf_counter(), 0.0, 0
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = _loss(model, batch, device)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            running += float(loss)
            steps += 1
            if steps % 100 == 0:
                print(f"epoch {epoch} step {steps}/{len(train_loader)} loss {running / steps:.4f} "
                      f"elapsed {time.perf_counter() - started:.0f}s", flush=True)
        history.append({"epoch": epoch, "train_loss": running / steps, "search_loss": search_loss(),
                        "elapsed_seconds": time.perf_counter() - started})
        print(json.dumps(history[-1]), flush=True)
        if history[-1]["search_loss"] < best["search_loss"]:
            best = {"epoch": epoch, "search_loss": history[-1]["search_loss"]}
            torch.save(model.state_dict(), checkpoint)

    if best["epoch"] == 0:
        raise RuntimeError("Fine-tuning did not lower the search loss. No checkpoint was saved.")
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    rows = []
    for row in predict_rows:
        image = _open(row["image_path"])
        torch.cuda.synchronize() if device == "cuda" else None
        started = time.perf_counter()
        inputs = processor(images=image, return_tensors="pt").to(device)
        with torch.inference_mode():
            generated = model.generate(**inputs, **GENERATION)
        caption = processor.decode(generated[0], skip_special_tokens=True).strip()
        torch.cuda.synchronize() if device == "cuda" else None
        rows.append({
            "image_id": row["image_id"], "model": args.model_name, "caption": caption,
            "model_id": f"{model_id} fine-tuned on Flickr8k train", "revision": digest[:40],
            "latency_seconds": time.perf_counter() - started, "device": device,
            "device_name": torch.cuda.get_device_name(0) if device == "cuda" else "CPU",
            "latency_scope": VISUAL_LATENCY_SCOPE, "generation": dict(GENERATION), "precision": "float32",
            "torch_version": version("torch"), "transformers_version": version("transformers"),
            "image_sha256": row["image_sha256"], "baseline_type": "fine-tuned captioner",
        })
    write_jsonl(args.output, rows)
    record = {
        "schema": "captionbench-finetune/v1", "base_model": model_id, "base_revision": base_revision,
        "checkpoint_sha256": digest, "best": best, "history": history,
        "train_images": len(train_rows), "train_pairs": len(train_loader.dataset),
        "search_images": len(search_rows), "predicted_images": len(rows),
        "settings": {key: getattr(args, key) for key in ("epochs", "batch_size", "learning_rate", "weight_decay",
                                                         "max_length", "seed")},
        "schedule": "AdamW, one-cycle cosine, 10% warm-up, gradient clip 1.0, fp16 autocast for training",
        "text_normalization": "lowercase and strip the reference captions",
        "device": torch.cuda.get_device_name(0) if device == "cuda" else "CPU",
        "torch_version": version("torch"), "transformers_version": version("transformers"),
    }
    (run_dir / "finetune.json").write_text(json.dumps(record, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({"best": best, "checkpoint_sha256": digest, "output": str(Path(args.output).resolve())}))


if __name__ == "__main__":
    main()
