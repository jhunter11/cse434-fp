"""Command-line entry points for local caption experiments."""

import argparse
import json
from pathlib import Path

from .data import load_manifest, prepare_flickr8k, read_jsonl, validate_manifest, write_jsonl
from .models import MODEL_IDS, generate_predictions, generate_to_file


def _json_output(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _evaluation_inputs(args):
    records = [row for row in load_manifest(args.manifest) if row["split"] == "test"]
    predictions = [row for path in args.predictions for row in read_jsonl(path)]
    return records, predictions


def _create_demo_manifest(output_dir):
    from PIL import Image, ImageDraw

    output_dir = Path(output_dir).resolve()
    images_dir = output_dir / "synthetic-images"
    images_dir.mkdir(parents=True, exist_ok=True)
    examples = [
        ("red-square.png", "red", "square", ["a red square on a white background", "a white image with a red square", "one red square is shown", "a red geometric square", "a square colored red on white"]),
        ("blue-circle.png", "blue", "circle", ["a blue circle on a white background", "a white image with a blue circle", "one blue circle is shown", "a blue geometric circle", "a circle colored blue on white"]),
        ("green-triangle.png", "green", "triangle", ["a green triangle on a white background", "a white image with a green triangle", "one green triangle is shown", "a green geometric triangle", "a triangle colored green on white"]),
    ]
    rows = []
    for filename, color, shape, references in examples:
        image = Image.new("RGB", (192, 192), "white")
        draw = ImageDraw.Draw(image)
        if shape == "square":
            draw.rectangle((48, 48, 144, 144), fill=color)
        elif shape == "circle":
            draw.ellipse((48, 48, 144, 144), fill=color)
        else:
            draw.polygon([(96, 40), (40, 144), (152, 144)], fill=color)
        path = images_dir / filename
        image.save(path)
        rows.append({"image_id": filename, "image_path": str(path), "split": "test", "references": references,
                     "source": "original programmatically drawn synthetic geometry fixture"})
    manifest = output_dir / "synthetic-manifest.jsonl"
    write_jsonl(manifest, rows)
    return manifest


def _build_parser():
    parser = argparse.ArgumentParser(prog="captionbench", description="Local, reproducible image-captioning comparison")
    sub = parser.add_subparsers(dest="command", required=True)
    validate = sub.add_parser("validate", help="Check manifest identities, paths, splits and five references")
    validate.add_argument("--manifest", required=True)
    validate.add_argument("--skip-images", action="store_true", help="Validate metadata without requiring image files")

    prepare = sub.add_parser("prepare-flickr8k", help="Prepare local Flickr8k files using original split lists")
    prepare.add_argument("--images-dir", "--images", dest="images_dir", required=True)
    prepare.add_argument("--captions", required=True, help="CSV captions.txt or legacy Flickr8k.token.txt")
    prepare.add_argument("--train-split", required=True)
    prepare.add_argument("--val-split", required=True, help="Original dev split file")
    prepare.add_argument("--test-split", required=True)
    prepare.add_argument("--output", required=True)

    generate = sub.add_parser("generate", help="Generate captions; real model weights load only when selected")
    generate.add_argument("--manifest", required=True)
    generate.add_argument("--model", choices=list(MODEL_IDS), default="generic")
    generate.add_argument("--split", choices=["train", "val", "test"], default="test")
    generate.add_argument("--limit", type=int)
    generate.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    generate.add_argument("--revision", help="Model commit hash or revision; resolved hash is recorded")
    generate.add_argument("--output", required=True)
    generate.add_argument("--no-resume", action="store_true", help="Require a new output file instead of resuming")

    evaluate = sub.add_parser("evaluate", help="Score exact shared test-image coverage with all five references")
    evaluate.add_argument("--manifest", required=True)
    evaluate.add_argument("--predictions", nargs="+", required=True)
    evaluate.add_argument("--include-meteor", action="store_true")
    evaluate.add_argument("--output", required=True)

    ratings = sub.add_parser("ratings-template", help="Create a blinded human-rating CSV and private model key")
    ratings.add_argument("--manifest", required=True)
    ratings.add_argument("--predictions", nargs="+", required=True)
    ratings.add_argument("--sample-size", type=int, default=100)
    ratings.add_argument("--seed", type=int, default=42)
    ratings.add_argument("--output-dir", required=True)

    correlate = sub.add_parser("correlate", help="Analyze completed ratings; empty cells are not fabricated")
    correlate.add_argument("--ratings", required=True)
    correlate.add_argument("--private-key", required=True)
    correlate.add_argument("--evaluation", required=True)
    correlate.add_argument("--bootstrap-samples", type=int, default=1000)
    correlate.add_argument("--seed", type=int, default=42)
    correlate.add_argument("--output", required=True)

    demo = sub.add_parser("demo-pipeline", help="Run a no-network synthetic plumbing check with the generic control")
    demo.add_argument("--manifest", help="Optional local fixture manifest; default creates three original geometry images")
    demo.add_argument("--output-dir", "--output", dest="output_dir", default="outputs/demo")
    return parser


def main(argv=None):
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "validate":
            summary = validate_manifest(args.manifest, require_images=not args.skip_images)
        elif args.command == "prepare-flickr8k":
            summary = prepare_flickr8k(args.images_dir, args.captions, args.train_split,
                                      args.val_split, args.test_split, args.output)
        elif args.command == "generate":
            records = load_manifest(args.manifest)
            predictions = generate_to_file(records, args.output, args.model, args.split, args.limit,
                                           args.device, args.revision, resume=not args.no_resume)
            summary = {"output": str(Path(args.output).resolve()), "prediction_count": len(predictions),
                       "model": args.model, "split": args.split, "revision": predictions[0]["revision"]}
        elif args.command == "evaluate":
            from .evaluation import evaluate
            result = evaluate(*_evaluation_inputs(args), include_meteor=args.include_meteor)
            _json_output(args.output, result)
            summary = {"output": str(Path(args.output).resolve()), "per_model": result["per_model"],
                       "unmeasured_metrics": result["unmeasured_metrics"]}
        elif args.command == "ratings-template":
            from .ratings import create_rating_packet, write_rating_packet
            packet = create_rating_packet(*_evaluation_inputs(args), sample_size=args.sample_size, seed=args.seed)
            write_rating_packet(packet, args.output_dir)
            summary = {"output_dir": str(Path(args.output_dir).resolve()), "sample_size": args.sample_size,
                       "rating_rows": len(packet["public_rows"]), "completed_ratings": 0}
        elif args.command == "correlate":
            from .ratings import analyze_ratings, read_ratings_csv
            private_key = json.loads(Path(args.private_key).read_text(encoding="utf-8"))
            evaluation_result = json.loads(Path(args.evaluation).read_text(encoding="utf-8"))
            result = analyze_ratings(read_ratings_csv(args.ratings), private_key, evaluation_result,
                                     bootstrap_samples=args.bootstrap_samples, seed=args.seed)
            _json_output(args.output, result)
            summary = {"output": str(Path(args.output).resolve()), "analysis": result}
        else:
            output_dir = Path(args.output_dir).resolve()
            rating_dir = output_dir / "ratings"
            if any((rating_dir / filename).exists() for filename in
                   ("ratings.csv", "private-model-key.json", "selection.json")):
                raise ValueError("Rating packet files already exist. Use a new output directory before running the demo.")
            from .evaluation import evaluate
            from .ratings import create_rating_packet, write_rating_packet
            manifest = Path(args.manifest) if args.manifest else _create_demo_manifest(output_dir)
            records = [row for row in load_manifest(manifest) if row["split"] == "test"]
            predictions = generate_predictions(records, model_key="generic")
            write_jsonl(output_dir / "generic-predictions.jsonl", predictions)
            result = evaluate(records, predictions)
            source_scope = "supplied local fixture" if args.manifest else "original synthetic geometry fixture"
            result["evidence_scope"] = f"{source_scope}; generic nonvisual control; no visual model executed; not a model-quality benchmark"
            _json_output(output_dir / "evaluation.json", result)
            sample_size = min(100, len(records))
            packet = create_rating_packet(records, predictions, sample_size=sample_size, seed=42)
            write_rating_packet(packet, rating_dir)
            summary = {"output_dir": str(output_dir), "manifest": str(manifest.resolve()),
                       "image_count": len(records), "model": "generic", "evidence_scope": result["evidence_scope"],
                       "metrics": result["per_model"], "rating_rows": len(packet["public_rows"]), "completed_ratings": 0}
    except (ValueError, OSError, RuntimeError, ImportError) as exc:
        parser.error(str(exc))
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
