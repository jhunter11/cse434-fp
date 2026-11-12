"""Measure how much of each metric a caption earns without its own image.

For every model the script scores each caption twice on one fixed image set:

* matched: against the references and pixels of its own image;
* mismatched: against the references and pixels of a different image.

The mismatched score is the part of a metric that the caption earns from
dataset-level wording alone. The difference is the grounding gap. The script
averages the mismatched score over several fixed cyclic shifts of the image
order, so every image receives another image's captions and no pair repeats.

Document frequencies for CIDEr-D stay the same in both conditions, because a
shift keeps the same reference corpus.

Example:
    python scripts/grounding_gap.py --manifest data/flickr8k.jsonl --split val \
        --image-ids outputs/midterm/confirm-ids.json \
        --predictions outputs/midterm/confirm-*.jsonl --output results/midterm/grounding-gap.json
"""

import argparse
import glob
import json
import random
import statistics
from pathlib import Path

from captionbench.data import load_manifest, read_jsonl
from captionbench.evaluation import evaluate

PER_IMAGE = {"bleu4_sentence": "bleu4_sentence", "meteor": "meteor", "cider_d": "cider_d", "clipscore": "clipscore"}
SHIFTS = (1, 37, 73, 109, 151)


def _strip(rows):
    return [{key: value for key, value in row.items() if key != "image_sha256"} for row in rows]


def _bootstrap(values, samples, seed):
    rng = random.Random(seed)
    count = len(values)
    means = sorted(statistics.fmean(values[rng.randrange(count)] for _ in range(count)) for _ in range(samples))
    return [means[int(0.025 * samples)], means[int(0.975 * samples) - 1]]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--image-ids", required=True)
    parser.add_argument("--predictions", nargs="+", required=True)
    parser.add_argument("--donor-maps", nargs="+", help="JSON files that map each image ID to a donor image ID. "
                        "Each map must be a permutation with no fixed point. They replace the cyclic shifts.")
    parser.add_argument("--no-clipscore", action="store_true")
    parser.add_argument("--no-meteor", action="store_true", help="Skip METEOR when WordNet is absent")
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20261111)
    parser.add_argument("--output", required=True)
    parser.add_argument("--per-image-output", help="Optional JSON file with matched per-image scores")
    args = parser.parse_args(argv)

    image_ids = sorted(json.loads(Path(args.image_ids).read_text(encoding="utf-8")))
    keep = set(image_ids)
    by_id = {row["image_id"]: row for row in load_manifest(args.manifest) if row["split"] == args.split}
    records = [by_id[image_id] for image_id in image_ids]
    paths = sorted({path for pattern in args.predictions for path in glob.glob(pattern)})
    predictions = [row for path in paths for row in read_jsonl(path) if row["image_id"] in keep]
    clip, meteor = not args.no_clipscore, not args.no_meteor
    matched = evaluate(records, predictions, include_meteor=meteor, split=args.split, include_clipscore=clip)

    donor_lists = []
    for path in args.donor_maps or []:
        mapping = json.loads(Path(path).read_text(encoding="utf-8"))
        donors = [mapping[image_id] for image_id in image_ids]
        if sorted(donors) != image_ids or any(a == b for a, b in zip(image_ids, donors, strict=True)):
            raise ValueError(f"{path} must be a permutation of the image IDs with no fixed point")
        donor_lists.append(donors)
    for shift in () if args.donor_maps else SHIFTS:
        shift %= len(image_ids)
        if shift == 0:
            raise ValueError("A shift must move every image")
        donor_lists.append(image_ids[shift:] + image_ids[:shift])
    shifted_runs = []
    for donors in donor_lists:
        swapped = [{"image_id": image_id, "split": args.split, "image_path": by_id[donor]["image_path"],
                    "references": by_id[donor]["references"]}
                   for image_id, donor in zip(image_ids, donors, strict=True)]
        shifted_runs.append(evaluate(swapped, _strip(predictions), include_meteor=meteor, split=args.split,
                                     include_clipscore=clip))

    def per_image(result, model):
        return {row["image_id"]: row for row in result["per_image"] if row["model"] == model}

    models = {}
    for model, summary in matched["per_model"].items():
        own = per_image(matched, model)
        others = [per_image(run, model) for run in shifted_runs]
        entry = {"image_count": summary["image_count"], "caption_tokens_mean": summary["metrics"]["caption_tokens_mean"],
                 "mean_latency_seconds": summary["mean_latency_seconds"], "condition": summary["condition"]["values"],
                 "metrics": {}}
        corpus_mismatched = statistics.fmean(run["per_model"][model]["metrics"]["bleu4_corpus"] for run in shifted_runs)
        entry["metrics"]["bleu4_corpus"] = {
            "matched": summary["metrics"]["bleu4_corpus"], "mismatched": corpus_mismatched,
            "gap": summary["metrics"]["bleu4_corpus"] - corpus_mismatched}
        for name, key in PER_IMAGE.items():
            if own[image_ids[0]].get(key) is None:
                continue
            matched_values = [own[i][key] for i in image_ids]
            mismatched_values = [statistics.fmean(other[i][key] for other in others) for i in image_ids]
            gaps = [a - b for a, b in zip(matched_values, mismatched_values, strict=True)]
            entry["metrics"][name] = {
                "matched": statistics.fmean(matched_values), "mismatched": statistics.fmean(mismatched_values),
                "gap": statistics.fmean(gaps),
                "matched_ci95": _bootstrap(matched_values, args.bootstrap_samples, args.seed),
                "gap_ci95": _bootstrap(gaps, args.bootstrap_samples, args.seed)}
        for values in entry["metrics"].values():
            values["blind_share"] = values["mismatched"] / values["matched"] if values["matched"] else None
        models[model] = entry

    result = {
        "schema": "captionbench-grounding-gap/v1", "split": args.split, "image_count": len(image_ids),
        "image_ids_sha256": matched["provenance"]["image_ids_sha256"],
        "shifts": None if args.donor_maps else list(SHIFTS),
        "donor_maps": [Path(path).name for path in args.donor_maps] if args.donor_maps else None,
        "definition": "matched = caption scored on its own image; mismatched = mean over the donor assignments (fixed cyclic "
                      "shifts of the sorted image order, or the donor maps); gap = matched - mismatched; blind_share = mismatched / matched",
        "interval": "95% percentile bootstrap over images; fixed captions; excludes training-seed variation",
        "scales": {"bleu4": "0-1", "meteor": "0-1", "cider_d": "0-10", "clipscore": "0-2.5"},
        "prediction_files": [Path(path).name for path in paths],
        "scorer_provenance": {key: matched["provenance"][key] for key in
                              ("tokenizer", "normalization", "bleu4_corpus", "bleu4_sentence", "meteor", "cider_d",
                               "clipscore", "nltk_version")},
        "models": models,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    if args.per_image_output:
        Path(args.per_image_output).write_text(json.dumps(matched["per_image"]) + "\n", encoding="utf-8")
    for model, entry in sorted(models.items()):
        print(model, {name: (round(v["matched"], 4), round(v["mismatched"], 4)) for name, v in entry["metrics"].items()})


if __name__ == "__main__":
    main()
