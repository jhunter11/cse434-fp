"""Run the whole grounding-gap study with one command.

The runner needs a GPU, an internet connection, and nothing else. It uses no
cloud drive and no account. It downloads Flickr8k, generates and trains every
condition, scores each caption on its own image and on other images, and
writes the tables, the figures, and the caption gallery.

Every stage skips work whose output file exists. Run the same command again
after a disconnect to continue.

Profiles:

* ``quick``: a wiring check with one small decoder and a short fine-tune.
  Its scores are not study results.
* ``midterm``: the conditions in the midterm report.
* ``full``: ``midterm`` plus a wider decoder on ConvNeXt features and a
  fine-tuned BLIP large. Use a GPU with 16 GiB or more.
* ``improve``: our decoders only, against released BLIP. It tests a wider
  decoder, three stronger frozen encoders, and beam search. No fine-tuning.

The test split stays sealed. No stage reads it.

Example:
    python scripts/run_study.py --profile quick --work outputs/study
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
SEARCH_IDS = ROOT / "configs" / "partition" / "search-ids.json"
CONFIRM_IDS = ROOT / "configs" / "partition" / "confirm-ids.json"
CLI = "from captionbench.cli import main; raise SystemExit(main())"

PROFILES = {
    "quick": {
        "zero_shot": ["generic", "blip-base"],
        "decoders": [("decoder-plain", "plain", ["--epochs", "3", "--limit-train-images", "1500"])],
        "decoder_seeds": [17],
        "finetunes": [("blip-base-ft", "blip-base", ["--epochs", "1", "--limit-train-images", "200"])],
        "donor_rules": False, "bootstrap": 200,
        # Every quick stage that reads a zero-shot caption reads only the 200 confirmation images.
        "zero_shot_ids": CONFIRM_IDS,
    },
    "midterm": {
        "zero_shot": ["generic", "blip-base", "blip-large"],
        "decoders": [("decoder-plain", "plain", []), ("decoder-relation", "relation", [])],
        "decoder_seeds": [17, 29, 43],
        "finetunes": [("blip-base-ft", "blip-base", [])],
        "donor_rules": True, "bootstrap": 2000, "zero_shot_ids": None,
    },
}
PROFILES["full"] = {
    **PROFILES["midterm"],
    "decoders": [*PROFILES["midterm"]["decoders"],
                 ("decoder-convnext", "plain", ["--encoder", "convnext_base", "--width", "256", "--dropout", "0.3"])],
    "finetunes": [("blip-base-ft", "blip-base", []), ("blip-large-ft", "blip-large", ["--single-seed"])],
}
# Each family below is also captioned with three beams from the same weights (family name + "-beam").
WIDE = ["--width", "256", "--dropout", "0.3"]
PROFILES["improve"] = {
    "zero_shot": ["generic", "blip-base", "blip-large"],
    "decoders": [("decoder-plain", "plain", []), ("decoder-wide", "plain", WIDE),
                 ("decoder-mobile", "plain", ["--encoder", "mobilenet_v3_large", *WIDE]),
                 ("decoder-convnext", "plain", ["--encoder", "convnext_base", *WIDE]),
                 ("decoder-dinov2", "plain", ["--encoder", "dinov2_base", *WIDE])],
    "decoder_seeds": [17, 29, 43], "beams": 3,
    "finetunes": [], "donor_rules": False, "bootstrap": 2000, "zero_shot_ids": CONFIRM_IDS,
}


def run(label, command):
    print(f"\n=== {label}", flush=True)
    started = time.perf_counter()
    subprocess.run([sys.executable, *map(str, command)], check=True, cwd=ROOT)
    print(f"=== {label}: done in {time.perf_counter() - started:.0f}s", flush=True)


def wordnet_ready():
    try:
        from nltk.corpus import wordnet
        wordnet.ensure_loaded()
        return True
    except LookupError:
        return False


def repair_token_captions(path):
    """Join captions that an earlier trainer saved as word-token lists. Return the number of repaired rows."""
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    broken = [row for row in rows if isinstance(row.get("caption"), list)]
    for row in broken:
        row["caption"] = " ".join(row["caption"]).strip()
    if broken:
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8", newline="\n")
    return len(broken)


def training_summary(work, names):
    """Collect the decoder loss histories in the layout that the figure script reads."""
    runs = []
    for name in names:
        history = json.loads((work / "runs" / name / "history.json").read_text(encoding="utf-8"))
        family, seed = name.rsplit("-s", 1)
        runs.append({"model": name, "family": family, "seed": int(seed), "best_epoch": history["best_epoch"],
                     "best_search_token_loss": history["best_validation_loss"],
                     "epochs": history["epochs"]})
    return {"schema": "captionbench-decoder-training/v1", "source": "scripts/run_study.py", "runs": runs}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--profile", choices=list(PROFILES), default="quick")
    parser.add_argument("--work", default="outputs/study", help="Folder for every output of this run")
    parser.add_argument("--manifest", default="data/flickr8k.jsonl")
    parser.add_argument("--data-root", default="data/flickr8k")
    parser.add_argument("--finetune-seeds", type=int, nargs="+", default=[17, 29, 43])
    parser.add_argument("--finetune-batch-size", type=int, default=8)
    parser.add_argument("--skip-finetune", action="store_true")
    args = parser.parse_args(argv)

    profile = PROFILES[args.profile]
    work, manifest = (ROOT / args.work).resolve(), (ROOT / args.manifest).resolve()
    results, report = work / "results", work / "report"
    for folder in (work / "runs", results, report / "figures"):
        folder.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    if not manifest.is_file():
        run("Download Flickr8k and write the manifest",
            [SCRIPTS / "fetch_flickr8k.py", "--root", ROOT / args.data_root, "--manifest", manifest])
    if not wordnet_ready():
        run("Download WordNet for METEOR", ["-m", "nltk.downloader", "-q", "wordnet"])

    subset = profile["zero_shot_ids"]
    for model in profile["zero_shot"]:
        output = work / f"val-{model}.jsonl"
        if not (output.with_suffix(".done")).is_file():
            run(f"Caption the {'confirmation images' if subset else 'validation split'}: {model}",
                ["-c", CLI, "generate", "--manifest", manifest, "--model", model, "--split", "val",
                 "--output", output, *(["--image-ids", subset] if subset else [])])
            output.with_suffix(".done").write_text("ok\n", encoding="utf-8")

    decoder_names = []
    for family, variant, extra in profile["decoders"]:
        encoder = extra[extra.index("--encoder") + 1] if "--encoder" in extra else "resnet18"
        for seed in profile["decoder_seeds"]:
            name = f"{family}-s{seed}"
            decoder_names.append(name)
            output = work / f"confirm-{name}.jsonl"
            beam_name = f"{family}-beam-s{seed}"
            beam_output = work / f"confirm-{beam_name}.jsonl"
            beam = (["--beam-output", beam_output, "--beam-model-name", beam_name, "--beams", profile["beams"]]
                    if profile.get("beams") else [])
            if output.is_file() and (not beam or beam_output.is_file()):
                repaired = repair_token_captions(output)
                if repaired:
                    print(f"Joined {repaired} token-list captions in {output.name}", flush=True)
                continue
            run_dir = work / "runs" / name
            if run_dir.exists():
                # An interrupted stage left its folder, and the trainer never overwrites one. Keep it and train again.
                stale = run_dir.with_name(f"{name}.interrupted-{time.strftime('%Y%m%d-%H%M%S')}")
                run_dir.rename(stale)
                print(f"Moved the folder of an interrupted run to {stale.name}", flush=True)
            run(f"Train decoder {name}",
                [SCRIPTS / "train_decoder.py", "--manifest", manifest, "--search-ids", SEARCH_IDS,
                 "--predict-ids", CONFIRM_IDS, "--run-dir", work / "runs" / name, "--output", output,
                 "--model-name", name, "--variant", variant, "--seed", seed,
                 "--feature-cache", work / f"features-{encoder}.pt", *extra, *beam])

    def score_and_draw(label, out, patterns, bootstrap, donor_rules):
        """Score the prediction files and draw every output below ``out``."""
        out_results, out_report = out / "results", out / "report"
        for folder in (out_results, out_report / "figures"):
            folder.mkdir(parents=True, exist_ok=True)
        gap = [SCRIPTS / "grounding_gap.py", "--manifest", manifest, "--split", "val", "--image-ids", CONFIRM_IDS,
               "--predictions", *patterns, "--bootstrap-samples", bootstrap]
        run(f"{label}Score every caption on its own image and on other images",
            [*gap, "--output", out_results / "grounding-gap.json", "--per-image-output", out / "per-image.json"])
        if donor_rules:
            run(f"{label}Pair images by CLIP similarity",
                [SCRIPTS / "donor_rules.py", "--manifest", manifest, "--image-ids", CONFIRM_IDS,
                 "--output-dir", out])
            (out_results / "donor-rules.json").write_text((out / "donor-rules.json").read_text(encoding="utf-8"),
                                                          encoding="utf-8")
            for rule in ("nearest", "farthest"):
                run(f"{label}Score on the {rule} other image",
                    [*gap, "--donor-maps", out / f"donors-{rule}.json",
                     "--output", out_results / f"grounding-gap-{rule}.json"])
        (out_results / "decoder-training.json").write_text(
            json.dumps(training_summary(work, decoder_names), indent=1) + "\n", encoding="utf-8")
        run(f"{label}Draw the figures and the results table",
            [SCRIPTS / "make_midterm_figures.py", "--results", out_results, "--report", out_report,
             "--image-format", "png"])
        for suffix in ("pdf", "png"):
            run(f"{label}Draw the caption gallery ({suffix})",
                [SCRIPTS / "make_gallery.py", "--manifest", manifest, "--image-ids", CONFIRM_IDS,
                 "--predictions-dir", work, "--output", out / f"gallery.{suffix}"])

    finetunes = [] if args.skip_finetune else profile["finetunes"]
    preflight = work / "preflight"
    if args.profile != "quick" and finetunes and not (preflight / "passed").is_file():
        # A long run must not fail in its last stage. Run every later stage once on a tiny fine-tune first.
        preflight.mkdir(parents=True, exist_ok=True)
        smoke = preflight / "confirm-blip-base-ft-s0.jsonl"
        if not smoke.is_file():
            run("Preflight: fine-tune for one short epoch",
                [SCRIPTS / "finetune_blip.py", "--manifest", manifest, "--search-ids", SEARCH_IDS,
                 "--predict-ids", CONFIRM_IDS, "--run-dir", preflight / "run", "--output", smoke,
                 "--model-name", "blip-base-ft-s0", "--epochs", "1", "--limit-train-images", "200",
                 "--batch-size", args.finetune_batch_size])
        score_and_draw("Preflight: ", preflight, [work / "confirm-*.jsonl", work / "val-*.jsonl", smoke], 100, True)
        (preflight / "passed").write_text("ok\n", encoding="utf-8")
        print("\nPreflight passed. Every later stage ran once. The long fine-tuning runs start now.", flush=True)

    for family, base, extra in finetunes:
        seeds =args.finetune_seeds[:1] if "--single-seed" in extra or args.profile == "quick" else args.finetune_seeds
        for seed in seeds:
            name = f"{family}-s{seed}"
            output = work / f"confirm-{name}.jsonl"
            if output.is_file():
                continue
            run(f"Fine-tune {name}",
                [SCRIPTS / "finetune_blip.py", "--manifest", manifest, "--search-ids", SEARCH_IDS,
                 "--predict-ids", CONFIRM_IDS, "--run-dir", work / "runs" / name, "--output", output,
                 "--model-name", name, "--base-model", base, "--seed", seed,
                 "--batch-size", args.finetune_batch_size, *[item for item in extra if item != "--single-seed"]])

    score_and_draw("", work, [work / "confirm-*.jsonl", work / "val-*.jsonl"], profile["bootstrap"],
                   profile["donor_rules"])
    print(f"\nStudy profile '{args.profile}' finished in {(time.perf_counter() - started) / 60:.1f} minutes.")
    print(f"Tables and figures: {report}\nAggregate results: {results}\nGallery: {work / 'gallery.pdf'}")
    if args.profile == "quick":
        print("The quick profile checks the wiring. Its scores are not study results.")


if __name__ == "__main__":
    main()
