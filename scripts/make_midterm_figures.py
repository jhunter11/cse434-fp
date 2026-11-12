"""Draw the midterm figures and the LaTeX results table from saved result files.

The script reads ``results/midterm/grounding-gap.json`` and
``results/midterm/decoder-training.json``. It computes nothing new about a
model. It averages the three decoder seeds per family and writes:

* ``report/figures/blind-share.pdf``: matched and mismatched score per metric;
* ``report/figures/finetune-split.pdf``: fine-tuning gain split by source;
* ``report/figures/decoder-curves.pdf``: decoder loss per epoch;
* ``report/results-table.tex``: the table body that the report includes;
* ``results/midterm/summary.json``: every number the report cites.

Example:
    python scripts/make_midterm_figures.py --results results/midterm --report report
"""

import argparse
import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

METRICS = [("bleu4_corpus", "BLEU-4"), ("meteor", "METEOR"), ("cider_d", "CIDEr-D"), ("clipscore", "CLIPScore")]
FAMILIES = [("generic", "Image-blind control"), ("decoder-plain", "Decoder, plain"),
            ("decoder-relation", "Decoder, relation"), ("decoder-convnext", "Decoder, ConvNeXt grid"),
            ("blip-base", "BLIP base, zero-shot"), ("blip-large", "BLIP large, zero-shot"),
            ("blip-base-ft", "BLIP base, fine-tuned"), ("blip-large-ft", "BLIP large, fine-tuned")]
INK, MUTED, GRID = "#1f2933", "#52606d", "#d9dee3"
MATCHED, MISMATCHED = "#0072b2", "#e69f00"
STYLE, GROUND = "#e69f00", "#0072b2"


EXTRA_FORMATS = []


def _save(figure, path):
    figure.savefig(path)
    for suffix in EXTRA_FORMATS:
        figure.savefig(Path(path).with_suffix(f".{suffix}"), dpi=200)


def family_rows(models):
    """Average each family over its seeds. Return ``{family: {metric: {...}}}``."""
    rows = {}
    for key, _ in FAMILIES:
        names = [key] if key in models else sorted(n for n in models if n.rsplit("-s", 1)[0] == key)
        if not names:
            continue
        entry = {"runs": len(names), "caption_tokens_mean": statistics.fmean(models[n]["caption_tokens_mean"] for n in names),
                 "metrics": {}}
        for metric, _ in METRICS:
            cells = [models[n]["metrics"][metric] for n in names]
            matched = statistics.fmean(c["matched"] for c in cells)
            mismatched = statistics.fmean(c["mismatched"] for c in cells)
            entry["metrics"][metric] = {
                "matched": matched, "mismatched": mismatched, "gap": matched - mismatched,
                "blind_share": mismatched / matched,
                "matched_seed_sd": statistics.stdev(c["matched"] for c in cells) if len(cells) > 1 else None,
                "matched_ci95": cells[0].get("matched_ci95") if len(cells) == 1 else None}
        rows[key] = entry
    return rows


def style(axis):
    axis.spines[["top", "right"]].set_visible(False)
    axis.spines[["left", "bottom"]].set_color(GRID)
    axis.tick_params(colors=MUTED, labelsize=7, length=0)
    axis.yaxis.grid(True, color=GRID, linewidth=0.6)
    axis.set_axisbelow(True)


def blind_share_figure(rows, path):
    keys = [key for key, _ in FAMILIES if key in rows]
    short = {"generic": "Blind\nctrl.", "decoder-plain": "Dec.\nplain", "decoder-relation": "Dec.\nrel.",
             "blip-base": "BLIP\nbase", "blip-large": "BLIP\nlarge", "blip-base-ft": "BLIP\ntuned"}
    figure, axes = plt.subplots(1, 4, figsize=(7.0, 1.8))
    for axis, (metric, label) in zip(axes, METRICS, strict=True):
        style(axis)
        for index, key in enumerate(keys):
            cell = rows[key]["metrics"][metric]
            axis.bar(index - 0.19, cell["matched"], 0.36, color=MATCHED, label="own image" if index == 0 else None)
            axis.bar(index + 0.19, cell["mismatched"], 0.36, color=MISMATCHED,
                     label="other image" if index == 0 else None)
            axis.text(index, max(cell["matched"], cell["mismatched"]) * 1.03, f"{100 * cell['blind_share']:.0f}%",
                      ha="center", va="bottom", fontsize=6, color=INK)
        axis.set_xticks(range(len(keys)), [short[key] for key in keys], fontsize=5.2, color=INK)
        axis.set_title(label, fontsize=8, color=INK, loc="left")
        axis.set_ylim(0, axis.get_ylim()[1] * 1.12)
    axes[2].legend(fontsize=6, frameon=False, loc="upper left", handlelength=1)
    figure.tight_layout(pad=0.4, w_pad=0.8)
    _save(figure, path)
    plt.close(figure)


def finetune_figure(rows, path):
    """Split the fine-tuning gain per metric into the mismatched part and the gap part."""
    before, after = rows["blip-base"]["metrics"], rows["blip-base-ft"]["metrics"]
    figure, axis = plt.subplots(figsize=(3.3, 1.9))
    style(axis)
    split = {}
    for index, (metric, _label) in enumerate(METRICS):
        base = before[metric]["matched"]
        style_gain = (after[metric]["mismatched"] - before[metric]["mismatched"]) / base
        ground_gain = (after[metric]["gap"] - before[metric]["gap"]) / base
        split[metric] = {"relative_gain": style_gain + ground_gain, "from_mismatched": style_gain,
                         "from_gap": ground_gain}
        axis.bar(index - 0.19, 100 * style_gain, 0.36, color=STYLE, label="earned on other images" if index == 0 else None)
        axis.bar(index + 0.19, 100 * ground_gain, 0.36, color=GROUND, label="earned on own image only" if index == 0 else None)
    axis.axhline(0, color=MUTED, linewidth=0.6)
    axis.set_xticks(range(len(METRICS)), [label for _, label in METRICS], fontsize=7, color=INK)
    axis.set_ylabel("Change, % of zero-shot score", fontsize=7, color=MUTED)
    axis.legend(fontsize=6, frameon=False, loc="best", handlelength=1)
    figure.tight_layout(pad=0.4)
    _save(figure, path)
    plt.close(figure)
    return split


def curves_figure(training, path):
    figure, axis = plt.subplots(figsize=(3.3, 1.9))
    style(axis)
    colors = {"decoder-plain": "#0072b2", "decoder-relation": "#d55e00"}
    for run in training["runs"]:
        if run["family"] not in colors:
            continue
        epochs = [epoch["epoch"] for epoch in run["epochs"]]
        axis.plot(epochs, [epoch["validation_token_loss"] for epoch in run["epochs"]], color=colors[run["family"]],
                  linewidth=1.0, label={"decoder-plain": "plain attention", "decoder-relation": "relation attention"}
                  [run["family"]] if run["seed"] == 17 else None)
        axis.plot(epochs, [epoch["train_token_loss"] for epoch in run["epochs"]], color=colors[run["family"]],
                  linewidth=0.7, linestyle=":")
    axis.set_xlabel("Epoch", fontsize=7, color=MUTED)
    axis.set_ylabel("Token loss (solid: search set)", fontsize=7, color=MUTED)
    axis.legend(fontsize=6, frameon=False)
    figure.tight_layout(pad=0.4)
    _save(figure, path)
    plt.close(figure)


def table(rows, path):
    lines = []
    decoders = [key for key, _ in FAMILIES if key in rows and key.startswith("decoder-")]
    breaks = {"generic", *decoders[-1:]}
    for key, label in FAMILIES:
        if key not in rows:
            continue
        cells = []
        for metric, _ in METRICS:
            cell = rows[key]["metrics"][metric]
            scale = 1 if metric in ("cider_d", "clipscore") else 100
            digits = 2 if scale == 1 else 1
            cells.append(f"{scale * cell['matched']:.{digits}f} & {100 * cell['blind_share']:.0f}")
        if rows[key]["runs"] > 1:
            label = f"{label} ({rows[key]['runs']} seeds)"
        lines.append(f"{label} & " + " & ".join(cells) + f" & {rows[key]['caption_tokens_mean']:.1f} \\\\")
        if key in breaks:
            lines.append("\\addlinespace[2pt]")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--results", default="results/midterm")
    parser.add_argument("--report", default="report")
    parser.add_argument("--image-format", choices=["png", "svg"], help="Also save each figure in this format")
    args = parser.parse_args(argv)
    EXTRA_FORMATS[:] = [args.image_format] if args.image_format else []
    results, report = Path(args.results), Path(args.report)
    (report / "figures").mkdir(parents=True, exist_ok=True)
    gap = json.loads((results / "grounding-gap.json").read_text(encoding="utf-8"))
    training = json.loads((results / "decoder-training.json").read_text(encoding="utf-8"))
    rows = family_rows(gap["models"])
    plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42})
    blind_share_figure(rows, report / "figures" / "blind-share.pdf")
    split = finetune_figure(rows, report / "figures" / "finetune-split.pdf") if "blip-base-ft" in rows else None
    curves_figure(training, report / "figures" / "decoder-curves.pdf")
    table(rows, report / "results-table.tex")
    summary = {"schema": "captionbench-midterm-summary/v1", "image_count": gap["image_count"], "shifts": gap["shifts"],
               "families": rows, "finetune_gain_split": split,
               "note": "Family rows average the seeds of one decoder family. blind_share = mismatched / matched."}
    (results / "summary.json").write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
    for key, entry in rows.items():
        print(key, {m: (round(c["matched"], 4), round(c["blind_share"], 3)) for m, c in entry["metrics"].items()})
    if split:
        print("finetune split", {m: {k: round(v, 3) for k, v in s.items()} for m, s in split.items()})


if __name__ == "__main__":
    main()
