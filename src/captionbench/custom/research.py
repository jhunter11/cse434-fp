"""Relation probes and image-cluster uncertainty for custom captioners."""
import hashlib
import json
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F


def validate_relation_pairs(pairs, records, vocab, max_length=30):
    """Require independently labeled, matched-length, held-out contrast pairs."""
    by_id = {r["image_id"]: r for r in records}
    seen = set()
    for pair in pairs:
        required = {"pair_id", "image_id", "correct", "foil", "category", "annotation_source"}
        if not required <= pair.keys():
            raise ValueError("Each relation pair needs its labels, category, and annotation source.")
        if pair["pair_id"] in seen:
            raise ValueError("Relation pair IDs must be unique.")
        seen.add(pair["pair_id"])
        record = by_id.get(pair["image_id"])
        if record is None or record["split"] not in {"val", "test"}:
            raise ValueError("Use held-out images for relation probes.")
        if pair["category"] not in {"role_swap", "relation_flip", "attribute_binding"}:
            raise ValueError("Use a declared relation-probe category.")
        if not str(pair["annotation_source"]).strip():
            raise ValueError("Record who verified each caption against the image.")
        good = vocab.encode(pair["correct"], max_length=max_length)
        bad = vocab.encode(pair["foil"], max_length=max_length)
        # A longer encode must not expose hidden truncation in either caption.
        if len(vocab.encode(pair["correct"], max_length=10000)) != len(good):
            raise ValueError("A correct caption exceeds the model context.")
        if len(vocab.encode(pair["foil"], max_length=10000)) != len(bad):
            raise ValueError("A foil exceeds the model context.")
        if len(good) != len(bad):
            raise ValueError("Correct and foil captions need matched token counts.")
        if vocab.unk_id in good or vocab.unk_id in bad:
            raise ValueError("A probe contains an unknown token. Revise or report coverage exclusions.")
        if good == bad or len(good) < 4:
            raise ValueError("Correct and foil captions must differ and contain content words.")
    if not pairs:
        raise ValueError("Provide independently annotated pairs before running a relation probe.")
    return by_id


@torch.no_grad()
def caption_log_probability(model, feature, text, vocab, device="cpu", max_length=30):
    ids = torch.tensor([vocab.encode(text, max_length=max_length)], device=device)
    logits = model(feature.unsqueeze(0).to(device), ids[:, :-1])
    targets = ids[:, 1:]
    # Average over the same number of predicted tokens, including EOS.
    nll = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), reduction="mean")
    return -float(nll)


def run_relation_probe(model, features, records, vocab, pairs, device="cpu", seed=42, max_length=30):
    """Compare correct-vs-foil margins on real and mismatched images.

    This is a conditional likelihood diagnostic, not free-generation accuracy.
    Label validity still requires an annotator to inspect the actual image.
    """
    validate_relation_pairs(pairs, records, vocab, max_length)
    image_ids = sorted({p["image_id"] for p in pairs})
    if len(image_ids) < 2:
        raise ValueError("Use at least two distinct images for the mismatched-image control.")
    rng = np.random.default_rng(seed)
    order = list(rng.permutation(image_ids))
    # A random cycle is a derangement, so no image controls itself.
    mismatch = {order[i]: order[(i + 1) % len(order)] for i in range(len(order))}
    previous_mode = model.training
    model.eval()
    rows = []
    try:
        for pair in pairs:
            image_id = pair["image_id"]
            def margin(feature):
                return (caption_log_probability(model, feature, pair["correct"], vocab, device, max_length)
                        - caption_log_probability(model, feature, pair["foil"], vocab, device, max_length))
            observed = margin(features[image_id])
            shuffled = margin(features[mismatch[image_id]])
            rows.append({**pair, "margin": observed, "mismatched_margin": shuffled,
                         "visual_gain": observed - shuffled, "correct_preferred": float(observed > 0),
                         "tie": bool(observed == 0), "control_image_id": mismatch[image_id]})
    finally:
        model.train(previous_mode)
    return rows


def cluster_bootstrap(rows, value_key, samples=2000, seed=42):
    """Resample images, after averaging all probes attached to each image."""
    if value_key in {"visual_gain", "mismatched_margin"}:
        raise ValueError("Shared image controls induce dependence. Report this point estimate without an image-only confidence interval.")
    if samples < 1 or not rows:
        raise ValueError("Provide observations and a positive bootstrap sample count.")
    grouped = defaultdict(list)
    for row in rows:
        value = float(row[value_key])
        if not np.isfinite(value):
            raise ValueError("Bootstrap values must be finite.")
        grouped[row["image_id"]].append(value)
    values = np.array([np.mean(grouped[k]) for k in sorted(grouped)], dtype=float)
    if len(values) < 2:
        raise ValueError("At least two independent images are required.")
    rng = np.random.default_rng(seed)
    estimates = np.array([rng.choice(values, size=len(values), replace=True).mean() for _ in range(samples)])
    low, high = np.quantile(estimates, [0.025, 0.975])
    return {"mean": float(values.mean()), "ci95": [float(low), float(high)],
            "image_count": len(values), "probe_count": len(rows), "samples": samples,
            "unit": "image", "note": "Conditional on these labels, model seed, and image-control assignment."}


def paired_image_difference(first, second, value_key, samples=2000, seed=42):
    """Paired model comparison with exactly matched probe IDs and images."""
    if value_key in {"visual_gain", "mismatched_margin"}:
        raise ValueError("Shared image controls need a separate dependence-aware uncertainty analysis.")
    a = {r["pair_id"]: r for r in first}
    b = {r["pair_id"]: r for r in second}
    if len(a) != len(first) or len(b) != len(second) or a.keys() != b.keys():
        raise ValueError("Model comparisons require unique, identical probe IDs.")
    rows = []
    for key in sorted(a):
        fields = ("image_id", "correct", "foil", "category", "annotation_source", "control_image_id")
        if any(a[key].get(f) != b[key].get(f) for f in fields):
            raise ValueError("Model comparisons require identical labels and image controls.")
        rows.append({"image_id": a[key]["image_id"], "difference": a[key][value_key] - b[key][value_key]})
    return cluster_bootstrap(rows, "difference", samples, seed)


def relation_pair_digest(pairs):
    text = json.dumps(sorted(pairs, key=lambda x: x["pair_id"]), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def factorial_relation_interaction(plain, cnn, relation, combined, value_key="margin", samples=2000, seed=42):
    """Estimate combined - CNN - relation + plain on matched relation probes."""
    if value_key not in {"margin", "correct_preferred"}:
        raise ValueError("Use real-image margin or preference accuracy for the factorial contrast.")
    conditions = [{r["pair_id"]: r for r in rows} for rows in (plain, cnn, relation, combined)]
    if any(len(c) != len(rows) for c, rows in zip(conditions, (plain, cnn, relation, combined))):
        raise ValueError("Probe IDs must be unique within every condition.")
    if any(c.keys() != conditions[0].keys() for c in conditions[1:]):
        raise ValueError("All four conditions need the same probes.")
    rows = []
    for pair_id in sorted(conditions[0]):
        a, b, c, d = [condition[pair_id] for condition in conditions]
        fields = ("image_id", "correct", "foil", "category", "annotation_source")
        if any(any(row.get(f) != a.get(f) for f in fields) for row in (b, c, d)):
            raise ValueError("All four conditions need identical images and annotations.")
        rows.append({"image_id": a["image_id"], "interaction": d[value_key] - b[value_key] - c[value_key] + a[value_key]})
    return cluster_bootstrap(rows, "interaction", samples, seed)
