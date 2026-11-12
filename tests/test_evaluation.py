import copy

import pytest
from nltk.tokenize import wordpunct_tokenize
from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu, sentence_bleu

from captionbench.evaluation import EvaluationError, MetricSetupError, cider_d, evaluate


def fixtures():
    manifest = [
        {"image_id": "one", "image_path": "one.jpg", "split": "test", "references": [
            "a black dog runs in a field", "a dog is running outside", "an animal runs through the grass",
            "the dog plays on a lawn", "a black dog runs quickly in a field"]},
        {"image_id": "two", "image_path": "two.jpg", "split": "test", "references": [
            "a child wears a red hat", "a little child in a red hat", "the child has a red hat on",
            "a kid is wearing a red cap", "a child wears a red hat outdoors"]},
    ]
    predictions = [{"image_id": "one", "model": "baseline", "caption": "A black dog runs in a field.", "latency_seconds": 0.2},
                   {"image_id": "two", "model": "baseline", "caption": "a child", "latency_seconds": 0.4}]
    return manifest, predictions


def test_bleu_matches_nltk_and_corpus_is_not_sentence_mean():
    manifest, predictions = fixtures()
    result = evaluate(manifest, predictions)
    references = [[wordpunct_tokenize(s.lower()) for s in row["references"]] for row in manifest]
    hypotheses = [wordpunct_tokenize(row["caption"].lower()) for row in predictions]
    expected_corpus = corpus_bleu(references, hypotheses, smoothing_function=None, auto_reweigh=False)
    expected_sentence = [sentence_bleu(r, h, smoothing_function=SmoothingFunction(epsilon=0.1).method1,
                                       auto_reweigh=False) for r, h in zip(references, hypotheses)]
    scores = result["per_model"]["baseline"]["metrics"]
    assert scores["bleu4_corpus"] == pytest.approx(expected_corpus)
    assert scores["bleu4_sentence_mean"] == pytest.approx(sum(expected_sentence) / 2)
    assert scores["bleu4_corpus"] != pytest.approx(scores["bleu4_sentence_mean"])
    assert result["per_model"]["baseline"]["mean_latency_seconds"] == pytest.approx(0.3)
    assert result["provenance"]["references_per_image"] == 5


def test_uses_fifth_reference_and_is_order_independent():
    manifest, predictions = fixtures()
    predictions[0]["caption"] = manifest[0]["references"][4]
    predictions[1]["caption"] = manifest[1]["references"][4]
    result = evaluate(manifest, predictions)
    assert result["per_model"]["baseline"]["metrics"]["bleu4_corpus"] == pytest.approx(1.0)
    assert result == evaluate(reversed(manifest), reversed(predictions))


def test_short_exact_caption_does_not_auto_reweight():
    manifest, predictions = fixtures()
    manifest = [manifest[0]]
    manifest[0]["references"] = ["red dog"] * 5
    predictions = [{**predictions[0], "caption": "red dog"}]
    result = evaluate(manifest, predictions)
    assert 0 < result["per_image"][0]["bleu4_sentence"] < 1


@pytest.mark.parametrize("mutation,match", [
    (lambda m, p: p.append(copy.deepcopy(p[0])), "Duplicate prediction"),
    (lambda m, p: p.pop(), "coverage mismatch"),
    (lambda m, p: p.append({**p[0], "model": "second"}), "coverage mismatch"),
    (lambda m, p: p[0].update(image_id="unknown"), "unknown image_id"),
    (lambda m, p: p[0].update(caption="  "), "nonempty string"),
    (lambda m, p: p[0].update(latency_seconds=float("nan")), "finite and nonnegative"),
    (lambda m, p: p[0].update(latency_seconds=-1), "finite and nonnegative"),
    (lambda m, p: m.append(copy.deepcopy(m[0])), "Duplicate manifest"),
    (lambda m, p: m[0].update(split="train"), "split=test"),
    (lambda m, p: m[0].update(references=["a caption"]), "Exactly five"),
])
def test_rejects_invalid_comparisons(mutation, match):
    manifest, predictions = fixtures()
    mutation(manifest, predictions)
    with pytest.raises(EvaluationError, match=match):
        evaluate(manifest, predictions)


def test_optional_metrics_are_explicitly_unmeasured():
    result = evaluate(*fixtures())
    scores = result["per_model"]["baseline"]["metrics"]
    for metric in ("meteor_mean", "bertscore_f1", "clipscore"):
        assert scores[metric] is None
        assert metric in result["unmeasured_metrics"]
    assert "cider_d" not in result["unmeasured_metrics"]
    assert result["per_model"]["baseline"]["condition"]["status"] == "unknown"


@pytest.mark.parametrize("field,value", [("revision", "different-commit"),
                                         ("generation", {"num_beams": 5}),
                                         ("precision", "float16")])
def test_mixed_conditions_cannot_be_pooled_under_one_model(field, value):
    manifest, predictions = fixtures()
    for prediction in predictions:
        prediction.update(model_id="org/captioner", revision="same-commit", precision="float32",
                          generation={"num_beams": 3})
    predictions[1][field] = value
    with pytest.raises(EvaluationError, match="Mixed model conditions"):
        evaluate(manifest, predictions)


def test_mixed_absent_and_supplied_condition_metadata_fails():
    manifest, predictions = fixtures()
    predictions[0]["revision"] = "commit-a"
    with pytest.raises(EvaluationError, match="Mixed supplied/missing"):
        evaluate(manifest, predictions)


def test_uniform_conditions_are_reported_with_unknown_fields_explicit():
    manifest, predictions = fixtures()
    for prediction in predictions:
        prediction.update(revision="commit-a", generation={"num_beams": 3})
    result = evaluate(manifest, predictions)
    condition = result["per_model"]["baseline"]["condition"]
    assert condition["status"] == "partial"
    assert condition["values"]["revision"] == "commit-a"
    assert "precision" in condition["missing_fields"]
    assert result["provenance"]["model_conditions"]["baseline"] == condition


def test_known_image_hashes_are_bound_to_predictions_and_preserved():
    manifest, predictions = fixtures()
    for image, prediction in zip(manifest, predictions):
        image["image_sha256"] = "a" * 64
        prediction["image_sha256"] = "a" * 64
    result = evaluate(manifest, predictions)
    assert all(row["image_sha256"] == "a" * 64 for row in result["per_image"])
    assert result["provenance"]["image_content_hashes"]["one"]["source"] == "manifest_and_predictions"


def test_legacy_fixture_hashes_are_explicitly_unknown():
    result = evaluate(*fixtures())
    assert all(row["image_sha256"] is None for row in result["per_image"])
    assert result["provenance"]["image_content_hashes"]["one"]["source"] == "unknown"


@pytest.mark.parametrize("mutation,match", [
    (lambda m, p: m[0].update(image_sha256="INVALID"), "lowercase 64-character"),
    (lambda m, p: p[0].update(image_sha256="A" * 64), "lowercase 64-character"),
    (lambda m, p: m[0].update(image_sha256="a" * 64), "requires matching prediction"),
    (lambda m, p: (m[0].update(image_sha256="a" * 64), p[0].update(image_sha256="b" * 64)), "does not match the manifest"),
])
def test_invalid_or_mismatched_image_digests_fail(mutation, match):
    manifest, predictions = fixtures()
    mutation(manifest, predictions)
    with pytest.raises(EvaluationError, match=match):
        evaluate(manifest, predictions)


def test_cross_model_image_hashes_must_match_even_without_manifest_hashes():
    manifest, predictions = fixtures()
    predictions.extend([{**row, "model": "second"} for row in predictions])
    for row in predictions:
        row["image_sha256"] = "a" * 64
    assert evaluate(manifest, predictions)["provenance"]["image_content_hashes"]["one"]["source"] == "predictions_only"
    predictions[2]["image_sha256"] = "b" * 64
    with pytest.raises(EvaluationError, match="mismatch across models"):
        evaluate(manifest, predictions)
    del predictions[2]["image_sha256"]
    with pytest.raises(EvaluationError, match="Mixed supplied/missing image hashes"):
        evaluate(manifest, predictions)


def test_normalized_reference_hash_ignores_order_but_detects_content_changes():
    manifest, predictions = fixtures()
    first = evaluate(manifest, predictions)
    manifest[0]["references"].reverse()
    reordered = evaluate(manifest, predictions)
    assert first["provenance"]["normalized_references_sha256"] == reordered["provenance"]["normalized_references_sha256"]
    manifest[0]["references"][0] = "a different reference caption"
    changed = evaluate(manifest, predictions)
    assert first["provenance"]["normalized_references_sha256"] != changed["provenance"]["normalized_references_sha256"]


def test_meteor_resource_error_is_not_silently_scored(monkeypatch):
    import captionbench.evaluation as evaluation

    def missing_resource():
        raise MetricSetupError("WordNet unavailable")

    monkeypatch.setattr(evaluation, "_load_meteor", missing_resource)
    with pytest.raises(MetricSetupError, match="WordNet"):
        evaluate(*fixtures(), include_meteor=True)


def test_meteor_uses_multi_reference_implementation(monkeypatch):
    import captionbench.evaluation as evaluation

    calls = []

    def score(references, hypothesis):
        calls.append((references, hypothesis))
        return 0.75

    monkeypatch.setattr(evaluation, "_load_meteor", lambda: score)
    result = evaluate(*fixtures(), include_meteor=True)
    assert len(calls) == 2 and all(len(r) == 5 for r, _ in calls)
    assert result["per_model"]["baseline"]["metrics"]["meteor_mean"] == 0.75
    assert "meteor_mean" not in result["unmeasured_metrics"]


def test_cider_d_gives_ten_for_exact_consensus_and_zero_for_no_overlap():
    dog, hat = "a black dog runs".split(), "the child wears red".split()
    references = [[dog] * 5, [hat] * 5]
    exact = cider_d(references, [dog, hat])
    assert exact == pytest.approx([10.0, 10.0])
    swapped = cider_d(references, [hat, dog])
    assert swapped == [0.0, 0.0]


def test_cider_d_ranks_consensus_above_partial_match_and_penalizes_length():
    manifest, predictions = fixtures()
    result = evaluate(manifest, predictions)
    by_image = {row["image_id"]: row["cider_d"] for row in result["per_image"]}
    assert 0.0 < by_image["two"] < by_image["one"] <= 10.0
    assert result["per_model"]["baseline"]["metrics"]["cider_d"] == pytest.approx(sum(by_image.values()) / 2)
    references = [["a black dog runs in a field".split()] * 5, ["the child wears red".split()] * 5]
    short = cider_d(references, ["a black dog runs in a field".split(), "the child".split()])
    padded = cider_d(references, [("a black dog runs in a field " * 3).split(), "the child".split()])
    assert padded[0] < short[0]


def test_cider_d_document_frequency_removes_words_that_every_image_shares():
    references = [["a dog".split()] * 5, ["a cat".split()] * 5]
    scores = cider_d(references, ["a".split(), "a cat".split()])
    assert scores[0] == 0.0
    assert scores[1] > 0.0


def test_validation_split_needs_an_explicit_request():
    manifest, predictions = fixtures()
    for row in manifest:
        row["split"] = "val"
    with pytest.raises(EvaluationError, match="split=test"):
        evaluate(manifest, predictions)
    result = evaluate(manifest, predictions, split="val")
    assert result["provenance"]["split"] == "val"
    with pytest.raises(EvaluationError, match="val or test"):
        evaluate(manifest, predictions, split="train")
