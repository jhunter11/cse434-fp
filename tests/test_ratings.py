import copy
import json

import pytest
from scipy.stats import spearmanr

from captionbench.evaluation import EvaluationError
from captionbench.ratings import (analyze_ratings, create_rating_packet, read_ratings_csv,
                                  spearman_summary, write_rating_packet)


def fixtures():
    manifest = [{"image_id": f"image-{i}", "image_path": f"image-{i}.png", "split": "test",
                 "references": ["a dog runs in a field"] * 5} for i in range(5)]
    predictions = [{"image_id": row["image_id"], "model": model, "caption": f"{model} caption {i}"}
                   for i, row in enumerate(manifest) for model in ("base", "large")]
    return manifest, predictions


def completed():
    manifest, predictions = fixtures()
    packet = create_rating_packet(manifest, predictions, sample_size=5, blinding_key="test-private-key-123")
    mapping = {r["assignment_id"]: r for r in packet["private_key"]["assignments"]}
    ratings = []
    for rater_id in ("r1", "r2"):
        for row in packet["public_rows"]:
            score = int(row["image_id"].split("-")[1]) + 1
            ratings.append({**row, "rater_id": rater_id, "accuracy": str(score), "detail": str(6 - score)})
    evaluation = {"per_image": [{**row, "bleu4_sentence": (int(row["image_id"].split("-")[1]) + 1) / 5,
                                 "meteor": None} for row in predictions]}
    return packet, ratings, evaluation, mapping


def test_selection_is_common_and_reproducible_while_model_key_is_private():
    manifest, predictions = fixtures()
    first = create_rating_packet(manifest, predictions, sample_size=3, blinding_key="test-private-key-123")
    second = create_rating_packet(reversed(manifest), reversed(predictions), sample_size=3,
                                  blinding_key="test-private-key-123")
    assert first == second
    assert len(first["public_rows"]) == 6
    assert all("model" not in row for row in first["public_rows"])
    selected = set(first["selection"]["image_ids"])
    for model in ("base", "large"):
        assert {r["image_id"] for r in first["private_key"]["assignments"] if r["model"] == model} == selected
    new_key = create_rating_packet(manifest, predictions, sample_size=3, blinding_key="other-private-key-789")
    assert new_key["selection"] == first["selection"]
    assert {r["assignment_id"] for r in new_key["public_rows"]}.isdisjoint(r["assignment_id"] for r in first["public_rows"])


def test_default_hundred_requires_enough_images():
    with pytest.raises(EvaluationError, match="Requested 100"):
        create_rating_packet(*fixtures())


def test_write_separate_key_and_csv_roundtrip(tmp_path):
    packet, rows, evaluation, _ = completed()
    packet["public_rows"] = rows
    paths = write_rating_packet(packet, tmp_path)
    csv_text = (tmp_path / "ratings.csv").read_text()
    assert "blinding_key" not in csv_text and ",model," not in csv_text
    assert json.loads((tmp_path / "private-model-key.json").read_text())["assignments"]
    result = analyze_ratings(read_ratings_csv(paths["ratings"]), packet["private_key"], evaluation,
                             bootstrap_samples=0)
    assert result["rating_count"] == 20
    with pytest.raises(EvaluationError, match="already exist"):
        write_rating_packet(packet, tmp_path)


def test_formula_like_caption_is_escaped_and_still_matches(tmp_path):
    manifest, predictions = fixtures()
    predictions[0]["caption"] = "=SUM(1,2)"
    packet = create_rating_packet(manifest, predictions, sample_size=5, blinding_key="test-private-key-123")
    for row in packet["public_rows"]:
        row.update(rater_id="r1", accuracy=3, detail=3)
    paths = write_rating_packet(packet, tmp_path)
    rows = read_ratings_csv(paths["ratings"])
    assert any(r["caption"].startswith("'=") for r in rows)
    evaluation = {"per_image": [{**r, "bleu4_sentence": 0.4, "meteor": None} for r in predictions]}
    assert analyze_ratings(rows, packet["private_key"], evaluation, bootstrap_samples=0)["complete"]


def test_aggregate_and_per_rater_correlations_and_equal_caption_weight():
    packet, rows, evaluation, _ = completed()
    result = analyze_ratings(rows, packet["private_key"], evaluation, bootstrap_samples=0)
    assert result["complete"] and result["paired_image_count"] == 5
    assert result["rater_count"] == 2 and result["rated_caption_count"] == 10
    assert result["per_model"]["base"]["mean_accuracy"] == 3
    assert all(r["rater_count"] == 2 for r in result["per_caption"])
    assert result["aggregate_correlations"]["bleu4_sentence"]["accuracy"]["rho"] == pytest.approx(1)
    assert result["aggregate_correlations"]["bleu4_sentence"]["detail"]["rho"] == pytest.approx(-1)
    assert result["per_rater"]["r1"]["correlations"]["bleu4_sentence"]["accuracy"]["rho"] == pytest.approx(1)
    assert result["aggregate_correlations"]["meteor"]["accuracy"]["rho"] is None
    assert "pvalue" not in json.dumps(result)


@pytest.mark.parametrize("change,match", [
    (lambda rows: rows.append(copy.deepcopy(rows[0])), "Duplicate rating"),
    (lambda rows: rows[0].update(assignment_id="not-real"), "Unknown rating"),
    (lambda rows: rows[0].update(rater_id=" "), "rater_id"),
    (lambda rows: rows[0].update(accuracy=""), "integer from 1 through 5"),
    (lambda rows: rows[0].update(accuracy="6"), "integer from 1 through 5"),
    (lambda rows: rows[0].update(detail=1.5), "integer from 1 through 5"),
    (lambda rows: rows[0].update(caption="edited caption"), "altered caption"),
    (lambda rows: rows[0].update(image_id="another-image"), "mismatched image_id"),
])
def test_bad_ratings_fail(change, match):
    packet, rows, evaluation, _ = completed()
    change(rows)
    with pytest.raises(EvaluationError, match=match):
        analyze_ratings(rows, packet["private_key"], evaluation, bootstrap_samples=0)


def test_stale_evaluation_cannot_be_joined_to_ratings():
    packet, rows, evaluation, _ = completed()
    evaluation["per_image"][0]["caption"] = "a newer generation"
    with pytest.raises(EvaluationError, match="does not match"):
        analyze_ratings(rows, packet["private_key"], evaluation, bootstrap_samples=0)


def test_unbalanced_private_key_cannot_change_comparison_population():
    packet, rows, evaluation, _ = completed()
    packet["private_key"]["assignments"].pop()
    with pytest.raises(EvaluationError, match="same selected images"):
        analyze_ratings(rows, packet["private_key"], evaluation, bootstrap_samples=0)


def test_nonfinite_or_out_of_range_automatic_scores_are_rejected():
    packet, rows, evaluation, _ = completed()
    evaluation["per_image"][0]["bleu4_sentence"] = float("nan")
    with pytest.raises(EvaluationError, match="finite values"):
        analyze_ratings(rows, packet["private_key"], evaluation, bootstrap_samples=0)


def test_known_rating_image_hash_cannot_be_joined_to_different_image_content():
    manifest, predictions = fixtures()
    for prediction in predictions:
        prediction["image_sha256"] = "a" * 64
    packet = create_rating_packet(manifest, predictions, sample_size=5, blinding_key="test-private-key-123")
    rows = [{**row, "rater_id": "r1", "accuracy": 3, "detail": 3} for row in packet["public_rows"]]
    evaluation = {"per_image": [{**r, "bleu4_sentence": 0.4, "meteor": None} for r in predictions]}
    assert analyze_ratings(rows, packet["private_key"], evaluation, bootstrap_samples=0)["complete"]
    evaluation["per_image"][0]["image_sha256"] = "b" * 64
    with pytest.raises(EvaluationError, match="evaluated image content"):
        analyze_ratings(rows, packet["private_key"], evaluation, bootstrap_samples=0)


def test_partial_forms_report_coverage_and_do_not_fabricate_ratings():
    packet, rows, evaluation, _ = completed()
    result = analyze_ratings(rows[:2], packet["private_key"], evaluation, bootstrap_samples=0)
    assert not result["complete"]
    assert result["rating_count"] == 2
    assert result["expected_caption_count"] == 10


def test_spearman_ties_match_scipy_and_undefined_cases_are_null():
    x, y = [1, 1, 2, 3, 4], [2, 1, 2, 4, 3]
    summary = spearman_summary([(str(i), a, b) for i, (a, b) in enumerate(zip(x, y))], bootstrap_samples=0)
    assert summary["rho"] == pytest.approx(spearmanr(x, y).statistic)
    assert spearman_summary([("a", 1, 1)] * 3)["status"] == "constant_input"
    assert spearman_summary([("a", 1, 1), ("b", 2, 2)])["status"] == "insufficient_pairs"
    assert spearman_summary([("a", 1, 1)] * 3)["rho"] is None


def test_image_cluster_bootstrap_is_reproducible_and_keeps_caption_count():
    pairs = [(str(i), float(i), float(i)) for i in range(20) for _ in range(2)]
    summary = spearman_summary(pairs, bootstrap_samples=100)
    assert summary == spearman_summary(pairs, bootstrap_samples=100)
    assert summary["caption_count"] == 40 and summary["image_cluster_count"] == 20
    assert summary["ci95"] == pytest.approx([1, 1])
    assert summary["bootstrap_valid_samples"] == 100
    with pytest.raises(EvaluationError, match="finite"):
        spearman_summary([("a", float("nan"), 1)])


def test_malformed_csv_is_rejected(tmp_path):
    path = tmp_path / "malformed.csv"
    path.write_text("assignment_id,rater_id,accuracy,detail\na,r,3,4,extra\n")
    with pytest.raises(EvaluationError, match="wrong number"):
        read_ratings_csv(path)
