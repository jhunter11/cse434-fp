import pytest

pytest.importorskip("torch")

from captionbench.custom.research import cluster_bootstrap, paired_image_difference, validate_relation_pairs


def test_bootstrap_uses_images_instead_of_repeated_captions():
    rows = [{"image_id": "a", "gain": 1.0}] * 20 + [{"image_id": "b", "gain": 0.0}]
    result = cluster_bootstrap(rows, "gain", samples=100, seed=4)
    assert result["mean"] == 0.5
    assert result["image_count"] == 2 and result["probe_count"] == 21


def test_paired_comparison_rejects_changed_controls():
    rows = [{"pair_id": str(i), "image_id": str(i), "gain": 0.5, "control_image_id": str(1-i)} for i in range(2)]
    other = [dict(x) for x in rows]
    other[0]["control_image_id"] = "different"
    with pytest.raises(ValueError, match="identical labels"):
        paired_image_difference(rows, other, "gain", samples=20)


def test_paired_comparison_preserves_sign():
    rows = [{"pair_id": str(i), "image_id": str(i), "gain": 1.0} for i in range(4)]
    other = [{**r, "gain": 0.25} for r in rows]
    result = paired_image_difference(rows, other, "gain", samples=20)
    assert result["mean"] == 0.75 and result["ci95"] == [0.75, 0.75]


def test_probe_rejects_training_images():
    class Vocab:
        pass
    pair = dict(pair_id="p", image_id="a", correct="dog follows boy", foil="boy follows dog",
                category="role_swap", annotation_source="unit fixture")
    with pytest.raises(ValueError, match="held-out"):
        validate_relation_pairs([pair], [{"image_id": "a", "split": "train"}], Vocab())


def test_shuffled_control_ci_rejects_dependent_resampling():
    with pytest.raises(ValueError, match="dependence"):
        cluster_bootstrap([{"image_id": "a", "visual_gain": 1}], "visual_gain")


def test_factorial_interaction_matches_prespecified_contrast():
    from captionbench.custom.research import factorial_relation_interaction
    base = [{"pair_id": str(i), "image_id": str(i), "margin": 0.0} for i in range(4)]
    cnn = [{**r, "margin": 1} for r in base]
    relation = [{**r, "margin": 2} for r in base]
    combined = [{**r, "margin": 4} for r in base]
    result = factorial_relation_interaction(base, cnn, relation, combined, samples=20)
    assert result["mean"] == 1 and result["ci95"] == [1, 1]


def test_real_probe_path_preserves_model_mode_and_deranges_images():
    import torch
    from captionbench.custom.models import Captioner, ModelConfig
    from captionbench.custom.experiment import build_vocab
    from captionbench.custom.research import run_relation_probe
    torch.manual_seed(9)
    vocab = build_vocab([{"split": "train", "references": ["the dog follows the boy", "the boy follows the dog"]}], min_freq=1)
    cfg = ModelConfig(feature_channels=4, d_model=16, n_heads=2, n_layers=1, ff_dim=32, dropout=0.0, max_length=12)
    model = Captioner(len(vocab), vocab.pad_id, cfg, "hybrid_relation").train()
    pairs = [dict(pair_id=str(i), image_id=str(i), correct="the dog follows the boy",
                  foil="the boy follows the dog", category="role_swap", annotation_source="synthetic software fixture") for i in range(2)]
    records = [{"image_id": str(i), "split": "test"} for i in range(2)]
    features = {str(i): torch.randn(4, 2, 2) for i in range(2)}
    rows = run_relation_probe(model, features, records, vocab, pairs, max_length=12)
    assert model.training and len(rows) == 2
    assert all(r["image_id"] != r["control_image_id"] for r in rows)
    assert all(isinstance(r["margin"], float) for r in rows)
