"""Offline behavioral checks; these tests do not evaluate caption quality."""

import unittest
from dataclasses import replace

import pytest

torch = pytest.importorskip("torch")
from torch import nn

from captionbench.custom.models import Captioner, ModelConfig, exclusive_coverage


class CustomCaptionerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(17)
        self.config = ModelConfig(
            feature_channels=8, d_model=16, n_heads=2, n_layers=2,
            ff_dim=32, dropout=0.0, max_length=12,
        )
        self.features = torch.randn(2, 8, 3, 4)
        self.tokens = torch.tensor([[1, 4, 7, 2, 0], [1, 9, 5, 8, 2]])
        self.variants = Captioner.variants

    def make_model(self, variant="hybrid"):
        return Captioner(23, pad_id=0, config=self.config, variant=variant)

    def test_forward_backward_and_custom_branch_gradients(self):
        for variant in self.variants:
            with self.subTest(variant=variant):
                model = self.make_model(variant)
                logits = model(self.features, self.tokens)
                self.assertEqual(tuple(logits.shape), (2, 5, 23))
                targets = torch.tensor([[4, 7, 2, 0, 0], [9, 5, 8, 2, 0]])
                loss = nn.functional.cross_entropy(
                    logits.flatten(0, 1), targets.flatten(), ignore_index=0,
                )
                loss.backward()
                self.assertTrue(torch.isfinite(loss))
                self.assertGreater(model.token_embedding.weight.grad.abs().sum().item(), 0)
                self.assertGreater(model.feature_projection.weight.grad.abs().sum().item(), 0)
                if variant not in ("plain", "relation_only"):
                    cross = model.layers[-1].cross_attention
                    conv_grad = cross.spatial_branches[0][0].weight.grad
                    self.assertTrue(torch.isfinite(conv_grad).all())
                    self.assertGreater(conv_grad.abs().sum().item(), 0)
                    if variant != "fixed_kernel":
                        self.assertGreater(cross.token_gate.weight.grad.abs().sum().item(), 0)
                    if variant in ("hybrid", "hybrid_relation", "hybrid_coverage", "fixed_kernel"):
                        self.assertGreater(cross.raw_cnn_strength.grad.abs().sum().item(), 0)
                if variant in ("relation_only", "hybrid_relation"):
                    relation = model.visual_relation
                    self.assertGreater(relation.qkv.weight.grad.abs().sum().item(), 0)
                    self.assertGreater(relation.geometry_bias[0].weight.grad.abs().sum().item(), 0)

    def test_causal_prefix_is_invariant_to_future_tokens_in_every_variant(self):
        original = torch.tensor([[1, 4, 7, 9, 8, 2], [1, 5, 4, 8, 6, 2]])
        changed = original.clone()
        changed[:, 3:] = torch.tensor([[15, 16, 17], [18, 19, 20]])
        for variant in self.variants:
            with self.subTest(variant=variant):
                model = self.make_model(variant).eval()
                with torch.no_grad():
                    first, attn_first = model(self.features, original, True)
                    second, attn_second = model(self.features, changed, True)
                torch.testing.assert_close(first[:, :3], second[:, :3], atol=1e-6, rtol=1e-5)
                torch.testing.assert_close(attn_first[:, :, :3], attn_second[:, :, :3], atol=1e-6, rtol=1e-5)

    def test_every_variant_uses_image_features(self):
        for variant in self.variants:
            with self.subTest(variant=variant):
                model = self.make_model(variant).eval()
                with torch.no_grad():
                    first = model(self.features, self.tokens)
                    second = model(self.features.flip(0), self.tokens)
                self.assertGreater((first - second).abs().max().item(), 1e-5)

    def test_attention_is_normalized_and_pad_queries_are_zero(self):
        valid = self.tokens.ne(0)
        for variant in self.variants:
            with self.subTest(variant=variant):
                model = self.make_model(variant).eval()
                logits, attention = model(self.features, self.tokens, return_attention=True)
                self.assertEqual(tuple(attention.shape), (2, 2, 5, 12))
                torch.testing.assert_close(
                    attention.sum(-1), valid[:, None, :].expand(-1, 2, -1).float(),
                    atol=1e-6, rtol=1e-5,
                )
                self.assertTrue((attention >= 0).all())
                self.assertEqual(logits[0, -1].abs().sum().item(), 0)

    def test_self_attention_ignores_a_padding_key_even_before_a_valid_token(self):
        model = self.make_model().eval()
        block = model.layers[0].self_attention
        hidden = torch.randn(1, 3, self.config.d_model)
        changed = hidden.clone()
        changed[:, 1] = 1000 * torch.randn_like(changed[:, 1])
        valid = torch.tensor([[True, False, True]])
        first = block(hidden, valid)
        second = block(changed, valid)
        torch.testing.assert_close(first[:, 2], second[:, 2])
        self.assertEqual(first[:, 1].abs().sum().item(), 0)

    def test_all_pad_and_left_pad_are_finite(self):
        tokens = torch.tensor([[0, 0, 0, 0], [0, 1, 4, 2]])
        for variant in self.variants:
            with self.subTest(variant=variant):
                logits, attention = self.make_model(variant)(self.features, tokens, True)
                self.assertTrue(torch.isfinite(logits).all())
                self.assertTrue(torch.isfinite(attention).all())
                self.assertEqual(logits[0].abs().sum().item(), 0)
                self.assertEqual(attention[0].abs().sum().item(), 0)

    def test_exclusive_coverage_has_no_same_step_or_future_dependency(self):
        probabilities = torch.softmax(torch.randn(1, 2, 5, 6), -1).requires_grad_()
        coverage = exclusive_coverage(probabilities)
        self.assertEqual(coverage[:, :, 0].abs().sum().item(), 0)
        torch.testing.assert_close(coverage[:, :, 2], probabilities[:, :, :2].sum(2))
        coverage[:, :, 2].sum().backward()
        self.assertGreater(probabilities.grad[:, :, :2].abs().sum().item(), 0)
        self.assertEqual(probabilities.grad[:, :, 2:].abs().sum().item(), 0)
        changed = probabilities.detach().clone()
        changed[:, :, 2:] = torch.softmax(torch.randn_like(changed[:, :, 2:]), -1)
        torch.testing.assert_close(exclusive_coverage(changed)[:, :, :3], coverage[:, :, :3])

    def test_coverage_skips_padding_and_penalizes_previous_focus(self):
        model = self.make_model("hybrid_coverage").eval()
        cross = model.layers[0].cross_attention
        hidden = torch.randn(1, 3, 16)
        grid = torch.randn(1, 16, 3, 4)
        valid = torch.tensor([[True, False, True]])
        _, padded_attention = cross(hidden, grid, valid)
        _, short_attention = cross(hidden[:, [0, 2]], grid, torch.ones(1, 2, dtype=torch.bool))
        torch.testing.assert_close(padded_attention[:, :, 2], short_attention[:, :, 1])
        # Repeated identical queries share first-pass scores. Coverage should
        # reduce the second step's probability at the first step's favorite cell.
        repeated = hidden[:, :1].expand(-1, 2, -1)
        _, repeated_attention = cross(repeated, grid, torch.ones(1, 2, dtype=torch.bool))
        favorite = repeated_attention[:, :, 0].argmax(-1, keepdim=True)
        first = repeated_attention[:, :, 0].gather(-1, favorite)
        second = repeated_attention[:, :, 1].gather(-1, favorite)
        self.assertTrue((second < first).all())

    def test_generation_stops_on_eos_and_restores_training_mode(self):
        model = self.make_model().train()
        with torch.no_grad():
            model.output_projection.weight.zero_()
            model.output_projection.bias.fill_(-20)
            model.output_projection.bias[2] = 20
        generated = model.generate(self.features, bos_id=1, eos_id=2, max_new_tokens=5)
        torch.testing.assert_close(generated, torch.tensor([[1, 2], [1, 2]]))
        self.assertTrue(model.training)

    def test_generation_cannot_emit_pad_or_repeat_bos_before_eos(self):
        model = self.make_model().eval()
        with torch.no_grad():
            model.output_projection.weight.zero_()
            model.output_projection.bias.fill_(-20)
            model.output_projection.bias[0] = 20
            model.output_projection.bias[1] = 19
            model.output_projection.bias[4] = 18
        generated = model.generate(self.features, 1, 2, max_new_tokens=3)
        torch.testing.assert_close(generated, torch.tensor([[1, 4, 4, 4], [1, 4, 4, 4]]))

    def test_one_beam_gives_the_greedy_caption(self):
        model = self.make_model("plain").eval()
        greedy = model.generate(self.features, 1, 2, max_new_tokens=8)
        beam = model.generate_beam(self.features, 1, 2, num_beams=1, max_new_tokens=8)
        torch.testing.assert_close(beam, greedy)

    def _caption_log_prob(self, model, features, tokens):
        """Sum of token log-probabilities and generated length for one caption row (PAD excluded)."""
        tokens = tokens[tokens.ne(0)][None]
        log_probs = torch.log_softmax(model(features, tokens[:, :-1]), dim=-1)
        picked = log_probs.gather(-1, tokens[:, 1:, None]).sum()
        return picked.item(), tokens.shape[1] - 1

    def test_beam_search_matches_exhaustive_search_when_the_beam_holds_every_prefix(self):
        # Three free words (EOS, 3, 4) and two steps: at most 3 live prefixes, so 9 beams lose nothing.
        model = self.make_model("plain").eval()
        with torch.no_grad():
            model.output_projection.bias[5:] = -1e4
        beam = model.generate_beam(self.features, 1, 2, num_beams=9, max_new_tokens=2)
        for row in range(2):
            features = self.features[row:row + 1]
            options = [[1, 2], *[[1, a, b] for a in (3, 4) for b in (2, 3, 4)]]
            scores = []
            for option in options:
                total, length = self._caption_log_prob(model, features, torch.tensor(option))
                scores.append(total / length)
            best = options[max(range(len(options)), key=scores.__getitem__)]
            self.assertEqual(beam[row][beam[row].ne(0)].tolist(), best)

    def test_beam_search_pads_after_eos_and_restores_training_mode(self):
        model = self.make_model("plain").train()
        with torch.no_grad():
            model.output_projection.weight.zero_()
            model.output_projection.bias.fill_(-20)
            model.output_projection.bias[2] = 20
        generated = model.generate_beam(self.features, 1, 2, num_beams=3, max_new_tokens=5)
        torch.testing.assert_close(generated, torch.tensor([[1, 2], [1, 2]]))
        self.assertTrue(model.training)

    def test_relative_geometry_has_direction_distance_and_reverse_symmetry(self):
        relation = self.make_model("relation_only").visual_relation
        geometry = relation.relative_geometry(2, 2, torch.device("cpu"), torch.float32)
        torch.testing.assert_close(geometry[0, 0], torch.tensor([0.0, 0.0, 0.0]))
        torch.testing.assert_close(geometry[0, 1], torch.tensor([2.0, 0.0, 2.0]))
        torch.testing.assert_close(geometry[0, 2], torch.tensor([0.0, 2.0, 2.0]))
        torch.testing.assert_close(geometry[0, 3], torch.tensor([2.0, 2.0, 8.0 ** 0.5]))
        torch.testing.assert_close(geometry[:, :, :2], -geometry.transpose(0, 1)[:, :, :2])
        torch.testing.assert_close(geometry[:, :, 2], geometry[:, :, 2].T)

    def test_relative_geometry_bias_changes_visual_attention(self):
        relation = self.make_model("relation_only").visual_relation.eval()
        with torch.no_grad():
            # Remove QK scores and assign an increasing bias to spatial distance.
            relation.qkv.weight[:32].zero_()
            relation.qkv.bias[:32].zero_()
            relation.geometry_bias[0].weight.zero_()
            relation.geometry_bias[0].bias.zero_()
            relation.geometry_bias[0].weight[0, 2] = 1
            relation.geometry_bias[2].weight.zero_()
            relation.geometry_bias[2].bias.zero_()
            relation.geometry_bias[2].weight[:, 0] = 1
            _, attention = relation(torch.randn(1, 16, 2, 2), return_attention=True)
        torch.testing.assert_close(attention.sum(-1), torch.ones(1, 2, 4))
        self.assertTrue((attention[0, :, 0, 3] > attention[0, :, 0, 1]).all())
        self.assertTrue((attention[0, :, 0, 1] > attention[0, :, 0, 0]).all())

    def test_content_only_control_keeps_visual_attention_and_skips_geometry(self):
        enabled = self.make_model("hybrid_relation").eval()
        disabled = Captioner(
            23, 0, replace(self.config, use_relation_geometry=False), "hybrid_relation",
        ).eval()
        disabled.load_state_dict(enabled.state_dict())
        enabled_logits = enabled(self.features, self.tokens)
        disabled_logits = disabled(self.features, self.tokens)
        self.assertGreater((enabled_logits - disabled_logits).abs().max().item(), 1e-7)
        disabled_logits.square().mean().backward()
        self.assertIsNone(disabled.visual_relation.geometry_bias[0].weight.grad)
        self.assertGreater(disabled.visual_relation.qkv.weight.grad.abs().sum().item(), 0)

    def test_tiny_image_conditioned_fixture_can_be_fit(self):
        # Both images have the same text prefix and different target words.
        # This detects a detached, ignored, or otherwise untrainable visual path.
        torch.manual_seed(33)
        model = self.make_model("hybrid_coverage")
        images = torch.randn(2, 8, 3, 4)
        prefix = torch.ones(2, 1, dtype=torch.long)
        targets = torch.tensor([4, 9])
        optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
        initial = nn.functional.cross_entropy(model(images, prefix)[:, 0], targets).item()
        for _ in range(30):
            optimizer.zero_grad(set_to_none=True)
            logits = model(images, prefix)[:, 0]
            loss = nn.functional.cross_entropy(logits, targets)
            loss.backward()
            optimizer.step()
        final_logits = model(images, prefix)[:, 0]
        final = nn.functional.cross_entropy(final_logits, targets).item()
        self.assertLess(final, initial * 0.2)
        self.assertTrue(torch.equal(final_logits.argmax(-1), targets))

    def test_invalid_shapes_lengths_and_variant_fail_clearly(self):
        with self.assertRaises(ValueError):
            Captioner(23, 0, self.config, "unknown")
        model = self.make_model()
        with self.assertRaises(ValueError):
            model(self.features.flatten(2), self.tokens)
        with self.assertRaises(ValueError):
            model(self.features, torch.ones(2, 13, dtype=torch.long))
        with self.assertRaises(ValueError):
            model.generate(self.features, 1, 2, max_new_tokens=12)


if __name__ == "__main__":
    unittest.main(verbosity=2)
