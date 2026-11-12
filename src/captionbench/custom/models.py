"""Small caption decoders with token-gated spatial CNN attention.

The input is a frozen encoder's spatial feature grid, not a pooled image vector.
All decoder weights and the projection of that grid are trained from scratch.
These are project model candidates, with no claim of research novelty.
"""

from dataclasses import dataclass
import math

import torch
from torch import nn


@dataclass(frozen=True)
class ModelConfig:
    feature_channels: int
    d_model: int = 256
    n_heads: int = 4
    n_layers: int = 2
    ff_dim: int = 512
    dropout: float = 0.1
    max_length: int = 40
    kernels: tuple[int, ...] = (1, 3, 5)
    coverage_strength: float = 1.0
    max_cnn_strength: float = 2.0
    relation_hidden: int = 32
    use_relation_geometry: bool = True
    embedding_init_std: float | None = None

    def __post_init__(self):
        if min(self.feature_channels, self.d_model, self.n_heads, self.n_layers,
               self.ff_dim, self.max_length, self.relation_hidden) <= 0:
            raise ValueError("Model dimensions and max_length must be positive.")
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads.")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1).")
        if not self.kernels or any(k <= 0 or k % 2 == 0 for k in self.kernels):
            raise ValueError("kernels must contain positive odd integers.")
        if self.coverage_strength < 0 or self.max_cnn_strength <= 0:
            raise ValueError("Coverage must be nonnegative and max CNN strength positive.")
        if self.embedding_init_std is not None and (not math.isfinite(self.embedding_init_std) or self.embedding_init_std <= 0):
            raise ValueError("Embedding initialization standard deviation must be finite and positive.")


def masked_softmax(scores, allowed):
    """Return zero for fully masked rows, including an all-padding prefix."""
    probabilities = scores.masked_fill(~allowed, torch.finfo(scores.dtype).min).softmax(-1)
    probabilities = probabilities * allowed.to(scores.dtype)
    return probabilities / probabilities.sum(-1, keepdim=True).clamp_min(
        torch.finfo(scores.dtype).tiny
    )


def exclusive_coverage(first_pass_attention):
    """C_t = sum_{s < t} P_s; the current query never contributes to C_t."""
    zeros = torch.zeros_like(first_pass_attention[:, :, :1])
    return torch.cat((zeros, first_pass_attention[:, :, :-1].cumsum(dim=2)), dim=2)


def spatial_coordinates(height, width, device, dtype):
    """Coordinates use x to the right and y downward, both in [-1, 1]."""
    y = torch.linspace(-1, 1, height, device=device, dtype=dtype)
    x = torch.linspace(-1, 1, width, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    return torch.stack((xx, yy), dim=-1)


class VisualRelationBlock(nn.Module):
    """Mix spatial cells with content attention and a learned geometry bias.

    R[h,i,j] = MLP_h(x_j - x_i, y_j - y_i, distance(i,j)).
    A = softmax(QK^T / sqrt(d_head) + R); output = LN(grid + W_o A V).
    Cells are spatial features, not detected objects or supervised relations.
    """

    def __init__(self, config):
        super().__init__()
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        self.use_relation_geometry = config.use_relation_geometry
        self.qkv = nn.Linear(config.d_model, 3 * config.d_model)
        self.geometry_bias = nn.Sequential(
            nn.Linear(3, config.relation_hidden), nn.GELU(),
            nn.Linear(config.relation_hidden, config.n_heads),
        )
        self.output = nn.Linear(config.d_model, config.d_model)
        self.norm = nn.LayerNorm(config.d_model)
        self.dropout = nn.Dropout(config.dropout)

    @staticmethod
    def relative_geometry(height, width, device, dtype):
        coordinates = spatial_coordinates(height, width, device, dtype).reshape(-1, 2)
        # Axes are [query cell i, key cell j, coordinate].
        delta = coordinates[None, :, :] - coordinates[:, None, :]
        distance = torch.linalg.vector_norm(delta, dim=-1, keepdim=True)
        return torch.cat((delta, distance), dim=-1)

    def forward(self, image_grid, return_attention=False):
        batch, width, height, columns = image_grid.shape
        memory = image_grid.flatten(2).transpose(1, 2)
        cells = memory.shape[1]
        query, key, value = self.qkv(memory).chunk(3, dim=-1)
        query, key, value = (
            item.reshape(batch, cells, self.n_heads, self.head_dim).transpose(1, 2)
            for item in (query, key, value)
        )
        logits = (query @ key.transpose(-1, -2)) / math.sqrt(self.head_dim)
        if self.use_relation_geometry:
            geometry = self.relative_geometry(height, columns, image_grid.device, image_grid.dtype)
            bias = self.geometry_bias(geometry).permute(2, 0, 1)
            logits = logits + bias.unsqueeze(0)
        attention = logits.softmax(dim=-1)
        context = (self.dropout(attention) @ value).transpose(1, 2).reshape(batch, cells, width)
        enriched = self.norm(memory + self.dropout(self.output(context)))
        enriched = enriched.transpose(1, 2).reshape(batch, width, height, columns)
        return (enriched, attention) if return_attention else enriched


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        self.qkv = nn.Linear(config.d_model, 3 * config.d_model)
        self.output = nn.Linear(config.d_model, config.d_model)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, hidden, valid):
        batch, length, width = hidden.shape
        query, key, value = self.qkv(hidden).chunk(3, dim=-1)
        query, key, value = (
            item.reshape(batch, length, self.n_heads, self.head_dim).transpose(1, 2)
            for item in (query, key, value)
        )
        scores = (query @ key.transpose(-1, -2)) / math.sqrt(self.head_dim)
        causal = torch.ones(length, length, device=hidden.device, dtype=torch.bool).tril()
        allowed = (
            causal[None, None] & valid[:, None, None, :] & valid[:, None, :, None]
        )
        probabilities = masked_softmax(scores, allowed)
        context = self.dropout(probabilities) @ value
        context = context.transpose(1, 2).reshape(batch, length, width)
        return self.output(context) * valid.unsqueeze(-1)


class SpatialCrossAttention(nn.Module):
    """CNN basis maps produce spatial attention scores for each text query.

    B[h,k,n] = pointwise_h(GELU(depthwise_k(image)))[n]
    g[t,h,:] = softmax(token_gate(text[t,h]))
    S[t,h,n] = sum_k g[t,h,k] * B[h,k,n]

    plain: L = QK^T / sqrt(d_head)
    cnn_only: L = S
    hybrid and fixed_kernel: L = QK^T / sqrt(d_head) + alpha[h] * S
    hybrid_coverage: P0 = softmax(L), C_t = sum_{s<t} P0_s,
                     P = softmax(L - coverage_strength * C)

    alpha[h] = max_cnn_strength * sigmoid(raw_cnn_strength[h]).
    Coverage is a parallel two-pass approximation; it accumulates first-pass
    probabilities, not recursively corrected attention from the final pass.
    """

    def __init__(self, config, variant):
        super().__init__()
        self.variant = variant
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        self.coverage_strength = config.coverage_strength
        self.max_cnn_strength = config.max_cnn_strength
        self.value = nn.Linear(config.d_model, config.d_model)
        self.output = nn.Linear(config.d_model, config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        if variant != "cnn_only":
            self.query = nn.Linear(config.d_model, config.d_model)
            self.key = nn.Linear(config.d_model, config.d_model)
        if variant != "plain":
            self.kernels = (3,) if variant == "fixed_kernel" else config.kernels
            self.spatial_branches = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(config.d_model, config.d_model, kernel,
                              padding=kernel // 2, groups=config.d_model),
                    nn.GELU(),
                    nn.Conv2d(config.d_model, config.n_heads, kernel_size=1),
                )
                for kernel in self.kernels
            ])
            self.token_gate = nn.Linear(config.d_model, config.n_heads * len(self.kernels))
            if variant != "cnn_only":
                self.raw_cnn_strength = nn.Parameter(torch.zeros(config.n_heads))

    def forward(self, hidden, image_grid, valid):
        batch, length, width = hidden.shape
        memory = image_grid.flatten(2).transpose(1, 2)
        cells = memory.shape[1]
        value = self.value(memory).reshape(batch, cells, self.n_heads, self.head_dim)
        value = value.transpose(1, 2)
        if self.variant != "cnn_only":
            query = self.query(hidden).reshape(batch, length, self.n_heads, self.head_dim)
            key = self.key(memory).reshape(batch, cells, self.n_heads, self.head_dim)
            logits = (query.transpose(1, 2) @ key.transpose(1, 2).transpose(-1, -2))
            logits = logits / math.sqrt(self.head_dim)
        if self.variant != "plain":
            # Each kernel has a map for each head; text chooses its kernel mixture.
            basis = torch.stack([branch(image_grid).flatten(2)
                                 for branch in self.spatial_branches], dim=2)
            gate = self.token_gate(hidden).reshape(batch, length, self.n_heads, len(self.kernels))
            gate = gate.softmax(dim=-1)
            spatial_scores = torch.einsum("bthk,bhkn->bhtn", gate, basis)
            if self.variant == "cnn_only":
                logits = spatial_scores
            else:
                strength = self.max_cnn_strength * self.raw_cnn_strength.sigmoid()
                logits = logits + strength[None, :, None, None] * spatial_scores
        allowed = valid[:, None, :, None].expand(-1, self.n_heads, -1, cells)
        if self.variant == "hybrid_coverage":
            first_pass = masked_softmax(logits, allowed)
            coverage = exclusive_coverage(first_pass)
            logits = logits - self.coverage_strength * coverage
        attention = masked_softmax(logits, allowed)
        context = self.dropout(attention) @ value
        context = context.transpose(1, 2).reshape(batch, length, width)
        return self.output(context) * valid.unsqueeze(-1), attention


class CaptionDecoderLayer(nn.Module):
    def __init__(self, config, variant):
        super().__init__()
        self.self_attention = CausalSelfAttention(config)
        self.cross_attention = SpatialCrossAttention(config, variant)
        self.self_norm = nn.LayerNorm(config.d_model)
        self.cross_norm = nn.LayerNorm(config.d_model)
        self.feedforward_norm = nn.LayerNorm(config.d_model)
        self.feedforward = nn.Sequential(
            nn.Linear(config.d_model, config.ff_dim), nn.GELU(),
            nn.Dropout(config.dropout), nn.Linear(config.ff_dim, config.d_model),
        )
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, hidden, image_grid, valid):
        hidden = hidden + self.dropout(self.self_attention(self.self_norm(hidden), valid))
        context, attention = self.cross_attention(self.cross_norm(hidden), image_grid, valid)
        hidden = hidden + self.dropout(context)
        hidden = hidden + self.dropout(self.feedforward(self.feedforward_norm(hidden)))
        return hidden * valid.unsqueeze(-1), attention


class Captioner(nn.Module):
    """Image-grid captioner; forward predicts the next word at each input token."""

    variants = (
        "plain", "hybrid", "relation_only", "hybrid_relation",
        "fixed_kernel", "cnn_only", "hybrid_coverage",
    )

    def __init__(self, vocab_size, pad_id, config, variant="hybrid"):
        super().__init__()
        if variant not in self.variants:
            raise ValueError(f"Unknown variant {variant!r}; choose one of {self.variants}.")
        if vocab_size <= 0 or not 0 <= pad_id < vocab_size:
            raise ValueError("vocab_size must be positive and pad_id must be in the vocabulary.")
        self.config = config
        self.variant = variant
        self.vocab_size = vocab_size
        self.pad_id = pad_id
        self.feature_projection = nn.Conv2d(config.feature_channels, config.d_model, 1)
        self.grid_position = nn.Linear(2, config.d_model, bias=False)
        self.image_norm = nn.LayerNorm(config.d_model)
        if variant in ("relation_only", "hybrid_relation"):
            self.visual_relation = VisualRelationBlock(config)
        decoder_variant = {"relation_only": "plain", "hybrid_relation": "hybrid"}.get(variant, variant)
        self.token_embedding = nn.Embedding(vocab_size, config.d_model, padding_idx=pad_id)
        self.text_position = nn.Embedding(config.max_length, config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList([
            CaptionDecoderLayer(config, decoder_variant) for _ in range(config.n_layers)
        ])
        self.final_norm = nn.LayerNorm(config.d_model)
        self.output_projection = nn.Linear(config.d_model, vocab_size)
        if config.embedding_init_std is not None:
            # Initialize last so every other parameter keeps its original seeded value.
            with torch.random.fork_rng(devices=[]), torch.no_grad():
                self.token_embedding.weight.normal_(mean=0.0, std=config.embedding_init_std)
                self.token_embedding.weight[pad_id].zero_()

    def _image_grid(self, features):
        grid = self.feature_projection(features)
        height, width = grid.shape[-2:]
        coordinates = spatial_coordinates(height, width, grid.device, grid.dtype)
        positions = self.grid_position(coordinates).permute(2, 0, 1)
        grid = grid + positions.unsqueeze(0)
        grid = self.image_norm(grid.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        if self.variant in ("relation_only", "hybrid_relation"):
            # Both the CNN basis and cross-attention K/V use this same grid.
            grid = self.visual_relation(grid)
        return grid

    def forward(self, features, tokens, return_attention=False):
        if features.ndim != 4 or features.shape[1] != self.config.feature_channels:
            raise ValueError("features must have shape [batch, feature_channels, height, width].")
        if min(features.shape[-2:]) < 1:
            raise ValueError("The feature grid must contain at least one cell.")
        if tokens.ndim != 2 or tokens.shape[0] != features.shape[0]:
            raise ValueError("tokens must have shape [batch, length] and match the feature batch.")
        if not 1 <= tokens.shape[1] <= self.config.max_length:
            raise ValueError("Token length must be between 1 and config.max_length.")
        image_grid = self._image_grid(features)
        valid = tokens.ne(self.pad_id)
        positions = torch.arange(tokens.shape[1], device=tokens.device)
        hidden = self.token_embedding(tokens) * math.sqrt(self.config.d_model)
        hidden = self.dropout(hidden + self.text_position(positions)) * valid.unsqueeze(-1)
        for layer in self.layers:
            hidden, attention = layer(hidden, image_grid, valid)
        logits = self.output_projection(self.final_norm(hidden)) * valid.unsqueeze(-1)
        return (logits, attention) if return_attention else logits

    @torch.no_grad()
    def generate(self, features, bos_id, eos_id, max_new_tokens=24):
        """Greedy decoding with BOS included and PAD after an early EOS."""
        if not 0 <= max_new_tokens <= self.config.max_length - 1:
            raise ValueError("max_new_tokens must be between 0 and config.max_length - 1.")
        if not (0 <= bos_id < self.vocab_size and 0 <= eos_id < self.vocab_size):
            raise ValueError("BOS and EOS must be vocabulary IDs.")
        if bos_id == self.pad_id or eos_id == self.pad_id or bos_id == eos_id:
            raise ValueError("BOS, EOS, and PAD must have distinct IDs.")
        tokens = torch.full((features.shape[0], 1), bos_id, dtype=torch.long, device=features.device)
        finished = torch.zeros(features.shape[0], dtype=torch.bool, device=features.device)
        was_training = self.training
        self.eval()
        try:
            for _ in range(max_new_tokens):
                next_logits = self(features, tokens)[:, -1]
                # PAD is reserved for finished rows; BOS starts a caption once.
                next_logits[:, [self.pad_id, bos_id]] = -torch.inf
                next_token = next_logits.argmax(dim=-1)
                next_token = torch.where(finished, self.pad_id, next_token)
                tokens = torch.cat((tokens, next_token[:, None]), dim=1)
                finished = finished | next_token.eq(eos_id)
                if finished.all():
                    break
        finally:
            self.train(was_training)
        return tokens
