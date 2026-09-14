"""The shared Transformer encoder and the supervised classifier around it.

The encoder is identical in every arm -- same depth, width, heads, dropout and
initialisation -- and takes its tokens from whichever embedder the config names.
It returns token-level representations rather than a pooled vector, because SSL
pretraining (next stage) reconstructs per-token targets while the supervised
head pools. Keeping the pooling out of the encoder means the two tasks share one
module with no branching inside it.
"""

from __future__ import annotations

import torch
from torch import nn

from ecg.models.config import ModelConfig
from ecg.models.embeddings import build_embedder

#: Standard deviation for positional-embedding initialisation, the usual
#: Transformer value. Small enough that early training is driven by the signal
#: rather than by position.
POSITION_INIT_STD: float = 0.02


class EcgEncoder(nn.Module):
    """Patch embedder, learned positions, and a pre-norm Transformer stack.

    Pre-norm (``norm_first=True``) rather than post-norm: it trains without a
    learning-rate warmup schedule, which removes one hyperparameter that would
    otherwise have to be tuned identically across four arms.

    Positions are learned rather than sinusoidal. Every record is the same
    length, so there is nothing to extrapolate to, and 20 learned vectors cost
    2,560 parameters.

    Args:
        config: Model configuration. Its ``embedder`` field selects the stem.

    Attributes:
        embedder: The patch embedder under comparison.
        positions: Learned positional embedding, ``(1, n_tokens, d_model)``.
        blocks: The Transformer encoder stack.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.embedder = build_embedder(config)
        self.positions = nn.Parameter(
            torch.zeros(1, config.n_tokens, config.d_model)
        )
        nn.init.trunc_normal_(self.positions, std=POSITION_INIT_STD)
        self.dropout = nn.Dropout(config.dropout)

        layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.d_ffn,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(
            layer,
            num_layers=config.n_layers,
            norm=nn.LayerNorm(config.d_model),
            enable_nested_tensor=False,
        )

    @property
    def d_model(self) -> int:
        """Width of a token representation."""
        return self.config.d_model

    @property
    def n_tokens(self) -> int:
        """Tokens per record."""
        return self.config.n_tokens

    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        """Encode raw signal into token representations.

        Args:
            signal: ``(batch, n_leads, n_samples)``. For SSL, masking must
                already have been applied to this tensor -- see
                :class:`~ecg.models.embeddings.ConvPatchEmbedding` for why
                masking after the stem would leak.

        Returns:
            ``(batch, n_tokens, d_model)`` token representations.
        """
        tokens = self.embedder(signal) + self.positions
        return self.blocks(self.dropout(tokens))


class EcgClassifier(nn.Module):
    """Supervised head over :class:`EcgEncoder`.

    Mean-pools the token representations, normalises, and projects to five
    logits -- one per diagnostic superclass. The labels are multi-label, so
    these are independent logits for per-class BCE, never a softmax.

    Args:
        config: Model configuration.

    Attributes:
        encoder: The shared encoder, reusable as an SSL pretraining target.
        head: LayerNorm, dropout and the output projection.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = EcgEncoder(config)
        self.head = nn.Sequential(
            nn.LayerNorm(config.d_model),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model, config.n_classes),
        )

    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        """Score a batch of records.

        Args:
            signal: ``(batch, n_leads, n_samples)``.

        Returns:
            ``(batch, n_classes)`` logits. Apply a sigmoid, not a softmax.
        """
        tokens = self.encoder(signal)
        return self.head(tokens.mean(dim=1))


def build_classifier(config: ModelConfig) -> EcgClassifier:
    """Construct the supervised model for one arm.

    Args:
        config: Model configuration.

    Returns:
        An :class:`EcgClassifier`.
    """
    return EcgClassifier(config)


def count_parameters(module: nn.Module, *, trainable_only: bool = True) -> int:
    """Count parameters in a module.

    Args:
        module: Any module.
        trainable_only: Count only parameters that require gradients.

    Returns:
        Parameter count.
    """
    params = module.parameters()
    if trainable_only:
        params = (p for p in params if p.requires_grad)
    return sum(int(p.numel()) for p in params)


def parameter_report(config: ModelConfig) -> dict[str, int]:
    """Break a model's parameter count into embedder, encoder and head.

    Logged to MLflow for every run. The embedder counts are the ones that must
    match across arms: if a convolutional arm wins on more parameters, the
    result is about capacity rather than about inductive bias.

    Args:
        config: Model configuration.

    Returns:
        Mapping with ``"embedder"``, ``"positions"``, ``"transformer"``,
        ``"head"`` and ``"total"``.
    """
    model = build_classifier(config)
    embedder = count_parameters(model.encoder.embedder)
    positions = int(model.encoder.positions.numel())
    transformer = count_parameters(model.encoder.blocks)
    head = count_parameters(model.head)
    return {
        "embedder": embedder,
        "positions": positions,
        "transformer": transformer,
        "head": head,
        "total": count_parameters(model),
    }


def embedder_parity(config: ModelConfig) -> dict[str, float]:
    """Compare the two embedders' parameter counts at one configuration.

    Args:
        config: Model configuration; its ``embedder`` field is ignored, since
            both arms are built.

    Returns:
        Mapping with ``"linear"``, ``"conv"`` and ``"relative_difference"``,
        the last being ``|conv - linear| / linear``. The project requires it to
        stay below 0.10.
    """
    linear = count_parameters(build_embedder(config.for_arm("linear")))
    conv = count_parameters(build_embedder(config.for_arm("conv")))
    return {
        "linear": float(linear),
        "conv": float(conv),
        "relative_difference": abs(conv - linear) / linear,
    }
