"""Masked-patch reconstruction pretraining.

The pretext task: blank half the 500 ms patches of a record, run what is left
through the same encoder the supervised arms use, and predict the missing
samples. Loss is mean squared error **on the masked positions only** -- scoring
the visible ones as well would train an identity map over most of the sequence
and dilute the gradient that carries the actual task.

The decoder is a single linear projection from a token representation back to
its 12x50 samples. MAE found a deeper decoder helps linear probing and matters
much less for fine-tuning, which is what happens here; a shallow decoder also
pushes reconstruction detail into the encoder, which is the part being kept.
Either way it is identical in both arms, so it cannot explain a difference
between them.

One target choice worth stating because it departs from image MAE: the
reconstruction target is **not** normalised per patch. MAE normalises each
patch's target to zero mean and unit variance, which throws away absolute
amplitude. Amplitude is diagnostic here -- the HYP superclass rests on voltage
criteria -- so the target is the per-record normalised signal exactly as the
supervised arms see it.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from ecg.models.config import ModelConfig, SslConfig
from ecg.models.embeddings import patchify
from ecg.models.encoder import EcgClassifier, EcgEncoder
from ecg.models.masking import (
    apply_mask,
    mask_summary,
    reach_of,
    sample_block_mask,
)


@dataclass(frozen=True)
class SslStep:
    """Result of one pretraining step.

    Attributes:
        loss: Masked-position MSE, the tensor to call ``backward`` on.
        token_mask: The mask that was drawn, kept for plotting and debugging.
        metrics: Scalars for MLflow, including the contaminated-token fraction
            that quantifies how differently the two stems experience the mask.
    """

    loss: torch.Tensor
    token_mask: torch.Tensor
    metrics: dict[str, float]


class MaskedReconstruction(nn.Module):
    """Encoder plus a linear reconstruction decoder.

    Holds the same :class:`~ecg.models.encoder.EcgEncoder` the supervised arms
    use, so pretrained weights transfer with no surgery -- see
    :func:`transfer_encoder`.

    Args:
        config: Model configuration; its ``embedder`` field selects the arm.
        ssl_config: Masking settings. Identical across arms C and D.

    Attributes:
        encoder: The shared encoder, the only part kept after pretraining.
        decoder: Linear projection from ``d_model`` back to one patch.
    """

    def __init__(self, config: ModelConfig, ssl_config: SslConfig | None = None) -> None:
        super().__init__()
        self.config = config
        self.ssl_config = ssl_config or SslConfig()
        self.encoder = EcgEncoder(config)
        self.decoder = nn.Linear(config.d_model, config.patch_features)

    @property
    def reach(self) -> int:
        """Tokens either side that this arm's embedder can see into."""
        return reach_of(self.config, self.encoder.embedder.receptive_field)

    def sample_mask(
        self,
        batch_size: int,
        *,
        generator: torch.Generator | None = None,
        device: str | torch.device = "cpu",
    ) -> torch.Tensor:
        """Draw a block mask for a batch.

        Args:
            batch_size: Records in the batch.
            generator: CPU generator for reproducibility.
            device: Device for the returned mask.

        Returns:
            ``(batch_size, n_tokens)`` boolean tensor.
        """
        return sample_block_mask(
            batch_size,
            self.config.n_tokens,
            self.ssl_config,
            generator=generator,
            device=device,
        )

    def forward(
        self, signal: torch.Tensor, token_mask: torch.Tensor
    ) -> torch.Tensor:
        """Reconstruct every patch from the masked signal.

        Args:
            signal: ``(batch, n_leads, n_samples)`` unmasked raw signal. The
                mask is applied here rather than by the caller, so masking can
                never accidentally be skipped or applied twice.
            token_mask: ``(batch, n_tokens)`` boolean tensor.

        Returns:
            ``(batch, n_tokens, patch_features)`` predicted patches, in the
            layout :func:`~ecg.models.embeddings.patchify` produces.
        """
        masked = apply_mask(signal, token_mask, self.config.patch_samples)
        return self.decoder(self.encoder(masked))

    def loss(self, signal: torch.Tensor, token_mask: torch.Tensor) -> torch.Tensor:
        """Mean squared error over masked positions only.

        Args:
            signal: ``(batch, n_leads, n_samples)`` unmasked raw signal, which
                is also the reconstruction target.
            token_mask: ``(batch, n_tokens)`` boolean tensor.

        Returns:
            Scalar loss.
        """
        predicted = self(signal, token_mask)
        target = patchify(signal, self.config)
        squared_error = (predicted - target).pow(2).mean(dim=-1)  # (batch, n_tokens)
        return squared_error[token_mask].mean()

    def step(
        self,
        signal: torch.Tensor,
        *,
        generator: torch.Generator | None = None,
    ) -> SslStep:
        """Draw a mask, compute the loss, and collect metrics.

        Args:
            signal: ``(batch, n_leads, n_samples)`` unmasked raw signal.
            generator: CPU generator for reproducibility (integrity rule 4).

        Returns:
            An :class:`SslStep`.
        """
        token_mask = self.sample_mask(
            signal.shape[0], generator=generator, device=signal.device
        )
        loss = self.loss(signal, token_mask)
        metrics = mask_summary(token_mask, self.reach)
        metrics["ssl_loss"] = float(loss.detach())
        return SslStep(loss=loss, token_mask=token_mask, metrics=metrics)


def build_pretrainer(
    config: ModelConfig, ssl_config: SslConfig | None = None
) -> MaskedReconstruction:
    """Construct the pretraining model for one arm.

    Args:
        config: Model configuration.
        ssl_config: Masking settings; defaults to the project's.

    Returns:
        A :class:`MaskedReconstruction`.
    """
    return MaskedReconstruction(config, ssl_config)


def transfer_encoder(
    pretrained: MaskedReconstruction, classifier: EcgClassifier
) -> EcgClassifier:
    """Copy pretrained encoder weights into a fresh classifier.

    Strict loading on purpose: a silently partial transfer would show up as a
    weak SSL result rather than as an error, and arms C and D would be reported
    as "SSL did not help" when in fact it had never been applied.

    Args:
        pretrained: The pretrained model.
        classifier: A classifier built from the same configuration.

    Returns:
        The same classifier, with encoder weights replaced. The head keeps its
        random initialisation.

    Raises:
        ValueError: If the two models were built from different configurations.
    """
    if pretrained.config != classifier.config:
        raise ValueError(
            "configuration mismatch between pretrained encoder and classifier: "
            f"{pretrained.config} vs {classifier.config}"
        )
    classifier.encoder.load_state_dict(pretrained.encoder.state_dict(), strict=True)
    return classifier


def baseline_loss(signal: torch.Tensor, token_mask: torch.Tensor, config: ModelConfig) -> float:
    """Loss of the trivial predictor that outputs zero everywhere.

    The signal is per-record normalised, so predicting zero is predicting the
    record's mean, and this value is close to 1.0. It is the number a
    pretraining run has to beat before any of its loss curve means anything --
    without it, a loss of 0.4 is unreadable.

    Args:
        signal: ``(batch, n_leads, n_samples)`` raw signal.
        token_mask: ``(batch, n_tokens)`` boolean tensor.
        config: Model configuration.

    Returns:
        The baseline MSE over masked positions.
    """
    target = patchify(signal, config)
    squared_error = target.pow(2).mean(dim=-1)
    return float(squared_error[token_mask].mean())
