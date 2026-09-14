"""Tests for masked-reconstruction pretraining.

The load-bearing test here is
:meth:`TestNoLeak.test_masked_content_cannot_reach_the_prediction`: if anything
about the masked region reached the encoder, the pretext task would be
solvable by copying and every SSL result in the study would be meaningless.
"""

from __future__ import annotations

import pytest
import torch

from ecg.models.config import ModelConfig, SslConfig
from ecg.models.embeddings import patchify
from ecg.models.encoder import build_classifier, count_parameters
from ecg.models.masking import sample_block_mask
from ecg.models.ssl import (
    MaskedReconstruction,
    baseline_loss,
    build_pretrainer,
    transfer_encoder,
)

BATCH = 8


@pytest.fixture()
def config() -> ModelConfig:
    return ModelConfig()


@pytest.fixture()
def tiny() -> ModelConfig:
    """A small configuration for tests that actually optimise."""
    return ModelConfig(n_samples=200, d_model=32, n_layers=1, n_heads=2)


@pytest.fixture()
def signal(config: ModelConfig) -> torch.Tensor:
    """Per-record normalised signal, as :class:`EcgBatches` produces."""
    raw = torch.randn(BATCH, config.n_leads, config.n_samples, generator=_gen(0))
    flat = raw.reshape(BATCH, -1)
    mean = flat.mean(dim=1).view(-1, 1, 1)
    std = flat.std(dim=1, unbiased=False).view(-1, 1, 1)
    return (raw - mean) / std


class TestShapesAndLoss:
    @pytest.mark.parametrize("arm", ["linear", "conv"])
    def test_reconstructs_every_patch(self, config, signal, arm) -> None:
        model = build_pretrainer(config.for_arm(arm))
        mask = model.sample_mask(BATCH, generator=_gen(1))
        predicted = model(signal, mask)
        assert predicted.shape == (BATCH, config.n_tokens, config.patch_features)

    def test_prediction_layout_matches_the_target(self, config, signal) -> None:
        """Prediction and target must be patchified identically or the loss
        compares lead I against lead II."""
        model = build_pretrainer(config)
        mask = model.sample_mask(BATCH, generator=_gen(2))
        assert model(signal, mask).shape == patchify(signal, config).shape

    def test_loss_is_a_finite_scalar(self, config, signal) -> None:
        model = build_pretrainer(config)
        loss = model.loss(signal, model.sample_mask(BATCH, generator=_gen(3)))
        assert loss.shape == ()
        assert torch.isfinite(loss)

    def test_loss_covers_masked_positions_only(self, config, signal) -> None:
        """Checked against an independent baseline rather than by restating the
        implementation: with a zeroed decoder the loss must equal the
        predict-zero baseline over exactly the masked patches."""
        model = build_pretrainer(config).eval()
        with torch.no_grad():
            model.decoder.weight.zero_()
            model.decoder.bias.zero_()
        mask = model.sample_mask(BATCH, generator=_gen(4))
        with torch.no_grad():
            loss = float(model.loss(signal, mask))
        assert loss == pytest.approx(baseline_loss(signal, mask, config), rel=1e-5)

    def test_baseline_is_near_one_for_normalised_signal(self, config, signal) -> None:
        """Predicting zero is predicting the record mean; the run must beat it."""
        mask = sample_block_mask(BATCH, config.n_tokens, SslConfig(), generator=_gen(5))
        assert baseline_loss(signal, mask, config) == pytest.approx(1.0, abs=0.15)


class TestNoLeak:
    @pytest.mark.parametrize("arm", ["linear", "conv"])
    def test_masked_content_cannot_reach_the_prediction(self, config, signal, arm) -> None:
        """Two signals differing only inside the mask must predict identically.

        This is the whole justification for masking raw samples rather than
        token embeddings. If it failed for the conv arm, the stem's 127-sample
        receptive field would be carrying masked content into its neighbours
        and the pretext task would collapse into copying.
        """
        model = build_pretrainer(config.for_arm(arm)).eval()
        mask = model.sample_mask(BATCH, generator=_gen(6))
        sample_mask = mask.repeat_interleave(config.patch_samples, dim=1)

        altered = signal.clone()
        altered.permute(0, 2, 1)[sample_mask] += 7.0

        with torch.no_grad():
            torch.testing.assert_close(model(signal, mask), model(altered, mask))

    def test_visible_content_does_reach_the_prediction(self, config, signal) -> None:
        """The converse, so the test above cannot pass by the model ignoring
        its input entirely."""
        model = build_pretrainer(config).eval()
        mask = model.sample_mask(BATCH, generator=_gen(7))
        visible = ~mask.repeat_interleave(config.patch_samples, dim=1)

        altered = signal.clone()
        altered.permute(0, 2, 1)[visible] += 7.0

        with torch.no_grad():
            assert not torch.allclose(model(signal, mask), model(altered, mask))


class TestStepAndMetrics:
    def test_step_reports_the_mask_and_loss(self, config, signal) -> None:
        step = build_pretrainer(config).step(signal, generator=_gen(8))
        assert step.token_mask.shape == (BATCH, config.n_tokens)
        assert torch.isfinite(step.loss)
        assert step.metrics["mask_ratio"] == pytest.approx(0.5)
        assert step.metrics["ssl_loss"] == pytest.approx(float(step.loss.detach()))

    def test_contamination_separates_the_arms(self, config, signal) -> None:
        """The metric that keeps the arms' differing task honest and logged."""
        linear = build_pretrainer(config.for_arm("linear")).step(signal, generator=_gen(9))
        conv = build_pretrainer(config.for_arm("conv")).step(signal, generator=_gen(9))
        assert linear.metrics["contaminated_fraction"] == 0.0
        assert conv.metrics["contaminated_fraction"] > 0.0

    def test_reach_matches_the_embedder(self, config: ModelConfig) -> None:
        assert build_pretrainer(config.for_arm("linear")).reach == 0
        assert build_pretrainer(config.for_arm("conv")).reach == 1

    def test_step_is_reproducible(self, config, signal) -> None:
        a = build_pretrainer(config).step(signal, generator=_gen(10))
        b = build_pretrainer(config).step(signal, generator=_gen(10))
        assert torch.equal(a.token_mask, b.token_mask)

    def test_mask_ratio_is_configurable(self, config, signal) -> None:
        model = build_pretrainer(config, SslConfig(mask_ratio=0.7, mask_span=2))
        step = model.step(signal, generator=_gen(11))
        assert step.metrics["masked_tokens"] == pytest.approx(14.0)


class TestOptimisation:
    @pytest.mark.parametrize("arm", ["linear", "conv"])
    def test_gradients_reach_every_parameter(self, tiny, arm) -> None:
        model = build_pretrainer(tiny.for_arm(arm))
        batch = torch.randn(4, tiny.n_leads, tiny.n_samples, generator=_gen(12))
        model.step(batch, generator=_gen(13)).loss.backward()
        for name, parameter in model.named_parameters():
            assert parameter.grad is not None, name
            assert torch.isfinite(parameter.grad).all(), name
            assert parameter.grad.abs().sum() > 0, name

    @pytest.mark.parametrize("arm", ["linear", "conv"])
    def test_the_task_is_learnable(self, tiny, arm) -> None:
        """Overfit one batch: a pretext task that cannot be optimised at all
        would show up here rather than after an hour of cloud time."""
        torch.manual_seed(0)
        model = build_pretrainer(tiny.for_arm(arm))
        batch = torch.randn(4, tiny.n_leads, tiny.n_samples, generator=_gen(14))
        optimiser = torch.optim.AdamW(model.parameters(), lr=1e-3)
        mask = model.sample_mask(4, generator=_gen(15))

        first = float(model.loss(batch, mask).detach())
        for _ in range(60):
            loss = model.loss(batch, mask)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
        assert float(loss.detach()) < first * 0.8


class TestTransferEncoder:
    def test_weights_arrive_in_the_classifier(self, config: ModelConfig) -> None:
        pretrained = build_pretrainer(config)
        classifier = transfer_encoder(pretrained, build_classifier(config))
        for name, parameter in pretrained.encoder.named_parameters():
            torch.testing.assert_close(
                parameter, dict(classifier.encoder.named_parameters())[name]
            )

    def test_head_stays_random(self, config: ModelConfig) -> None:
        """Fine-tuning must learn its own head, not inherit the decoder's."""
        classifier = build_classifier(config)
        before = classifier.head[2].weight.clone()
        transfer_encoder(build_pretrainer(config), classifier)
        torch.testing.assert_close(classifier.head[2].weight, before)

    def test_decoder_is_not_carried_over(self, config: ModelConfig) -> None:
        """The decoder is pretraining scaffolding and is discarded."""
        pretrained = build_pretrainer(config)
        classifier = transfer_encoder(pretrained, build_classifier(config))
        assert count_parameters(classifier) < count_parameters(pretrained)

    def test_mismatched_config_is_rejected(self, config: ModelConfig) -> None:
        """A silent partial transfer would read as 'SSL did not help'."""
        pretrained = build_pretrainer(config.for_arm("conv"))
        with pytest.raises(ValueError, match="configuration mismatch"):
            transfer_encoder(pretrained, build_classifier(config.for_arm("linear")))

    @pytest.mark.parametrize("arm", ["linear", "conv"])
    def test_transferred_model_still_classifies(self, config, signal, arm) -> None:
        pretrained = build_pretrainer(config.for_arm(arm))
        classifier = transfer_encoder(
            pretrained, build_classifier(config.for_arm(arm))
        ).eval()
        with torch.no_grad():
            logits = classifier(signal)
        assert logits.shape == (BATCH, config.n_classes)
        assert torch.isfinite(logits).all()


class TestSharedArchitecture:
    def test_decoder_is_identical_across_arms(self, config: ModelConfig) -> None:
        """Integrity rule 3 extends to the SSL decoder."""
        linear = build_pretrainer(config.for_arm("linear"))
        conv = build_pretrainer(config.for_arm("conv"))
        assert count_parameters(linear.decoder) == count_parameters(conv.decoder)

    def test_ssl_config_is_shared_not_per_arm(self, config: ModelConfig) -> None:
        """Tuning the mask per arm would make SSL part of the comparison."""
        ssl_config = SslConfig(mask_ratio=0.3)
        for arm in ("linear", "conv"):
            assert build_pretrainer(config.for_arm(arm), ssl_config).ssl_config is ssl_config

    def test_encoder_matches_the_supervised_one(self, config: ModelConfig) -> None:
        pretrained = build_pretrainer(config)
        classifier = build_classifier(config)
        assert {n: tuple(p.shape) for n, p in pretrained.encoder.named_parameters()} == {
            n: tuple(p.shape) for n, p in classifier.encoder.named_parameters()
        }

    def test_default_pretrainer_uses_the_project_defaults(self, config) -> None:
        model = MaskedReconstruction(config)
        assert model.ssl_config == SslConfig()


def _gen(seed: int) -> torch.Generator:
    """A seeded CPU generator."""
    return torch.Generator().manual_seed(seed)
