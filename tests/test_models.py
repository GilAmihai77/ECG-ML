"""Tests for the patch embedders and the shared encoder.

Three things are load-bearing for the research question and are tested harder
than the rest:

* the two embedders emit the **same number of tokens**, so the encoder sees the
  same sequence length in every arm;
* their **parameter counts match within 10%**, so a conv win cannot be capacity;
* the conv stem is **not** the degenerate ``kernel == stride == patch`` case,
  which is mathematically identical to the linear projection and would make the
  comparison vacuous.
"""

from __future__ import annotations

import pytest
import torch

from ecg.models.config import DEFAULT_CONV_HIDDEN, ModelConfig
from ecg.models.embeddings import (
    ConvPatchEmbedding,
    LinearPatchEmbedding,
    build_embedder,
)
from ecg.models.encoder import (
    EcgClassifier,
    EcgEncoder,
    build_classifier,
    count_parameters,
    embedder_parity,
    parameter_report,
)

BATCH = 4
#: Parameter-count agreement the project requires between the two embedders.
PARITY_TOLERANCE = 0.10


@pytest.fixture()
def config() -> ModelConfig:
    """The default configuration: 10 s at 100 Hz, 500 ms patches, 20 tokens."""
    return ModelConfig()


@pytest.fixture()
def signal(config: ModelConfig) -> torch.Tensor:
    """A batch shaped like one from :class:`ecg.data.datasets.EcgBatches`."""
    generator = torch.Generator().manual_seed(0)
    return torch.randn(
        BATCH, config.n_leads, config.n_samples, generator=generator
    )


class TestModelConfig:
    def test_token_geometry(self, config: ModelConfig) -> None:
        assert config.n_tokens == 20
        assert config.patch_features == 600
        assert config.d_ffn == 512

    def test_for_arm_changes_only_the_embedder(self, config: ModelConfig) -> None:
        """Integrity rule 3, as a property of the type."""
        conv = config.for_arm("conv")
        assert conv.embedder == "conv"
        for field in ("n_leads", "n_samples", "patch_samples", "d_model",
                      "n_layers", "n_heads", "ffn_mult", "dropout", "n_classes"):
            assert getattr(conv, field) == getattr(config, field)

    def test_config_is_frozen(self, config: ModelConfig) -> None:
        """Checkpoints record the config; a mutable one could not be trusted."""
        with pytest.raises(Exception):
            config.d_model = 256  # type: ignore[misc]

    def test_indivisible_patch_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="whole number of"):
            ModelConfig(n_samples=1000, patch_samples=64)

    def test_heads_must_divide_d_model(self) -> None:
        with pytest.raises(ValueError, match="divisible by n_heads"):
            ModelConfig(d_model=128, n_heads=5)

    def test_mismatched_conv_strides_are_rejected(self) -> None:
        """Strides that miss patch_samples would give the arms different lengths."""
        with pytest.raises(ValueError, match="strides multiply to"):
            ModelConfig(conv_hidden=((48, 7, 5, 1), (80, 5, 4, 0)))

    def test_unknown_embedder_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="unknown embedder"):
            ModelConfig(embedder="mlp")  # type: ignore[arg-type]


class TestEmbedders:
    @pytest.mark.parametrize("arm", ["linear", "conv"])
    def test_output_shape(self, config: ModelConfig, signal, arm: str) -> None:
        embedder = build_embedder(config.for_arm(arm))
        tokens = embedder(signal)
        assert tokens.shape == (BATCH, config.n_tokens, config.d_model)

    def test_both_arms_emit_the_same_token_count(self, config, signal) -> None:
        """The encoder must see one sequence length, whichever stem ran."""
        linear = LinearPatchEmbedding(config)(signal)
        conv = ConvPatchEmbedding(config)(signal)
        assert linear.shape == conv.shape

    def test_build_embedder_selects_by_config(self, config: ModelConfig) -> None:
        assert isinstance(build_embedder(config.for_arm("linear")), LinearPatchEmbedding)
        assert isinstance(build_embedder(config.for_arm("conv")), ConvPatchEmbedding)

    def test_parameter_counts_match_within_tolerance(self, config) -> None:
        """Otherwise a conv win is extra capacity, not inductive bias."""
        parity = embedder_parity(config)
        assert parity["relative_difference"] < PARITY_TOLERANCE, parity

    def test_linear_embedder_parameter_count_is_exact(self, config) -> None:
        linear = LinearPatchEmbedding(config)
        assert count_parameters(linear) == config.patch_features * config.d_model + config.d_model

    def test_conv_stem_is_not_the_degenerate_case(self, config) -> None:
        """Conv1d(kernel=stride=patch) IS the linear projection. Guard against it."""
        conv = ConvPatchEmbedding(config)
        assert conv.receptive_field > config.patch_samples
        assert conv.receptive_field == 127  # 1.27 s at 100 Hz

    def test_a_degenerate_stem_would_fail_this_guard(self) -> None:
        """Document the failure mode by constructing it deliberately."""
        degenerate = ModelConfig(
            embedder="conv", conv_hidden=(), conv_final=(50, 50, 0)
        )
        assert ConvPatchEmbedding(degenerate).receptive_field == 50

    def test_conv_stem_is_nonlinear(self, config, signal) -> None:
        """A stem without nonlinearity collapses to one linear map."""
        conv = ConvPatchEmbedding(config).eval()
        with torch.no_grad():
            doubled = conv(signal * 2.0)
            scaled = conv(signal) * 2.0
        assert not torch.allclose(doubled, scaled, atol=1e-3)

    def test_linear_embedder_sees_only_its_own_patch(self, config) -> None:
        """The contrast under test: a linear token has no view of its neighbours."""
        embedder = LinearPatchEmbedding(config).eval()
        a = torch.zeros(1, config.n_leads, config.n_samples)
        b = a.clone()
        b[0, :, config.patch_samples :] = 5.0  # disturb everything after patch 0
        with torch.no_grad():
            assert torch.allclose(embedder(a)[:, 0], embedder(b)[:, 0])

    def test_conv_token_sees_beyond_its_patch(self, config) -> None:
        """The same probe on the conv stem must change token 0."""
        embedder = ConvPatchEmbedding(config).eval()
        a = torch.zeros(1, config.n_leads, config.n_samples)
        b = a.clone()
        b[0, :, config.patch_samples :] = 5.0
        with torch.no_grad():
            assert not torch.allclose(embedder(a)[:, 0], embedder(b)[:, 0], atol=1e-4)

    def test_wrong_lead_count_is_loud(self, config: ModelConfig) -> None:
        embedder = LinearPatchEmbedding(config)
        with pytest.raises(ValueError, match="expected \\(batch, 12, 1000\\)"):
            embedder(torch.zeros(2, 8, config.n_samples))

    def test_wrong_length_is_loud(self, config: ModelConfig) -> None:
        embedder = ConvPatchEmbedding(config)
        with pytest.raises(ValueError, match="expected \\(batch, 12, 1000\\)"):
            embedder(torch.zeros(2, config.n_leads, 900))

    def test_missing_batch_dimension_is_loud(self, config: ModelConfig) -> None:
        with pytest.raises(ValueError, match="batch, n_leads, n_samples"):
            LinearPatchEmbedding(config)(torch.zeros(config.n_leads, config.n_samples))

    def test_patches_are_contiguous_windows(self, config: ModelConfig) -> None:
        """Token p must hold samples [p*patch, (p+1)*patch), not a stride pattern."""
        embedder = LinearPatchEmbedding(config)
        with torch.no_grad():
            embedder.project.weight.zero_()
            embedder.project.bias.zero_()
            embedder.project.weight[0, 0] = 1.0  # lead 0, first sample of patch
        signal = torch.zeros(1, config.n_leads, config.n_samples)
        signal[0, 0, :] = torch.arange(config.n_samples, dtype=torch.float32)
        with torch.no_grad():
            out = embedder(signal)[0, :, 0]
        expected = torch.arange(0, config.n_samples, config.patch_samples).float()
        torch.testing.assert_close(out, expected)


class TestEncoder:
    def test_returns_token_representations(self, config, signal) -> None:
        """SSL needs per-token output; pooling belongs to the supervised head."""
        encoder = EcgEncoder(config)
        assert encoder(signal).shape == (BATCH, config.n_tokens, config.d_model)

    @pytest.mark.parametrize("arm", ["linear", "conv"])
    def test_classifier_emits_one_logit_per_superclass(self, config, signal, arm) -> None:
        model = build_classifier(config.for_arm(arm))
        logits = model(signal)
        assert logits.shape == (BATCH, config.n_classes)
        assert torch.isfinite(logits).all()

    def test_logits_are_not_normalised(self, config, signal) -> None:
        """Labels are multi-label, so a softmax here would be a silent bug."""
        model = build_classifier(config).eval()
        with torch.no_grad():
            probabilities = torch.sigmoid(model(signal))
        assert not torch.allclose(
            probabilities.sum(dim=1), torch.ones(BATCH), atol=1e-3
        )

    def test_transformer_is_identical_across_arms(self, config: ModelConfig) -> None:
        """Integrity rule 3: only the embedder may differ."""
        linear = build_classifier(config.for_arm("linear"))
        conv = build_classifier(config.for_arm("conv"))
        assert count_parameters(linear.encoder.blocks) == count_parameters(
            conv.encoder.blocks
        )
        assert count_parameters(linear.head) == count_parameters(conv.head)
        linear_shapes = {n: tuple(p.shape) for n, p in linear.encoder.blocks.named_parameters()}
        conv_shapes = {n: tuple(p.shape) for n, p in conv.encoder.blocks.named_parameters()}
        assert linear_shapes == conv_shapes

    def test_eval_mode_is_deterministic(self, config, signal) -> None:
        """Dropout off means a checkpoint scores a record the same way twice."""
        model = build_classifier(config).eval()
        with torch.no_grad():
            torch.testing.assert_close(model(signal), model(signal))

    def test_train_mode_dropout_is_active(self, config, signal) -> None:
        model = build_classifier(config).train()
        torch.manual_seed(0)
        first = model(signal)
        second = model(signal)
        assert not torch.allclose(first, second)

    def test_same_seed_builds_the_same_weights(self, config: ModelConfig) -> None:
        """Integrity rule 6: a run must be reproducible from its config."""
        torch.manual_seed(11)
        a = build_classifier(config)
        torch.manual_seed(11)
        b = build_classifier(config)
        for (_, left), (_, right) in zip(a.named_parameters(), b.named_parameters()):
            torch.testing.assert_close(left, right)

    @pytest.mark.parametrize("arm", ["linear", "conv"])
    def test_gradients_reach_every_parameter(self, config, signal, arm) -> None:
        """A stem whose gradient never arrives would train as a frozen random map."""
        model = build_classifier(config.for_arm(arm))
        targets = torch.zeros(BATCH, config.n_classes)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            model(signal), targets
        )
        loss.backward()
        for name, parameter in model.named_parameters():
            assert parameter.grad is not None, name
            assert torch.isfinite(parameter.grad).all(), name
            assert parameter.grad.abs().sum() > 0, name

    def test_batch_size_one_works(self, config: ModelConfig) -> None:
        """GroupNorm rather than BatchNorm, so a single record is fine."""
        model = build_classifier(config.for_arm("conv")).eval()
        with torch.no_grad():
            out = model(torch.randn(1, config.n_leads, config.n_samples))
        assert out.shape == (1, config.n_classes)

    def test_records_are_scored_independently(self, config, signal) -> None:
        """No cross-record statistics anywhere in the model."""
        model = build_classifier(config.for_arm("conv")).eval()
        with torch.no_grad():
            batched = model(signal)
            alone = model(signal[:1])
        torch.testing.assert_close(batched[:1], alone, atol=1e-5, rtol=1e-4)


class TestParameterReport:
    def test_report_adds_up(self, config: ModelConfig) -> None:
        report = parameter_report(config)
        parts = report["embedder"] + report["positions"] + report["transformer"] + report["head"]
        assert parts == report["total"]

    def test_only_the_embedder_differs_between_arms(self, config: ModelConfig) -> None:
        linear = parameter_report(config.for_arm("linear"))
        conv = parameter_report(config.for_arm("conv"))
        for part in ("positions", "transformer", "head"):
            assert linear[part] == conv[part]
        assert linear["embedder"] != conv["embedder"]

    def test_parity_holds_at_the_cloud_width(self) -> None:
        """The cloud runs use d_model=256; parity must survive the change."""
        parity = embedder_parity(ModelConfig(d_model=256, n_heads=8))
        assert parity["relative_difference"] < PARITY_TOLERANCE, parity

    def test_default_stem_is_the_documented_one(self) -> None:
        assert DEFAULT_CONV_HIDDEN == ((48, 7, 5, 1), (80, 5, 5, 0))

    @pytest.mark.parametrize(("d_model", "n_heads"), [(64, 4), (128, 4), (256, 8), (384, 8)])
    def test_parity_holds_at_every_width_we_might_use(self, d_model, n_heads) -> None:
        """The stem width is solved for, so parity is not tied to one d_model."""
        parity = embedder_parity(ModelConfig(d_model=d_model, n_heads=n_heads))
        assert parity["relative_difference"] < PARITY_TOLERANCE, (d_model, parity)

    def test_stem_width_tracks_d_model(self) -> None:
        wide = ModelConfig(d_model=256, n_heads=8)
        assert wide.conv_hidden[1][0] != ModelConfig().conv_hidden[1][0]

    def test_an_explicit_stem_is_not_overridden(self) -> None:
        """Integrity rule 7: a spec written down in a config is the one that runs."""
        explicit = ((48, 7, 5, 1), (88, 5, 5, 0))
        assert ModelConfig(d_model=256, n_heads=8, conv_hidden=explicit).conv_hidden == explicit

    def test_for_arm_keeps_the_resolved_stem(self) -> None:
        config = ModelConfig(d_model=256, n_heads=8)
        assert config.for_arm("conv").conv_hidden == config.conv_hidden
