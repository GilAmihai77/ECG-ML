"""Tests for block masking of raw ECG signal.

The properties that matter are exactness (every record masked to the same
count, so losses are comparable), contiguity (blocks span a cardiac cycle
rather than scattering single patches), and that masking genuinely destroys the
information rather than merely marking it.
"""

from __future__ import annotations

import pytest
import torch

from ecg.models.config import ModelConfig, SslConfig
from ecg.models.embeddings import ConvPatchEmbedding, LinearPatchEmbedding
from ecg.models.masking import (
    MASK_VALUE,
    apply_mask,
    contaminated_fraction,
    expand_to_samples,
    mask_summary,
    reach_of,
    receptive_reach,
    sample_block_mask,
)

BATCH = 32
N_TOKENS = 20


def runs(row: torch.Tensor) -> list[int]:
    """Lengths of the contiguous ``True`` runs in a 1-D boolean tensor."""
    lengths: list[int] = []
    current = 0
    for value in row.tolist():
        if value:
            current += 1
        elif current:
            lengths.append(current)
            current = 0
    if current:
        lengths.append(current)
    return lengths


@pytest.fixture()
def config() -> ModelConfig:
    return ModelConfig()


@pytest.fixture()
def ssl_config() -> SslConfig:
    return SslConfig()


class TestSslConfig:
    def test_defaults_are_the_argued_ones(self, ssl_config: SslConfig) -> None:
        assert ssl_config.mask_ratio == 0.5
        assert ssl_config.mask_span == 2

    def test_masked_token_count(self, ssl_config: SslConfig) -> None:
        assert ssl_config.n_masked_tokens(N_TOKENS) == 10

    def test_block_sizes_sum_to_the_masked_count(self, ssl_config: SslConfig) -> None:
        assert ssl_config.block_sizes(N_TOKENS) == (2, 2, 2, 2, 2)

    def test_remainder_becomes_a_short_final_block(self) -> None:
        """Masked count stays exact rather than rounding to a span multiple."""
        sizes = SslConfig(mask_ratio=0.5, mask_span=4).block_sizes(N_TOKENS)
        assert sizes == (4, 4, 2)
        assert sum(sizes) == 10

    @pytest.mark.parametrize("ratio", [0.0, 1.0, -0.1, 1.5])
    def test_degenerate_ratio_is_rejected(self, ratio: float) -> None:
        with pytest.raises(ValueError, match=r"\(0, 1\)"):
            SslConfig(mask_ratio=ratio)

    def test_zero_span_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="at least 1"):
            SslConfig(mask_span=0)

    def test_ratio_that_rounds_away_is_loud(self) -> None:
        """At 20 tokens a small ratio rounds to zero masked patches."""
        with pytest.raises(ValueError, match="at least one masked"):
            SslConfig(mask_ratio=0.01).n_masked_tokens(N_TOKENS)

    def test_config_is_frozen(self, ssl_config: SslConfig) -> None:
        with pytest.raises(Exception):
            ssl_config.mask_ratio = 0.7  # type: ignore[misc]


class TestSampleBlockMask:
    def test_shape_and_dtype(self, ssl_config: SslConfig) -> None:
        mask = sample_block_mask(BATCH, N_TOKENS, ssl_config)
        assert mask.shape == (BATCH, N_TOKENS)
        assert mask.dtype == torch.bool

    def test_every_record_masks_the_same_count(self, ssl_config: SslConfig) -> None:
        """A varying count would make per-record losses incomparable."""
        mask = sample_block_mask(BATCH, N_TOKENS, ssl_config)
        assert mask.sum(dim=1).unique().tolist() == [10]

    def test_masks_are_contiguous_blocks(self, ssl_config: SslConfig) -> None:
        """Span 2 covers a cardiac cycle; adjacent blocks merge to even runs."""
        mask = sample_block_mask(BATCH, N_TOKENS, ssl_config, generator=_gen(0))
        for row in mask:
            assert all(length % 2 == 0 for length in runs(row)), row.tolist()

    def test_span_four_gives_runs_of_at_least_four(self) -> None:
        config = SslConfig(mask_ratio=0.4, mask_span=4)
        mask = sample_block_mask(BATCH, N_TOKENS, config, generator=_gen(1))
        for row in mask:
            assert all(length % 4 == 0 for length in runs(row)), row.tolist()

    def test_blocks_never_run_past_the_end(self) -> None:
        """The stars-and-bars placement must not overflow the sequence."""
        for ratio in (0.1, 0.3, 0.5, 0.7, 0.9):
            config = SslConfig(mask_ratio=ratio, mask_span=2)
            expected = config.n_masked_tokens(N_TOKENS)
            mask = sample_block_mask(256, N_TOKENS, config, generator=_gen(2))
            assert mask.sum(dim=1).unique().tolist() == [expected], ratio

    def test_every_position_is_reachable(self, ssl_config: SslConfig) -> None:
        """A placement bug that never masks the last token would be silent."""
        mask = sample_block_mask(512, N_TOKENS, ssl_config, generator=_gen(3))
        assert bool(mask.any(dim=0).all())

    def test_edge_tokens_are_masked_less_often(self, ssl_config: SslConfig) -> None:
        """Documented, not accidental.

        A block of length L covers token 0 only by starting there, while an
        interior token is covered by L different block positions, so contiguous
        non-wrapping blocks cannot have uniform marginals. The exact values
        follow from the stars-and-bars count: C(14,4)/C(15,5) = 1/3 at each end
        and (C(14,4)+C(13,4))/C(15,5) = 4/7 one token in. Pinned here so a
        future change to the sampler has to face the arithmetic.
        """
        mask = sample_block_mask(8192, N_TOKENS, ssl_config, generator=_gen(4))
        frequency = mask.float().mean(dim=0)
        assert float(frequency[0]) == pytest.approx(1 / 3, abs=0.02)
        assert float(frequency[-1]) == pytest.approx(1 / 3, abs=0.02)
        assert float(frequency[1]) == pytest.approx(4 / 7, abs=0.02)
        assert float(frequency[5:15].mean()) == pytest.approx(0.5, abs=0.02)

    def test_no_position_is_starved(self, ssl_config: SslConfig) -> None:
        """The edge bias must stay a bias, not become a blind spot."""
        mask = sample_block_mask(8192, N_TOKENS, ssl_config, generator=_gen(16))
        assert float(mask.float().mean(dim=0).min()) > 0.25

    def test_records_get_different_masks(self, ssl_config: SslConfig) -> None:
        mask = sample_block_mask(BATCH, N_TOKENS, ssl_config, generator=_gen(5))
        assert len({tuple(row.tolist()) for row in mask}) > 1

    def test_same_generator_seed_reproduces(self, ssl_config: SslConfig) -> None:
        """Integrity rule 4: the mask stream is part of the run's state."""
        a = sample_block_mask(BATCH, N_TOKENS, ssl_config, generator=_gen(7))
        b = sample_block_mask(BATCH, N_TOKENS, ssl_config, generator=_gen(7))
        assert torch.equal(a, b)

    def test_successive_draws_differ(self, ssl_config: SslConfig) -> None:
        generator = _gen(8)
        a = sample_block_mask(BATCH, N_TOKENS, ssl_config, generator=generator)
        b = sample_block_mask(BATCH, N_TOKENS, ssl_config, generator=generator)
        assert not torch.equal(a, b)


class TestApplyMask:
    def test_expand_to_samples(self) -> None:
        token_mask = torch.tensor([[True, False, True]])
        expanded = expand_to_samples(token_mask, 4)
        assert expanded.tolist() == [[True] * 4 + [False] * 4 + [True] * 4]

    def test_masked_samples_are_blanked_across_all_leads(self, config, ssl_config) -> None:
        """A token is the whole 12-lead column, so leads cannot be masked apart."""
        signal = torch.randn(BATCH, config.n_leads, config.n_samples)
        token_mask = sample_block_mask(BATCH, N_TOKENS, ssl_config, generator=_gen(9))
        masked = apply_mask(signal, token_mask, config.patch_samples)
        sample_mask = expand_to_samples(token_mask, config.patch_samples)
        assert float(masked.permute(0, 2, 1)[sample_mask].abs().max()) == MASK_VALUE

    def test_visible_samples_are_untouched(self, config, ssl_config) -> None:
        signal = torch.randn(BATCH, config.n_leads, config.n_samples)
        token_mask = sample_block_mask(BATCH, N_TOKENS, ssl_config, generator=_gen(10))
        masked = apply_mask(signal, token_mask, config.patch_samples)
        visible = ~expand_to_samples(token_mask, config.patch_samples)
        torch.testing.assert_close(
            masked.permute(0, 2, 1)[visible], signal.permute(0, 2, 1)[visible]
        )

    def test_input_is_not_modified_in_place(self, config, ssl_config) -> None:
        """The unmasked signal is also the reconstruction target."""
        signal = torch.randn(BATCH, config.n_leads, config.n_samples)
        original = signal.clone()
        apply_mask(
            signal,
            sample_block_mask(BATCH, N_TOKENS, ssl_config, generator=_gen(11)),
            config.patch_samples,
        )
        torch.testing.assert_close(signal, original)


class TestReceptiveReach:
    def test_linear_embedder_reaches_nowhere(self, config: ModelConfig) -> None:
        embedder = LinearPatchEmbedding(config)
        assert reach_of(config, embedder.receptive_field) == 0

    def test_conv_embedder_reaches_one_token(self, config: ModelConfig) -> None:
        """127 samples against a 50-sample patch: 38.5 past each edge."""
        embedder = ConvPatchEmbedding(config)
        assert reach_of(config, embedder.receptive_field) == 1

    def test_reach_grows_with_receptive_field(self) -> None:
        assert receptive_reach(50, 50) == 0
        assert receptive_reach(150, 50) == 1
        assert receptive_reach(250, 50) == 2


class TestContaminatedFraction:
    def test_zero_when_nothing_reaches(self) -> None:
        mask = torch.tensor([[True, True, False, False]])
        assert contaminated_fraction(mask, reach=0) == 0.0

    def test_counts_visible_neighbours_of_masked_tokens(self) -> None:
        """Hand-checked: only token 2 borders the masked run."""
        mask = torch.tensor([[True, True, False, False, False, False]])
        assert contaminated_fraction(mask, reach=1) == pytest.approx(0.25)

    def test_scattered_masks_contaminate_more_than_blocks(self) -> None:
        """Part of why blocks are preferred: fewer boundaries per masked token."""
        blocked = torch.tensor([[True] * 4 + [False] * 8])
        scattered = torch.tensor([[True, False] * 4 + [False] * 4])
        assert contaminated_fraction(scattered, reach=1) > contaminated_fraction(
            blocked, reach=1
        )

    def test_summary_reports_the_realised_ratio(self, ssl_config: SslConfig) -> None:
        mask = sample_block_mask(BATCH, N_TOKENS, ssl_config, generator=_gen(12))
        summary = mask_summary(mask, reach=1)
        assert summary["mask_ratio"] == pytest.approx(0.5)
        assert summary["masked_tokens"] == pytest.approx(10.0)
        assert 0.0 < summary["contaminated_fraction"] <= 1.0


def _gen(seed: int) -> torch.Generator:
    """A seeded CPU generator."""
    return torch.Generator().manual_seed(seed)
