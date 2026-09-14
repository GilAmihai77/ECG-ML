"""Tests for cohorts, label-efficiency subsets and batching.

Built on a synthetic store and a synthetic metadata frame, so the suite stays
offline and needs neither PTB-XL nor WFDB.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from ecg.data.datasets import (
    Cohort,
    EcgBatches,
    build_cohorts,
    describe_cohorts,
    nested_subsets,
)
from ecg.data.preprocess import LEADS, SCALE, WaveformStore, normalize_per_record
from ecg.data.ptbxl import SUPERCLASSES

N_SAMPLES = 40
N_RECORDS = 60


@pytest.fixture()
def store() -> WaveformStore:
    """A store whose row r holds a signal scaled by r, so rows are traceable."""
    rng = np.random.default_rng(0)
    base = rng.normal(0.0, 0.3, size=(N_RECORDS, N_SAMPLES, len(LEADS)))
    base *= (1.0 + np.arange(N_RECORDS))[:, None, None] * 0.01
    return WaveformStore(
        waveforms=np.round(base * SCALE).astype(np.int16),
        ecg_ids=np.arange(1, N_RECORDS + 1, dtype=np.int64),
        sampling_rate=100,
    )


@pytest.fixture()
def cohort() -> Cohort:
    """A 60-record cohort with a deterministic multi-label matrix."""
    rng = np.random.default_rng(1)
    labels = (rng.random((N_RECORDS, len(SUPERCLASSES))) < 0.3).astype(np.float32)
    return Cohort(
        name="train",
        ecg_ids=np.arange(1, N_RECORDS + 1, dtype=np.int64),
        labels=labels,
    )


class TestCohort:
    def test_length_and_positives(self, cohort: Cohort) -> None:
        assert len(cohort) == N_RECORDS
        assert set(cohort.positives) == set(SUPERCLASSES)
        assert sum(cohort.positives.values()) == int(cohort.labels.sum())

    def test_subset_keeps_labels_aligned(self, cohort: Cohort) -> None:
        picked = np.array([5, 0, 9])
        sub = cohort.subset(picked, name="s")
        assert sub.ecg_ids.tolist() == cohort.ecg_ids[picked].tolist()
        np.testing.assert_array_equal(sub.labels, cohort.labels[picked])


class TestNestedSubsets:
    def test_fractions_are_nested(self, cohort: Cohort) -> None:
        """20% must sit inside 50% inside 100%, or the curve is confounded."""
        subsets = nested_subsets(cohort, (0.2, 0.5, 1.0), seed=0)
        small = set(subsets[0.2].ecg_ids.tolist())
        medium = set(subsets[0.5].ecg_ids.tolist())
        full = set(subsets[1.0].ecg_ids.tolist())
        assert small < medium < full
        assert full == set(cohort.ecg_ids.tolist())

    def test_sizes_are_right(self, cohort: Cohort) -> None:
        subsets = nested_subsets(cohort, (0.2, 0.5, 1.0), seed=0)
        assert len(subsets[0.2]) == 12
        assert len(subsets[0.5]) == 30
        assert len(subsets[1.0]) == 60

    def test_same_seed_reproduces(self, cohort: Cohort) -> None:
        a = nested_subsets(cohort, (0.2,), seed=7)[0.2]
        b = nested_subsets(cohort, (0.2,), seed=7)[0.2]
        np.testing.assert_array_equal(a.ecg_ids, b.ecg_ids)

    def test_different_seed_selects_differently(self, cohort: Cohort) -> None:
        a = nested_subsets(cohort, (0.2,), seed=1)[0.2]
        b = nested_subsets(cohort, (0.2,), seed=2)[0.2]
        assert a.ecg_ids.tolist() != b.ecg_ids.tolist()

    def test_labels_follow_the_records(self, cohort: Cohort) -> None:
        sub = nested_subsets(cohort, (0.5,), seed=3)[0.5]
        lookup = dict(zip(cohort.ecg_ids.tolist(), cohort.labels))
        for ecg_id, labels in zip(sub.ecg_ids.tolist(), sub.labels):
            np.testing.assert_array_equal(labels, lookup[ecg_id])

    def test_invalid_fraction_is_rejected(self, cohort: Cohort) -> None:
        with pytest.raises(ValueError, match=r"\(0, 1\]"):
            nested_subsets(cohort, (0.0,))
        with pytest.raises(ValueError, match=r"\(0, 1\]"):
            nested_subsets(cohort, (1.5,))

    def test_empty_subset_is_loud(self) -> None:
        tiny = Cohort("t", np.array([1, 2]), np.zeros((2, 5), dtype=np.float32))
        with pytest.raises(ValueError, match="selects 0"):
            nested_subsets(tiny, (0.01,))


class TestEcgBatches:
    def test_shape_is_channels_first_raw_signal(self, store, cohort) -> None:
        """The model does the patching, so batches are (B, 12, T)."""
        batches = EcgBatches(store, cohort, batch_size=16)
        signal, labels = next(iter(batches))
        assert signal.shape == (16, len(LEADS), N_SAMPLES)
        assert labels.shape == (16, len(SUPERCLASSES))
        assert signal.dtype == torch.float32

    def test_every_record_appears_exactly_once_per_epoch(self, store, cohort) -> None:
        batches = EcgBatches(store, cohort, batch_size=16)
        seen = sum(int(s.shape[0]) for s, _ in batches)
        assert seen == N_RECORDS
        assert len(batches) == 4  # 60 -> 16+16+16+12

    def test_drop_last_drops_the_short_batch(self, store, cohort) -> None:
        batches = EcgBatches(store, cohort, batch_size=16, drop_last=True)
        sizes = [int(s.shape[0]) for s, _ in batches]
        assert sizes == [16, 16, 16]
        assert len(batches) == 3

    def test_normalization_matches_the_numpy_path(self, store, cohort) -> None:
        """The torch and numpy implementations must not drift apart."""
        batches = EcgBatches(store, cohort, batch_size=8, shuffle=False)
        signal, _ = next(iter(batches))
        rows = store.rows_of(cohort.ecg_ids[:8])
        expected = normalize_per_record(store.millivolts(rows))  # (8, T, 12)
        np.testing.assert_allclose(
            signal.numpy(), expected.transpose(0, 2, 1), atol=1e-5
        )

    def test_each_record_is_normalised_independently(self, store, cohort) -> None:
        batches = EcgBatches(store, cohort, batch_size=32, shuffle=False)
        signal, _ = next(iter(batches))
        per_record_mean = signal.reshape(signal.shape[0], -1).mean(dim=1)
        per_record_std = signal.reshape(signal.shape[0], -1).std(dim=1, unbiased=False)
        assert torch.allclose(per_record_mean, torch.zeros_like(per_record_mean), atol=1e-4)
        assert torch.allclose(per_record_std, torch.ones_like(per_record_std), atol=1e-3)

    def test_labels_stay_aligned_with_signal_when_shuffled(self, cohort) -> None:
        """A shuffle that desynchronised x and y would be silent and fatal.

        Every sample of record ``r`` is the constant ``r``, so a batch's signal
        names the record it came from and misalignment cannot hide.
        """
        counts = np.tile(
            np.arange(1, N_RECORDS + 1, dtype=np.int16)[:, None, None],
            (1, N_SAMPLES, len(LEADS)),
        )
        marked = WaveformStore(
            waveforms=counts,
            ecg_ids=np.arange(1, N_RECORDS + 1, dtype=np.int64),
            sampling_rate=100,
        )
        label_lookup = dict(zip(cohort.ecg_ids.tolist(), cohort.labels))
        batches = EcgBatches(
            marked, cohort, batch_size=7, shuffle=True, seed=5, normalize=False
        )
        seen: list[int] = []
        for signal, labels in batches:
            for row in range(signal.shape[0]):
                ecg_id = int(round(float(signal[row, 0, 0]) * SCALE))
                seen.append(ecg_id)
                np.testing.assert_array_equal(
                    labels[row].numpy(), label_lookup[ecg_id]
                )
        assert sorted(seen) == cohort.ecg_ids.tolist()

    def test_shuffle_is_reproducible_and_actually_shuffles(self, store, cohort) -> None:
        a = next(iter(EcgBatches(store, cohort, batch_size=60, shuffle=True, seed=3)))[1]
        b = next(iter(EcgBatches(store, cohort, batch_size=60, shuffle=True, seed=3)))[1]
        ordered = next(iter(EcgBatches(store, cohort, batch_size=60, shuffle=False)))[1]
        assert torch.equal(a, b)
        assert not torch.equal(a, ordered)

    def test_unshuffled_order_matches_cohort_order(self, store, cohort) -> None:
        """Evaluation relies on this to line predictions up with ecg_ids."""
        batches = EcgBatches(store, cohort, batch_size=60, shuffle=False)
        _, labels = next(iter(batches))
        np.testing.assert_array_equal(labels.numpy(), cohort.labels)

    def test_epochs_differ_under_shuffle(self, store, cohort) -> None:
        batches = EcgBatches(store, cohort, batch_size=60, shuffle=True, seed=0)
        first = next(iter(batches))[1].clone()
        second = next(iter(batches))[1].clone()
        assert not torch.equal(first, second)

    def test_stored_dtype_stays_int16(self, store, cohort) -> None:
        """Holding float32 on device would cost 4x for no benefit."""
        batches = EcgBatches(store, cohort, batch_size=8)
        assert batches.waveforms.dtype == torch.int16


class TestBuildCohorts:
    def test_splits_and_ssl_pool(self, ptbxl_metadata: pd.DataFrame) -> None:
        cohorts = build_cohorts(ptbxl_metadata)
        assert set(cohorts) == {"train", "val", "test", "ssl"}
        for name in ("train", "val", "test", "ssl"):
            assert cohorts[name].labels.shape[1] == len(SUPERCLASSES)
            assert len(cohorts[name]) == cohorts[name].labels.shape[0]

    def test_ssl_pool_is_training_folds_and_unfiltered(
        self, ptbxl_metadata: pd.DataFrame
    ) -> None:
        cohorts = build_cohorts(ptbxl_metadata)
        train_folds = ptbxl_metadata[ptbxl_metadata["split"] == "train"]
        assert len(cohorts["ssl"]) == len(train_folds)
        assert len(cohorts["ssl"]) >= len(cohorts["train"])

    def test_describe_reports_positives(self, ptbxl_metadata: pd.DataFrame) -> None:
        table = describe_cohorts(build_cohorts(ptbxl_metadata))
        assert "n_records" in table.columns
        assert set(SUPERCLASSES) <= set(table.columns)
