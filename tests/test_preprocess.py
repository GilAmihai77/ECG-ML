"""Tests for the waveform store that feeds training.

Waveforms are synthesised rather than read from disk: the build takes its
reader as an argument precisely so the conversion can be tested without WFDB
files, which keeps this suite offline like the rest.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ecg.data.preprocess import (
    LEADS,
    SCALE,
    WaveformStore,
    build_waveform_store,
    normalize_per_record,
    to_counts,
)

N_SAMPLES = 1000  # 10 s at 100 Hz


def fake_signal(seed: int, n_samples: int = N_SAMPLES) -> np.ndarray:
    """A deterministic 12-lead waveform in plausible millivolt range."""
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, 0.3, size=(n_samples, len(LEADS))).astype(np.float32)


@pytest.fixture()
def metadata() -> pd.DataFrame:
    """Three records, indexed like the real metadata frame."""
    return pd.DataFrame(
        {"patient_id": [10, 11, 12], "split": ["train", "val", "test"]},
        index=pd.Index([1, 2, 3], name="ecg_id"),
    )


@pytest.fixture()
def loader():
    """A reader returning a distinct waveform per record."""

    def _loader(layout, row, *, sampling_rate):
        return fake_signal(int(row["patient_id"])), list(LEADS)

    return _loader


class TestToCounts:
    def test_millivolts_round_trip_exactly(self) -> None:
        """PTB-XL's own gain is 1000, so integer counts are microvolts."""
        mv = np.array([[0.001, -0.002], [1.234, -3.456]], dtype=np.float32)
        counts = to_counts(mv)
        assert counts.dtype == np.int16
        np.testing.assert_allclose(counts / SCALE, mv, atol=1e-6)

    def test_nan_becomes_zero(self) -> None:
        counts = to_counts(np.array([np.nan, 0.5], dtype=np.float32))
        assert counts.tolist() == [0, 500]

    def test_overflow_is_loud(self) -> None:
        """Silent wraparound would corrupt a record invisibly."""
        with pytest.raises(ValueError, match="exceeds int16"):
            to_counts(np.array([40.0], dtype=np.float32))


class TestNormalizePerRecord:
    def test_zero_mean_unit_variance(self) -> None:
        out = normalize_per_record(fake_signal(1))
        assert abs(float(out.mean())) < 1e-4
        assert abs(float(out.std()) - 1.0) < 1e-3

    def test_lead_ratios_are_preserved(self) -> None:
        """Per-lead scaling would destroy axis and the HYP voltage criteria."""
        signal = fake_signal(2)
        signal[:, 0] *= 4.0  # make one lead much larger
        out = normalize_per_record(signal)
        before = signal[:, 0].std() / signal[:, 1].std()
        after = out[:, 0].std() / out[:, 1].std()
        assert abs(before - after) < 1e-3

    def test_flat_record_does_not_divide_by_zero(self) -> None:
        out = normalize_per_record(np.zeros((10, len(LEADS)), dtype=np.float32))
        assert np.isfinite(out).all()
        assert float(np.abs(out).max()) == 0.0

    def test_batched_records_are_normalised_independently(self) -> None:
        batch = np.stack([fake_signal(3), fake_signal(4) * 10.0])
        out = normalize_per_record(batch)
        for row in out:
            assert abs(float(row.mean())) < 1e-4
            assert abs(float(row.std()) - 1.0) < 1e-3

    def test_statistics_never_cross_records(self) -> None:
        """Integrity rule 2: nothing may leak between records or splits."""
        a, b = fake_signal(5), fake_signal(6)
        alone = normalize_per_record(a)
        together = normalize_per_record(np.stack([a, b]))[0]
        np.testing.assert_allclose(alone, together, atol=1e-5)


class TestBuildWaveformStore:
    def test_builds_one_row_per_record(self, metadata, loader) -> None:
        store, failures = build_waveform_store(
            None, metadata, loader=loader, progress_every=0
        )
        assert len(store) == 3
        assert store.waveforms.shape == (3, N_SAMPLES, 12)
        assert store.waveforms.dtype == np.int16
        assert failures.empty

    def test_row_order_follows_the_metadata_frame(self, metadata, loader) -> None:
        store, _ = build_waveform_store(None, metadata, loader=loader, progress_every=0)
        assert store.ecg_ids.tolist() == [1, 2, 3]
        np.testing.assert_allclose(
            store.millivolts(store.row_of(2)), fake_signal(11), atol=1e-3
        )

    def test_unreadable_record_is_reported_not_dropped_silently(self, metadata) -> None:
        """Integrity rule 8."""

        def flaky(layout, row, *, sampling_rate):
            if row["patient_id"] == 11:
                raise OSError("file is missing")
            return fake_signal(int(row["patient_id"])), list(LEADS)

        store, failures = build_waveform_store(
            None, metadata, loader=flaky, progress_every=0
        )
        assert len(store) == 2
        assert store.ecg_ids.tolist() == [1, 3]
        assert failures["ecg_id"].tolist() == [2]
        assert "file is missing" in failures.loc[0, "reason"]

    def test_wrong_lead_order_is_rejected(self, metadata) -> None:
        """Silently mislabelled leads would be invisible downstream."""

        def scrambled(layout, row, *, sampling_rate):
            return fake_signal(1), list(reversed(LEADS))

        store, failures = build_waveform_store(
            None, metadata, loader=scrambled, progress_every=0
        )
        assert len(store) == 0
        assert len(failures) == 3
        assert "unexpected lead order" in failures.loc[0, "reason"]

    def test_wrong_length_is_rejected(self, metadata) -> None:
        def short(layout, row, *, sampling_rate):
            return fake_signal(1, n_samples=999), list(LEADS)

        _, failures = build_waveform_store(
            None, metadata, loader=short, progress_every=0
        )
        assert len(failures) == 3
        assert "expected shape" in failures.loc[0, "reason"]

    def test_non_finite_samples_are_zeroed_and_reported(self, metadata) -> None:
        def with_nans(layout, row, *, sampling_rate):
            signal = fake_signal(int(row["patient_id"]))
            signal[0, 0] = np.nan
            return signal, list(LEADS)

        store, failures = build_waveform_store(
            None, metadata, loader=with_nans, progress_every=0
        )
        assert len(store) == 3  # kept, but flagged
        assert len(failures) == 3
        assert "non-finite" in failures.loc[0, "reason"]
        assert store.waveforms[0, 0, 0] == 0

    def test_sampling_rate_sets_the_length(self, metadata) -> None:
        def at_500(layout, row, *, sampling_rate):
            return fake_signal(1, n_samples=5000), list(LEADS)

        store, failures = build_waveform_store(
            None, metadata, sampling_rate=500, loader=at_500, progress_every=0
        )
        assert failures.empty
        assert store.n_samples == 5000


class TestStoreRoundTrip:
    def test_save_and_load(self, tmp_path: Path, metadata, loader) -> None:
        store, _ = build_waveform_store(None, metadata, loader=loader, progress_every=0)
        store.save(tmp_path / "store")
        loaded = WaveformStore.load(tmp_path / "store")

        np.testing.assert_array_equal(loaded.waveforms, store.waveforms)
        np.testing.assert_array_equal(loaded.ecg_ids, store.ecg_ids)
        assert loaded.sampling_rate == 100
        assert loaded.scale == SCALE
        assert loaded.leads == LEADS

    def test_load_rejects_a_directory_that_is_not_a_store(self, tmp_path: Path) -> None:
        (tmp_path / "empty").mkdir()
        with pytest.raises(FileNotFoundError, match="not a waveform store"):
            WaveformStore.load(tmp_path / "empty")

    def test_rows_of_preserves_request_order(self, metadata, loader) -> None:
        store, _ = build_waveform_store(None, metadata, loader=loader, progress_every=0)
        assert store.rows_of([3, 1]).tolist() == [2, 0]

    def test_missing_id_is_loud(self, metadata, loader) -> None:
        store, _ = build_waveform_store(None, metadata, loader=loader, progress_every=0)
        with pytest.raises(KeyError, match="not in store"):
            store.rows_of([1, 999])
        with pytest.raises(KeyError, match="not in this store"):
            store.row_of(999)
