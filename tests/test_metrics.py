"""Tests for the multi-label metrics.

The one that matters for research integrity is
:meth:`TestThresholds.test_thresholds_are_reusable_across_cohorts`: F1
thresholds fitted on validation must be applicable unchanged to test, because
fitting them on test would inflate the headline number by an amount nobody can
recover afterwards.
"""

from __future__ import annotations

import numpy as np
import pytest

from ecg.data.ptbxl import SUPERCLASSES
from ecg.training.metrics import evaluate, select_thresholds

N = 400
N_CLASSES = len(SUPERCLASSES)


@pytest.fixture()
def labels() -> np.ndarray:
    rng = np.random.default_rng(0)
    return (rng.random((N, N_CLASSES)) < 0.3).astype(np.float64)


@pytest.fixture()
def informative(labels: np.ndarray) -> np.ndarray:
    """Scores correlated with the labels, as a trained model's would be."""
    rng = np.random.default_rng(1)
    return np.clip(labels * 0.6 + rng.normal(0.2, 0.2, labels.shape), 0.0, 1.0)


class TestEvaluate:
    def test_reports_every_superclass(self, labels, informative) -> None:
        metrics = evaluate(labels, informative)
        assert set(metrics.auroc) == set(SUPERCLASSES)
        assert metrics.n_records == N

    def test_informative_scores_beat_chance(self, labels, informative) -> None:
        assert evaluate(labels, informative).macro_auroc > 0.7

    def test_random_scores_are_near_chance(self, labels) -> None:
        rng = np.random.default_rng(2)
        metrics = evaluate(labels, rng.random(labels.shape))
        assert 0.4 < metrics.macro_auroc < 0.6

    def test_perfect_scores_are_perfect(self, labels) -> None:
        metrics = evaluate(labels, labels.copy())
        assert metrics.macro_auroc == pytest.approx(1.0)
        assert metrics.macro_f1 == pytest.approx(1.0)

    def test_macro_is_unweighted(self, labels, informative) -> None:
        """A rare class must count as much as a common one."""
        metrics = evaluate(labels, informative)
        expected = np.mean([v for v in metrics.auroc.values() if not np.isnan(v)])
        assert metrics.macro_auroc == pytest.approx(expected)

    def test_class_without_positives_is_nan_not_zero(self) -> None:
        """Reporting 0.5 would look like a real, poor score."""
        labels = np.zeros((50, N_CLASSES))
        labels[:20, 0] = 1.0
        scores = np.random.default_rng(3).random((50, N_CLASSES))
        metrics = evaluate(labels, scores)
        assert np.isnan(metrics.auroc[SUPERCLASSES[1]])
        assert not np.isnan(metrics.auroc[SUPERCLASSES[0]])
        assert not np.isnan(metrics.macro_auroc)

    def test_positives_are_counted(self, labels, informative) -> None:
        metrics = evaluate(labels, informative)
        for index, name in enumerate(SUPERCLASSES):
            assert metrics.n_positives[name] == int(labels[:, index].sum())

    def test_shape_mismatch_is_loud(self, labels) -> None:
        with pytest.raises(ValueError, match="shape mismatch"):
            evaluate(labels, np.zeros((N, N_CLASSES - 1)))

    def test_wrong_class_count_is_loud(self) -> None:
        with pytest.raises(ValueError, match="expected 5 classes"):
            evaluate(np.zeros((10, 3)), np.zeros((10, 3)))

    def test_mlflow_keys_are_prefixed_and_finite(self, labels, informative) -> None:
        flat = evaluate(labels, informative).to_mlflow("val")
        assert "val_macro_auroc" in flat
        assert all(key.startswith("val_") for key in flat)
        assert all(np.isfinite(value) for value in flat.values())

    def test_nan_classes_are_omitted_from_mlflow(self) -> None:
        labels = np.zeros((50, N_CLASSES))
        labels[:20, 0] = 1.0
        flat = evaluate(labels, np.random.default_rng(4).random((50, N_CLASSES))).to_mlflow("val")
        assert f"val_auroc_{SUPERCLASSES[1]}" not in flat


class TestThresholds:
    def test_one_threshold_per_class(self, labels, informative) -> None:
        assert select_thresholds(labels, informative).shape == (N_CLASSES,)

    def test_selected_thresholds_beat_the_default(self, labels, informative) -> None:
        chosen = select_thresholds(labels, informative)
        assert (
            evaluate(labels, informative, chosen).macro_f1
            >= evaluate(labels, informative).macro_f1
        )

    def test_thresholds_are_reusable_across_cohorts(self, labels, informative) -> None:
        """Integrity rule 2: fit on val, apply unchanged to test."""
        val, test = slice(0, 200), slice(200, N)
        chosen = select_thresholds(labels[val], informative[val])
        metrics = evaluate(labels[test], informative[test], chosen)
        assert np.isfinite(metrics.macro_f1)
        assert metrics.thresholds == pytest.approx(
            dict(zip(SUPERCLASSES, chosen.tolist()))
        )

    def test_fitting_on_test_would_inflate_f1(self, labels, informative) -> None:
        """Quantifies what integrity rule 2 is protecting against."""
        val, test = slice(0, 200), slice(200, N)
        honest = evaluate(
            labels[test], informative[test], select_thresholds(labels[val], informative[val])
        ).macro_f1
        cheating = evaluate(
            labels[test], informative[test], select_thresholds(labels[test], informative[test])
        ).macro_f1
        assert cheating >= honest

    def test_class_without_positives_keeps_the_default(self) -> None:
        labels = np.zeros((50, N_CLASSES))
        scores = np.random.default_rng(5).random((50, N_CLASSES))
        assert select_thresholds(labels, scores).tolist() == [0.5] * N_CLASSES
