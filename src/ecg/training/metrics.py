"""Multi-label classification metrics for the five diagnostic superclasses.

The labels are multi-label, not mutually exclusive, so every metric is computed
per class and then averaged unweighted. Macro averaging rather than micro is
the point of the study: HYP has 260 positives in the 2,105-record test split
against NORM's 910, and micro averaging would let the common classes hide a
failure on the rare one.

Thresholds are a separate concern from ranking. AUROC and PR-AUC need no
threshold; F1 does. :func:`select_thresholds` picks them **on validation only**
-- integrity rule 2 -- and :func:`evaluate` applies those fixed numbers to
test. Choosing thresholds on test would inflate F1 by an amount nobody can
estimate afterwards.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

from ecg.data.ptbxl import SUPERCLASSES

#: Threshold grid searched per class. 0.01 steps are finer than the noise in a
#: 2,000-record validation split, so a denser grid would only overfit it.
THRESHOLD_GRID: np.ndarray = np.arange(0.05, 0.96, 0.01)


@dataclass(frozen=True)
class ClassificationMetrics:
    """Per-class and macro-averaged scores.

    Attributes:
        auroc: Per-class AUROC, ``nan`` for a class with no positives.
        pr_auc: Per-class average precision.
        f1: Per-class F1 at the supplied thresholds.
        thresholds: The thresholds used.
        n_records: Records scored.
        n_positives: Positives per class.
    """

    auroc: dict[str, float]
    pr_auc: dict[str, float]
    f1: dict[str, float]
    thresholds: dict[str, float]
    n_records: int
    n_positives: dict[str, int]

    @property
    def macro_auroc(self) -> float:
        """Unweighted mean AUROC over classes that have positives."""
        return _macro(self.auroc)

    @property
    def macro_pr_auc(self) -> float:
        """Unweighted mean average precision."""
        return _macro(self.pr_auc)

    @property
    def macro_f1(self) -> float:
        """Unweighted mean F1."""
        return _macro(self.f1)

    def to_mlflow(self, prefix: str) -> dict[str, float]:
        """Flatten for metric logging.

        Args:
            prefix: Prepended to every key, e.g. ``"val"``.

        Returns:
            A flat mapping of metric name to value, skipping ``nan``.
        """
        flat: dict[str, float] = {
            f"{prefix}_macro_auroc": self.macro_auroc,
            f"{prefix}_macro_pr_auc": self.macro_pr_auc,
            f"{prefix}_macro_f1": self.macro_f1,
        }
        for name, table in (("auroc", self.auroc), ("pr_auc", self.pr_auc), ("f1", self.f1)):
            for label, value in table.items():
                if not np.isnan(value):
                    flat[f"{prefix}_{name}_{label}"] = float(value)
        return flat

    def summary(self) -> str:
        """One-line summary for progress output."""
        return (
            f"AUROC {self.macro_auroc:.4f}  PR-AUC {self.macro_pr_auc:.4f}  "
            f"F1 {self.macro_f1:.4f}"
        )


def select_thresholds(
    y_true: np.ndarray, y_score: np.ndarray, *, grid: np.ndarray = THRESHOLD_GRID
) -> np.ndarray:
    """Pick the per-class threshold maximising F1.

    Must be called on validation predictions only. The returned array is then
    passed unchanged to :func:`evaluate` on test.

    Args:
        y_true: ``(n_records, n_classes)`` binary labels.
        y_score: ``(n_records, n_classes)`` predicted probabilities.
        grid: Candidate thresholds.

    Returns:
        ``(n_classes,)`` thresholds. A class with no positives keeps 0.5, since
        there is nothing to tune against.
    """
    thresholds = np.full(y_true.shape[1], 0.5, dtype=np.float64)
    for index in range(y_true.shape[1]):
        truth = y_true[:, index]
        if truth.sum() == 0:
            continue
        scores = [
            f1_score(truth, (y_score[:, index] >= value).astype(int), zero_division=0)
            for value in grid
        ]
        thresholds[index] = float(grid[int(np.argmax(scores))])
    return thresholds


def evaluate(
    y_true: np.ndarray,
    y_score: np.ndarray,
    thresholds: np.ndarray | None = None,
    *,
    labels: tuple[str, ...] = SUPERCLASSES,
) -> ClassificationMetrics:
    """Score predictions against labels.

    Args:
        y_true: ``(n_records, n_classes)`` binary labels.
        y_score: ``(n_records, n_classes)`` predicted probabilities.
        thresholds: Per-class thresholds for F1. Defaults to 0.5 everywhere;
            pass the values from :func:`select_thresholds` on validation.
        labels: Class names, in column order.

    Returns:
        A :class:`ClassificationMetrics`.

    Raises:
        ValueError: If the shapes disagree or do not match ``labels``.
    """
    if y_true.shape != y_score.shape:
        raise ValueError(
            f"shape mismatch: labels {y_true.shape} vs scores {y_score.shape}"
        )
    if y_true.shape[1] != len(labels):
        raise ValueError(
            f"expected {len(labels)} classes, got {y_true.shape[1]}"
        )
    if thresholds is None:
        thresholds = np.full(len(labels), 0.5, dtype=np.float64)

    auroc: dict[str, float] = {}
    pr_auc: dict[str, float] = {}
    f1: dict[str, float] = {}
    positives: dict[str, int] = {}

    for index, name in enumerate(labels):
        truth = y_true[:, index].astype(int)
        score = y_score[:, index]
        positives[name] = int(truth.sum())
        # A class that is all-positive or all-negative has no defined ROC; report
        # nan rather than a misleading 0.5, and leave it out of the macro mean.
        if positives[name] == 0 or positives[name] == truth.shape[0]:
            auroc[name] = float("nan")
            pr_auc[name] = float("nan")
            f1[name] = float("nan")
            continue
        auroc[name] = float(roc_auc_score(truth, score))
        pr_auc[name] = float(average_precision_score(truth, score))
        f1[name] = float(
            f1_score(truth, (score >= thresholds[index]).astype(int), zero_division=0)
        )

    return ClassificationMetrics(
        auroc=auroc,
        pr_auc=pr_auc,
        f1=f1,
        thresholds=dict(zip(labels, (float(t) for t in thresholds))),
        n_records=int(y_true.shape[0]),
        n_positives=positives,
    )


def _macro(table: dict[str, float]) -> float:
    """Mean over the classes that have a defined value."""
    values = [value for value in table.values() if not np.isnan(value)]
    return float(np.mean(values)) if values else float("nan")
