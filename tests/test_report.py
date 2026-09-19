"""Tests for the research-summary tables and plots.

The one that matters for the write-up is
:meth:`TestDetailedMetrics.test_macro_matches_the_shipped_definition`. This
module recomputes AUROC, PR-AUC and F1 that :mod:`ecg.training.metrics` already
computes, and if the two ever disagreed the summary tables would quietly
contradict the numbers logged to MLflow during the run.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from ecg.data.datasets import Cohort
from ecg.data.ptbxl import SUPERCLASSES
from ecg.experiments.report import (
    CLASS_COLOURS,
    METRIC_COLUMNS,
    SPLIT_COLOURS,
    RunPredictions,
    comparison_table,
    curve,
    detailed_metrics,
    label_distribution,
    per_class_table,
    plot_label_distribution,
    plot_pr_curves,
    plot_training_curves,
    select_seed,
    style_table,
    training_history,
)
from ecg.training.metrics import evaluate, select_thresholds

N = 600
N_CLASSES = len(SUPERCLASSES)


@pytest.fixture()
def labels() -> np.ndarray:
    """Imbalanced multi-label targets, rarest class last, as in PTB-XL."""
    rng = np.random.default_rng(0)
    prevalence = np.array([0.44, 0.25, 0.24, 0.23, 0.12])
    return (rng.random((N, N_CLASSES)) < prevalence).astype(np.float64)


@pytest.fixture()
def scores(labels: np.ndarray) -> np.ndarray:
    """Weak but informative scores, heavily overlapping between classes."""
    rng = np.random.default_rng(1)
    return 1.0 / (1.0 + np.exp(-(1.1 * labels - 0.5 + rng.normal(0, 1.0, labels.shape))))


@pytest.fixture()
def thresholds(labels: np.ndarray, scores: np.ndarray) -> np.ndarray:
    return select_thresholds(labels, scores)


def _run(name: str, arm: str, fraction: float, labels, scores, thresholds,
         variant: str = "base", seed: int = 0):
    return RunPredictions(
        name=name,
        arm=arm,
        embedder=arm.split("+")[0],
        pretrained="ssl" in arm,
        label_fraction=fraction,
        thresholds=thresholds,
        y_true={"val": labels, "test": labels},
        y_score={"val": scores, "test": scores},
        variant=variant,
        seed=seed,
    )


def _replicates(labels, scores, thresholds, *, seeds=(0, 1, 2), arm="linear"):
    """One arm at several seeds, each seed's scores nudged a little."""
    rng = np.random.default_rng(7)
    return [
        _run(
            f"{arm}-20-s{seed}",
            arm,
            0.2,
            labels,
            np.clip(scores + rng.normal(0, 0.05, scores.shape), 0.0, 1.0),
            thresholds,
            seed=seed,
        )
        for seed in seeds
    ]


@pytest.fixture()
def cohorts(labels: np.ndarray) -> dict[str, Cohort]:
    """Three splits of unequal size, as the real ones are."""
    ids = np.arange(N, dtype=np.int64)
    return {
        "train": Cohort("train", ids[:400], labels[:400]),
        "val": Cohort("val", ids[400:500], labels[400:500]),
        "test": Cohort("test", ids[500:], labels[500:]),
    }


class TestLabelDistribution:
    def test_counts_and_percentages_agree(self, cohorts) -> None:
        frame = label_distribution(cohorts)
        for split, cohort in cohorts.items():
            assert frame.loc[split, "n_records"] == len(cohort)
            for index, name in enumerate(SUPERCLASSES):
                positives = cohort.labels[:, index].sum()
                assert frame.loc[split, f"{name}_n"] == positives
                assert frame.loc[split, f"{name}_pct"] == pytest.approx(
                    100.0 * positives / len(cohort)
                )

    def test_prevalence_is_relative_not_absolute(self, cohorts) -> None:
        # The whole point of percentages here: train is 4x val, so a count chart
        # would show cohort size rather than imbalance.
        frame = label_distribution(cohorts)
        assert frame.loc["train", "n_records"] > 3 * frame.loc["val", "n_records"]
        for name in SUPERCLASSES:
            assert 0.0 <= frame.loc["train", f"{name}_pct"] <= 100.0

    def test_unlabelled_records_are_counted_not_hidden(self) -> None:
        labels = np.zeros((10, len(SUPERCLASSES)), dtype=np.float32)
        labels[:4, 0] = 1.0
        cohort = {"train": Cohort("train", np.arange(10), labels)}
        frame = label_distribution(cohort, splits=("train",))
        assert frame.loc["train", "unlabelled"] == 6
        assert frame.loc["train", "labels_per_record"] == pytest.approx(0.4)

    def test_multi_label_rows_can_exceed_one_hundred_percent(self, cohorts) -> None:
        frame = label_distribution(cohorts)
        total = sum(frame.loc["train", f"{n}_pct"] for n in SUPERCLASSES)
        assert total > 100.0
        assert frame.loc["train", "labels_per_record"] > 1.0

    def test_unknown_split_is_named(self, cohorts) -> None:
        with pytest.raises(KeyError, match="no cohort named 'ssl'"):
            label_distribution(cohorts, splits=("train", "ssl"))

    def test_plot_draws_a_bar_per_split_and_class(self, cohorts) -> None:
        import matplotlib

        matplotlib.use("Agg")
        figure = plot_label_distribution(cohorts)
        axis = figure.axes[0]
        assert len(axis.containers) == 3
        assert all(len(container) == N_CLASSES for container in axis.containers)
        # One direct count label per bar: the absolute numbers were the ask.
        assert len(axis.texts) >= 3 * N_CLASSES

    def test_one_colour_per_split(self) -> None:
        assert len(SPLIT_COLOURS) >= 3


class TestDetailedMetrics:
    def test_every_per_class_number_matches_sklearn(
        self, labels, scores, thresholds
    ) -> None:
        table = detailed_metrics(labels, scores, thresholds)
        for index, name in enumerate(SUPERCLASSES):
            truth = labels[:, index]
            predicted = (scores[:, index] >= thresholds[index]).astype(int)
            expected = {
                "accuracy": accuracy_score(truth, predicted),
                "precision": precision_score(truth, predicted, zero_division=0),
                "recall": recall_score(truth, predicted, zero_division=0),
                "f1": f1_score(truth, predicted, zero_division=0),
                "auroc": roc_auc_score(truth, scores[:, index]),
                "pr_auc": average_precision_score(truth, scores[:, index]),
            }
            for key, value in expected.items():
                assert table.loc[name, key] == pytest.approx(value, abs=1e-9)

    def test_macro_matches_the_shipped_definition(
        self, labels, scores, thresholds
    ) -> None:
        table = detailed_metrics(labels, scores, thresholds)
        reference = evaluate(labels, scores, thresholds)
        assert table.loc["macro", "auroc"] == pytest.approx(reference.macro_auroc)
        assert table.loc["macro", "pr_auc"] == pytest.approx(reference.macro_pr_auc)
        assert table.loc["macro", "f1"] == pytest.approx(reference.macro_f1)

    def test_macro_is_unweighted_and_weighted_is_not(
        self, labels, scores, thresholds
    ) -> None:
        table = detailed_metrics(labels, scores, thresholds)
        per_class = table.loc[list(SUPERCLASSES)]
        assert table.loc["macro", "f1"] == pytest.approx(per_class["f1"].mean())
        assert table.loc["weighted", "f1"] == pytest.approx(
            np.average(per_class["f1"], weights=per_class["support"])
        )
        # With a rare, hard class the two averagings must actually differ, or
        # the weighted row would be decoration rather than a check.
        assert table.loc["macro", "f1"] != pytest.approx(table.loc["weighted", "f1"])

    def test_degenerate_class_is_nan_and_leaves_the_averages(
        self, labels, scores, thresholds
    ) -> None:
        blanked = labels.copy()
        blanked[:, -1] = 0.0
        table = detailed_metrics(blanked, scores, thresholds)
        rare = SUPERCLASSES[-1]
        assert table.loc[rare, list(METRIC_COLUMNS[3:])].isna().all()
        assert not table.loc["macro", list(METRIC_COLUMNS[3:])].isna().any()
        assert table.loc["macro", "f1"] == pytest.approx(
            detailed_metrics(blanked, scores, thresholds)
            .loc[list(SUPERCLASSES[:-1]), "f1"]
            .mean()
        )

    def test_support_and_prevalence_describe_the_cohort(
        self, labels, scores, thresholds
    ) -> None:
        table = detailed_metrics(labels, scores, thresholds)
        for index, name in enumerate(SUPERCLASSES):
            assert table.loc[name, "support"] == labels[:, index].sum()
            assert table.loc[name, "prevalence"] == pytest.approx(
                labels[:, index].mean()
            )

    def test_rejects_mismatched_shapes(self, labels, scores, thresholds) -> None:
        with pytest.raises(ValueError, match="shape mismatch"):
            detailed_metrics(labels, scores[:10], thresholds)
        with pytest.raises(ValueError, match="expected 5 classes"):
            detailed_metrics(labels[:, :3], scores[:, :3], thresholds[:3])


class TestTables:
    def test_comparison_table_has_one_row_per_run(
        self, labels, scores, thresholds
    ) -> None:
        runs = [
            _run(f"{arm}-20", arm, 0.2, labels, scores, thresholds)
            for arm in ("linear", "linear+ssl", "conv", "conv+ssl")
        ]
        table = comparison_table(runs)
        assert len(table) == 4
        assert table.index.names == ["label_fraction", "arm"]
        for column in ("accuracy", "precision", "recall", "f1", "auroc", "pr_auc"):
            assert f"{column}_mean" in table.columns
            assert f"{column}_sd" in table.columns
        assert (table["exact_match_mean"].between(0, 1)).all()
        assert (table["n_seeds"] == 1).all()

    def test_per_class_table_covers_every_superclass(
        self, labels, scores, thresholds
    ) -> None:
        runs = [_run("linear-20", "linear", 0.2, labels, scores, thresholds)]
        table = per_class_table(runs, metric="recall")
        assert list(table.columns) == [*SUPERCLASSES, "macro"]
        assert "weighted" not in table.columns

    def test_exact_match_is_stricter_than_per_class_accuracy(
        self, labels, scores, thresholds
    ) -> None:
        run = _run("linear-20", "linear", 0.2, labels, scores, thresholds)
        assert run.exact_match("test") < run.table("test").loc["macro", "accuracy"]

    def test_two_variants_are_kept_apart_not_overwritten(
        self, labels, scores, thresholds
    ) -> None:
        """(label_fraction, arm) stops identifying a run once a study directory
        holds two architectures -- and per_class_table keys a dict with it, so
        the second would silently replace the first."""
        runs = [
            _run(f"{variant}-linear-20", "linear", 0.2, labels, scores, thresholds,
                 variant=variant)
            for variant in ("base", "deep6")
        ]
        per_class = per_class_table(runs)
        assert len(per_class) == 2
        assert per_class.index.names == ["variant", "label_fraction", "arm"]

        comparison = comparison_table(runs)
        assert len(comparison) == 2
        assert comparison.index.names == ["variant", "label_fraction", "arm"]

    def test_one_variant_keeps_the_original_index(
        self, labels, scores, thresholds
    ) -> None:
        runs = [_run("linear-20", "linear", 0.2, labels, scores, thresholds)]
        assert per_class_table(runs).index.names == ["label_fraction", "arm"]
        assert comparison_table(runs).index.names == ["label_fraction", "arm"]

    def test_style_table_renders(self, labels, scores, thresholds) -> None:
        runs = [_run("linear-20", "linear", 0.2, labels, scores, thresholds)]
        assert "<table" in style_table(comparison_table(runs)).to_html()


class TestSeeds:
    """A five-seed study has five runs per (label_fraction, arm), and these
    tables used to key on that pair alone -- so four of them would have been
    dropped on the floor with nothing saying so."""

    def test_replicates_collapse_into_mean_and_sd(
        self, labels, scores, thresholds
    ) -> None:
        table = comparison_table(_replicates(labels, scores, thresholds))
        assert len(table) == 1
        assert table.iloc[0]["n_seeds"] == 3
        assert table.iloc[0]["auroc_sd"] > 0

    def test_per_seed_opens_the_row_back_up(self, labels, scores, thresholds) -> None:
        table = comparison_table(
            _replicates(labels, scores, thresholds), per_seed=True
        )
        assert len(table) == 3
        assert table.index.names == ["label_fraction", "arm", "seed"]

    def test_per_class_table_keeps_every_replicate(
        self, labels, scores, thresholds
    ) -> None:
        runs = _replicates(labels, scores, thresholds)
        per_seed = per_class_table(runs, per_seed=True)
        assert len(per_seed) == 3
        assert per_seed.index.names == ["label_fraction", "arm", "seed"]

        averaged = per_class_table(runs)
        assert len(averaged) == 1
        assert averaged.loc[(0.2, "linear"), "macro"] == pytest.approx(
            per_seed["macro"].mean()
        )

    def test_a_single_seed_keeps_the_original_index(
        self, labels, scores, thresholds
    ) -> None:
        """Nothing grows a constant level it does not need."""
        runs = [_run("linear-20", "linear", 0.2, labels, scores, thresholds)]
        assert comparison_table(runs).index.names == ["label_fraction", "arm"]
        assert per_class_table(runs).index.names == ["label_fraction", "arm"]

    def test_select_seed_narrows_a_study(self, labels, scores, thresholds) -> None:
        runs = _replicates(labels, scores, thresholds)
        assert [run.seed for run in select_seed(runs)] == [0]
        assert [run.seed for run in select_seed(runs, 2)] == [2]

    def test_select_seed_names_a_seed_that_is_not_there(
        self, labels, scores, thresholds
    ) -> None:
        with pytest.raises(ValueError, match=r"no run at seed 9"):
            select_seed(_replicates(labels, scores, thresholds), 9)

    def test_pr_panels_say_which_seed_they_are(
        self, labels, scores, thresholds
    ) -> None:
        import matplotlib

        matplotlib.use("Agg")
        runs = _replicates(labels, scores, thresholds, seeds=(0, 1))
        figure = plot_pr_curves(runs, ncols=2)
        assert "seed 0" in figure.axes[0].get_title(loc="left")
        alone = plot_pr_curves(runs[:1], ncols=1)
        assert "seed" not in alone.axes[0].get_title(loc="left")


class TestTrackingHistory:
    def test_missing_database_is_reported(self, tmp_path) -> None:
        with pytest.raises(FileNotFoundError, match="no tracking database"):
            training_history(tmp_path / "absent.db")

    def test_curve_is_empty_for_an_unlogged_key(self) -> None:
        history = pd.DataFrame(
            {"run": ["a"] * 3, "key": ["train_loss"] * 3, "step": [0, 1, 2],
             "value": [0.6, 0.5, 0.4]}
        )
        assert list(curve(history, "a", "train_loss")) == [0.6, 0.5, 0.4]
        assert curve(history, "a", "val_loss").empty


class TestPlots:
    """Plots are smoke-tested only: that they render, and survive missing data."""

    def test_pr_curves_render_one_panel_per_run(
        self, labels, scores, thresholds
    ) -> None:
        import matplotlib

        matplotlib.use("Agg")
        runs = [
            _run(f"{arm}-20", arm, 0.2, labels, scores, thresholds)
            for arm in ("linear", "conv")
        ]
        figure = plot_pr_curves(runs, ncols=2)
        assert len(figure.axes) == 2
        assert len(figure.axes[0].lines) == 2 * N_CLASSES  # curve + prevalence

    def test_training_curves_survive_a_key_no_run_logged(self) -> None:
        import matplotlib

        matplotlib.use("Agg")
        history = pd.DataFrame(
            {"run": ["a"] * 4, "key": ["train_loss"] * 4, "step": [0, 1, 2, 3],
             "value": [0.6, 0.5, 0.45, 0.43]}
        )
        figure = plot_training_curves(
            history, ["a"], keys=("train_loss", "val_loss"), ylabels=("t", "v")
        )
        assert len(figure.axes) == 2
        assert not figure.axes[1].lines
        assert figure.axes[1].texts  # the "no run logged" message

    def test_one_colour_per_superclass(self) -> None:
        assert len(CLASS_COLOURS) >= N_CLASSES
