"""Research summary: per-class metrics, PR curves and training curves.

Three things the training path deliberately does not produce are assembled here,
which is why this module exists rather than reading ``results.csv``.

**Accuracy, precision and recall.** :mod:`ecg.training.metrics` computes AUROC,
PR-AUC and F1 only, and :func:`ecg.experiments.runner._macro` narrows that
further -- ``result.json`` carries the three macro numbers plus per-class AUROC,
nothing else. The rest is recomputed here from predictions.

**Predictions.** No run saves ``y_score``, and a PR *curve* cannot be rebuilt
from a scalar average precision. So :func:`collect_predictions` reloads each
run's selected checkpoint and scores the cohorts again. The reload is the point:
the same weights validation selected, not the last epoch's.

**Thresholds come from validation, never test** (integrity rule 2). They are not
stored in the checkpoint despite what :func:`ecg.training.loops.train_supervised`
says, so they are refitted on validation here exactly as the runner did, then
applied unchanged to test.

Macro averaging is **unweighted** throughout -- a plain mean over the classes
with a defined value, matching :func:`ecg.training.metrics._macro`. That is the
study's choice, not an oversight: HYP is the rarest superclass, and support
weighting would let NORM's thousands hide a failure on it. A support-weighted
row is reported alongside so the difference is visible rather than assumed.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

from ecg.data.datasets import Cohort
from ecg.data.ptbxl import SUPERCLASSES
from ecg.experiments.runner import RESULT_FILE, RunOutcome, Workspace, _batches
from ecg.models.encoder import build_classifier
from ecg.training.checkpoints import load_checkpoint
from ecg.training.config import DEFAULT_VARIANT, RunConfig
from ecg.training.loops import predict, resolve_device
from ecg.training.metrics import select_thresholds

#: Categorical slots 1-5, one per superclass. Fixed order, never cycled: a class
#: keeps its colour across every panel, so NORM is the same blue in all of them.
#: Validated for adjacent-pair CVD separation on a light surface (worst pair
#: dE 9.1 protan). Three slots fall below 3:1 contrast, so every panel carries a
#: legend and a table -- identity is never colour alone.
CLASS_COLOURS: tuple[str, ...] = (
    "#2a78d6",  # NORM  blue
    "#eb6834",  # MI    orange
    "#1baf7a",  # STTC  aqua
    "#eda100",  # CD    yellow
    "#e87ba4",  # HYP   magenta
)

#: Categorical slots 1-3, one per split. These three validate on the all-pairs
#: list in both modes, so they stay distinguishable however the bars are read.
SPLIT_COLOURS: tuple[str, ...] = ("#2a78d6", "#eb6834", "#1baf7a")

#: Chart chrome. Recessive grid and axes, ink for all text.
INK: dict[str, str] = {
    "surface": "#fcfcfb",
    "primary": "#0b0b0b",
    "secondary": "#52514e",
    "muted": "#898781",
    "grid": "#e1e0d9",
    "baseline": "#c3c2b7",
}

#: Columns of a detailed metrics table, in display order.
METRIC_COLUMNS: tuple[str, ...] = (
    "support",
    "prevalence",
    "threshold",
    "accuracy",
    "precision",
    "recall",
    "f1",
    "auroc",
    "pr_auc",
)


def label_distribution(
    cohorts: dict[str, Cohort],
    *,
    splits: Sequence[str] = ("train", "val", "test"),
    labels: Sequence[str] = SUPERCLASSES,
) -> pd.DataFrame:
    """Positives and prevalence per class, per split.

    Prevalences do not sum to 100%: the labels are multi-label, so one record
    contributes to every superclass it carries and to none if it carries no
    scorable label. ``n_records`` and ``unlabelled`` are reported alongside so
    a prevalence can always be traced back to a count.

    Args:
        cohorts: Cohorts from :func:`ecg.data.datasets.build_cohorts`.
        splits: Cohort keys to report, in display order.
        labels: Class names in column order.

    Returns:
        Rows indexed by split: ``n_records``, then ``<class>_n`` and
        ``<class>_pct`` for each class, then ``labels_per_record`` and
        ``unlabelled``.

    Raises:
        KeyError: If a requested split is not in ``cohorts``.
    """
    rows: dict[str, dict[str, float]] = {}
    for split in splits:
        if split not in cohorts:
            raise KeyError(f"no cohort named {split!r}; have {sorted(cohorts)}")
        cohort = cohorts[split]
        total = len(cohort)
        row: dict[str, float] = {"n_records": float(total)}
        for index, name in enumerate(labels):
            positives = float(cohort.labels[:, index].sum())
            row[f"{name}_n"] = positives
            row[f"{name}_pct"] = 100.0 * positives / total if total else float("nan")
        row["labels_per_record"] = float(cohort.labels.sum(axis=1).mean())
        # Retained, not dropped (integrity rule 8) -- so they must be visible
        # here, or every prevalence below silently uses a larger denominator
        # than the reader assumes.
        row["unlabelled"] = float((cohort.labels.sum(axis=1) == 0).sum())
        rows[split] = row
    return pd.DataFrame(rows).T.rename_axis("split")


def plot_label_distribution(
    cohorts: dict[str, Cohort],
    *,
    splits: Sequence[str] = ("train", "val", "test"),
    labels: Sequence[str] = SUPERCLASSES,
):
    """Class prevalence per split, as grouped horizontal bars.

    Prevalence rather than raw counts, because the splits differ in size by
    roughly five to one and a count chart would show that difference instead of
    the imbalance it is meant to show. The absolute count sits at the end of
    each bar, so nothing has to be recovered from a percentage.

    Args:
        cohorts: Cohorts from :func:`ecg.data.datasets.build_cohorts`.
        splits: Cohort keys to compare.
        labels: Class names, plotted top to bottom in this order.

    Returns:
        The matplotlib figure.
    """
    import matplotlib.pyplot as plt

    frame = label_distribution(cohorts, splits=splits, labels=labels)
    positions = np.arange(len(labels))
    height = 0.8 / len(splits)
    figure, axis = plt.subplots(figsize=(10, 0.95 * len(labels) + 2.0))

    widest = max(
        frame.loc[split, f"{name}_pct"] for split in splits for name in labels
    )
    for order, split in enumerate(splits):
        # The axis is inverted below, so a larger y sits lower: adding the
        # offset puts the first split at the top of each group, matching the
        # order the legend lists them in.
        offset = (order - (len(splits) - 1) / 2) * height
        values = [frame.loc[split, f"{name}_pct"] for name in labels]
        counts = [frame.loc[split, f"{name}_n"] for name in labels]
        bars = axis.barh(
            positions + offset,
            values,
            height=height * 0.88,  # the gap is the separator, not a rule
            color=SPLIT_COLOURS[order % len(SPLIT_COLOURS)],
            label=f"{split}  (n={int(frame.loc[split, 'n_records']):,})",
            zorder=3,
        )
        for bar, count in zip(bars, counts):
            axis.text(
                bar.get_width() + widest * 0.012,
                bar.get_y() + bar.get_height() / 2,
                f"{int(count):,}",
                va="center",
                fontsize=8.5,
                color=INK["secondary"],
                zorder=4,
            )

    axis.set_yticks(positions, labels=list(labels), fontsize=10.5)
    axis.invert_yaxis()
    axis.set_xlim(0, widest * 1.16)
    axis.xaxis.set_major_formatter(lambda value, _: f"{value:.0f}%")
    _style_axis(
        axis, xlabel="records carrying the label (% of split)", ylabel="", title=""
    )
    # Padded off the axes so the multi-label note below it has its own line.
    axis.set_title(
        "Superclass prevalence by split",
        fontsize=11,
        color=INK["primary"],
        loc="left",
        pad=26,
    )
    # The class name is the chart's key; it outranks the axis furniture.
    for tick in axis.get_yticklabels():
        tick.set_color(INK["primary"])
    axis.grid(axis="y", visible=False)
    axis.legend(
        fontsize=9,
        frameon=False,
        labelcolor=INK["secondary"],
        loc="lower right",
    )
    axis.text(
        0.0,
        1.012,
        "multi-label: a record can carry several superclasses, so the bars "
        "within a split sum past 100%",
        transform=axis.transAxes,
        fontsize=8.5,
        color=INK["muted"],
    )
    figure.tight_layout()
    return figure


@dataclass(frozen=True)
class RunPredictions:
    """One supervised run's predictions on validation and test.

    Attributes:
        name: Run name.
        arm: Arm label, e.g. ``"conv+ssl"``.
        embedder: ``"linear"`` or ``"conv"``.
        pretrained: Whether the encoder started from SSL weights.
        label_fraction: Fraction of the labelled training cohort used.
        thresholds: Per-class F1 thresholds, fitted on validation.
        y_true: Binary labels per split, ``(n_records, n_classes)``.
        y_score: Predicted probabilities per split, same shape.
        variant: Architecture variant. Part of a run's identity, because
            ``(label_fraction, arm)`` alone is not unique once a study
            directory holds more than one architecture.
    """

    name: str
    arm: str
    embedder: str
    pretrained: bool
    label_fraction: float
    thresholds: np.ndarray
    y_true: dict[str, np.ndarray]
    y_score: dict[str, np.ndarray]
    variant: str = DEFAULT_VARIANT

    def table(self, split: str = "test") -> pd.DataFrame:
        """Per-class and averaged metrics for one split.

        Args:
            split: ``"val"`` or ``"test"``.

        Returns:
            The table from :func:`detailed_metrics`.
        """
        return detailed_metrics(
            self.y_true[split], self.y_score[split], self.thresholds
        )

    def exact_match(self, split: str = "test") -> float:
        """Subset accuracy: the fraction of records whose five labels are all correct.

        Reported separately from per-class accuracy because it is the only
        accuracy that is hard to game on an imbalanced multi-label problem.

        Args:
            split: ``"val"`` or ``"test"``.

        Returns:
            The fraction in ``[0, 1]``.
        """
        predicted = self.y_score[split] >= self.thresholds
        return float((predicted == self.y_true[split].astype(bool)).all(axis=1).mean())


def detailed_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    thresholds: np.ndarray,
    *,
    labels: Sequence[str] = SUPERCLASSES,
) -> pd.DataFrame:
    """Per-class metrics plus both averagings.

    Accuracy, precision and recall are threshold-dependent and use the supplied
    thresholds; AUROC and PR-AUC are ranking metrics and ignore them. A class
    with no positives, or with no negatives, gets ``nan`` for every metric
    rather than a misleading number, and is excluded from both averages.

    Args:
        y_true: ``(n_records, n_classes)`` binary labels.
        y_score: ``(n_records, n_classes)`` predicted probabilities.
        thresholds: ``(n_classes,)`` decision thresholds, fitted on validation.
        labels: Class names in column order.

    Returns:
        One row per class, then ``macro`` (unweighted mean, the study's headline
        averaging) and ``weighted`` (mean weighted by positives, shown for
        contrast only).

    Raises:
        ValueError: If the shapes disagree or do not match ``labels``.
    """
    if y_true.shape != y_score.shape:
        raise ValueError(
            f"shape mismatch: labels {y_true.shape} vs scores {y_score.shape}"
        )
    if y_true.shape[1] != len(labels):
        raise ValueError(f"expected {len(labels)} classes, got {y_true.shape[1]}")

    rows: dict[str, dict[str, float]] = {}
    for index, name in enumerate(labels):
        truth = y_true[:, index].astype(int)
        score = y_score[:, index]
        positives = int(truth.sum())
        row = {
            "support": float(positives),
            "prevalence": positives / truth.shape[0],
            "threshold": float(thresholds[index]),
        }
        if positives in (0, truth.shape[0]):
            row.update(dict.fromkeys(METRIC_COLUMNS[3:], float("nan")))
            rows[name] = row
            continue

        predicted = (score >= thresholds[index]).astype(int)
        denominator = 2 * (predicted & truth).sum() + (predicted != truth).sum()
        row.update(
            accuracy=float((predicted == truth).mean()),
            precision=float(precision_score(truth, predicted, zero_division=0)),
            recall=float(recall_score(truth, predicted, zero_division=0)),
            # f1 from the confusion counts directly, so it cannot drift from the
            # precision and recall printed beside it.
            f1=float(2 * (predicted & truth).sum() / denominator) if denominator else 0.0,
            auroc=float(roc_auc_score(truth, score)),
            pr_auc=float(average_precision_score(truth, score)),
        )
        rows[name] = row

    frame = pd.DataFrame(rows).T[list(METRIC_COLUMNS)]
    scored = frame.dropna(subset=["auroc"])
    frame.loc["macro"] = scored.mean()
    frame.loc["weighted"] = scored.mul(scored["support"], axis=0).sum() / scored[
        "support"
    ].sum()
    for average in ("macro", "weighted"):
        frame.loc[average, ["support", "prevalence", "threshold"]] = float("nan")
    frame.loc["macro", "support"] = scored["support"].sum()
    return frame


def collect_predictions(
    study_dir: str | Path,
    workspace: Workspace,
    *,
    splits: Sequence[str] = ("val", "test"),
    progress: bool = True,
) -> list[RunPredictions]:
    """Reload every finished supervised run and re-score its cohorts.

    Each run's ``config.yaml`` supplies the architecture, so a checkpoint is
    never paired with a guessed one (integrity rule 5). Runs without a
    ``result.json`` are unfinished and skipped; pretraining runs have no
    classifier head and are skipped too.

    Args:
        study_dir: Directory holding one subdirectory per run.
        workspace: Loaded store and cohorts.
        splits: Cohorts to score. ``"val"`` must be present -- thresholds come
            from it.
        progress: Print one line per run.

    Returns:
        One entry per supervised run, in directory order.

    Raises:
        ValueError: If ``"val"`` is not among ``splits``.
        FileNotFoundError: If ``study_dir`` does not exist.
    """
    if "val" not in splits:
        raise ValueError("splits must include 'val': thresholds are fitted there")
    root = Path(study_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"no study directory at {root}")

    collected: list[RunPredictions] = []
    for directory in sorted(p for p in root.iterdir() if p.is_dir()):
        if not (directory / RESULT_FILE).exists():
            continue
        outcome = RunOutcome.load(directory)
        if outcome.kind != "supervised" or not outcome.checkpoint:
            continue

        config = RunConfig.from_yaml(directory / "config.yaml")
        device = resolve_device(config.train.device)
        model = build_classifier(config.model)
        model.load_state_dict(load_checkpoint(outcome.checkpoint)["model"])
        model.to(device)

        truths: dict[str, np.ndarray] = {}
        scores: dict[str, np.ndarray] = {}
        for split in splits:
            batches = _batches(
                workspace, workspace.cohorts[split], config, device, shuffle=False
            )
            truths[split], scores[split] = predict(model, batches)

        thresholds = select_thresholds(truths["val"], scores["val"])
        collected.append(
            RunPredictions(
                name=outcome.name,
                arm=outcome.arm,
                embedder=outcome.embedder,
                pretrained=outcome.pretrained,
                label_fraction=outcome.label_fraction,
                thresholds=thresholds,
                y_true=truths,
                y_score=scores,
                variant=outcome.variant,
            )
        )
        if progress:
            print(f"scored {outcome.name:<28} thresholds {np.round(thresholds, 2)}")
    return collected


def comparison_table(
    runs: Iterable[RunPredictions], *, split: str = "test"
) -> pd.DataFrame:
    """One row per run: the macro numbers side by side.

    Args:
        runs: Runs from :func:`collect_predictions`.
        split: Cohort to report.

    Returns:
        Rows indexed by ``(label_fraction, arm)``, sorted, with macro accuracy,
        precision, recall, F1, AUROC, PR-AUC and exact-match accuracy. If the
        runs span several architecture variants, ``variant`` becomes the outer
        index level, since the pair alone would no longer identify a run.
    """
    rows = []
    for run in runs:
        macro = run.table(split).loc["macro"]
        rows.append(
            {
                "variant": run.variant,
                "label_fraction": run.label_fraction,
                "arm": run.arm,
                "embedder": run.embedder,
                "pretrained": run.pretrained,
                **{name: macro[name] for name in METRIC_COLUMNS[3:]},
                "exact_match": run.exact_match(split),
            }
        )
    frame = pd.DataFrame(rows).sort_values(["variant", "label_fraction", "arm"])
    index = ["label_fraction", "arm"]
    if frame["variant"].nunique() > 1:
        index = ["variant", *index]
    return frame.set_index(index)


def per_class_table(
    runs: Iterable[RunPredictions], *, split: str = "test", metric: str = "f1"
) -> pd.DataFrame:
    """One metric, every class, every run.

    Args:
        runs: Runs from :func:`collect_predictions`.
        split: Cohort to report.
        metric: Column of :func:`detailed_metrics` to extract.

    Returns:
        Runs as rows, classes plus ``macro`` as columns. Rows are keyed by
        ``(label_fraction, arm)``, with ``variant`` prepended when the runs
        span more than one -- without it two architectures' runs would share a
        key and one would silently replace the other.
    """
    collected = list(runs)
    several = len({run.variant for run in collected}) > 1
    rows = {}
    for run in collected:
        key = (run.label_fraction, run.arm)
        rows[(run.variant, *key) if several else key] = run.table(split)[metric]
    frame = pd.DataFrame(rows).T.drop(columns=["weighted"])
    frame.index.names = (["variant"] if several else []) + ["label_fraction", "arm"]
    return frame.sort_index()


def training_history(
    database: str | Path, *, experiment: str | None = None
) -> pd.DataFrame:
    """Read per-epoch metrics out of the MLflow SQLite backend.

    Reads with a read-only connection so a database sitting on a Drive mount is
    never written to -- SQLite's locking assumes POSIX semantics FUSE does not
    honour, and a write is how that file corrupts.

    Args:
        database: Path to ``mlflow.db``. Copy it off Drive first.
        experiment: Restrict to this experiment name. ``None`` reads all.

    Returns:
        Long-form: ``run``, ``key``, ``step``, ``value``.

    Raises:
        FileNotFoundError: If the database does not exist.
    """
    path = Path(database)
    if not path.exists():
        raise FileNotFoundError(f"no tracking database at {path}")

    query = """
        SELECT r.name AS run, m.key, m.step, m.value
        FROM metrics m
        JOIN runs r ON r.run_uuid = m.run_uuid
        JOIN experiments e ON e.experiment_id = r.experiment_id
        WHERE m.is_nan = 0 {clause}
        ORDER BY r.name, m.key, m.step
    """
    clause = "AND e.name = ?" if experiment else ""
    connection = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
    try:
        return pd.read_sql(
            query.format(clause=clause),
            connection,
            params=(experiment,) if experiment else None,
        )
    finally:
        connection.close()


def curve(history: pd.DataFrame, run: str, key: str) -> pd.Series:
    """Pull one metric's epoch series out of a history frame.

    Args:
        history: Frame from :func:`training_history`.
        run: Run name.
        key: Metric key, e.g. ``"train_loss"``.

    Returns:
        Values indexed by step. Empty if the run never logged that key --
        which is the normal case for ``val_loss``, see the notebook.
    """
    rows = history[(history["run"] == run) & (history["key"] == key)]
    return rows.set_index("step")["value"].sort_index()


def _style_axis(axis, *, xlabel: str, ylabel: str, title: str) -> None:
    """Apply the shared chrome: recessive grid, no top or right spine."""
    axis.set_title(title, fontsize=10, color=INK["primary"], loc="left")
    axis.set_xlabel(xlabel, fontsize=9, color=INK["secondary"])
    axis.set_ylabel(ylabel, fontsize=9, color=INK["secondary"])
    axis.grid(color=INK["grid"], linewidth=0.7, alpha=0.9)
    axis.set_axisbelow(True)
    axis.tick_params(colors=INK["muted"], labelsize=8)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axis.spines[side].set_color(INK["baseline"])


def plot_pr_curves(
    runs: Sequence[RunPredictions],
    *,
    split: str = "test",
    labels: Sequence[str] = SUPERCLASSES,
    ncols: int = 4,
):
    """Precision-recall curves, one panel per run, one line per class.

    The dashed horizontal line in each panel is that class's prevalence -- the
    precision a coin-flip classifier would reach. Without it a PR curve is
    unreadable across classes of different base rates, which is exactly the
    comparison being made here.

    Args:
        runs: Runs from :func:`collect_predictions`.
        split: Cohort to plot.
        labels: Class names in column order.
        ncols: Panels per row.

    Returns:
        The matplotlib figure.
    """
    import matplotlib.pyplot as plt

    nrows = int(np.ceil(len(runs) / ncols))
    figure, axes = plt.subplots(
        nrows, ncols, figsize=(3.6 * ncols, 3.4 * nrows), squeeze=False
    )
    flat = axes.ravel()
    # Only when it disambiguates: on a single-variant study it would repeat the
    # same word across every panel heading.
    several = len({run.variant for run in runs}) > 1

    for axis, run in zip(flat, runs):
        truth, score = run.y_true[split], run.y_score[split]
        for index, name in enumerate(labels):
            column = truth[:, index].astype(int)
            if column.sum() in (0, column.shape[0]):
                continue
            precision, recall, _ = precision_recall_curve(column, score[:, index])
            average = average_precision_score(column, score[:, index])
            axis.plot(
                recall,
                precision,
                color=CLASS_COLOURS[index % len(CLASS_COLOURS)],
                linewidth=2,
                label=f"{name}  {average:.3f}",
            )
            axis.axhline(
                column.mean(),
                color=CLASS_COLOURS[index % len(CLASS_COLOURS)],
                linewidth=0.8,
                linestyle=":",
                alpha=0.6,
            )
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1.02)
        _style_axis(
            axis,
            xlabel="recall",
            ylabel="precision",
            title=(
                f"{run.variant}  ·  " if several else ""
            ) + f"{run.arm}  ·  {run.label_fraction:.0%} labels",
        )
        # Framed against the surface, not frameless: PR curves decay through the
        # lower left, so a transparent legend sits on top of the lines it names.
        legend = axis.legend(
            fontsize=7.5,
            loc="lower left",
            frameon=True,
            facecolor=INK["surface"],
            edgecolor=INK["grid"],
            framealpha=0.92,
            title="class · AP",
            title_fontsize=7.5,
            labelcolor=INK["secondary"],
        )
        legend.get_title().set_color(INK["muted"])

    for axis in flat[len(runs) :]:
        axis.set_visible(False)
    figure.suptitle(
        f"Precision-recall by class ({split}); dotted line = class prevalence",
        fontsize=11,
        color=INK["primary"],
        x=0.01,
        ha="left",
    )
    figure.tight_layout()
    return figure


def plot_training_curves(
    history: pd.DataFrame,
    run_names: Sequence[str],
    *,
    keys: Sequence[str] = ("train_loss", "val_macro_auroc"),
    ylabels: Sequence[str] = ("BCE loss (train)", "macro AUROC (val)"),
    title: str = "",
):
    """Training curves as small multiples -- one panel per metric.

    Two panels rather than two y-axes on one. A dual-axis plot invites the eye
    to read a crossing point that is an artefact of two arbitrary scales.

    Args:
        history: Frame from :func:`training_history`.
        run_names: Runs to overlay, in legend order.
        keys: Metric keys, one panel each.
        ylabels: Panel heading per key.
        title: Optional figure heading.

    Returns:
        The matplotlib figure.
    """
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(
        1, len(keys), figsize=(5.4 * len(keys), 3.8), squeeze=False
    )
    for axis, key, ylabel in zip(axes[0], keys, ylabels):
        plotted = 0
        for index, name in enumerate(run_names):
            series = curve(history, name, key)
            if series.empty:
                continue
            axis.plot(
                series.index + 1,
                series.to_numpy(),
                color=CLASS_COLOURS[index % len(CLASS_COLOURS)],
                linewidth=2,
                label=name,
            )
            plotted += 1
        if not plotted:
            axis.text(
                0.5,
                0.5,
                f"no run logged {key!r}",
                ha="center",
                va="center",
                color=INK["muted"],
                fontsize=9,
                transform=axis.transAxes,
            )
        # The heading carries the metric name, so the y-axis does not repeat it.
        _style_axis(axis, xlabel="epoch", ylabel="", title=ylabel)
        if plotted > 1:
            axis.legend(fontsize=8, frameon=False, labelcolor=INK["secondary"])
    if title:
        figure.suptitle(title, fontsize=11, color=INK["primary"], x=0.01, ha="left")
    figure.tight_layout()
    return figure


def style_table(frame: pd.DataFrame, *, precision: int = 3):
    """Format a metrics table for notebook display.

    Args:
        frame: Any table from this module.
        precision: Decimal places for float columns.

    Returns:
        A pandas ``Styler``, shaded per column so the best run in each is
        visible without reading every number.
    """
    numeric = frame.select_dtypes("number").columns
    shaded = [c for c in numeric if c not in ("support", "prevalence", "threshold")]
    return (
        frame.style.format(precision=precision, na_rep="-")
        .format({"support": "{:.0f}"}, na_rep="-")
        .background_gradient(cmap="Blues", subset=shaded, axis=0, vmin=0, vmax=1)
        .set_properties(**{"font-size": "9.5pt"})
    )
