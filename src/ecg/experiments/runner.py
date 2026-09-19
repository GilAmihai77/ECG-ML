"""Executing a plan: one run at a time, resumable, test touched once.

Two properties are deliberate.

**Resumability.** Colab disconnects mid-study, and re-running a twelve-run grid
from the start would be the most expensive possible response. Each finished run
writes ``result.json`` into its own output directory, and :func:`run_plan`
skips any run that already has one. Restarting after a disconnect therefore
costs only the run that was interrupted.

**Test is read once per run, after model selection.** Training and early
stopping see validation only; the best checkpoint is then reloaded and scored
on test using thresholds fitted on validation. Nothing in the loop can consult
test, because the loop is never handed it.

A run is one seed. The study runs each experiment at several
(:data:`ecg.experiments.plan.DEFAULT_SEEDS`) and reports the mean with its
standard deviation, so the summary functions here take a frame of individual
runs and collapse it: :func:`aggregate_runs` for the per-arm numbers, and
:func:`paired_contrast` -- behind :func:`ssl_benefit` and
:func:`embedder_benefit` -- for the two comparisons the study exists to make.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from ecg.data.datasets import Cohort, EcgBatches, build_cohorts, holdout_split, nested_subsets
from ecg.data.preprocess import WaveformStore
from ecg.data.ptbxl import load_metadata
from ecg.models.encoder import build_classifier
from ecg.models.ssl import build_pretrainer
from ecg.training.checkpoints import load_encoder_weights
from ecg.training.config import DEFAULT_VARIANT, RunConfig
from ecg.training.loops import evaluate_classifier, pretrain, resolve_device, train_supervised
from ecg.training.metrics import ClassificationMetrics
from ecg.training.tracking import UNKNOWN, Tracker, git_provenance
from ecg.experiments.plan import RunSpec

#: Written into a run's output directory when it completes successfully.
RESULT_FILE: str = "result.json"


@dataclass
class RunOutcome:
    """What a finished run produced.

    Attributes:
        name: Run name.
        kind: ``"pretrain"`` or ``"supervised"``.
        variant: Architecture variant the run belongs to.
        git_sha: Short commit the run was produced from, so ``results.csv`` is
            self-describing without consulting MLflow. Defaults keep an older
            ``result.json`` loadable.
        arm: Arm label, e.g. ``"conv+ssl"``.
        embedder: Which patch embedder was used.
        pretrained: Whether the encoder started from SSL weights.
        label_fraction: Fraction of the labelled training cohort used.
        mask_ratio: SSL masking ratio.
        seed: Training seed.
        n_train: Records trained on.
        best_epoch: Epoch selected on validation.
        epochs_run: Epochs actually executed.
        seconds: Wall-clock training time.
        checkpoint: Path to the selected checkpoint.
        metrics: Flat metrics; validation and test for supervised runs,
            reconstruction loss for pretraining runs.
    """

    name: str
    kind: str
    arm: str
    embedder: str
    pretrained: bool
    label_fraction: float
    mask_ratio: float
    seed: int
    n_train: int
    best_epoch: int
    epochs_run: int
    seconds: float
    checkpoint: str
    metrics: dict[str, float] = field(default_factory=dict)
    variant: str = DEFAULT_VARIANT
    git_sha: str = UNKNOWN

    def save(self, directory: str | Path) -> Path:
        """Write the outcome as JSON, marking the run complete.

        Args:
            directory: The run's output directory.

        Returns:
            The path written.
        """
        path = Path(directory) / RESULT_FILE
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")
        return path

    @classmethod
    def load(cls, directory: str | Path) -> RunOutcome:
        """Read an outcome written by :meth:`save`.

        Args:
            directory: The run's output directory.

        Returns:
            The outcome.
        """
        payload = json.loads(
            (Path(directory) / RESULT_FILE).read_text(encoding="utf-8")
        )
        return cls(**payload)

    def to_row(self) -> dict[str, Any]:
        """Flatten for a results table."""
        row: dict[str, Any] = {
            "run": self.name,
            "variant": self.variant,
            "git_sha": self.git_sha,
            "kind": self.kind,
            "arm": self.arm,
            "embedder": self.embedder,
            "pretrained": self.pretrained,
            "label_fraction": self.label_fraction,
            "mask_ratio": self.mask_ratio,
            "seed": self.seed,
            "n_train": self.n_train,
            "best_epoch": self.best_epoch,
            "minutes": round(self.seconds / 60, 2),
        }
        row.update(self.metrics)
        return row


@dataclass
class Workspace:
    """The data every run in a study shares.

    Loading the store once and reusing it across runs matters: it is 523 MB at
    100 Hz, and re-reading it per run would dominate a short fine-tune.

    Attributes:
        store: The waveform store.
        cohorts: Train, val, test and the SSL pool.
    """

    store: WaveformStore
    cohorts: dict[str, Cohort]

    @classmethod
    def load(cls, config: RunConfig) -> Workspace:
        """Load the store and cohorts named by a configuration.

        Args:
            config: Any run configuration from the study.

        Returns:
            The workspace.
        """
        store = WaveformStore.load(config.store_path)
        cohorts = build_cohorts(load_metadata(config.metadata_path))
        return cls(store=store, cohorts=cohorts)


def run_one(
    spec: RunSpec,
    workspace: Workspace,
    *,
    output_root: str | Path = "runs",
    tracking_uri: str | None = None,
    experiment: str = "ecg-ssl",
    track: bool = True,
    progress: bool = True,
) -> RunOutcome:
    """Execute a single run end to end.

    Args:
        spec: The run to execute. For a supervised run with a dependency,
            ``spec.config.pretrained_from`` must already name a checkpoint path
            rather than a run name -- :func:`run_plan` resolves that.
        workspace: Shared store and cohorts.
        output_root: Directory holding one subdirectory per run.
        tracking_uri: Tracking database. Defaults to the config's.
        experiment: MLflow experiment name.
        track: Set ``False`` to skip MLflow entirely.
        progress: Print per-epoch lines.

    Returns:
        The outcome, already saved to the run's directory.
    """
    directory = Path(output_root) / spec.name
    config = spec.config.with_(
        output_dir=str(directory),
        tracking_uri=tracking_uri or spec.config.tracking_uri,
    )
    config.to_yaml(directory / "config.yaml")

    # Asked once per run, not once per epoch: it shells out to git.
    provenance = git_provenance()

    with Tracker(
        config.tracking_uri, experiment=experiment, run_name=spec.name, enabled=track
    ) as tracker:
        tracker.log_params(config.mlflow_params())
        tracker.set_tags(
            {"kind": spec.kind, "arm": spec.arm, "variant": config.variant, **provenance}
        )
        outcome = (
            _run_pretrain(spec, config, workspace, tracker, progress)
            if spec.kind == "pretrain"
            else _run_supervised(spec, config, workspace, tracker, progress)
        )
        tracker.log_metrics(outcome.metrics, step=outcome.best_epoch)

    outcome.variant = config.variant
    outcome.git_sha = provenance["git_sha"]
    outcome.save(directory)
    return outcome


def run_plan(
    specs: list[RunSpec],
    workspace: Workspace,
    *,
    output_root: str | Path = "runs",
    tracking_uri: str | None = None,
    experiment: str = "ecg-ssl",
    track: bool = True,
    resume: bool = True,
    progress: bool = True,
) -> list[RunOutcome]:
    """Execute a plan, skipping runs that already finished.

    Args:
        specs: The plan, pretraining runs first.
        workspace: Shared store and cohorts.
        output_root: Directory holding one subdirectory per run.
        tracking_uri: Tracking database. Defaults to each config's.
        experiment: MLflow experiment name.
        track: Set ``False`` to skip MLflow entirely.
        resume: Skip runs whose ``result.json`` exists. The reason a Colab
            disconnect costs one run rather than the whole study.
        progress: Print per-run and per-epoch lines.

    Returns:
        One outcome per run, in plan order.

    Raises:
        FileNotFoundError: If a run depends on a pretraining run whose
            checkpoint is missing -- which means the plan was reordered or a
            dependency was never executed.
    """
    root = Path(output_root)
    outcomes: list[RunOutcome] = []
    checkpoints: dict[str, str] = {}
    current_sha = git_provenance()["git_sha"]
    stale: list[str] = []

    for index, spec in enumerate(specs, start=1):
        directory = root / spec.name
        if resume and (directory / RESULT_FILE).exists():
            outcome = RunOutcome.load(directory)
            checkpoints[spec.name] = outcome.checkpoint
            outcomes.append(outcome)
            if outcome.git_sha not in (current_sha, UNKNOWN):
                stale.append(spec.name)
            if progress:
                print(
                    f"[{index}/{len(specs)}] {spec.name}: already complete "
                    f"(variant {outcome.variant}, git {outcome.git_sha}), skipping"
                )
            continue

        resolved = spec
        if spec.depends_on:
            source = checkpoints.get(spec.depends_on)
            if source is None:
                source = str(root / spec.depends_on / "pretrain_best.pt")
            if not Path(source).exists():
                raise FileNotFoundError(
                    f"run {spec.name!r} needs the checkpoint from "
                    f"{spec.depends_on!r}, but {source} does not exist"
                )
            resolved = RunSpec(
                name=spec.name,
                kind=spec.kind,
                config=spec.config.with_(pretrained_from=source),
                depends_on=spec.depends_on,
            )

        if progress:
            print(f"\n[{index}/{len(specs)}] {spec.name}  ({spec.arm})")
        outcome = run_one(
            resolved,
            workspace,
            output_root=root,
            tracking_uri=tracking_uri,
            experiment=experiment,
            track=track,
            progress=progress,
        )
        checkpoints[spec.name] = outcome.checkpoint
        outcomes.append(outcome)

    if stale:
        # The one failure this scheme exists to catch: the code changed, the
        # run names did not, so resume returns the previous architecture's
        # numbers as if they were the new one's. Pass --variant to separate
        # them, or --no-resume to overwrite.
        warnings.warn(
            f"{len(stale)} run(s) were skipped that were produced by a "
            f"different commit than the current {current_sha}: "
            f"{', '.join(stale[:4])}{' ...' if len(stale) > 4 else ''}. "
            "If you changed the model, these results are the OLD model's -- "
            "re-run under a new --variant.",
            RuntimeWarning,
            stacklevel=2,
        )
    return outcomes


def results_frame(outcomes: list[RunOutcome]) -> pd.DataFrame:
    """Tabulate outcomes.

    Args:
        outcomes: Finished runs.

    Returns:
        One row per run.
    """
    return pd.DataFrame([outcome.to_row() for outcome in outcomes])


def load_results(output_root: str | Path) -> pd.DataFrame:
    """Rebuild the results table from every finished run under a directory.

    ``results.csv`` holds only the plan that last executed, so running a second
    variant overwrites the first variant's summary. The per-run
    ``result.json`` files are not overwritten -- each variant has its own
    directories -- so the full history is always recoverable from them, and
    this is what a comparison across variants should read.

    Args:
        output_root: Directory holding one subdirectory per run.

    Returns:
        One row per finished run, every variant included, sorted by variant
        then run name. Empty if nothing has finished.

    Raises:
        FileNotFoundError: If the directory does not exist.
    """
    root = Path(output_root)
    if not root.is_dir():
        raise FileNotFoundError(f"no output directory at {root}")

    outcomes = [
        RunOutcome.load(directory)
        for directory in sorted(p for p in root.iterdir() if p.is_dir())
        if (directory / RESULT_FILE).exists()
    ]
    if not outcomes:
        return pd.DataFrame()
    return results_frame(outcomes).sort_values(["variant", "run"], ignore_index=True)


def _grouped(frame: pd.DataFrame, index: list[str]) -> list[str]:
    """Prepend ``variant`` to a pivot index when the frame holds more than one.

    Without this, a table covering two architectures would average them into a
    single row -- silently, and into a number that describes neither.
    """
    if "variant" in frame.columns and frame["variant"].nunique() > 1:
        return ["variant", *index]
    return index


#: Columns that say *which* run a row is, as opposed to how it scored. Anything
#: numeric outside this set is treated as a metric and averaged across seeds.
IDENTITY_COLUMNS: tuple[str, ...] = (
    "run",
    "variant",
    "git_sha",
    "kind",
    "arm",
    "embedder",
    "pretrained",
    "label_fraction",
    "mask_ratio",
    "seed",
    "n_train",
)

#: What makes two rows the same experiment at different seeds.
REPLICATE_KEYS: tuple[str, ...] = (
    "variant",
    "kind",
    "arm",
    "label_fraction",
    "mask_ratio",
)

#: Two-sided 95% critical values of Student's t, by degrees of freedom. A table
#: rather than ``scipy.stats``: scipy is a preprocessing-only extra here, and
#: the cloud training image is kept to what a run actually needs. Lookups round
#: *down* to a tabulated df, so an untabulated one gets the wider interval.
_T95: tuple[tuple[int, float], ...] = (
    (1, 12.706),
    (2, 4.303),
    (3, 3.182),
    (4, 2.776),
    (5, 2.571),
    (6, 2.447),
    (7, 2.365),
    (8, 2.306),
    (9, 2.262),
    (10, 2.228),
    (12, 2.179),
    (15, 2.131),
    (20, 2.086),
    (30, 2.042),
    (60, 2.000),
)


def _t95(df: int) -> float:
    """Two-sided 95% critical value of t for ``df`` degrees of freedom."""
    for tabulated, value in _T95:
        if df <= tabulated:
            return value
    return 1.960


def _ci95(sd: pd.Series, n: pd.Series) -> pd.Series:
    """Half-width of a 95% confidence interval for a mean.

    Args:
        sd: Sample standard deviations (``ddof=1``).
        n: Sample sizes.

    Returns:
        ``t * sd / sqrt(n)``, ``nan`` wherever ``n < 2``.
    """
    critical = n.map(lambda size: _t95(int(size) - 1) if size >= 2 else np.nan)
    return critical * sd / np.sqrt(n)


def metric_columns(frame: pd.DataFrame) -> list[str]:
    """The numeric columns of a results frame that are measurements.

    Args:
        frame: Table from :func:`results_frame`.

    Returns:
        Column names, in frame order, excluding :data:`IDENTITY_COLUMNS`.
    """
    return [
        column
        for column in frame.columns
        if column not in IDENTITY_COLUMNS
        and pd.api.types.is_numeric_dtype(frame[column])
    ]


def aggregate_runs(
    frame: pd.DataFrame, metrics: list[str] | None = None
) -> pd.DataFrame:
    """Collapse replicate seeds into mean, standard deviation and n.

    This is the table the study reports. A single run's macro AUROC moves by
    more than the differences being compared, so one number per arm cannot
    support a claim about the embedder or about SSL; a mean over seeds with its
    spread beside it can.

    Args:
        frame: Table from :func:`results_frame` or :func:`load_results`.
        metrics: Columns to summarise. Defaults to :func:`metric_columns`.

    Returns:
        Rows indexed by :data:`REPLICATE_KEYS`, with ``n_seeds``, ``seeds`` (the
        seeds actually present, so a part-finished study is visible rather than
        being read as a complete one) and ``<metric>_mean`` / ``<metric>_sd``
        for each metric. The deviation is the sample one (``ddof=1``) and is
        ``nan`` at ``n_seeds == 1``.

    Raises:
        ValueError: If the frame is missing a replicate key.

    Warns:
        RuntimeWarning: If one experiment holds two runs at the same seed --
            two directories for one configuration, which would be reported as
            independent replicates and shrink the interval for nothing.
    """
    if frame.empty:
        return pd.DataFrame()
    missing = [key for key in REPLICATE_KEYS if key not in frame.columns]
    if missing:
        raise ValueError(f"results frame is missing {missing}")

    working = frame.copy()
    if "seed" not in working.columns:
        working["seed"] = 0
    keys = list(REPLICATE_KEYS)
    # dropna=False: pretraining rows carry a nan label_fraction, and the
    # default would drop every one of them from the summary.
    grouped = working.groupby(keys, dropna=False, sort=True)

    duplicated = grouped["seed"].apply(lambda s: s.duplicated().any())
    if bool(duplicated.any()):
        warnings.warn(
            f"{int(duplicated.sum())} experiment(s) hold more than one run at "
            "the same seed, so their n_seeds counts repeats as replicates: "
            f"{[str(key) for key in duplicated[duplicated].index[:3]]}. "
            "Usually a study re-run under a changed plan; delete the stale run "
            "directories or separate them with --variant.",
            RuntimeWarning,
            stacklevel=2,
        )

    summary = pd.DataFrame(
        {
            "n_seeds": grouped["seed"].nunique(),
            "seeds": grouped["seed"].apply(
                lambda s: ",".join(str(v) for v in sorted(s.unique()))
            ),
        }
    )
    for metric in metrics if metrics is not None else metric_columns(working):
        summary[f"{metric}_mean"] = grouped[metric].mean()
        summary[f"{metric}_sd"] = grouped[metric].std(ddof=1)
    return summary


def format_mean_sd(
    mean: float, sd: float, *, precision: int = 3, na_rep: str = "-"
) -> str:
    """Render one ``mean +/- sd`` cell.

    Args:
        mean: The mean.
        sd: The standard deviation; ``nan`` for a single seed.
        precision: Decimal places.
        na_rep: Stand-in for a missing mean.

    Returns:
        ``"0.912 ± 0.004"``, or ``"0.912"`` when the deviation is undefined.
    """
    if mean is None or (isinstance(mean, float) and np.isnan(mean)):
        return na_rep
    if sd is None or (isinstance(sd, float) and np.isnan(sd)):
        return f"{mean:.{precision}f}"
    return f"{mean:.{precision}f} ± {sd:.{precision}f}"


def label_efficiency_table(
    frame: pd.DataFrame, metric: str = "test_macro_auroc", *, stat: str = "mean"
) -> pd.DataFrame:
    """Pivot supervised results into the label-efficiency curve.

    Args:
        frame: Table from :func:`results_frame`.
        metric: Column to display.
        stat: ``"mean"`` across seeds, ``"sd"`` for the sample deviation (error
            bars for the curve), or ``"n"`` for the seed count.

    Returns:
        Label fractions as rows, arms as columns. If the frame covers several
        architecture variants, they become the outer row level rather than
        being averaged together.

    Raises:
        ValueError: If ``stat`` is not one of the three.
    """
    aggregate = {"mean": "mean", "sd": lambda s: s.std(ddof=1), "n": "count"}
    if stat not in aggregate:
        raise ValueError(f"stat must be 'mean', 'sd' or 'n', got {stat!r}")
    supervised = frame[frame["kind"] == "supervised"]
    return supervised.pivot_table(
        index=_grouped(supervised, ["label_fraction"]),
        columns="arm",
        values=metric,
        aggfunc=aggregate[stat],
    )


def label_efficiency_report(
    frame: pd.DataFrame, metric: str = "test_macro_auroc", *, precision: int = 3
) -> pd.DataFrame:
    """The label-efficiency curve as ``mean ± sd`` text, one cell per arm.

    The reporting counterpart of :func:`label_efficiency_table`, which stays
    numeric so it can still be plotted.

    Args:
        frame: Table from :func:`results_frame`.
        metric: Column to display.
        precision: Decimal places.

    Returns:
        Same shape as :func:`label_efficiency_table`, with string cells.
    """
    means = label_efficiency_table(frame, metric, stat="mean")
    deviations = label_efficiency_table(frame, metric, stat="sd").reindex(
        index=means.index, columns=means.columns
    )
    report = means.astype(object)
    for column in means.columns:
        report[column] = [
            format_mean_sd(mean, sd, precision=precision)
            for mean, sd in zip(means[column], deviations[column])
        ]
    return report


def paired_contrast(
    frame: pd.DataFrame,
    *,
    contrast: str,
    levels: tuple[Any, Any],
    names: tuple[str, str],
    index: list[str],
    metric: str = "test_macro_auroc",
) -> pd.DataFrame:
    """Difference two arms **within each seed**, then average the differences.

    Pairing is what makes five seeds worth running rather than merely honest.
    The two means come out the same either way, but the spread does not: a seed
    that was a bad draw for one arm was the same bad draw for the other -- same
    labelled records, same batch order -- so differencing inside the seed
    removes that shared variation from the interval instead of leaving it in
    both terms. Unpaired, the between-seed noise can be several times the effect
    being measured and no interval would exclude zero; paired, the interval is
    about the contrast itself.

    Seeds missing either arm are dropped from the row entirely, so the two
    reported means always differ by exactly the reported gain.

    Args:
        frame: Table from :func:`results_frame`.
        contrast: Column holding the two arms, e.g. ``"pretrained"``.
        levels: The two values of that column, baseline first.
        names: Display names for those values, baseline first.
        index: Grouping columns. ``variant`` is prepended when the frame holds
            more than one.
        metric: Column to compare.

    Returns:
        One row per group. Columns: the two arm means and their deviations,
        ``<treatment>_gain`` (the paired mean difference), ``gain_sd``,
        ``gain_ci95`` (half-width of the 95% interval on that mean),
        ``gain_beats_noise`` (the interval excludes zero) and ``n_seeds``
        (complete pairs). At one seed the deviations are ``nan`` and
        ``gain_beats_noise`` is ``False`` -- a single run cannot show that
        anything beat noise.
    """
    supervised = frame[frame["kind"] == "supervised"].copy()
    if "seed" not in supervised.columns:
        supervised["seed"] = 0
    keys = _grouped(supervised, index)
    baseline, treatment = names

    paired = supervised.pivot_table(
        index=[*keys, "seed"], columns=contrast, values=metric, aggfunc="mean"
    ).rename(columns=dict(zip(levels, names)))
    if baseline not in paired.columns or treatment not in paired.columns:
        # Only one arm present -- nothing to pair, so report what there is.
        return paired.groupby(keys).mean()

    paired = paired.dropna(subset=[baseline, treatment])
    gain = f"{treatment}_gain"
    paired[gain] = paired[treatment] - paired[baseline]
    grouped = paired.groupby(keys)

    summary = pd.DataFrame(
        {
            baseline: grouped[baseline].mean(),
            f"{baseline}_sd": grouped[baseline].std(ddof=1),
            treatment: grouped[treatment].mean(),
            f"{treatment}_sd": grouped[treatment].std(ddof=1),
            gain: grouped[gain].mean(),
            "gain_sd": grouped[gain].std(ddof=1),
            "n_seeds": grouped[gain].count(),
        }
    )
    summary["gain_ci95"] = _ci95(summary["gain_sd"], summary["n_seeds"])
    summary["gain_beats_noise"] = (summary[gain].abs() > summary["gain_ci95"]).fillna(
        False
    )
    return summary[
        [
            baseline,
            f"{baseline}_sd",
            treatment,
            f"{treatment}_sd",
            gain,
            "gain_sd",
            "gain_ci95",
            "gain_beats_noise",
            "n_seeds",
        ]
    ]


def ssl_benefit(
    frame: pd.DataFrame, metric: str = "test_macro_auroc"
) -> pd.DataFrame:
    """The study's headline: what pretraining bought, per embedder and fraction.

    Arms C/D against A/B, paired by seed -- see :func:`paired_contrast` for why
    the pairing rather than a difference of two averages.

    Args:
        frame: Table from :func:`results_frame`.
        metric: Column to compare.

    Returns:
        Rows indexed by ``(embedder, label_fraction)``, variants kept apart as
        an outer level, with ``scratch``, ``ssl``, ``ssl_gain`` and the spread
        columns described in :func:`paired_contrast`.
    """
    return paired_contrast(
        frame,
        contrast="pretrained",
        levels=(False, True),
        names=("scratch", "ssl"),
        index=["embedder", "label_fraction"],
        metric=metric,
    )


def embedder_benefit(
    frame: pd.DataFrame, metric: str = "test_macro_auroc"
) -> pd.DataFrame:
    """The other half of the question: what the conv stem bought over linear.

    A against B, and C against D, paired by seed. The pairing is legitimate
    here for the same reason it is in :func:`ssl_benefit`: at a given seed the
    two embedders train on identical records in identical order under an
    identical budget, so the only thing left between them is the embedder.

    Read it beside the parameter counts. A conv win at matched capacity is an
    inductive-bias result; a conv win at 40% more parameters is not one.

    Args:
        frame: Table from :func:`results_frame`.
        metric: Column to compare.

    Returns:
        Rows indexed by ``(pretrained, label_fraction)``, variants kept apart as
        an outer level, with ``linear``, ``conv``, ``conv_gain`` and the spread
        columns described in :func:`paired_contrast`.
    """
    return paired_contrast(
        frame,
        contrast="embedder",
        levels=("linear", "conv"),
        names=("linear", "conv"),
        index=["pretrained", "label_fraction"],
        metric=metric,
    )


def _batches(
    workspace: Workspace,
    cohort: Cohort,
    config: RunConfig,
    device: torch.device,
    *,
    shuffle: bool,
) -> EcgBatches:
    """Build batches resident on the training device.

    Every cohort goes through here rather than constructing :class:`EcgBatches`
    inline, because the device is easy to omit and omitting it fails only on a
    GPU: the model is moved by the training loop, the batches are not, and the
    first matmul reports a device mismatch several frames deep.

    Args:
        workspace: Shared store and cohorts.
        cohort: Records to iterate.
        config: The run configuration, supplying batch size and seed.
        device: Device the model will train on.
        shuffle: Shuffle each epoch. ``False`` for evaluation cohorts, so
            predictions line up with :attr:`Cohort.ecg_ids`.

    Returns:
        Batches whose tensors already live on ``device``.
    """
    return EcgBatches(
        workspace.store,
        cohort,
        batch_size=config.train.batch_size,
        shuffle=shuffle,
        device=device,
        seed=config.train.seed,
    )


def _run_pretrain(
    spec: RunSpec,
    config: RunConfig,
    workspace: Workspace,
    tracker: Tracker,
    progress: bool,
) -> RunOutcome:
    """Execute a pretraining run."""
    pool = workspace.cohorts["ssl"]
    kept, held = holdout_split(pool, config.ssl_holdout, seed=config.subset_seed)
    model = build_pretrainer(config.model, config.ssl)
    device = resolve_device(config.train.device)

    result = pretrain(
        model,
        _batches(workspace, kept, config, device, shuffle=True),
        _batches(workspace, held, config, device, shuffle=False) if len(held) else None,
        config,
        tracker=tracker,
        progress=progress,
    )
    return RunOutcome(
        name=spec.name,
        kind="pretrain",
        arm=spec.arm,
        embedder=config.model.embedder,
        pretrained=False,
        label_fraction=float("nan"),
        mask_ratio=config.ssl.mask_ratio,
        seed=config.train.seed,
        n_train=len(kept),
        best_epoch=result.best_epoch,
        epochs_run=result.epochs_run,
        seconds=result.seconds,
        checkpoint=str(result.best_checkpoint) if result.best_checkpoint else "",
        metrics={
            "holdout_loss": -result.best_score,
            "final_ssl_loss": result.history[-1]["ssl_loss"] if result.history else float("nan"),
            "contaminated_fraction": (
                result.history[-1].get("contaminated_fraction", 0.0)
                if result.history
                else 0.0
            ),
        },
    )


def _run_supervised(
    spec: RunSpec,
    config: RunConfig,
    workspace: Workspace,
    tracker: Tracker,
    progress: bool,
) -> RunOutcome:
    """Execute a supervised run and score it on test exactly once."""
    train_cohort = nested_subsets(
        workspace.cohorts["train"], (config.label_fraction,), seed=config.subset_seed
    )[config.label_fraction]

    model = build_classifier(config.model)
    if config.pretrained_from:
        load_encoder_weights(model, config.pretrained_from)
    device = resolve_device(config.train.device)

    val_batches = _batches(workspace, workspace.cohorts["val"], config, device, shuffle=False)
    result = train_supervised(
        model,
        _batches(workspace, train_cohort, config, device, shuffle=True),
        val_batches,
        config,
        tracker=tracker,
        progress=progress,
    )

    # Model selection is finished. Reload the epoch validation chose -- the
    # weights in memory are the last epoch's, not the best -- then read test
    # once, with thresholds fitted on validation (integrity rule 2).
    selected = build_classifier(config.model)
    if result.best_checkpoint:
        from ecg.training.checkpoints import load_checkpoint

        selected.load_state_dict(load_checkpoint(result.best_checkpoint)["model"])
    else:
        selected.load_state_dict(model.state_dict())
    selected.to(device)

    val_metrics = evaluate_classifier(selected, val_batches)
    thresholds = np.array(
        [val_metrics.thresholds[name] for name in val_metrics.thresholds]
    )
    test_metrics = evaluate_classifier(
        selected,
        _batches(workspace, workspace.cohorts["test"], config, device, shuffle=False),
        thresholds,
    )

    metrics = _macro(val_metrics, "val")
    metrics.update(_macro(test_metrics, "test"))
    if progress:
        print(f"  test {test_metrics.summary()}   (thresholds from val)")

    return RunOutcome(
        name=spec.name,
        kind="supervised",
        arm=spec.arm,
        embedder=config.model.embedder,
        pretrained=bool(config.pretrained_from),
        label_fraction=config.label_fraction,
        mask_ratio=config.ssl.mask_ratio,
        seed=config.train.seed,
        n_train=len(train_cohort),
        best_epoch=result.best_epoch,
        epochs_run=result.epochs_run,
        seconds=result.seconds,
        checkpoint=str(result.best_checkpoint) if result.best_checkpoint else "",
        metrics=metrics,
    )


def _macro(metrics: ClassificationMetrics, prefix: str) -> dict[str, float]:
    """The three headline numbers, plus per-class AUROC."""
    flat = {
        f"{prefix}_macro_auroc": metrics.macro_auroc,
        f"{prefix}_macro_pr_auc": metrics.macro_pr_auc,
        f"{prefix}_macro_f1": metrics.macro_f1,
    }
    for name, value in metrics.auroc.items():
        if not np.isnan(value):
            flat[f"{prefix}_auroc_{name}"] = float(value)
    return flat
