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
"""

from __future__ import annotations

import json
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
from ecg.training.config import RunConfig
from ecg.training.loops import evaluate_classifier, pretrain, resolve_device, train_supervised
from ecg.training.metrics import ClassificationMetrics
from ecg.training.tracking import Tracker
from ecg.experiments.plan import RunSpec

#: Written into a run's output directory when it completes successfully.
RESULT_FILE: str = "result.json"


@dataclass
class RunOutcome:
    """What a finished run produced.

    Attributes:
        name: Run name.
        kind: ``"pretrain"`` or ``"supervised"``.
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

    with Tracker(
        config.tracking_uri, experiment=experiment, run_name=spec.name, enabled=track
    ) as tracker:
        tracker.log_params(config.mlflow_params())
        tracker.set_tags({"kind": spec.kind, "arm": spec.arm})
        outcome = (
            _run_pretrain(spec, config, workspace, tracker, progress)
            if spec.kind == "pretrain"
            else _run_supervised(spec, config, workspace, tracker, progress)
        )
        tracker.log_metrics(outcome.metrics, step=outcome.best_epoch)

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

    for index, spec in enumerate(specs, start=1):
        directory = root / spec.name
        if resume and (directory / RESULT_FILE).exists():
            outcome = RunOutcome.load(directory)
            checkpoints[spec.name] = outcome.checkpoint
            outcomes.append(outcome)
            if progress:
                print(f"[{index}/{len(specs)}] {spec.name}: already complete, skipping")
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

    return outcomes


def results_frame(outcomes: list[RunOutcome]) -> pd.DataFrame:
    """Tabulate outcomes.

    Args:
        outcomes: Finished runs.

    Returns:
        One row per run.
    """
    return pd.DataFrame([outcome.to_row() for outcome in outcomes])


def label_efficiency_table(
    frame: pd.DataFrame, metric: str = "test_macro_auroc"
) -> pd.DataFrame:
    """Pivot supervised results into the label-efficiency curve.

    Args:
        frame: Table from :func:`results_frame`.
        metric: Column to display.

    Returns:
        Label fractions as rows, arms as columns.
    """
    supervised = frame[frame["kind"] == "supervised"]
    return supervised.pivot_table(
        index="label_fraction", columns="arm", values=metric, aggfunc="mean"
    )


def ssl_benefit(
    frame: pd.DataFrame, metric: str = "test_macro_auroc"
) -> pd.DataFrame:
    """The study's headline: what pretraining bought, per embedder and fraction.

    Args:
        frame: Table from :func:`results_frame`.
        metric: Column to compare.

    Returns:
        Rows indexed by ``(embedder, label_fraction)`` with the scratch score,
        the pretrained score and their difference.
    """
    supervised = frame[frame["kind"] == "supervised"]
    pivot = supervised.pivot_table(
        index=["embedder", "label_fraction"],
        columns="pretrained",
        values=metric,
        aggfunc="mean",
    )
    pivot = pivot.rename(columns={False: "scratch", True: "ssl"})
    if "scratch" in pivot and "ssl" in pivot:
        pivot["ssl_gain"] = pivot["ssl"] - pivot["scratch"]
    return pivot


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
