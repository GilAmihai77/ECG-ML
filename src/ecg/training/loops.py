"""The two training loops: SSL pretraining and supervised fine-tuning.

Both share the same skeleton -- cosine-decayed AdamW, gradient clipping, one
evaluation and one checkpoint per epoch -- because arms A/B and C/D must differ
only in the embedder and in whether the encoder was pretrained. The shared
pieces live in this module rather than being written twice, so they cannot
drift apart.

Each epoch ends with exactly two durable writes: the checkpoint and the MLflow
archive, both to ``output_dir``. That is the only place the loop touches the
Drive mount, and both writes are wrapped so a stale mount costs a warning
rather than the run.
"""

from __future__ import annotations

import math
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch import nn

from ecg.data.datasets import EcgBatches
from ecg.models.encoder import EcgClassifier
from ecg.models.ssl import MaskedReconstruction
from ecg.training.checkpoints import TRACKING_DB, save_checkpoint, sync_tracking
from ecg.training.config import RunConfig, TrainConfig
from ecg.training.metrics import ClassificationMetrics, evaluate, select_thresholds
from ecg.training.tracking import Tracker

#: Metric that selects the best supervised checkpoint. Validation only.
SELECTION_METRIC: str = "val_macro_auroc"


@dataclass
class TrainingResult:
    """Outcome of one training run.

    Attributes:
        history: One record per evaluated epoch, as logged to MLflow.
        best_epoch: Epoch with the best selection metric.
        best_score: That metric's value.
        best_checkpoint: Path to the best checkpoint, if one was written.
        seconds: Wall-clock training time.
    """

    history: list[dict[str, float]] = field(default_factory=list)
    best_epoch: int = 0
    best_score: float = float("-inf")
    best_checkpoint: Path | None = None
    seconds: float = 0.0

    @property
    def epochs_run(self) -> int:
        """Number of evaluated epochs."""
        return len(self.history)


def resolve_device(requested: str) -> torch.device:
    """Turn a configured device string into a device.

    Args:
        requested: ``"auto"``, ``"cpu"`` or an explicit device string.

    Returns:
        The device to train on.
    """
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def set_seed(seed: int) -> None:
    """Seed every generator the loops draw from (integrity rule 4).

    Args:
        seed: The run seed.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_optimiser(
    model: nn.Module, config: TrainConfig
) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler]:
    """Build AdamW and its cosine schedule.

    Weight decay is not applied to biases or normalisation parameters, the
    usual convention: decaying a LayerNorm gain pulls it toward zero and
    fights the normalisation it is there to provide.

    No warmup. The encoder is pre-norm, which trains stably without one, and
    dropping it removes a hyperparameter that would otherwise have to be held
    identical across four arms.

    Args:
        model: Model to optimise.
        config: Optimisation budget.

    Returns:
        ``(optimiser, scheduler)``. The scheduler steps once per epoch.
    """
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.ndim <= 1 or name.endswith(".bias") or "positions" in name:
            no_decay.append(parameter)
        else:
            decay.append(parameter)

    optimiser = torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": config.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=config.lr,
    )

    def factor(epoch: int) -> float:
        """Cosine decay from 1.0 to ``min_lr_ratio`` over the run."""
        if config.epochs <= 1:
            return 1.0
        progress = epoch / (config.epochs - 1)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return config.min_lr_ratio + (1.0 - config.min_lr_ratio) * cosine

    return optimiser, torch.optim.lr_scheduler.LambdaLR(optimiser, factor)


def autocast_context(device: torch.device, enabled: bool):
    """Autocast context for the device, or a no-op.

    bfloat16 rather than float16: it has the same exponent range as float32, so
    there is no loss scaling to configure and no scaler state to checkpoint.

    Args:
        device: Training device.
        enabled: Whether mixed precision was requested.

    Returns:
        A context manager.
    """
    if enabled and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return torch.autocast(device_type="cpu", enabled=False)


@torch.no_grad()
def predict(model: EcgClassifier, batches: EcgBatches) -> tuple[np.ndarray, np.ndarray]:
    """Score a cohort.

    Args:
        model: Classifier to evaluate.
        batches: Batches over the cohort. Must be unshuffled so predictions
            line up with the cohort's ``ecg_ids``.

    Returns:
        ``(y_true, y_score)`` as float arrays of shape ``(n_records, 5)``.
    """
    was_training = model.training
    model.eval()
    scores, truths = [], []
    for signal, labels in batches:
        scores.append(torch.sigmoid(model(signal)).float().cpu().numpy())
        truths.append(labels.float().cpu().numpy())
    model.train(was_training)
    return np.concatenate(truths), np.concatenate(scores)


def pretrain(
    model: MaskedReconstruction,
    batches: EcgBatches,
    holdout: EcgBatches | None,
    config: RunConfig,
    *,
    tracker: Tracker | None = None,
    progress: bool = True,
) -> TrainingResult:
    """Run masked-reconstruction pretraining.

    Args:
        model: The pretraining model for one arm.
        batches: Shuffled batches over the SSL pool.
        holdout: Unshuffled batches over the held-out slice, or ``None`` to
            skip reconstruction validation.
        config: The run configuration.
        tracker: Tracker to log to; tracking is skipped when ``None``.
        progress: Print one line per epoch.

    Early stopping needs a held-out signal, so it is active only when
    ``holdout`` is given. Stopping on the *training* reconstruction loss would
    be worse than not stopping: it falls almost monotonically, so patience
    would never fire, and on the occasion it did it would be measuring noise.
    Pretraining evaluates every epoch, so patience counts epochs here, where in
    :func:`train_supervised` it counts evaluations.

    Returns:
        A :class:`TrainingResult`. The selection metric is the held-out
        reconstruction loss, negated so that larger is better throughout.

    Warns:
        RuntimeWarning: If ``patience`` is set but no holdout was supplied, so
            the run silently has no early stopping; or if early stopping fired
            while the cosine schedule was still near peak learning rate -- see
            :func:`_warn_if_schedule_incomplete`.
    """
    device = resolve_device(config.train.device)
    model.to(device)
    set_seed(config.train.seed)
    optimiser, scheduler = build_optimiser(model, config.train)
    generator = torch.Generator().manual_seed(config.train.seed)
    output = Path(config.output_dir)
    result = TrainingResult()
    stale = 0
    stopping = bool(config.train.patience) and holdout is not None
    if config.train.patience and holdout is None:
        warnings.warn(
            "patience is set but this pretraining run has no holdout "
            "(ssl_holdout=0), so early stopping is disabled: the only "
            "available signal is the training reconstruction loss, which "
            "falls almost monotonically. Set ssl_holdout above 0 to get the "
            "early stop, or set patience to 0 to say you did not want one.",
            RuntimeWarning,
            stacklevel=2,
        )
    started = time.perf_counter()

    for epoch in range(config.train.epochs):
        model.train()
        totals: dict[str, float] = {}
        for signal, _ in batches:
            with autocast_context(device, config.train.amp):
                step = model.step(signal, generator=generator)
            optimiser.zero_grad(set_to_none=True)
            step.loss.backward()
            _clip(model, config.train.grad_clip)
            optimiser.step()
            for key, value in step.metrics.items():
                totals[key] = totals.get(key, 0.0) + value
        scheduler.step()

        n_steps = max(1, len(batches))
        record = {key: value / n_steps for key, value in totals.items()}
        record["lr"] = float(optimiser.param_groups[0]["lr"])
        record["epoch"] = float(epoch)
        if holdout is not None:
            record["holdout_loss"] = _reconstruction_loss(model, holdout, config)

        score = -record.get("holdout_loss", record["ssl_loss"])
        result.history.append(record)
        if tracker is not None:
            tracker.log_metrics(record, step=epoch)
        if progress:
            print(
                f"  epoch {epoch + 1:3d}/{config.train.epochs}  "
                f"ssl_loss {record['ssl_loss']:.4f}"
                + (
                    f"  holdout {record['holdout_loss']:.4f}"
                    if holdout is not None
                    else ""
                )
            )

        if score > result.best_score:
            result.best_score = score
            result.best_epoch = epoch
            stale = 0
            result.best_checkpoint = _persist(
                output / "pretrain_best.pt", epoch, model, optimiser, config,
                record, with_optimiser=False,
            )
        else:
            stale += 1
        _persist(output / "pretrain_last.pt", epoch, model, optimiser, config, record)

        # After the durable write, not before it: the encoder the SSL arms
        # fine-tune from is already safe either way, but breaking first would
        # leave pretrain_last.pt and the synced database a few epochs behind
        # the run that actually happened.
        if stopping and stale >= config.train.patience:
            if progress:
                print(f"  early stop: {stale} epochs without holdout improvement")
            _warn_if_schedule_incomplete(optimiser, config.train, epoch)
            break

    result.seconds = time.perf_counter() - started
    return result


def train_supervised(
    model: EcgClassifier,
    train_batches: EcgBatches,
    val_batches: EcgBatches,
    config: RunConfig,
    *,
    tracker: Tracker | None = None,
    progress: bool = True,
) -> TrainingResult:
    """Train or fine-tune a classifier.

    Model selection uses validation macro AUROC only; test is never read here
    (integrity rule 2). Per-class F1 thresholds are chosen on validation at
    every evaluated epoch; the caller refits them on validation at the selected
    epoch and passes those fixed numbers to test, so the test evaluation never
    fits its own.

    ``val_loss`` is logged alongside the ranking metrics but nothing selects on
    it -- see :data:`SELECTION_METRIC`.

    Args:
        model: Classifier, optionally with a pretrained encoder already loaded.
        train_batches: Shuffled batches over the labelled training cohort.
        val_batches: Unshuffled batches over the validation cohort.
        config: The run configuration.
        tracker: Tracker to log to; tracking is skipped when ``None``.
        progress: Print one line per evaluated epoch.

    Returns:
        A :class:`TrainingResult`.
    """
    device = resolve_device(config.train.device)
    model.to(device)
    set_seed(config.train.seed)
    optimiser, scheduler = build_optimiser(model, config.train)
    criterion = nn.BCEWithLogitsLoss()
    output = Path(config.output_dir)
    result = TrainingResult()
    stale = 0
    started = time.perf_counter()

    for epoch in range(config.train.epochs):
        model.train()
        total, n_steps = 0.0, 0
        for signal, labels in train_batches:
            with autocast_context(device, config.train.amp):
                loss = criterion(model(signal), labels)
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            _clip(model, config.train.grad_clip)
            optimiser.step()
            total += float(loss.detach())
            n_steps += 1
        scheduler.step()

        if (epoch + 1) % config.train.eval_every and epoch + 1 != config.train.epochs:
            continue

        y_true, y_score = predict(model, val_batches)
        metrics = evaluate(y_true, y_score, select_thresholds(y_true, y_score))
        record = metrics.to_mlflow("val")
        record["train_loss"] = total / max(1, n_steps)
        # Logged, never selected on: SELECTION_METRIC stays val macro AUROC.
        # It costs nothing -- the scores are already in hand -- and without it a
        # training curve has no validation counterpart to plot against.
        record["val_loss"] = _binary_cross_entropy(y_true, y_score)
        record["lr"] = float(optimiser.param_groups[0]["lr"])
        record["epoch"] = float(epoch)
        result.history.append(record)
        if tracker is not None:
            tracker.log_metrics(record, step=epoch)
        if progress:
            print(
                f"  epoch {epoch + 1:3d}/{config.train.epochs}  "
                f"loss {record['train_loss']:.4f}  val {metrics.summary()}"
            )

        score = record[SELECTION_METRIC]
        if score > result.best_score:
            result.best_score = score
            result.best_epoch = epoch
            stale = 0
            result.best_checkpoint = _persist(
                output / "best.pt", epoch, model, optimiser, config, record,
                with_optimiser=False,
            )
        else:
            stale += 1
        _persist(output / "last.pt", epoch, model, optimiser, config, record)

        if config.train.patience and stale >= config.train.patience:
            if progress:
                print(f"  early stop: {stale} evaluations without improvement")
            _warn_if_schedule_incomplete(optimiser, config.train, epoch)
            break

    result.seconds = time.perf_counter() - started
    return result


def evaluate_classifier(
    model: EcgClassifier,
    batches: EcgBatches,
    thresholds: np.ndarray | None = None,
) -> ClassificationMetrics:
    """Score a classifier on a cohort.

    Args:
        model: Classifier to evaluate.
        batches: Unshuffled batches over the cohort.
        thresholds: Per-class F1 thresholds. When ``None`` they are fitted on
            *this* cohort, which is correct for validation and wrong for test
            -- pass validation's thresholds when scoring test.

    Returns:
        The metrics.
    """
    y_true, y_score = predict(model, batches)
    if thresholds is None:
        thresholds = select_thresholds(y_true, y_score)
    return evaluate(y_true, y_score, thresholds)


def _binary_cross_entropy(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Mean BCE over every record and class, from probabilities.

    Recomputed from the scores rather than accumulated during the validation
    pass, so it needs no extra forward pass. Probabilities are clipped away from
    0 and 1 because a confident-and-wrong prediction would otherwise send the
    mean to infinity and blank the curve it exists to draw.

    Args:
        y_true: ``(n_records, n_classes)`` binary labels.
        y_score: ``(n_records, n_classes)`` predicted probabilities.

    Returns:
        The mean loss.
    """
    scores = np.clip(y_score, 1e-7, 1 - 1e-7)
    return float(
        -np.mean(y_true * np.log(scores) + (1 - y_true) * np.log(1 - scores))
    )


def _reconstruction_loss(
    model: MaskedReconstruction, batches: EcgBatches, config: RunConfig
) -> float:
    """Mean held-out reconstruction loss under a fixed mask stream.

    The mask generator is re-seeded every call, so the held-out number moves
    only because the model moved. A fresh random mask each epoch would add
    noise that looks like progress.

    Args:
        model: The pretraining model.
        batches: Unshuffled held-out batches.
        config: The run configuration.

    Returns:
        Mean loss over the held-out slice.
    """
    was_training = model.training
    model.eval()
    generator = torch.Generator().manual_seed(config.train.seed)
    total, count = 0.0, 0
    with torch.no_grad():
        for signal, _ in batches:
            mask = model.sample_mask(
                signal.shape[0], generator=generator, device=signal.device
            )
            total += float(model.loss(signal, mask))
            count += 1
    model.train(was_training)
    return total / max(1, count)


def _clip(model: nn.Module, grad_clip: float) -> None:
    """Clip gradients by global norm, if enabled."""
    if grad_clip > 0:
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)


#: Fraction of the peak learning rate below which the cosine schedule counts as
#: having done its job. Above it, a run that stopped early never annealed.
_ANNEALED_BELOW: float = 0.5


def _warn_if_schedule_incomplete(
    optimiser: torch.optim.Optimizer, config: TrainConfig, epoch: int
) -> None:
    """Warn when early stopping cut the cosine schedule short.

    The trap that makes "set a huge epoch budget and let patience decide" a
    worse idea than it sounds. :func:`build_optimiser` spreads the cosine decay
    across ``config.epochs``, so raising that number does not merely add a
    ceiling -- it stretches the schedule. Stop at epoch 60 of 300 and the
    learning rate is still near peak, so the weights never got the low-rate
    phase where a transformer consolidates. The run does not fail; it quietly
    returns a worse encoder than the same number of epochs under a budget that
    matched.

    So set ``epochs`` to the length the schedule should span and treat patience
    as the safety net for a run that has plainly stopped improving, rather than
    as the normal way a run ends.

    Args:
        optimiser: The optimiser, read for its current learning rate.
        config: The optimisation budget, for the peak rate.
        epoch: The epoch the run stopped after, zero-based.
    """
    final = float(optimiser.param_groups[0]["lr"])
    if config.lr <= 0 or final <= _ANNEALED_BELOW * config.lr:
        return
    warnings.warn(
        f"early stop at epoch {epoch + 1} of a {config.epochs}-epoch budget, "
        f"with the learning rate still at {final / config.lr:.0%} of peak. The "
        "cosine schedule is spread over the full budget, so this run never "
        "reached its low-rate phase and is not equivalent to one trained with "
        f"epochs={epoch + 1}. Lower epochs to roughly the length runs actually "
        "take, and keep patience as the safety net rather than the usual exit.",
        RuntimeWarning,
        stacklevel=2,
    )


def _persist(
    path: Path,
    epoch: int,
    model: nn.Module,
    optimiser: torch.optim.Optimizer,
    config: RunConfig,
    metrics: dict[str, float],
    *,
    with_optimiser: bool = True,
) -> Path | None:
    """Write a checkpoint and copy the tracking database beside it.

    The one place the loop touches durable storage. Both writes are wrapped:
    on Colab a stale Drive mount raises from here, and losing an epoch of
    checkpointing must not cost the hours of training that preceded it.

    Args:
        path: Checkpoint destination.
        epoch: Completed epoch.
        model: Model to save.
        optimiser: Optimiser to save.
        config: The run configuration.
        metrics: Metrics at this epoch.
        with_optimiser: Include optimiser state. ``True`` for the ``last``
            checkpoint, which exists to resume from. ``False`` for ``best``,
            which is only ever read for its weights -- by
            :func:`~ecg.training.checkpoints.load_encoder_weights` and by the
            test evaluation -- so the optimiser state would be two thirds of a
            40 MB Drive write for nothing.

    Returns:
        The checkpoint path, or ``None`` if the write failed.
    """
    try:
        written = save_checkpoint(
            path,
            epoch=epoch,
            model=model,
            optimiser=optimiser if with_optimiser else None,
            config=config,
            metrics=metrics,
        )
    except Exception as error:  # noqa: BLE001 - never lose a run to a bad mount
        warnings.warn(
            f"could not write checkpoint {path}: {type(error).__name__}: {error}",
            RuntimeWarning,
            stacklevel=2,
        )
        return None

    try:
        sync_tracking(config.tracking_uri, path.parent / TRACKING_DB)
    except Exception as error:  # noqa: BLE001
        warnings.warn(
            f"could not sync tracking database: {type(error).__name__}: {error}",
            RuntimeWarning,
            stacklevel=2,
        )
    return written
