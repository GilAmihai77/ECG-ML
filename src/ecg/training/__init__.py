"""Training: configuration, metrics, durable state, and the two loops.

A run is fully described by a :class:`~ecg.training.config.RunConfig`, which
round-trips through YAML and is copied into every checkpoint, so any result can
be traced back to the settings that produced it.
"""

from __future__ import annotations

from ecg.training.checkpoints import (
    fullest_tracking,
    load_checkpoint,
    load_encoder_weights,
    restore_tracking,
    save_checkpoint,
    sync_tracking,
)
from ecg.training.config import RunConfig, TrainConfig
from ecg.training.loops import (
    TrainingResult,
    evaluate_classifier,
    predict,
    pretrain,
    resolve_device,
    set_seed,
    train_supervised,
)
from ecg.training.metrics import (
    ClassificationMetrics,
    evaluate,
    select_thresholds,
)
from ecg.training.tracking import Tracker

__all__ = [
    "ClassificationMetrics",
    "RunConfig",
    "TrainConfig",
    "Tracker",
    "TrainingResult",
    "evaluate",
    "evaluate_classifier",
    "fullest_tracking",
    "load_checkpoint",
    "load_encoder_weights",
    "predict",
    "pretrain",
    "resolve_device",
    "restore_tracking",
    "save_checkpoint",
    "select_thresholds",
    "set_seed",
    "sync_tracking",
    "train_supervised",
]
