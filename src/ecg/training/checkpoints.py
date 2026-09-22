"""Durable state: checkpoints and the tracking archive.

Colab disconnects, and its local disk goes with the session, so anything worth
keeping has to reach Drive. Drive is a FUSE mount whose costs are the opposite
of a local disk's: one large file is cheap, thousands of small ones are not.
Both functions here respect that: a checkpoint is one file, and tracking is one
SQLite database rather than the old file store's thousands of tiny ones.

Every write goes to a temporary name and is then moved into place. A rename is
not guaranteed atomic on FUSE the way it is on a local filesystem, but it still
turns "half-written checkpoint discovered after a disconnect" from the likely
outcome into an unlikely one.
"""

from __future__ import annotations

import os
import shutil
import sqlite3
from pathlib import Path
from typing import Any

import torch
from torch import nn

from ecg.training.config import RunConfig

#: Name of the tracking database copied beside the checkpoints.
TRACKING_DB: str = "mlflow.db"


def save_checkpoint(
    path: str | Path,
    *,
    epoch: int,
    model: nn.Module,
    optimiser: torch.optim.Optimizer | None,
    config: RunConfig,
    metrics: dict[str, float] | None = None,
    state: dict[str, Any] | None = None,
) -> Path:
    """Write a resumable checkpoint.

    The configuration travels with the weights (integrity rule 5), so a
    checkpoint found later is self-describing and cannot be silently paired
    with the wrong architecture.

    Args:
        path: Destination file.
        epoch: Completed epoch number.
        model: Model whose ``state_dict`` is saved, unless ``state`` is given.
        optimiser: Optimiser state, so a run resumes rather than restarts.
            ``None`` for a final, inference-only checkpoint.
        config: The run configuration.
        metrics: Metrics at this epoch, for choosing between checkpoints later.
        state: Weights to write instead of the model's current ones. Used when
            a loop holds the best weights in memory and flushes them later --
            by then the model has trained on, so ``model.state_dict()`` is no
            longer the thing being saved. ``model`` is still required, as the
            architecture the state belongs to.

    Returns:
        The path written.
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "epoch": epoch,
        "model": model.state_dict() if state is None else state,
        "optimiser": optimiser.state_dict() if optimiser is not None else None,
        "config": config.to_dict(),
        "metrics": metrics or {},
    }
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, destination)
    return destination


def load_checkpoint(
    path: str | Path, *, map_location: str | torch.device = "cpu"
) -> dict[str, Any]:
    """Read a checkpoint written by :func:`save_checkpoint`.

    Args:
        path: Checkpoint file.
        map_location: Device to map tensors onto.

    Returns:
        The payload, with ``"config"`` rebuilt into a :class:`RunConfig`.

    Raises:
        FileNotFoundError: If the checkpoint does not exist.
    """
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"no checkpoint at {source}")
    # weights_only=False: the payload carries the config mapping as well as
    # tensors, and the file is one this project wrote.
    payload = torch.load(source, map_location=map_location, weights_only=False)
    payload["config"] = RunConfig.from_dict(payload["config"])
    return payload


def load_encoder_weights(
    model: nn.Module, path: str | Path, *, map_location: str | torch.device = "cpu"
) -> dict[str, Any]:
    """Initialise a model's encoder from a pretraining checkpoint.

    Arms C and D start here. Loading is strict over the encoder's own keys: a
    partial transfer would present as "SSL did not help" rather than as an
    error, which is the most expensive kind of silent failure in this study.

    Args:
        model: A model exposing an ``encoder`` attribute.
        path: Pretraining checkpoint.
        map_location: Device to map tensors onto.

    Returns:
        The checkpoint payload, so the caller can log what it loaded.

    Raises:
        RuntimeError: If the checkpoint holds no encoder weights.
    """
    payload = load_checkpoint(path, map_location=map_location)
    prefix = "encoder."
    encoder_state = {
        key[len(prefix) :]: value
        for key, value in payload["model"].items()
        if key.startswith(prefix)
    }
    if not encoder_state:
        raise RuntimeError(
            f"checkpoint {path} contains no encoder.* weights; it was probably "
            "not written by a pretraining run"
        )
    model.encoder.load_state_dict(encoder_state, strict=True)
    return payload


def sync_tracking(source_db: str | Path, destination: str | Path) -> Path | None:
    """Copy the MLflow tracking database to durable storage.

    This is why tracking is written locally rather than pointed at Drive. The
    old MLflow file store appended one line per metric to one small file per
    metric, and Drive has no partial update, so every append re-uploaded the
    whole file. The SQLite backend is a single file instead, and copying it
    once per epoch rides along with the checkpoint write that has to happen
    anyway.

    The copy goes through SQLite's own backup API rather than
    :func:`shutil.copy`, so it is a consistent snapshot even if a write is in
    flight -- a byte copy of a live database can capture a torn page.

    Args:
        source_db: Local tracking database.
        destination: Destination path on durable storage.

    Returns:
        The destination, or ``None`` if the database does not exist yet --
        which is the normal case for a run with tracking disabled.
    """
    source = Path(source_db)
    if not source.exists():
        return None

    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")

    origin = sqlite3.connect(str(source))
    copy = sqlite3.connect(str(temporary))
    try:
        origin.backup(copy)
    finally:
        copy.close()
        origin.close()
    os.replace(temporary, target)
    return target


def fullest_tracking(root: str | Path) -> Path | None:
    """Find the most complete tracking snapshot beneath a directory.

    Every copy written by :func:`sync_tracking` is a copy of the *whole*
    database, not of one run's rows, so the largest file is the one holding the
    most runs.

    Size rather than modification time, deliberately. Within a session the
    snapshots grow monotonically and either rule agrees. Across sessions they
    do not: a reconnected Colab session starts from an empty database unless
    :func:`restore_tracking` runs first, so Drive accumulates several
    complete-but-disjoint databases, and the newest is merely the last session's
    -- not the fullest. Modification time is the weaker signal anyway, because
    Drive's FUSE layer does not promise it tracks write order and a re-upload
    can leave an old snapshot looking like the newest file on the mount.

    Only ``stat`` is called: nothing here opens a database on the mount, which
    is what corrupts one.

    Args:
        root: Directory searched recursively.

    Returns:
        The largest ``mlflow.db`` beneath ``root``, most recent breaking a tie,
        or ``None`` if there is none.
    """
    found = sorted(
        Path(root).rglob(TRACKING_DB),
        key=lambda path: (path.stat().st_size, path.stat().st_mtime),
    )
    return found[-1] if found else None


def restore_tracking(source: str | Path, destination: str | Path) -> Path:
    """Copy a synced tracking database back for a resumed session.

    Lets a reconnected Colab session append to the same MLflow runs instead of
    starting empty ones alongside them.

    Args:
        source: Database on durable storage.
        destination: Local path to restore to.

    Returns:
        The restored path.

    Raises:
        FileNotFoundError: If the source does not exist.
    """
    origin = Path(source)
    if not origin.exists():
        raise FileNotFoundError(f"no tracking database at {origin}")
    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(origin, target)
    return target


def free_space_mb(path: str | Path) -> float:
    """Free space at a path, in megabytes.

    Colab's Drive quota is a common and confusing failure: the mount stops
    accepting writes and the error surfaces far from the cause.

    Args:
        path: Any existing path on the filesystem of interest.

    Returns:
        Free megabytes.
    """
    return shutil.disk_usage(Path(path)).free / (1024 * 1024)
