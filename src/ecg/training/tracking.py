"""MLflow tracking that cannot kill a training run.

Every call here is wrapped. Tracking is instrumentation, not the experiment: an
MLflow failure two hours into an A100 run must cost a warning line, never the
run. On Colab the realistic failure is a stale Drive mount raising ``OSError``
from deep inside the file store, and the second-most realistic is a disk-quota
error at the same place.

The tracking URI is local by default. The MLflow tree reaches durable storage
as a tarball written alongside each checkpoint -- see
:func:`ecg.training.checkpoints.archive_mlruns` for why that is much cheaper
than pointing MLflow at Drive directly.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from types import TracebackType
from typing import Any

#: How many tracking failures to warn about before going quiet. A broken mount
#: would otherwise emit one warning per metric per epoch.
WARN_LIMIT: int = 3


def normalise_tracking_uri(uri: str | Path) -> str:
    """Turn a filesystem path into a tracking URI MLflow fully supports.

    A path ending in ``.db`` becomes a SQLite URI. That is the backend this
    project uses: as of MLflow 3.x the old file store raises unless
    ``MLFLOW_ALLOW_FILE_STORE`` is set, and it was the wrong shape for this
    setup anyway -- it writes one small file per metric, where SQLite writes a
    single file that can be snapshotted to Drive in one operation.

    Anything else is treated as a file-store directory and converted to a
    ``file:`` URI, since MLflow rejects a bare absolute path (on Windows
    ``C:\\...`` parses as a scheme, and Drive paths fail the same way).

    Args:
        uri: A path or an already-qualified URI.

    Returns:
        A URI MLflow accepts.
    """
    text = str(uri)
    if "://" in text or text.startswith("file:"):
        return text
    path = Path(text).resolve()
    if path.suffix == ".db":
        path.parent.mkdir(parents=True, exist_ok=True)
        return f"sqlite:///{path.as_posix()}"
    return path.as_uri()


class Tracker:
    """A small MLflow wrapper whose methods never raise.

    Args:
        tracking_uri: MLflow tracking URI or local directory.
        experiment: Experiment name.
        run_name: Run name.
        enabled: Set ``False`` to disable tracking entirely, which is what the
            tests and local smoke runs use.

    Attributes:
        failures: Count of suppressed tracking errors. Logged at the end of a
            run so a silently broken tracker is still visible.
    """

    def __init__(
        self,
        tracking_uri: str | Path = "mlruns",
        experiment: str = "ecg-ssl",
        run_name: str = "run",
        *,
        enabled: bool = True,
    ) -> None:
        self.tracking_uri = normalise_tracking_uri(tracking_uri)
        self.experiment = experiment
        self.run_name = run_name
        self.enabled = enabled
        self.failures = 0
        self._active = False
        self._mlflow: Any = None

    def __enter__(self) -> Tracker:
        """Start a run, degrading to a no-op tracker on any failure."""
        if not self.enabled:
            return self
        try:
            import mlflow

            self._mlflow = mlflow
            mlflow.set_tracking_uri(self.tracking_uri)
            mlflow.set_experiment(self.experiment)
            mlflow.start_run(run_name=self.run_name)
            self._active = True
        except Exception as error:  # noqa: BLE001 - tracking must not propagate
            self._warn("start run", error)
            self._mlflow = None
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """End the run, reporting how many tracking calls were suppressed."""
        if self.failures:
            warnings.warn(
                f"{self.failures} tracking call(s) failed and were suppressed; "
                "metrics for this run are incomplete",
                RuntimeWarning,
                stacklevel=2,
            )
        if not self._active or self._mlflow is None:
            return
        try:
            self._mlflow.end_run()
        except Exception as error:  # noqa: BLE001
            self._warn("end run", error)
        self._active = False

    @property
    def active(self) -> bool:
        """Whether a run is actually open."""
        return self._active

    def log_params(self, params: dict[str, Any]) -> None:
        """Log run parameters once (integrity rule 4).

        Args:
            params: Flat mapping, e.g. from
                :meth:`ecg.training.config.RunConfig.mlflow_params`.
        """
        self._call("log params", lambda: self._mlflow.log_params(params))

    def log_metrics(self, metrics: dict[str, float], step: int) -> None:
        """Log a batch of metrics at one step.

        Called once per epoch rather than once per training step. The file
        store writes one line per metric per call, so per-step logging would
        multiply the archive's size and write cost by the steps per epoch for
        curves nobody reads across 17 runs.

        Args:
            metrics: Metric name to value.
            step: Step index, normally the epoch.
        """
        clean = {key: float(value) for key, value in metrics.items()}
        self._call("log metrics", lambda: self._mlflow.log_metrics(clean, step=step))

    def log_artifact(self, path: str | Path) -> None:
        """Attach a file to the run.

        Args:
            path: File to copy into the run's artifact store.
        """
        self._call("log artifact", lambda: self._mlflow.log_artifact(str(path)))

    def set_tags(self, tags: dict[str, Any]) -> None:
        """Set run tags.

        Args:
            tags: Tag name to value.
        """
        self._call("set tags", lambda: self._mlflow.set_tags(tags))

    def _call(self, what: str, action: Any) -> None:
        """Run a tracking action, swallowing and counting any failure."""
        if not self._active or self._mlflow is None:
            return
        try:
            action()
        except Exception as error:  # noqa: BLE001 - tracking must not propagate
            self._warn(what, error)

    def _warn(self, what: str, error: Exception) -> None:
        """Count a failure and warn for the first few."""
        self.failures += 1
        if self.failures <= WARN_LIMIT:
            warnings.warn(
                f"mlflow: could not {what}: {type(error).__name__}: {error}",
                RuntimeWarning,
                stacklevel=3,
            )
