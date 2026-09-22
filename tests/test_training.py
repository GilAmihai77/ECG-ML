"""Tests for run configuration, checkpoints and tracking.

Integrity rules 4, 5 and 6 are the subject: a run must be reproducible from a
config file, configs travel with checkpoints, and seeds are recorded. The
tracking tests check the property that matters operationally -- that MLflow can
fail in any way at all without taking the training run with it.
"""

from __future__ import annotations

import math
import os
import sqlite3
import time
import warnings
from pathlib import Path

import pytest
import torch

from ecg.models.config import ModelConfig, SslConfig
from ecg.models.encoder import build_classifier
from ecg.models.ssl import build_pretrainer
from ecg.training.checkpoints import (
    fullest_tracking,
    load_checkpoint,
    load_encoder_weights,
    restore_tracking,
    save_checkpoint,
    sync_tracking,
)
from ecg.training.config import RunConfig, TrainConfig
from ecg.training.loops import build_optimiser, resolve_device, set_seed
from ecg.training.tracking import Tracker


@pytest.fixture()
def config() -> RunConfig:
    return RunConfig(
        name="unit",
        model=ModelConfig(n_samples=200, d_model=32, n_layers=1, n_heads=2),
        train=TrainConfig(epochs=2, batch_size=4, amp=False, device="cpu"),
    )


class TestTrainConfig:
    def test_rejects_empty_budget(self) -> None:
        with pytest.raises(ValueError, match="epochs must be positive"):
            TrainConfig(epochs=0)
        with pytest.raises(ValueError, match="batch_size must be positive"):
            TrainConfig(batch_size=0)
        with pytest.raises(ValueError, match="lr must be positive"):
            TrainConfig(lr=0.0)

    def test_is_frozen(self) -> None:
        with pytest.raises(Exception):
            TrainConfig().epochs = 5  # type: ignore[misc]


class TestRunConfig:
    def test_arm_label(self) -> None:
        assert RunConfig().arm == "linear"
        assert RunConfig(model=ModelConfig(embedder="conv")).arm == "conv"
        assert RunConfig(pretrained_from="x.pt").arm == "linear+ssl"

    def test_rejects_bad_fractions(self) -> None:
        with pytest.raises(ValueError, match=r"label_fraction must be in \(0, 1\]"):
            RunConfig(label_fraction=0.0)
        with pytest.raises(ValueError, match=r"ssl_holdout must be in \[0, 1\)"):
            RunConfig(ssl_holdout=1.0)

    def test_yaml_round_trip_is_exact(self, tmp_path: Path, config: RunConfig) -> None:
        """Integrity rule 6. YAML has no tuple type, so the conv spec has to be
        coerced back or the reloaded config compares unequal."""
        path = config.to_yaml(tmp_path / "run.yaml")
        assert RunConfig.from_yaml(path) == config

    def test_conv_spec_survives_the_round_trip(self, tmp_path: Path) -> None:
        config = RunConfig(model=ModelConfig(embedder="conv", d_model=256, n_heads=8))
        reloaded = RunConfig.from_yaml(config.to_yaml(tmp_path / "c.yaml"))
        assert reloaded.model.conv_hidden == config.model.conv_hidden
        assert isinstance(reloaded.model.conv_hidden[0], tuple)

    def test_mlflow_params_are_flat_scalars(self, config: RunConfig) -> None:
        params = config.mlflow_params()
        assert params["arm"] == "linear"
        assert params["model.embedder"] == "linear"
        assert params["train.seed"] == 0
        assert params["ssl.mask_ratio"] == 0.5
        assert not any(isinstance(value, dict) for value in params.values())

    def test_with_replaces_fields(self, config: RunConfig) -> None:
        assert config.with_(label_fraction=0.2).label_fraction == 0.2
        assert config.label_fraction == 1.0

    def test_seed_is_recorded(self, config: RunConfig) -> None:
        """Integrity rule 4."""
        assert "train.seed" in config.mlflow_params()
        assert "subset_seed" in config.mlflow_params()


class TestCheckpoints:
    def test_round_trip_restores_weights(self, tmp_path: Path, config: RunConfig) -> None:
        model = build_classifier(config.model)
        optimiser, _ = build_optimiser(model, config.train)
        save_checkpoint(
            tmp_path / "c.pt", epoch=3, model=model, optimiser=optimiser, config=config
        )
        payload = load_checkpoint(tmp_path / "c.pt")

        assert payload["epoch"] == 3
        restored = build_classifier(config.model)
        restored.load_state_dict(payload["model"])
        for (_, a), (_, b) in zip(model.named_parameters(), restored.named_parameters()):
            torch.testing.assert_close(a, b)

    def test_config_travels_with_the_weights(self, tmp_path, config) -> None:
        """Integrity rule 5: a checkpoint is self-describing."""
        save_checkpoint(
            tmp_path / "c.pt", epoch=0, model=build_classifier(config.model),
            optimiser=None, config=config,
        )
        assert load_checkpoint(tmp_path / "c.pt")["config"] == config

    def test_no_temporary_file_is_left_behind(self, tmp_path, config) -> None:
        save_checkpoint(
            tmp_path / "c.pt", epoch=0, model=build_classifier(config.model),
            optimiser=None, config=config,
        )
        assert [p.name for p in tmp_path.iterdir()] == ["c.pt"]

    def test_overwrite_keeps_one_file(self, tmp_path, config) -> None:
        for epoch in range(3):
            save_checkpoint(
                tmp_path / "c.pt", epoch=epoch, model=build_classifier(config.model),
                optimiser=None, config=config,
            )
        assert load_checkpoint(tmp_path / "c.pt")["epoch"] == 2

    def test_missing_checkpoint_is_loud(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="no checkpoint"):
            load_checkpoint(tmp_path / "absent.pt")


class TestEncoderTransfer:
    def test_pretrained_encoder_arrives(self, tmp_path, config) -> None:
        pretrained = build_pretrainer(config.model, SslConfig())
        save_checkpoint(
            tmp_path / "p.pt", epoch=0, model=pretrained, optimiser=None, config=config
        )
        classifier = build_classifier(config.model)
        load_encoder_weights(classifier, tmp_path / "p.pt")
        for name, parameter in pretrained.encoder.named_parameters():
            torch.testing.assert_close(
                parameter, dict(classifier.encoder.named_parameters())[name]
            )

    def test_supervised_checkpoint_is_rejected_as_a_source(self, tmp_path, config) -> None:
        """A checkpoint with no encoder.* keys means the wrong file was passed."""
        bare = torch.nn.Linear(2, 2)
        save_checkpoint(
            tmp_path / "s.pt", epoch=0, model=bare, optimiser=None, config=config
        )
        with pytest.raises(RuntimeError, match="no encoder"):
            load_encoder_weights(build_classifier(config.model), tmp_path / "s.pt")

    def test_mismatched_architecture_is_loud(self, tmp_path, config) -> None:
        """Silently skipping mismatched keys would read as 'SSL did not help'."""
        pretrained = build_pretrainer(config.model.for_arm("conv"))
        save_checkpoint(
            tmp_path / "p.pt", epoch=0, model=pretrained, optimiser=None, config=config
        )
        with pytest.raises(RuntimeError):
            load_encoder_weights(
                build_classifier(config.model.for_arm("linear")), tmp_path / "p.pt"
            )


class TestTrackingSync:
    def test_sync_and_restore_round_trip(self, tmp_path: Path) -> None:
        local = _tracking_db(tmp_path / "mlruns" / "mlflow.db")
        synced = sync_tracking(local, tmp_path / "drive" / "mlflow.db")
        assert synced is not None and synced.exists()

        restored = restore_tracking(synced, tmp_path / "resumed" / "mlflow.db")
        assert _read_back(restored) == [("val_macro_auroc", 0.8)]

    def test_sync_copies_the_contents_not_just_the_file(self, tmp_path: Path) -> None:
        local = _tracking_db(tmp_path / "mlflow.db")
        synced = sync_tracking(local, tmp_path / "drive" / "mlflow.db")
        assert _read_back(synced) == _read_back(local)

    def test_sync_is_a_single_file(self, tmp_path: Path) -> None:
        """The whole point: Drive is charged per file, not per byte."""
        local = _tracking_db(tmp_path / "mlflow.db")
        drive = tmp_path / "drive"
        sync_tracking(local, drive / "mlflow.db")
        assert [p.name for p in drive.iterdir()] == ["mlflow.db"]

    def test_no_temporary_is_left_behind(self, tmp_path: Path) -> None:
        local = _tracking_db(tmp_path / "mlflow.db")
        drive = tmp_path / "drive"
        for _ in range(3):
            sync_tracking(local, drive / "mlflow.db")
        assert [p.name for p in drive.iterdir()] == ["mlflow.db"]

    def test_missing_source_is_not_an_error(self, tmp_path: Path) -> None:
        """A run with tracking disabled still checkpoints normally."""
        assert sync_tracking(tmp_path / "absent.db", tmp_path / "out.db") is None

    def test_missing_source_is_loud_on_restore(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="no tracking database"):
            restore_tracking(tmp_path / "absent.db", tmp_path / "out.db")


class TestFullestTracking:
    """Which snapshot a reconnected session should resume from."""

    def _snapshot(self, path: Path, rows: int) -> Path:
        """Write a tracking database holding ``rows`` metric rows."""
        local = _tracking_db(path.parent / "source.db", rows=rows)
        synced = sync_tracking(local, path)
        assert synced is not None
        local.unlink()
        return synced

    def test_no_snapshot_is_not_an_error(self, tmp_path: Path) -> None:
        """The first session of all has nothing to resume, and says so."""
        assert fullest_tracking(tmp_path) is None

    def test_it_finds_snapshots_nested_under_run_directories(
        self, tmp_path: Path
    ) -> None:
        only = self._snapshot(tmp_path / "study" / "sup-linear-f020" / "mlflow.db", 4)
        assert fullest_tracking(tmp_path) == only

    def test_the_largest_wins_over_the_most_recent(self, tmp_path: Path) -> None:
        """The failure this exists to prevent.

        A session's snapshots grow monotonically, so within one session either
        rule agrees. Across sessions they do not: a small database written later
        is a *different* session, not a fuller one, and taking it silently hides
        every epoch the long session logged.
        """
        full = self._snapshot(tmp_path / "long-session" / "mlflow.db", 600)
        recent = self._snapshot(tmp_path / "short-session" / "mlflow.db", 50)
        os.utime(recent, (time.time() + 60, time.time() + 60))

        assert recent.stat().st_mtime > full.stat().st_mtime
        assert fullest_tracking(tmp_path) == full

    def test_it_never_opens_a_database(self, tmp_path: Path, monkeypatch) -> None:
        """Opening one on a Drive mount is what corrupts it."""
        self._snapshot(tmp_path / "study" / "mlflow.db", 4)
        monkeypatch.setattr(
            sqlite3, "connect", lambda *a, **k: pytest.fail("opened a database")
        )
        assert fullest_tracking(tmp_path) is not None


class TestTracker:
    def test_disabled_tracker_is_inert(self) -> None:
        with Tracker(enabled=False) as tracker:
            tracker.log_params({"a": 1})
            tracker.log_metrics({"loss": 1.0}, step=0)
        assert not tracker.active
        assert tracker.failures == 0

    def test_failures_are_swallowed_not_raised(self) -> None:
        """A broken tracker must cost a warning, never two hours of A100 time."""
        tracker = Tracker(enabled=False)
        tracker._active = True
        tracker._mlflow = _ExplodingMlflow()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            tracker.log_metrics({"loss": 1.0}, step=0)
            tracker.log_params({"a": 1})
            tracker.log_artifact("nowhere.txt")
        assert tracker.failures == 3
        assert caught

    def test_warnings_stop_after_the_limit(self) -> None:
        tracker = Tracker(enabled=False)
        tracker._active = True
        tracker._mlflow = _ExplodingMlflow()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            for _ in range(20):
                tracker.log_metrics({"loss": 1.0}, step=0)
        assert tracker.failures == 20
        assert len(caught) == 3

    def test_exit_reports_suppressed_failures(self) -> None:
        """A silently broken tracker would otherwise look like a clean run."""
        tracker = Tracker(enabled=False)
        tracker.failures = 5
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            tracker.__exit__(None, None, None)
        assert any("incomplete" in str(w.message) for w in caught)

    def test_real_run_writes_a_database(self, tmp_path: Path) -> None:
        """A plain path must work: it is what callers naturally pass."""
        database = tmp_path / "mlruns" / "mlflow.db"
        with Tracker(tracking_uri=str(database), run_name="t") as tracker:
            tracker.log_params({"arm": "linear"})
            tracker.log_metrics({"val_macro_auroc": 0.8}, step=0)
        assert tracker.failures == 0
        assert database.exists()

    def test_tracking_is_one_file(self, tmp_path: Path) -> None:
        """What makes the per-epoch Drive copy cheap."""
        database = tmp_path / "mlruns" / "mlflow.db"
        with Tracker(tracking_uri=str(database), run_name="t") as tracker:
            tracker.log_metrics({"loss": 1.0}, step=0)
        assert [p.name for p in database.parent.iterdir()] == ["mlflow.db"]

    def test_db_path_becomes_a_sqlite_uri(self, tmp_path: Path) -> None:
        """MLflow 3.x raises on the file store unless explicitly opted back in."""
        uri = Tracker(tracking_uri=str(tmp_path / "mlflow.db")).tracking_uri
        assert uri.startswith("sqlite:///")

    def test_directory_becomes_a_file_uri(self, tmp_path: Path) -> None:
        """MLflow rejects a bare absolute path; C:\\... parses as a scheme."""
        assert Tracker(tracking_uri=str(tmp_path)).tracking_uri.startswith("file:")

    def test_qualified_uri_is_left_alone(self) -> None:
        assert Tracker(tracking_uri="https://example.test").tracking_uri == (
            "https://example.test"
        )


class TestLoopHelpers:
    def test_resolve_device(self) -> None:
        assert resolve_device("cpu").type == "cpu"
        assert resolve_device("auto").type in {"cpu", "cuda"}

    def test_set_seed_is_reproducible(self) -> None:
        set_seed(3)
        a = torch.randn(4)
        set_seed(3)
        torch.testing.assert_close(a, torch.randn(4))

    def test_norms_and_biases_are_not_decayed(self, config: RunConfig) -> None:
        """Decaying a LayerNorm gain fights the normalisation it provides."""
        model = build_classifier(config.model)
        optimiser, _ = build_optimiser(model, config.train)
        decayed, undecayed = optimiser.param_groups
        assert decayed["weight_decay"] == config.train.weight_decay
        assert undecayed["weight_decay"] == 0.0
        assert all(p.ndim > 1 for p in decayed["params"])

    def test_positional_embedding_is_not_decayed(self, config: RunConfig) -> None:
        model = build_classifier(config.model)
        _, undecayed = build_optimiser(model, config.train)[0].param_groups
        assert any(p is model.encoder.positions for p in undecayed["params"])

    def test_cosine_schedule_decays_to_the_floor(self) -> None:
        train = TrainConfig(epochs=10, min_lr_ratio=0.01)
        model = build_classifier(ModelConfig(n_samples=200, d_model=32, n_heads=2))
        optimiser, scheduler = build_optimiser(model, train)
        seen = []
        for _ in range(train.epochs):
            seen.append(optimiser.param_groups[0]["lr"])
            optimiser.step()
            scheduler.step()
        assert seen[0] == pytest.approx(train.lr)
        assert seen[-1] < seen[0]
        assert seen[-1] >= train.lr * train.min_lr_ratio * 0.99
        assert seen == sorted(seen, reverse=True)


class TestScheduleSpan:
    """``epochs`` used to be both the ceiling and the decay's time axis, so
    raising it stretched the schedule instead of extending the run. These
    check that separating them works and that leaving them joined is unchanged.
    """

    @staticmethod
    def _rates(train: TrainConfig) -> list[float]:
        model = build_classifier(ModelConfig(n_samples=200, d_model=32, n_heads=2))
        optimiser, scheduler = build_optimiser(model, train)
        seen = []
        for _ in range(train.epochs):
            seen.append(optimiser.param_groups[0]["lr"])
            optimiser.step()
            scheduler.step()
        return seen

    def test_the_default_is_the_old_schedule_exactly(self) -> None:
        """Nothing already trained is invalidated by this field existing."""
        joined = self._rates(TrainConfig(epochs=20))
        explicit = self._rates(TrainConfig(epochs=20, schedule_epochs=20))
        assert joined == pytest.approx(explicit)
        # And the closed form the field replaced, recomputed here rather than
        # imported, so a change to either side shows up as a failure.
        train = TrainConfig(epochs=20)
        expected = [
            train.lr
            * (
                train.min_lr_ratio
                + (1 - train.min_lr_ratio)
                * 0.5
                * (1 + math.cos(math.pi * epoch / (train.epochs - 1)))
            )
            for epoch in range(train.epochs)
        ]
        assert joined == pytest.approx(expected)

    def test_a_short_span_decays_on_its_own_timetable(self) -> None:
        """The point: the first 10 epochs of a 10-epoch decay are the same
        whether the ceiling is 10 or 40."""
        short = self._rates(TrainConfig(epochs=10, schedule_epochs=10))
        capped = self._rates(TrainConfig(epochs=40, schedule_epochs=10))
        assert capped[:10] == pytest.approx(short)

    def test_past_the_span_it_holds_at_the_floor(self) -> None:
        """Not riding the cosine back up toward the peak."""
        rates = self._rates(TrainConfig(epochs=40, schedule_epochs=10))
        floor = TrainConfig().lr * TrainConfig().min_lr_ratio
        assert all(rate == pytest.approx(floor) for rate in rates[10:])

    def test_a_long_ceiling_no_longer_stretches_the_decay(self) -> None:
        """Without the field, epoch 20 of a 200-epoch budget sits near peak.
        With it, the decay is finished and the run is free to keep going."""
        stretched = self._rates(TrainConfig(epochs=200))
        decoupled = self._rates(TrainConfig(epochs=200, schedule_epochs=50))
        peak = TrainConfig().lr
        # Barely moved: 98% of peak, which is the whole complaint.
        assert stretched[20] > 0.95 * peak
        # Well into its decay at the same epoch -- 64% of peak, cos(pi*20/49).
        assert decoupled[20] < 0.7 * peak
        assert decoupled[20] < 0.7 * stretched[20]
        # And by epoch 60 the decay is done while the run is still free to go.
        assert decoupled[60] == pytest.approx(peak * TrainConfig().min_lr_ratio)

    def test_a_span_past_the_ceiling_is_rejected(self) -> None:
        """It would put the end of the decay past the end of the run, which is
        the problem this field exists to remove."""
        with pytest.raises(ValueError, match="must not exceed"):
            TrainConfig(epochs=10, schedule_epochs=20)

    def test_a_negative_span_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="must not be negative"):
            TrainConfig(schedule_epochs=-1)

    def test_it_round_trips_and_is_logged(self, config: RunConfig, tmp_path) -> None:
        """Integrity rules 4 and 6: it changes the run, so it is part of it."""
        changed = config.with_(
            train=TrainConfig(epochs=40, schedule_epochs=10)
        )
        restored = RunConfig.from_yaml(changed.to_yaml(tmp_path / "config.yaml"))
        assert restored.train.schedule_epochs == 10
        assert changed.mlflow_params()["train.schedule_epochs"] == 10

    def test_an_old_config_without_the_field_still_loads(
        self, config: RunConfig
    ) -> None:
        payload = config.to_dict()
        del payload["train"]["schedule_epochs"]
        assert RunConfig.from_dict(payload).train.schedule_epochs == 0


class TestPretrainingBudget:
    """Pretraining and fine-tuning want budgets an order of magnitude apart.
    One number used to set both, so a long SSL budget also bought long
    fine-tunes -- and those are 60 of the study's 70 runs."""

    def test_unset_leaves_the_budget_alone(self) -> None:
        train = TrainConfig(epochs=50)
        assert train.for_pretraining() is train

    def test_it_substitutes_only_the_two_epoch_counts(self) -> None:
        train = TrainConfig(epochs=50, ssl_epochs=600, lr=1e-3, patience=7)
        ssl = train.for_pretraining()
        assert (ssl.epochs, train.epochs) == (600, 50)
        # Everything that must stay identical across arms is untouched.
        assert (ssl.lr, ssl.patience, ssl.seed, ssl.batch_size) == (
            train.lr, train.patience, train.seed, train.batch_size
        )

    def test_the_ssl_schedule_defaults_to_the_ssl_budget(self) -> None:
        """Not to --epochs, which would stretch or truncate the decay."""
        ssl = TrainConfig(epochs=50, ssl_epochs=600).for_pretraining()
        assert ssl.schedule_epochs == 0  # 0 means "span epochs", now 600
        model = build_classifier(ModelConfig(n_samples=200, d_model=32, n_heads=2))
        _, scheduler = build_optimiser(model, ssl)
        assert scheduler.lr_lambdas[0](599) == pytest.approx(ssl.min_lr_ratio)
        assert scheduler.lr_lambdas[0](300) > ssl.min_lr_ratio

    def test_an_explicit_ssl_schedule_is_carried_over(self) -> None:
        ssl = TrainConfig(
            epochs=50, ssl_epochs=600, ssl_schedule_epochs=400
        ).for_pretraining()
        assert (ssl.epochs, ssl.schedule_epochs) == (600, 400)

    def test_a_schedule_past_the_ssl_budget_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="pretraining budget"):
            TrainConfig(epochs=50, ssl_epochs=100, ssl_schedule_epochs=200)

    def test_negative_budgets_are_rejected(self) -> None:
        with pytest.raises(ValueError, match="ssl_epochs must not be negative"):
            TrainConfig(ssl_epochs=-1)
        with pytest.raises(ValueError, match="checkpoint_every must be positive"):
            TrainConfig(checkpoint_every=0)
        with pytest.raises(ValueError, match="snapshot_every must not be negative"):
            TrainConfig(snapshot_every=-1)

    def test_the_new_fields_round_trip_and_are_logged(self, config, tmp_path) -> None:
        changed = config.with_(
            train=TrainConfig(
                epochs=50, ssl_epochs=600, checkpoint_every=10, snapshot_every=100
            )
        )
        restored = RunConfig.from_yaml(changed.to_yaml(tmp_path / "config.yaml"))
        assert restored.train.ssl_epochs == 600
        assert restored.train.checkpoint_every == 10
        params = changed.mlflow_params()
        assert params["train.ssl_epochs"] == 600
        assert params["train.snapshot_every"] == 100

    def test_an_old_config_without_them_still_loads(self, config: RunConfig) -> None:
        payload = config.to_dict()
        for field in ("ssl_epochs", "ssl_schedule_epochs", "checkpoint_every"):
            del payload["train"][field]
        restored = RunConfig.from_dict(payload).train
        assert (restored.ssl_epochs, restored.checkpoint_every) == (0, 1)


def _tracking_db(path: Path, rows: int = 1) -> Path:
    """A small SQLite file standing in for the MLflow tracking database.

    Args:
        path: Where to write it.
        rows: How many metric rows to insert. More rows means a larger file,
            which is what :func:`fullest_tracking` sorts on.

    Returns:
        The path written.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(str(path))
    try:
        connection.execute("CREATE TABLE metrics (key TEXT, value REAL)")
        connection.executemany(
            "INSERT INTO metrics VALUES ('val_macro_auroc', ?)",
            [(0.8,)] * rows,
        )
        connection.commit()
    finally:
        connection.close()
    return path


def _read_back(path: Path) -> list[tuple[str, float]]:
    """Read the rows of a database written by :func:`_tracking_db`."""
    connection = sqlite3.connect(str(path))
    try:
        return list(connection.execute("SELECT key, value FROM metrics"))
    finally:
        connection.close()


class _ExplodingMlflow:
    """Stands in for mlflow when every call fails, as a stale mount would."""

    def __getattr__(self, name: str):
        def explode(*args, **kwargs):
            raise OSError("[Errno 5] Input/output error")

        return explode
