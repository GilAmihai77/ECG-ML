"""Tests for run configuration, checkpoints and tracking.

Integrity rules 4, 5 and 6 are the subject: a run must be reproducible from a
config file, configs travel with checkpoints, and seeds are recorded. The
tracking tests check the property that matters operationally -- that MLflow can
fail in any way at all without taking the training run with it.
"""

from __future__ import annotations

import sqlite3
import warnings
from pathlib import Path

import pytest
import torch

from ecg.models.config import ModelConfig, SslConfig
from ecg.models.encoder import build_classifier
from ecg.models.ssl import build_pretrainer
from ecg.training.checkpoints import (
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


def _tracking_db(path: Path) -> Path:
    """A small SQLite file standing in for the MLflow tracking database."""
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(str(path))
    try:
        connection.execute("CREATE TABLE metrics (key TEXT, value REAL)")
        connection.execute("INSERT INTO metrics VALUES ('val_macro_auroc', 0.8)")
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
