"""End-to-end tests for the two training loops.

Everything runs on a tiny synthetic store so the suite stays offline and fast.
The loops are exercised for real -- optimiser steps, checkpoints, early
stopping -- rather than mocked, because the failures worth catching here are
the ones that only appear when the pieces are wired together.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pytest
import torch

from ecg.data.datasets import Cohort, EcgBatches, holdout_split
from ecg.data.preprocess import LEADS, SCALE, WaveformStore
from ecg.data.ptbxl import SUPERCLASSES
from ecg.models.config import ModelConfig, SslConfig
from ecg.models.encoder import build_classifier
from ecg.models.ssl import build_pretrainer
from ecg.training.checkpoints import load_checkpoint
from ecg.training.config import RunConfig, TrainConfig
from ecg.training.loops import (
    evaluate_classifier,
    predict,
    pretrain,
    train_supervised,
)

N_RECORDS = 48
N_SAMPLES = 200


@pytest.fixture()
def store() -> WaveformStore:
    rng = np.random.default_rng(0)
    waves = rng.normal(0.0, 0.3, size=(N_RECORDS, N_SAMPLES, len(LEADS)))
    return WaveformStore(
        waveforms=np.round(waves * SCALE).astype(np.int16),
        ecg_ids=np.arange(1, N_RECORDS + 1, dtype=np.int64),
        sampling_rate=100,
    )


@pytest.fixture()
def learnable(store: WaveformStore) -> tuple[WaveformStore, Cohort]:
    """A cohort whose label is readable from the signal.

    Records with an even index get a large offset on lead 0 and a positive
    label, so a working loop must reach a high AUROC. Against random labels a
    broken loop and a working one are indistinguishable.
    """
    waves = store.waveforms.copy()
    labels = np.zeros((N_RECORDS, len(SUPERCLASSES)), dtype=np.float32)
    for row in range(0, N_RECORDS, 2):
        waves[row, :, 0] += 4000
        labels[row, :] = 1.0
    marked = WaveformStore(
        waveforms=waves, ecg_ids=store.ecg_ids, sampling_rate=100
    )
    return marked, Cohort("train", store.ecg_ids.copy(), labels)


@pytest.fixture()
def config(tmp_path: Path) -> RunConfig:
    return RunConfig(
        name="unit",
        model=ModelConfig(n_samples=N_SAMPLES, d_model=32, n_layers=1, n_heads=2),
        train=TrainConfig(
            epochs=3, batch_size=16, lr=3e-3, amp=False, device="cpu", seed=0
        ),
        ssl=SslConfig(mask_ratio=0.5, mask_span=2),
        output_dir=str(tmp_path / "out"),
        tracking_uri=str(tmp_path / "mlruns"),
    )


class TestSupervisedLoop:
    def test_runs_and_checkpoints(self, learnable, config) -> None:
        store, cohort = learnable
        model = build_classifier(config.model)
        result = train_supervised(
            model,
            EcgBatches(store, cohort, batch_size=16, seed=0),
            EcgBatches(store, cohort, batch_size=16, shuffle=False),
            config,
            progress=False,
        )
        assert result.epochs_run == 3
        assert result.best_checkpoint is not None
        assert result.best_checkpoint.exists()
        assert (Path(config.output_dir) / "last.pt").exists()

    def test_it_actually_learns(self, learnable, config) -> None:
        """A loop that never calls optimiser.step() would still 'run'."""
        store, cohort = learnable
        batches = EcgBatches(store, cohort, batch_size=16, shuffle=False)
        model = build_classifier(config.model)
        before = evaluate_classifier(model, batches).macro_auroc
        train_supervised(
            model,
            EcgBatches(store, cohort, batch_size=16, seed=0),
            batches,
            config.with_(train=config.train.__class__(**{**vars(config.train), "epochs": 12})),
            progress=False,
        )
        assert evaluate_classifier(model, batches).macro_auroc > max(before, 0.9)

    def test_history_carries_the_selection_metric(self, learnable, config) -> None:
        store, cohort = learnable
        result = train_supervised(
            build_classifier(config.model),
            EcgBatches(store, cohort, batch_size=16, seed=0),
            EcgBatches(store, cohort, batch_size=16, shuffle=False),
            config,
            progress=False,
        )
        for record in result.history:
            assert "val_macro_auroc" in record
            assert "train_loss" in record
            assert "lr" in record

    def test_best_checkpoint_carries_no_optimiser_state(self, learnable, config) -> None:
        """best.pt is read only for its weights, by encoder transfer and by the
        test evaluation. Optimiser state would be two thirds of a 40 MB Drive
        write per improving epoch, for nothing."""
        store, cohort = learnable
        result = train_supervised(
            build_classifier(config.model),
            EcgBatches(store, cohort, batch_size=16, seed=0),
            EcgBatches(store, cohort, batch_size=16, shuffle=False),
            config,
            progress=False,
        )
        assert load_checkpoint(result.best_checkpoint)["optimiser"] is None
        assert load_checkpoint(Path(config.output_dir) / "last.pt")["optimiser"] is not None

    def test_checkpoint_is_self_describing(self, learnable, config) -> None:
        """Integrity rule 5."""
        store, cohort = learnable
        result = train_supervised(
            build_classifier(config.model),
            EcgBatches(store, cohort, batch_size=16, seed=0),
            EcgBatches(store, cohort, batch_size=16, shuffle=False),
            config,
            progress=False,
        )
        payload = load_checkpoint(result.best_checkpoint)
        assert payload["config"] == config
        assert "val_macro_auroc" in payload["metrics"]

    def test_early_stopping_halts(self, learnable, config) -> None:
        store, cohort = learnable
        patient = config.with_(
            train=TrainConfig(
                epochs=40, batch_size=16, lr=1e-9, amp=False, device="cpu", patience=2
            )
        )
        # Stopping at epoch 3 of a 40-epoch cosine leaves the rate near peak,
        # which the loop is required to say rather than let pass as equivalent
        # to a 3-epoch run.
        with pytest.warns(RuntimeWarning, match="never reached its low-rate"):
            result = train_supervised(
                build_classifier(patient.model),
                EcgBatches(store, cohort, batch_size=16, seed=0),
                EcgBatches(store, cohort, batch_size=16, shuffle=False),
                patient,
                progress=False,
            )
        assert result.epochs_run < 40

    def test_eval_every_reduces_evaluations(self, learnable, config) -> None:
        store, cohort = learnable
        sparse = config.with_(
            train=TrainConfig(
                epochs=6, batch_size=16, amp=False, device="cpu", eval_every=3
            )
        )
        result = train_supervised(
            build_classifier(sparse.model),
            EcgBatches(store, cohort, batch_size=16, seed=0),
            EcgBatches(store, cohort, batch_size=16, shuffle=False),
            sparse,
            progress=False,
        )
        assert result.epochs_run == 2

    def test_same_seed_reproduces_the_run(self, learnable, config) -> None:
        """Integrity rules 4 and 6."""
        store, cohort = learnable

        def run() -> list[float]:
            torch.manual_seed(0)
            return [
                record["val_macro_auroc"]
                for record in train_supervised(
                    build_classifier(config.model),
                    EcgBatches(store, cohort, batch_size=16, seed=0),
                    EcgBatches(store, cohort, batch_size=16, shuffle=False),
                    config,
                    progress=False,
                ).history
            ]

        assert run() == pytest.approx(run())

    def test_test_split_is_never_read(self, learnable, config) -> None:
        """Integrity rule 2, enforced by handing the loop no test batches at all.

        The loop's signature takes train and val only, so a test leak would
        require a change here rather than a slip in a notebook.
        """
        import inspect

        assert "test" not in inspect.signature(train_supervised).parameters


class TestPredictAndEvaluate:
    def test_predictions_line_up_with_the_cohort(self, learnable, config) -> None:
        store, cohort = learnable
        batches = EcgBatches(store, cohort, batch_size=16, shuffle=False)
        y_true, y_score = predict(build_classifier(config.model), batches)
        assert y_true.shape == (N_RECORDS, len(SUPERCLASSES))
        assert y_score.shape == y_true.shape
        np.testing.assert_allclose(y_true, cohort.labels)

    def test_scores_are_probabilities(self, learnable, config) -> None:
        store, cohort = learnable
        _, y_score = predict(
            build_classifier(config.model),
            EcgBatches(store, cohort, batch_size=16, shuffle=False),
        )
        assert float(y_score.min()) >= 0.0
        assert float(y_score.max()) <= 1.0

    def test_evaluation_leaves_the_model_in_training_mode(self, learnable, config) -> None:
        """A stray eval() would silently disable dropout for the rest of the run."""
        store, cohort = learnable
        model = build_classifier(config.model).train()
        evaluate_classifier(model, EcgBatches(store, cohort, batch_size=16, shuffle=False))
        assert model.training

    def test_supplied_thresholds_are_used_verbatim(self, learnable, config) -> None:
        store, cohort = learnable
        fixed = np.full(len(SUPERCLASSES), 0.42)
        metrics = evaluate_classifier(
            build_classifier(config.model),
            EcgBatches(store, cohort, batch_size=16, shuffle=False),
            fixed,
        )
        assert set(metrics.thresholds.values()) == {0.42}


class TestPretrainLoop:
    def test_runs_and_checkpoints(self, store, config) -> None:
        cohort = Cohort(
            "ssl", store.ecg_ids.copy(),
            np.zeros((N_RECORDS, len(SUPERCLASSES)), dtype=np.float32),
        )
        kept, held = holdout_split(cohort, 0.25, seed=0)
        result = pretrain(
            build_pretrainer(config.model, config.ssl),
            EcgBatches(store, kept, batch_size=16, seed=0),
            EcgBatches(store, held, batch_size=16, shuffle=False),
            config,
            progress=False,
        )
        assert result.epochs_run == 3
        assert result.best_checkpoint is not None
        assert (Path(config.output_dir) / "pretrain_last.pt").exists()
        for record in result.history:
            assert "ssl_loss" in record
            assert "holdout_loss" in record
            assert "contaminated_fraction" in record

    def test_loss_falls(self, store, config) -> None:
        cohort = Cohort(
            "ssl", store.ecg_ids.copy(),
            np.zeros((N_RECORDS, len(SUPERCLASSES)), dtype=np.float32),
        )
        longer = config.with_(
            train=TrainConfig(epochs=15, batch_size=16, lr=3e-3, amp=False, device="cpu")
        )
        result = pretrain(
            build_pretrainer(longer.model, longer.ssl),
            EcgBatches(store, cohort, batch_size=16, seed=0),
            None,
            longer,
            progress=False,
        )
        assert result.history[-1]["ssl_loss"] < result.history[0]["ssl_loss"]

    def test_holdout_loss_is_measured_under_a_fixed_mask(self, store, config) -> None:
        """A fresh random mask each epoch would add noise that reads as progress.

        Two evaluations of an unchanged model must agree exactly, which they
        only can if the mask generator is re-seeded per call.
        """
        from ecg.training.loops import _reconstruction_loss

        cohort = Cohort(
            "ssl", store.ecg_ids.copy(),
            np.zeros((N_RECORDS, len(SUPERCLASSES)), dtype=np.float32),
        )
        model = build_pretrainer(config.model, config.ssl)
        batches = EcgBatches(store, cohort, batch_size=16, shuffle=False)
        first = _reconstruction_loss(model, batches, config)
        second = _reconstruction_loss(model, batches, config)
        assert first == pytest.approx(second, rel=1e-9)

    def test_early_stopping_halts(self, store, config) -> None:
        """So a large epoch budget is a ceiling rather than a commitment."""
        cohort = Cohort(
            "ssl", store.ecg_ids.copy(),
            np.zeros((N_RECORDS, len(SUPERCLASSES)), dtype=np.float32),
        )
        kept, held = holdout_split(cohort, 0.25, seed=0)
        # lr small enough that the holdout loss cannot improve, so patience
        # fires rather than the budget running out.
        patient = config.with_(
            train=TrainConfig(
                epochs=40, batch_size=16, lr=1e-9, amp=False, device="cpu", patience=2
            )
        )
        with pytest.warns(RuntimeWarning, match="never reached its low-rate"):
            result = pretrain(
                build_pretrainer(patient.model, patient.ssl),
                EcgBatches(store, kept, batch_size=16, seed=0),
                EcgBatches(store, held, batch_size=16, shuffle=False),
                patient,
                progress=False,
            )
        assert result.epochs_run < 40

    def test_the_best_encoder_survives_an_early_stop(self, store, config) -> None:
        """The SSL arms fine-tune from this checkpoint, so stopping early must
        not hand them the last epoch's weights instead of the best."""
        cohort = Cohort(
            "ssl", store.ecg_ids.copy(),
            np.zeros((N_RECORDS, len(SUPERCLASSES)), dtype=np.float32),
        )
        kept, held = holdout_split(cohort, 0.25, seed=0)
        patient = config.with_(
            train=TrainConfig(
                epochs=40, batch_size=16, lr=1e-9, amp=False, device="cpu", patience=2
            )
        )
        with pytest.warns(RuntimeWarning):
            result = pretrain(
                build_pretrainer(patient.model, patient.ssl),
                EcgBatches(store, kept, batch_size=16, seed=0),
                EcgBatches(store, held, batch_size=16, shuffle=False),
                patient,
                progress=False,
            )
        assert result.best_checkpoint is not None
        assert Path(result.best_checkpoint).exists()
        assert result.best_epoch <= result.epochs_run
        # The durable writes happen before the break, so they describe the run
        # that actually ran rather than trailing it by `patience` epochs.
        assert (Path(config.output_dir) / "pretrain_last.pt").exists()

    def test_patience_without_a_holdout_says_it_is_inert(self, store, config) -> None:
        """Training reconstruction loss falls almost monotonically, so patience
        on it would never fire -- silently, unless it says so."""
        cohort = Cohort(
            "ssl", store.ecg_ids.copy(),
            np.zeros((N_RECORDS, len(SUPERCLASSES)), dtype=np.float32),
        )
        patient = config.with_(
            train=TrainConfig(
                epochs=3, batch_size=16, lr=1e-9, amp=False, device="cpu", patience=1
            )
        )
        with pytest.warns(RuntimeWarning, match="no holdout"):
            result = pretrain(
                build_pretrainer(patient.model, patient.ssl),
                EcgBatches(store, cohort, batch_size=16, seed=0),
                None,
                patient,
                progress=False,
            )
        assert result.epochs_run == 3

    def test_a_completed_budget_does_not_warn(self, store, config) -> None:
        """The warning is about stopping early, not about patience existing."""
        cohort = Cohort(
            "ssl", store.ecg_ids.copy(),
            np.zeros((N_RECORDS, len(SUPERCLASSES)), dtype=np.float32),
        )
        kept, held = holdout_split(cohort, 0.25, seed=0)
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            pretrain(
                build_pretrainer(config.model, config.ssl),
                EcgBatches(store, kept, batch_size=16, seed=0),
                EcgBatches(store, held, batch_size=16, shuffle=False),
                config.with_(
                    train=TrainConfig(
                        epochs=3, batch_size=16, lr=3e-3, amp=False,
                        device="cpu", patience=0,
                    )
                ),
                progress=False,
            )

    def test_checkpoint_can_initialise_a_classifier(self, store, config) -> None:
        """The whole point of arms C and D."""
        from ecg.training.checkpoints import load_encoder_weights

        cohort = Cohort(
            "ssl", store.ecg_ids.copy(),
            np.zeros((N_RECORDS, len(SUPERCLASSES)), dtype=np.float32),
        )
        result = pretrain(
            build_pretrainer(config.model, config.ssl),
            EcgBatches(store, cohort, batch_size=16, seed=0),
            None,
            config,
            progress=False,
        )
        classifier = build_classifier(config.model)
        payload = load_encoder_weights(classifier, result.best_checkpoint)
        assert payload["config"] == config


class TestHoldoutSplit:
    def test_parts_are_disjoint_and_complete(self) -> None:
        cohort = Cohort(
            "ssl", np.arange(1, 101), np.zeros((100, 5), dtype=np.float32)
        )
        kept, held = holdout_split(cohort, 0.1, seed=0)
        assert len(kept) == 90
        assert len(held) == 10
        assert not set(kept.ecg_ids.tolist()) & set(held.ecg_ids.tolist())

    def test_zero_fraction_holds_nothing_out(self) -> None:
        cohort = Cohort("ssl", np.arange(1, 11), np.zeros((10, 5), dtype=np.float32))
        kept, held = holdout_split(cohort, 0.0)
        assert len(kept) == 10
        assert len(held) == 0

    def test_is_reproducible(self) -> None:
        cohort = Cohort("ssl", np.arange(1, 101), np.zeros((100, 5), dtype=np.float32))
        a = holdout_split(cohort, 0.2, seed=7)[1]
        b = holdout_split(cohort, 0.2, seed=7)[1]
        np.testing.assert_array_equal(a.ecg_ids, b.ecg_ids)

    def test_invalid_fraction_is_rejected(self) -> None:
        cohort = Cohort("ssl", np.arange(1, 11), np.zeros((10, 5), dtype=np.float32))
        with pytest.raises(ValueError, match=r"\[0, 1\)"):
            holdout_split(cohort, 1.0)
