"""Tests for the experiment plan, the runner and the results tables.

The plan tests are where integrity rule 3 is checked at study scale: every run
must derive from one base configuration, so the arms cannot differ in the
encoder, the optimiser, the budget or the mask. The runner tests cover the two
operational properties that decide whether a disconnected Colab session costs
one run or the whole study.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ecg.data.datasets import Cohort
from ecg.data.preprocess import LEADS, SCALE, WaveformStore
from ecg.data.ptbxl import SUPERCLASSES
from ecg.experiments.cli import base_config, build_parser, main
from ecg.experiments.plan import (
    RunSpec,
    ablation_plan,
    describe_plan,
    experiment_plan,
    pretrain_name,
    supervised_name,
)
from ecg.experiments.runner import (
    RESULT_FILE,
    RunOutcome,
    Workspace,
    label_efficiency_table,
    results_frame,
    run_one,
    run_plan,
    ssl_benefit,
)
from ecg.models.config import ModelConfig, SslConfig
from ecg.training.config import RunConfig, TrainConfig

N_RECORDS = 64
N_SAMPLES = 200


@pytest.fixture()
def base(tmp_path: Path) -> RunConfig:
    return RunConfig(
        model=ModelConfig(n_samples=N_SAMPLES, d_model=32, n_layers=1, n_heads=2),
        train=TrainConfig(
            epochs=2, batch_size=16, lr=3e-3, amp=False, device="cpu", seed=0
        ),
        ssl=SslConfig(mask_ratio=0.5, mask_span=2),
        ssl_holdout=0.25,
        output_dir=str(tmp_path / "runs"),
        tracking_uri=str(tmp_path / "runs" / "mlflow.db"),
    )


@pytest.fixture()
def workspace() -> Workspace:
    """A synthetic workspace with a signal the classifier can actually read."""
    rng = np.random.default_rng(0)
    waves = rng.normal(0.0, 0.3, size=(N_RECORDS, N_SAMPLES, len(LEADS)))
    labels = np.zeros((N_RECORDS, len(SUPERCLASSES)), dtype=np.float32)
    for row in range(0, N_RECORDS, 2):
        waves[row, :, 0] += 4.0
        labels[row, :] = 1.0

    store = WaveformStore(
        waveforms=np.round(waves * SCALE).astype(np.int16),
        ecg_ids=np.arange(1, N_RECORDS + 1, dtype=np.int64),
        sampling_rate=100,
    )
    ids = store.ecg_ids
    cohorts = {
        "train": Cohort("train", ids[:32], labels[:32]),
        "val": Cohort("val", ids[32:48], labels[32:48]),
        "test": Cohort("test", ids[48:], labels[48:]),
        "ssl": Cohort("ssl", ids[:32], labels[:32]),
    }
    return Workspace(store=store, cohorts=cohorts)


class TestPlan:
    def test_study_has_pretrains_then_four_arms(self, base: RunConfig) -> None:
        specs = experiment_plan(base, fractions=(0.2, 1.0))
        assert len(specs) == 2 + 2 * 2 * 2
        assert [s.kind for s in specs[:2]] == ["pretrain", "pretrain"]
        assert {s.arm for s in specs if s.kind == "supervised"} == {
            "linear", "conv", "linear+ssl", "conv+ssl",
        }

    def test_pretrains_come_before_their_dependents(self, base: RunConfig) -> None:
        """run_plan relies on order rather than resolving a graph."""
        specs = experiment_plan(base)
        seen: set[str] = set()
        for spec in specs:
            if spec.depends_on:
                assert spec.depends_on in seen, spec.name
            seen.add(spec.name)

    def test_only_the_embedder_and_pretraining_vary(self, base: RunConfig) -> None:
        """Integrity rule 3, at study scale."""
        specs = experiment_plan(base)
        for spec in specs:
            assert spec.config.train == base.train
            assert spec.config.ssl == base.ssl
            assert spec.config.model.for_arm(base.model.embedder) == base.model

    def test_every_arm_sees_the_same_records_at_a_fraction(self, base: RunConfig) -> None:
        """A per-arm subset seed would confound sample size with which patients."""
        specs = experiment_plan(base)
        seeds = {s.config.subset_seed for s in specs}
        assert len(seeds) == 1

    def test_names_are_unique(self, base: RunConfig) -> None:
        specs = experiment_plan(base)
        assert len({s.name for s in specs}) == len(specs)

    def test_fractions_are_carried_through(self, base: RunConfig) -> None:
        specs = experiment_plan(base, fractions=(0.1, 0.9))
        fractions = {s.config.label_fraction for s in specs if s.kind == "supervised"}
        assert fractions == {0.1, 0.9}

    def test_ssl_arms_depend_on_the_matching_embedder(self, base: RunConfig) -> None:
        """A conv arm loading a linear encoder would fail at load time, but
        catching it in the plan is cheaper than catching it on an A100."""
        for spec in experiment_plan(base):
            if spec.depends_on:
                assert spec.config.model.embedder in spec.depends_on

    def test_naming_helpers(self) -> None:
        assert pretrain_name("conv", 0.5) == "pretrain-conv-m50"
        assert supervised_name("linear", True, 0.2) == "sup-linear-ssl-f020"
        assert supervised_name("linear", False, 1.0) == "sup-linear-scratch-f100"

    def test_describe_is_printable(self, base: RunConfig) -> None:
        text = describe_plan(experiment_plan(base))
        assert "14 runs" in text
        assert "sup-conv-ssl-f100" in text


class TestAblationPlan:
    def test_one_pretrain_and_one_finetune_per_ratio(self, base: RunConfig) -> None:
        specs = ablation_plan(base, mask_ratios=(0.3, 0.5, 0.7))
        assert len(specs) == 6
        assert {s.config.ssl.mask_ratio for s in specs} == {0.3, 0.5, 0.7}

    def test_uses_one_embedder_only(self, base: RunConfig) -> None:
        """Tuning the mask per arm would fold SSL into the comparison."""
        specs = ablation_plan(base, embedder="linear")
        assert {s.config.model.embedder for s in specs} == {"linear"}

    def test_uses_the_smallest_fraction(self, base: RunConfig) -> None:
        specs = ablation_plan(base, fraction=0.2)
        fractions = {s.config.label_fraction for s in specs if s.kind == "supervised"}
        assert fractions == {0.2}

    def test_span_is_inherited_not_reset(self, base: RunConfig) -> None:
        specs = ablation_plan(base.with_(ssl=SslConfig(mask_ratio=0.5, mask_span=4)))
        assert {s.config.ssl.mask_span for s in specs} == {4}


class TestRunner:
    def test_single_supervised_run(self, base, workspace, tmp_path) -> None:
        spec = RunSpec("sup", "supervised", base.with_(name="sup", label_fraction=1.0))
        outcome = run_one(
            spec, workspace, output_root=tmp_path / "runs", track=False, progress=False
        )
        assert outcome.kind == "supervised"
        assert "val_macro_auroc" in outcome.metrics
        assert "test_macro_auroc" in outcome.metrics
        assert (tmp_path / "runs" / "sup" / RESULT_FILE).exists()
        assert (tmp_path / "runs" / "sup" / "config.yaml").exists()

    def test_single_pretrain_run(self, base, workspace, tmp_path) -> None:
        spec = RunSpec("pre", "pretrain", base.with_(name="pre"))
        outcome = run_one(
            spec, workspace, output_root=tmp_path / "runs", track=False, progress=False
        )
        assert outcome.kind == "pretrain"
        assert "holdout_loss" in outcome.metrics
        assert Path(outcome.checkpoint).exists()

    def test_plan_runs_and_transfers_the_encoder(self, base, workspace, tmp_path) -> None:
        specs = experiment_plan(base, fractions=(1.0,), embedders=("linear",))
        outcomes = run_plan(
            specs, workspace, output_root=tmp_path / "runs", track=False, progress=False
        )
        assert len(outcomes) == 3
        ssl_run = next(o for o in outcomes if o.pretrained)
        assert ssl_run.arm == "linear+ssl"

    def test_resume_skips_finished_runs(self, base, workspace, tmp_path) -> None:
        """What makes a Colab disconnect cost one run instead of the study."""
        specs = experiment_plan(base, fractions=(1.0,), embedders=("linear",))
        root = tmp_path / "runs"
        first = run_plan(specs, workspace, output_root=root, track=False, progress=False)
        stamps = {
            path: path.stat().st_mtime_ns for path in root.rglob(RESULT_FILE)
        }
        assert len(stamps) == 3
        second = run_plan(specs, workspace, output_root=root, track=False, progress=False)
        assert [o.name for o in first] == [o.name for o in second]
        assert all(
            path.stat().st_mtime_ns == stamp for path, stamp in stamps.items()
        )

    def test_no_resume_reruns(self, base, workspace, tmp_path) -> None:
        specs = [RunSpec("sup", "supervised", base.with_(name="sup"))]
        root = tmp_path / "runs"
        run_plan(specs, workspace, output_root=root, track=False, progress=False)
        marker = root / "sup" / RESULT_FILE
        before = marker.stat().st_mtime_ns
        run_plan(
            specs, workspace, output_root=root, track=False, resume=False, progress=False
        )
        assert marker.stat().st_mtime_ns != before

    def test_missing_dependency_is_loud(self, base, workspace, tmp_path) -> None:
        """Better to fail here than to silently fine-tune a random encoder."""
        spec = RunSpec(
            "sup", "supervised", base.with_(name="sup"), depends_on="never-ran"
        )
        with pytest.raises(FileNotFoundError, match="never-ran"):
            run_plan(
                [spec], workspace, output_root=tmp_path / "runs", track=False,
                progress=False,
            )

    def test_outcome_round_trips_through_json(self, base, workspace, tmp_path) -> None:
        spec = RunSpec("sup", "supervised", base.with_(name="sup"))
        outcome = run_one(
            spec, workspace, output_root=tmp_path / "runs", track=False, progress=False
        )
        assert RunOutcome.load(tmp_path / "runs" / "sup") == outcome

    def test_test_is_scored_from_the_selected_checkpoint(
        self, base, workspace, tmp_path
    ) -> None:
        """The weights in memory after training are the last epoch's, not the
        best; scoring those would quietly report the wrong model."""
        longer = base.with_(
            name="sup",
            train=TrainConfig(
                epochs=4, batch_size=16, lr=3e-3, amp=False, device="cpu", seed=0
            ),
        )
        outcome = run_one(
            RunSpec("sup", "supervised", longer),
            workspace,
            output_root=tmp_path / "runs",
            track=False,
            progress=False,
        )
        assert Path(outcome.checkpoint).name == "best.pt"
        assert outcome.best_epoch <= outcome.epochs_run


class TestResults:
    def test_frame_has_one_row_per_run(self, base, workspace, tmp_path) -> None:
        specs = experiment_plan(base, fractions=(1.0,), embedders=("linear",))
        frame = results_frame(
            run_plan(specs, workspace, output_root=tmp_path / "r", track=False, progress=False)
        )
        assert len(frame) == 3
        assert {"run", "arm", "label_fraction", "test_macro_auroc"} <= set(frame.columns)

    def test_label_efficiency_pivots_fraction_by_arm(self) -> None:
        frame = _fake_results()
        table = label_efficiency_table(frame)
        assert list(table.index) == [0.2, 1.0]
        assert set(table.columns) == {"linear", "linear+ssl"}

    def test_ssl_benefit_reports_the_gain(self) -> None:
        frame = _fake_results()
        table = ssl_benefit(frame)
        assert "ssl_gain" in table.columns
        assert table.loc[("linear", 0.2), "ssl_gain"] == pytest.approx(0.05)

    def test_pretrain_rows_are_excluded_from_the_curve(self) -> None:
        frame = _fake_results()
        frame.loc[len(frame)] = {**frame.iloc[0].to_dict(), "kind": "pretrain"}
        assert len(label_efficiency_table(frame).index) == 2


class TestCli:
    def test_dry_run_prints_the_plan_and_stops(self, tmp_path, capsys) -> None:
        """No store is loaded, so a nonexistent path must not matter."""
        code = main(
            [
                "--dry-run", "--output", str(tmp_path / "runs"),
                "--store", "does/not/exist", "--fractions", "0.2", "1.0",
            ]
        )
        assert code == 0
        # 2 pretrains + 2 fractions x 2 embedders x {scratch, ssl}
        assert "10 runs" in capsys.readouterr().out

    def test_default_grid_is_fourteen_runs(self, tmp_path, capsys) -> None:
        """2 pretrains + 3 fractions x 4 arms, the study as planned."""
        main(["--dry-run", "--output", str(tmp_path / "runs")])
        assert "14 runs" in capsys.readouterr().out

    def test_ablation_plan_is_selectable(self, tmp_path, capsys) -> None:
        main(
            [
                "--dry-run", "--plan", "ablation", "--output", str(tmp_path),
                "--mask-ratios", "0.3", "0.5",
            ]
        )
        out = capsys.readouterr().out
        assert "pretrain-linear-m30" in out
        assert "pretrain-linear-m50" in out

    def test_arguments_reach_the_config(self) -> None:
        args = build_parser().parse_args(
            ["--epochs", "7", "--d-model", "64", "--mask-ratio", "0.7", "--no-amp"]
        )
        config = base_config(args)
        assert config.train.epochs == 7
        assert config.model.d_model == 64
        assert config.ssl.mask_ratio == 0.7
        assert config.train.amp is False

    def test_tracking_defaults_beside_the_output(self, tmp_path) -> None:
        args = build_parser().parse_args(["--output", str(tmp_path / "runs")])
        assert base_config(args).tracking_uri.endswith("mlflow.db")

    def test_cloud_defaults_are_the_intended_ones(self) -> None:
        """The command run on the A100 should need no extra flags."""
        config = base_config(build_parser().parse_args([]))
        assert config.model.d_model == 256
        assert config.model.n_heads == 8
        assert config.ssl.mask_ratio == 0.5
        assert config.ssl.mask_span == 2
        assert config.train.amp is True


def _fake_results():
    """A minimal results frame for the table functions."""
    import pandas as pd

    rows = []
    for fraction in (0.2, 1.0):
        for pretrained in (False, True):
            rows.append(
                {
                    "run": f"r{fraction}{pretrained}",
                    "kind": "supervised",
                    "arm": "linear+ssl" if pretrained else "linear",
                    "embedder": "linear",
                    "pretrained": pretrained,
                    "label_fraction": fraction,
                    "test_macro_auroc": 0.80 + (0.05 if pretrained else 0.0),
                }
            )
    return pd.DataFrame(rows)
