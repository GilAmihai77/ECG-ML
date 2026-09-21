"""Tests for the experiment plan, the runner and the results tables.

The plan tests are where integrity rule 3 is checked at study scale: every run
must derive from one base configuration, so the arms cannot differ in the
encoder, the optimiser, the budget or the mask. The runner tests cover the two
operational properties that decide whether a disconnected Colab session costs
one run or the whole study.
"""

from __future__ import annotations

import warnings
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

import ecg.experiments.runner as runner
import ecg.training.tracking as tracking
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
    seed_name,
    ssl_budget_plan,
    supervised_name,
    variant_name,
)
from ecg.experiments.runner import (
    RESULT_FILE,
    RunOutcome,
    Workspace,
    aggregate_runs,
    embedder_benefit,
    format_mean_sd,
    label_efficiency_report,
    label_efficiency_table,
    load_results,
    results_frame,
    run_one,
    run_plan,
    ssl_benefit,
)
from ecg.training.loops import resolve_device, snapshot_name
from ecg.models.config import ModelConfig, SslConfig
from ecg.training.config import DEFAULT_VARIANT, RunConfig, TrainConfig
from ecg.training.tracking import UNKNOWN, git_provenance

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
        specs = experiment_plan(base, fractions=(0.2, 1.0), seeds=(0,))
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
        """Integrity rule 3, at study scale. The seed varies too, by design, so
        it is normalised away before the budgets are compared."""
        specs = experiment_plan(base)
        for spec in specs:
            assert replace(spec.config.train, seed=base.train.seed) == base.train
            assert spec.config.ssl == base.ssl
            assert spec.config.model.for_arm(base.model.embedder) == base.model

    def test_every_arm_sees_the_same_records_at_a_fraction(self, base: RunConfig) -> None:
        """A per-arm subset seed would confound sample size with which patients.

        Per-*seed* is the opposite: it is what puts "which records you drew"
        inside the reported interval instead of outside it.
        """
        for spec in experiment_plan(base):
            assert spec.config.subset_seed == spec.config.train.seed

        by_seed: dict[int, set[int]] = {}
        for spec in experiment_plan(base):
            by_seed.setdefault(spec.config.train.seed, set()).add(
                spec.config.subset_seed
            )
        assert all(len(subsets) == 1 for subsets in by_seed.values())
        assert len(by_seed) == 5

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
        text = describe_plan(experiment_plan(base, seeds=(0,)))
        assert "14 runs" in text
        assert "sup-conv-ssl-f100" in text


class TestSeeds:
    """Five replicates per experiment, and what has to hold for the mean and
    deviation over them to mean anything."""

    def test_default_study_is_five_seeds(self, base: RunConfig) -> None:
        specs = experiment_plan(base)
        assert {s.config.train.seed for s in specs} == {0, 1, 2, 3, 4}
        # 2 embedders x 5 pretrains, then 3 fractions x 4 arms x 5 seeds.
        assert len(specs) == 10 + 60

    def test_each_arm_has_one_run_per_seed(self, base: RunConfig) -> None:
        counts: dict[tuple, list[int]] = {}
        for spec in experiment_plan(base):
            key = (spec.kind, spec.arm, spec.config.label_fraction)
            counts.setdefault(key, []).append(spec.config.train.seed)
        assert all(sorted(seeds) == [0, 1, 2, 3, 4] for seeds in counts.values())

    def test_seeds_get_their_own_directories(self, base: RunConfig) -> None:
        """Sharing one would make run_plan skip four replicates as complete and
        report the first five times, with nothing on screen saying so."""
        names = [s.name for s in experiment_plan(base)]
        assert len(set(names)) == len(names)
        assert "sup-conv-ssl-f100-s3" in names

    def test_a_single_seed_keeps_the_original_names(self, base: RunConfig) -> None:
        """So a study already on Drive still resumes rather than re-running."""
        assert seed_name("sup-linear-ssl-f020", 0, several=False) == (
            "sup-linear-ssl-f020"
        )
        assert "sup-conv-ssl-f100" in {s.name for s in experiment_plan(base, seeds=(0,))}

    def test_seed_suffix_sits_inside_the_variant_prefix(self, base: RunConfig) -> None:
        specs = experiment_plan(base.with_(variant="deep6"))
        assert "deep6-sup-linear-scratch-f100-s2" in {s.name for s in specs}

    def test_pretraining_is_replicated_per_seed(self, base: RunConfig) -> None:
        """Otherwise the SSL arm's interval omits pretraining variance and comes
        out narrower than the scratch arm's for a reason unrelated to SSL."""
        pretrains = [s for s in experiment_plan(base) if s.kind == "pretrain"]
        assert len(pretrains) == 10
        assert {s.config.train.seed for s in pretrains} == {0, 1, 2, 3, 4}

    def test_an_ssl_arm_fine_tunes_its_own_seeds_encoder(self, base: RunConfig) -> None:
        by_name = {s.name: s for s in experiment_plan(base)}
        for spec in by_name.values():
            if spec.depends_on:
                assert by_name[spec.depends_on].config.train.seed == (
                    spec.config.train.seed
                )

    def test_share_pretraining_collapses_the_encoders(self, base: RunConfig) -> None:
        specs = experiment_plan(base, share_pretraining=True)
        pretrains = [s for s in specs if s.kind == "pretrain"]
        assert len(pretrains) == 2
        # Unsuffixed, because with sharing there is only one per embedder.
        assert {s.name for s in pretrains} == {
            "pretrain-linear-m50", "pretrain-conv-m50",
        }
        assert {s.depends_on for s in specs if s.depends_on} == {
            "pretrain-linear-m50", "pretrain-conv-m50",
        }

    def test_share_pretraining_still_replicates_the_fine_tunes(
        self, base: RunConfig
    ) -> None:
        specs = experiment_plan(base, share_pretraining=True)
        supervised = [s for s in specs if s.kind == "supervised"]
        assert len(supervised) == 60
        assert {s.config.train.seed for s in supervised} == {0, 1, 2, 3, 4}

    def test_a_repeated_seed_is_an_error(self, base: RunConfig) -> None:
        """Two runs with one name: the second is skipped as already complete,
        and the study reports n=5 over four distinct runs."""
        with pytest.raises(ValueError, match="distinct"):
            experiment_plan(base, seeds=(0, 1, 1))

    def test_no_seeds_is_an_error(self, base: RunConfig) -> None:
        with pytest.raises(ValueError, match="at least one seed"):
            experiment_plan(base, seeds=())

    def test_describe_names_the_seed_count(self, base: RunConfig) -> None:
        """The count multiplies the GPU bill; --dry-run is where it is cheap."""
        text = describe_plan(experiment_plan(base))
        assert "70 runs" in text
        assert "5 seed(s) [0, 1, 2, 3, 4]" in text


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

    def test_selection_defaults_to_one_seed(self, base: RunConfig) -> None:
        """It is a selection step run before the study, not a reported result."""
        specs = ablation_plan(base, mask_ratios=(0.3, 0.5))
        assert len(specs) == 4
        assert {s.config.train.seed for s in specs} == {0}

    def test_seeds_replicate_every_ratio(self, base: RunConfig) -> None:
        specs = ablation_plan(base, mask_ratios=(0.3, 0.5), seeds=(0, 1, 2))
        assert len(specs) == 2 * 3 * 2
        assert len({s.name for s in specs}) == len(specs)
        by_name = {s.name: s for s in specs}
        for spec in specs:
            if spec.depends_on:
                assert by_name[spec.depends_on].config.train.seed == (
                    spec.config.train.seed
                )


class TestSslBudgetPlan:
    """How long is pretraining worth running, asked before the study
    multiplies that number by ten."""

    def test_one_annealed_pretrain_per_budget(self, base: RunConfig) -> None:
        specs = ssl_budget_plan(base, budgets=(100, 200, 400))
        pretrains = [s for s in specs if s.kind == "pretrain"]
        assert len(specs) == 6
        assert [s.config.train.for_pretraining().epochs for s in pretrains] == [
            100, 200, 400
        ]
        # Each rung anneals over its own budget: 0 means "span the budget".
        assert {s.config.train.ssl_schedule_epochs for s in pretrains} == {0}

    def test_the_fine_tunes_keep_the_supervised_budget(self, base: RunConfig) -> None:
        """The SSL budget must not leak into the fine-tune, which is the whole
        reason ssl_epochs exists."""
        specs = ssl_budget_plan(base, budgets=(100, 400))
        for spec in specs:
            if spec.kind == "supervised":
                assert spec.config.train.epochs == base.train.epochs

    def test_each_fine_tune_follows_its_own_pretrain(self, base: RunConfig) -> None:
        by_name = {s.name: s for s in ssl_budget_plan(base, budgets=(100, 200))}
        for spec in by_name.values():
            if spec.depends_on:
                source = by_name[spec.depends_on]
                assert source.kind == "pretrain"
                assert spec.name.startswith(source.name)

    def test_snapshot_mode_pretrains_once(self, base: RunConfig) -> None:
        specs = ssl_budget_plan(
            base, budgets=(100, 200, 400, 600), from_snapshots=True
        )
        pretrains = [s for s in specs if s.kind == "pretrain"]
        assert len(pretrains) == 1
        assert pretrains[0].config.train.for_pretraining().epochs == 600
        assert len([s for s in specs if s.kind == "supervised"]) == 4

    def test_snapshot_mode_names_the_file_each_rung_loads(
        self, base: RunConfig
    ) -> None:
        specs = ssl_budget_plan(base, budgets=(100, 400), from_snapshots=True)
        loaded = [s.checkpoint_name for s in specs if s.kind == "supervised"]
        assert loaded == [snapshot_name(100), snapshot_name(400)]

    def test_snapshots_land_on_every_rung(self, base: RunConfig) -> None:
        """A period that missed a budget would leave that fine-tune with no
        checkpoint to load, discovered only after the pretraining had run."""
        for budgets in ((100, 200, 400, 600), (150, 450), (7, 13)):
            specs = ssl_budget_plan(base, budgets=budgets, from_snapshots=True)
            every = specs[0].config.train.snapshot_every
            assert every > 0
            assert all(rung % every == 0 for rung in budgets)

    def test_separate_runs_cost_more_than_snapshots(self, base: RunConfig) -> None:
        """The trade the flag exists to make."""
        def ssl_epochs(specs):
            return sum(
                s.config.train.for_pretraining().epochs
                for s in specs
                if s.kind == "pretrain"
            )

        budgets = (100, 200, 400, 600)
        assert ssl_epochs(ssl_budget_plan(base, budgets=budgets)) == 1300
        assert ssl_epochs(
            ssl_budget_plan(base, budgets=budgets, from_snapshots=True)
        ) == 600

    def test_names_are_unique_across_seeds(self, base: RunConfig) -> None:
        for snapshots in (False, True):
            specs = ssl_budget_plan(
                base, budgets=(100, 200), seeds=(0, 1), from_snapshots=snapshots
            )
            assert len({s.name for s in specs}) == len(specs)

    def test_it_is_prefixed_by_the_variant(self, base: RunConfig) -> None:
        specs = ssl_budget_plan(base.with_(variant="deep6"), budgets=(100,))
        assert all(s.name.startswith("deep6-") for s in specs)

    def test_bad_budgets_are_rejected(self, base: RunConfig) -> None:
        with pytest.raises(ValueError, match="at least one budget"):
            ssl_budget_plan(base, budgets=())
        with pytest.raises(ValueError, match="budgets must be positive"):
            ssl_budget_plan(base, budgets=(0, 100))


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
        specs = experiment_plan(
            base, fractions=(1.0,), embedders=("linear",), seeds=(0,)
        )
        outcomes = run_plan(
            specs, workspace, output_root=tmp_path / "runs", track=False, progress=False
        )
        assert len(outcomes) == 3
        ssl_run = next(o for o in outcomes if o.pretrained)
        assert ssl_run.arm == "linear+ssl"

    def test_resume_skips_finished_runs(self, base, workspace, tmp_path) -> None:
        """What makes a Colab disconnect cost one run instead of the study."""
        specs = experiment_plan(
            base, fractions=(1.0,), embedders=("linear",), seeds=(0,)
        )
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


class TestSeededInitialisation:
    """Weight initialisation draws from the global generator, and the loops
    seed only after their caller has already built the model. Left alone, a
    run's weights depended on how much randomness every earlier run in the
    plan consumed -- so a run was not reproducible from its own config, and
    two arms at one seed did not in fact start from comparable places."""

    def test_a_run_ignores_the_ambient_generator(
        self, base, workspace, tmp_path
    ) -> None:
        spec = RunSpec("sup", "supervised", base.with_(name="sup"))
        torch.manual_seed(0)
        first = run_one(
            spec, workspace, output_root=tmp_path / "a", track=False, progress=False
        )
        # Stand in for other runs having executed in between.
        torch.randn(5000)
        second = run_one(
            spec, workspace, output_root=tmp_path / "b", track=False, progress=False
        )
        assert first.metrics == pytest.approx(second.metrics)

    def test_pretraining_is_seeded_the_same_way(self, base, workspace, tmp_path) -> None:
        spec = RunSpec("pre", "pretrain", base.with_(name="pre"))
        torch.manual_seed(0)
        first = run_one(
            spec, workspace, output_root=tmp_path / "a", track=False, progress=False
        )
        torch.randn(5000)
        second = run_one(
            spec, workspace, output_root=tmp_path / "b", track=False, progress=False
        )
        assert first.metrics == pytest.approx(second.metrics)

    def test_different_seeds_still_differ(self, base, workspace, tmp_path) -> None:
        """The fix must not collapse the replicates into one another."""
        outcomes = [
            run_one(
                RunSpec("sup", "supervised", _replicated(base, seed)),
                workspace,
                output_root=tmp_path / f"s{seed}",
                track=False,
                progress=False,
            )
            for seed in (0, 1)
        ]
        assert outcomes[0].metrics != pytest.approx(outcomes[1].metrics)


def _replicated(base: RunConfig, seed: int) -> RunConfig:
    """The base config at one replicate seed, as the plan builds it."""
    return base.with_(
        name="sup",
        train=TrainConfig(**{**vars(base.train), "seed": seed}),
        subset_seed=seed,
    )


class TestVariant:
    """Telling one architecture's results from another's.

    The failure this guards against is silent: a run name says which embedder
    and which label fraction, nothing about the architecture, and a run name is
    also its output directory. Change the model, re-run, and every run is
    skipped as already complete -- handing back the old model's numbers with no
    error anywhere.
    """

    def test_default_variant_leaves_names_untouched(self, base: RunConfig) -> None:
        """An existing study must keep resuming, not re-run under new names."""
        assert variant_name(DEFAULT_VARIANT, "sup-linear-ssl-f020") == (
            "sup-linear-ssl-f020"
        )
        names = {s.name for s in experiment_plan(base, seeds=(0,))}
        assert "sup-conv-ssl-f100" in names

    def test_a_named_variant_prefixes_every_run(self, base: RunConfig) -> None:
        specs = experiment_plan(base.with_(variant="deep6"))
        assert all(s.name.startswith("deep6-") for s in specs)
        assert all(s.config.variant == "deep6" for s in specs)

    def test_variants_never_share_a_directory(self, base: RunConfig) -> None:
        """The whole point: different directories, so resume cannot confuse them."""
        old = {s.name for s in experiment_plan(base)}
        new = {s.name for s in experiment_plan(base.with_(variant="deep6"))}
        assert not (old & new)

    def test_ssl_dependencies_stay_inside_the_variant(self, base: RunConfig) -> None:
        """A deep6 arm must not fine-tune the base variant's encoder."""
        for spec in experiment_plan(base.with_(variant="deep6")):
            if spec.depends_on:
                assert spec.depends_on.startswith("deep6-")
                assert spec.depends_on in {s.name for s in experiment_plan(
                    base.with_(variant="deep6")
                )}

    def test_ablation_plan_is_prefixed_too(self, base: RunConfig) -> None:
        specs = ablation_plan(base.with_(variant="deep6"), mask_ratios=(0.3,))
        assert all(s.name.startswith("deep6-") for s in specs)

    def test_variant_must_be_usable_as_a_directory_name(self, base: RunConfig) -> None:
        for bad in ("", "two words", "a/b", "a\\b"):
            with pytest.raises(ValueError, match="variant must be"):
                base.with_(variant=bad)

    def test_variant_round_trips_through_yaml(self, base, tmp_path) -> None:
        """Integrity rule 6: the variant is part of the reproducible config."""
        path = base.with_(variant="deep6").to_yaml(tmp_path / "config.yaml")
        assert RunConfig.from_yaml(path).variant == "deep6"

    def test_an_old_config_without_a_variant_still_loads(self, base, tmp_path) -> None:
        payload = base.to_dict()
        del payload["variant"]
        assert RunConfig.from_dict(payload).variant == DEFAULT_VARIANT

    def test_describe_plan_names_the_variant(self, base: RunConfig) -> None:
        text = describe_plan(experiment_plan(base.with_(variant="deep6")))
        assert "variant deep6" in text

    def test_outcome_records_variant_and_commit(self, base, workspace, tmp_path) -> None:
        """So results.csv is self-describing without consulting MLflow."""
        spec = RunSpec("sup", "supervised", base.with_(name="sup", variant="deep6"))
        outcome = run_one(
            spec, workspace, output_root=tmp_path / "runs", track=False, progress=False
        )
        assert outcome.variant == "deep6"
        assert outcome.git_sha == git_provenance()["git_sha"]
        assert RunOutcome.load(tmp_path / "runs" / "sup") == outcome
        assert results_frame([outcome]).loc[0, "variant"] == "deep6"

    def test_resuming_across_a_code_change_warns(
        self, base, workspace, tmp_path, monkeypatch
    ) -> None:
        """The one case the variant exists to prevent, caught if it happens anyway."""
        specs = [RunSpec("sup", "supervised", base.with_(name="sup"))]
        root = tmp_path / "runs"
        run_plan(specs, workspace, output_root=root, track=False, progress=False)

        monkeypatch.setattr(
            runner, "git_provenance", lambda *a, **k: {"git_sha": "deadbee"}
        )
        with pytest.warns(RuntimeWarning, match="OLD model"):
            run_plan(specs, workspace, output_root=root, track=False, progress=False)

    def test_resuming_on_the_same_commit_is_quiet(
        self, base, workspace, tmp_path
    ) -> None:
        """A Colab disconnect is the normal case and must not cry wolf."""
        specs = [RunSpec("sup", "supervised", base.with_(name="sup"))]
        root = tmp_path / "runs"
        run_plan(specs, workspace, output_root=root, track=False, progress=False)
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            run_plan(specs, workspace, output_root=root, track=False, progress=False)


class TestVariantResults:
    """Comparing across variants must not silently average them together."""

    def test_load_results_reads_every_variant(self, base, workspace, tmp_path) -> None:
        """results.csv holds only the last plan; result.json files hold all of them."""
        root = tmp_path / "runs"
        for variant in ("base", "deep6"):
            spec = RunSpec(
                variant_name(variant, "sup"),
                "supervised",
                base.with_(name="sup", variant=variant),
            )
            run_one(spec, workspace, output_root=root, track=False, progress=False)

        frame = load_results(root)
        assert sorted(frame["variant"]) == ["base", "deep6"]

    def test_missing_directory_is_named(self, tmp_path) -> None:
        with pytest.raises(FileNotFoundError, match="no output directory"):
            load_results(tmp_path / "absent")

    def test_one_variant_keeps_the_original_shape(self) -> None:
        frame = _fake_results()
        assert label_efficiency_table(frame).index.name == "label_fraction"
        assert ssl_benefit(frame).index.names == ["embedder", "label_fraction"]

    def test_two_variants_are_split_not_averaged(self) -> None:
        one = _fake_results()
        other = _fake_results()
        other["variant"] = "deep6"
        other["test_macro_auroc"] = other["test_macro_auroc"] + 0.10
        both = pd.concat([one, other], ignore_index=True)

        curve = label_efficiency_table(both)
        assert curve.index.names == ["variant", "label_fraction"]
        # The averaged value would sit between the two; neither row may be it.
        assert curve.loc["deep6"].to_numpy().max() > curve.loc["base"].to_numpy().max()
        assert ssl_benefit(both).index.names == [
            "variant", "embedder", "label_fraction",
        ]


class TestGitProvenance:
    def test_reports_this_repository(self) -> None:
        provenance = git_provenance()
        assert set(provenance) == {"git_sha", "git_dirty"}
        assert provenance["git_dirty"] in ("true", "false", UNKNOWN)

    def test_a_directory_with_no_repository_is_not_an_error(self, tmp_path) -> None:
        """Provenance is instrumentation; it must never cost a run."""
        assert git_provenance(tmp_path) == {"git_sha": UNKNOWN, "git_dirty": UNKNOWN}

    def test_a_missing_git_binary_is_not_an_error(self, monkeypatch, tmp_path) -> None:
        def explode(*args, **kwargs):
            raise FileNotFoundError("git")

        monkeypatch.setattr(tracking.subprocess, "run", explode)
        assert git_provenance(tmp_path)["git_sha"] == UNKNOWN


class TestDevicePlacement:
    """The loops move the *model* with ``.to(device)`` and never touch the
    batches, so a cohort built without a device is invisible on a CPU-only box
    and dies on the first matmul of a GPU run. Local torch here is CPU-only, so
    these check placement rather than run on a GPU."""

    def test_batches_land_on_the_device_they_are_given(self, base, workspace) -> None:
        batches = runner._batches(
            workspace, workspace.cohorts["val"], base, torch.device("meta"), shuffle=False
        )
        assert batches.waveforms.device.type == "meta"
        assert batches.labels.device.type == "meta"

    def test_every_cohort_in_a_run_is_placed_explicitly(
        self, base, workspace, tmp_path, monkeypatch
    ) -> None:
        seen: list[object] = []
        original = runner.EcgBatches

        def recording(*args, **kwargs):
            seen.append(kwargs.get("device", "MISSING"))
            return original(*args, **kwargs)

        monkeypatch.setattr(runner, "EcgBatches", recording)
        specs = experiment_plan(
            base, fractions=(1.0,), embedders=("linear",), seeds=(0,)
        )
        run_plan(
            specs, workspace, output_root=tmp_path / "runs", track=False, progress=False
        )

        expected = resolve_device(base.train.device)
        assert len(seen) >= 5, "expected ssl train/holdout plus train/val/test"
        assert all(device == expected for device in seen), seen


class TestResults:
    def test_frame_has_one_row_per_run(self, base, workspace, tmp_path) -> None:
        specs = experiment_plan(
            base, fractions=(1.0,), embedders=("linear",), seeds=(0,)
        )
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


class TestAggregation:
    """Five runs per arm reported as one number plus its spread."""

    def test_mean_and_sd_over_seeds(self) -> None:
        summary = aggregate_runs(_fake_results(seeds=(0, 1, 2, 3, 4)))
        row = summary.loc[("base", "supervised", "linear", 0.2, 0.5)]
        assert row["n_seeds"] == 5
        assert row["seeds"] == "0,1,2,3,4"
        # 0.80 .. 0.84 in steps of 0.01.
        assert row["test_macro_auroc_mean"] == pytest.approx(0.82)
        assert row["test_macro_auroc_sd"] == pytest.approx(np.std(
            [0.80, 0.81, 0.82, 0.83, 0.84], ddof=1
        ))

    def test_a_single_seed_has_no_deviation(self) -> None:
        """nan, not zero: one run says nothing about the spread."""
        summary = aggregate_runs(_fake_results(seeds=(0,)))
        assert summary["test_macro_auroc_sd"].isna().all()
        assert (summary["n_seeds"] == 1).all()

    def test_seeds_are_listed_so_a_part_finished_study_is_visible(self) -> None:
        """Four of five finished reads as a complete study unless it says which."""
        summary = aggregate_runs(_fake_results(seeds=(0, 1, 3, 4)))
        assert (summary["seeds"] == "0,1,3,4").all()
        assert (summary["n_seeds"] == 4).all()

    def test_pretrain_rows_survive_the_grouping(self) -> None:
        """Their label_fraction is nan, which the default groupby would drop."""
        frame = _fake_results(seeds=(0, 1))
        for seed in (0, 1):
            frame.loc[len(frame)] = {
                **frame.iloc[0].to_dict(),
                "kind": "pretrain",
                "label_fraction": float("nan"),
                "seed": seed,
            }
        summary = aggregate_runs(frame)
        assert "pretrain" in summary.index.get_level_values("kind")

    def test_two_runs_at_one_seed_warn(self) -> None:
        """A repeat is not a replicate, and an interval built on it is wrong."""
        frame = _fake_results(seeds=(0, 1))
        frame.loc[len(frame)] = frame.iloc[0].to_dict()
        with pytest.warns(RuntimeWarning, match="same seed"):
            aggregate_runs(frame)

    def test_missing_key_is_named(self) -> None:
        with pytest.raises(ValueError, match="mask_ratio"):
            aggregate_runs(_fake_results().drop(columns=["mask_ratio"]))

    def test_report_renders_mean_and_sd(self) -> None:
        report = label_efficiency_report(_fake_results(seeds=(0, 1, 2, 3, 4)))
        assert report.loc[0.2, "linear"] == "0.820 ± 0.016"

    def test_report_omits_an_undefined_deviation(self) -> None:
        assert label_efficiency_report(_fake_results()).loc[0.2, "linear"] == "0.800"
        assert format_mean_sd(0.8, float("nan")) == "0.800"
        assert format_mean_sd(float("nan"), 0.1) == "-"

    def test_curve_stays_numeric_for_plotting(self) -> None:
        curve = label_efficiency_table(_fake_results(seeds=(0, 1, 2, 3, 4)))
        assert curve.loc[0.2, "linear"] == pytest.approx(0.82)
        errors = label_efficiency_table(_fake_results(seeds=(0, 1)), stat="sd")
        assert errors.loc[0.2, "linear"] == pytest.approx(0.01 / np.sqrt(2))

    def test_unknown_stat_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="stat must be"):
            label_efficiency_table(_fake_results(), stat="median")


class TestPairedContrast:
    """The seeds are shared between arms, so the comparison is paired."""

    def test_pairing_removes_the_shared_seed_variation(self) -> None:
        """Every seed shifts both arms together, so the gain is exactly 0.05
        with no spread -- while the arms themselves vary by much more. Unpaired,
        that shared movement would sit in the interval and hide the effect."""
        table = ssl_benefit(_fake_results(seeds=(0, 1, 2, 3, 4)))
        row = table.loc[("linear", 0.2)]
        assert row["ssl_gain"] == pytest.approx(0.05)
        assert row["gain_sd"] == pytest.approx(0.0)
        assert row["scratch_sd"] > 0.01
        assert row["n_seeds"] == 5
        assert bool(row["gain_beats_noise"])

    def test_the_two_means_differ_by_the_reported_gain(self) -> None:
        """Otherwise a reader who subtracts the columns gets a third number."""
        table = ssl_benefit(_fake_results(seeds=(0, 1, 2, 3, 4)))
        assert (table["ssl"] - table["scratch"]).values == pytest.approx(
            table["ssl_gain"].values
        )

    def test_a_seed_missing_one_arm_is_dropped_from_both(self) -> None:
        frame = _fake_results(seeds=(0, 1, 2))
        unfinished = (frame["seed"] == 2) & frame["pretrained"]
        table = ssl_benefit(frame[~unfinished])
        assert (table["n_seeds"] == 2).all()
        assert (table["ssl"] - table["scratch"]).values == pytest.approx(
            table["ssl_gain"].values
        )

    def test_one_seed_cannot_beat_noise(self) -> None:
        table = ssl_benefit(_fake_results(seeds=(0,)))
        assert table["gain_sd"].isna().all()
        assert not table["gain_beats_noise"].any()

    def test_a_gain_inside_the_noise_is_not_claimed(self) -> None:
        frame = _fake_results(seeds=(0, 1, 2, 3, 4))
        rng = np.random.default_rng(0)
        # Swamp the flat 0.05 with per-run noise that pairing cannot remove.
        frame["test_macro_auroc"] += rng.normal(0.0, 0.2, len(frame))
        assert not ssl_benefit(frame).loc[("linear", 0.2), "gain_beats_noise"]

    def test_embedder_contrast_is_paired_the_same_way(self) -> None:
        table = embedder_benefit(_fake_results(
            seeds=(0, 1, 2, 3, 4), embedders=("linear", "conv")
        ))
        assert table.index.names == ["pretrained", "label_fraction"]
        assert table.loc[(False, 0.2), "conv_gain"] == pytest.approx(0.02)
        assert table.loc[(False, 0.2), "gain_sd"] == pytest.approx(0.0)

    def test_variants_stay_apart_in_both_contrasts(self) -> None:
        one = _fake_results(seeds=(0, 1), embedders=("linear", "conv"))
        other = one.copy()
        other["variant"] = "deep6"
        both = pd.concat([one, other], ignore_index=True)
        assert ssl_benefit(both).index.names == [
            "variant", "embedder", "label_fraction",
        ]
        assert embedder_benefit(both).index.names == [
            "variant", "pretrained", "label_fraction",
        ]


class TestCli:
    def test_dry_run_prints_the_plan_and_stops(self, tmp_path, capsys) -> None:
        """No store is loaded, so a nonexistent path must not matter."""
        code = main(
            [
                "--dry-run", "--output", str(tmp_path / "runs"),
                "--store", "does/not/exist", "--fractions", "0.2", "1.0",
                "--seeds", "0",
            ]
        )
        assert code == 0
        # 2 pretrains + 2 fractions x 2 embedders x {scratch, ssl}
        assert "10 runs" in capsys.readouterr().out

    def test_default_grid_is_seventy_runs(self, tmp_path, capsys) -> None:
        """(2 pretrains + 3 fractions x 4 arms) x 5 seeds, the study as planned."""
        main(["--dry-run", "--output", str(tmp_path / "runs")])
        out = capsys.readouterr().out
        assert "70 runs" in out
        assert "5 seed(s)" in out

    def test_one_seed_gets_the_old_grid_back(self, tmp_path, capsys) -> None:
        """The cheap smoke-test path, and the names an existing study resumes."""
        main(["--dry-run", "--output", str(tmp_path / "runs"), "--seeds", "0"])
        assert "14 runs" in capsys.readouterr().out

    def test_sharing_pretraining_drops_eight_runs(self, tmp_path, capsys) -> None:
        main(["--dry-run", "--output", str(tmp_path / "runs"), "--share-pretraining"])
        assert "62 runs" in capsys.readouterr().out

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

    def test_seeds_reach_the_plan(self, tmp_path, capsys) -> None:
        main(
            [
                "--dry-run", "--output", str(tmp_path / "runs"),
                "--seeds", "7", "8", "--fractions", "1.0",
            ]
        )
        out = capsys.readouterr().out
        assert "2 seed(s) [7, 8]" in out
        assert "sup-conv-ssl-f100-s8" in out

    def test_arguments_reach_the_config(self) -> None:
        args = build_parser().parse_args(
            ["--epochs", "7", "--d-model", "64", "--mask-ratio", "0.7", "--no-amp"]
        )
        config = base_config(args)
        assert config.train.epochs == 7
        assert config.model.d_model == 64
        assert config.ssl.mask_ratio == 0.7
        assert config.train.amp is False

    def test_tracking_defaults_to_local_disk_not_the_output(self, tmp_path) -> None:
        """The output goes to Drive; SQLite must not follow it there."""
        args = build_parser().parse_args(["--output", str(tmp_path / "runs")])
        tracking = base_config(args).tracking_uri
        assert tracking == "mlruns/mlflow.db"
        assert str(tmp_path) not in tracking

    def test_tracking_on_a_mounted_drive_warns(self) -> None:
        """SQLite over FUSE corrupts silently; an error would be preferable,
        but the path cannot be proven remote, so warn."""
        args = build_parser().parse_args(
            ["--tracking", "/content/drive/MyDrive/ecg/mlflow.db"]
        )
        with pytest.warns(RuntimeWarning, match="mounted drive"):
            base_config(args)

    def test_local_tracking_does_not_warn(self, recwarn) -> None:
        base_config(build_parser().parse_args(["--tracking", "/content/mlflow.db"]))
        assert not [w for w in recwarn if "mounted drive" in str(w.message)]

    def test_cloud_defaults_are_the_intended_ones(self) -> None:
        """The command run on the A100 should need no extra flags."""
        config = base_config(build_parser().parse_args([]))
        assert config.model.d_model == 256
        assert config.model.n_heads == 8
        assert config.ssl.mask_ratio == 0.5
        assert config.ssl.mask_span == 2
        assert config.train.amp is True


def _fake_results(seeds=(0,), embedders=("linear",)):
    """A minimal results frame for the table functions.

    Each seed shifts every arm by the same 0.01, and pretraining is worth a
    flat 0.05 on top. So the marginal spread across seeds is real while the
    paired SSL gain is exactly 0.05 with zero deviation -- which is the whole
    argument for pairing, made small enough to assert on.
    """
    import pandas as pd

    rows = []
    for fraction in (0.2, 1.0):
        for embedder in embedders:
            for pretrained in (False, True):
                for seed in seeds:
                    rows.append(
                        {
                            "run": f"r{embedder}{fraction}{pretrained}s{seed}",
                            "variant": "base",
                            "kind": "supervised",
                            "arm": f"{embedder}+ssl" if pretrained else embedder,
                            "embedder": embedder,
                            "pretrained": pretrained,
                            "label_fraction": fraction,
                            "mask_ratio": 0.5,
                            "seed": seed,
                            "test_macro_auroc": (
                                0.80
                                + (0.05 if pretrained else 0.0)
                                + (0.02 if embedder == "conv" else 0.0)
                                + 0.01 * seed
                            ),
                        }
                    )
    return pd.DataFrame(rows)
