"""Command line entry point: ``ecg-run``.

One command runs the study, and ``--dry-run`` prints the grid without touching
a GPU -- worth doing before every cloud session, because the cheapest place to
notice a wrong fraction or a wrong mask ratio is here.

Resuming is the default. After a Colab disconnect, re-issuing the same command
skips every run that already wrote ``result.json`` and continues from the one
that was interrupted.

Every experiment runs at five seeds and is reported as mean +/- sd, which is
also why ``--dry-run`` matters more than it used to: the default study is 70
runs, not 14. ``--seeds 0`` gets the old single-seed grid back for a smoke test.
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

from ecg.experiments.plan import (
    DEFAULT_FRACTIONS,
    DEFAULT_MASK_RATIOS,
    DEFAULT_SEEDS,
    DEFAULT_SSL_BUDGETS,
    ablation_plan,
    describe_plan,
    experiment_plan,
    ssl_budget_plan,
)
from ecg.experiments.runner import (
    Workspace,
    aggregate_runs,
    embedder_benefit,
    label_efficiency_report,
    results_frame,
    run_plan,
    ssl_benefit,
)
from ecg.models.config import ModelConfig, SslConfig
from ecg.training.config import DEFAULT_VARIANT, RunConfig, TrainConfig


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser.

    Returns:
        The parser.
    """
    parser = argparse.ArgumentParser(
        prog="ecg-run",
        description="Run the SSL / patch-embedder study on PTB-XL.",
    )
    parser.add_argument(
        "--plan",
        choices=("study", "ablation", "ssl-budget"),
        default="study",
        help=(
            "'study' is the four arms; 'ablation' selects the mask ratio "
            "first; 'ssl-budget' asks how long pretraining is worth running "
            "before the study multiplies that cost by ten."
        ),
    )
    parser.add_argument("--store", default="data/store_100hz", help="Waveform store.")
    parser.add_argument("--metadata", default="data/ptbxl", help="PTB-XL root.")
    parser.add_argument("--output", default="runs", help="Directory for run outputs.")
    parser.add_argument(
        "--tracking",
        default="mlruns/mlflow.db",
        help=(
            "Tracking database. Keep it on LOCAL disk: SQLite locking assumes "
            "POSIX semantics a Drive FUSE mount does not honour. A consistent "
            "copy is written to the output directory every epoch."
        ),
    )
    parser.add_argument("--experiment", default="ecg-ssl", help="MLflow experiment.")
    parser.add_argument(
        "--variant",
        default=DEFAULT_VARIANT,
        help=(
            "Architecture variant, e.g. 'deep6'. Set this whenever you change "
            "the model: it prefixes every run name, so the new runs get their "
            "own directories instead of being skipped as already complete, and "
            "they are filterable in MLflow. Name what changed, not a version "
            "number. Keep --experiment the same so old and new stay comparable."
        ),
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument(
        "--schedule-epochs",
        type=int,
        default=0,
        help=(
            "Epochs the cosine learning-rate decay spans; 0 means --epochs, "
            "which is what every run before this flag existed did. Set it and "
            "--epochs becomes a pure ceiling: '--schedule-epochs 50 --epochs "
            "300 --patience 10' decays exactly as a 50-epoch run, then holds "
            "at the floor until the run stops improving. Leave it at 0 and a "
            "large --epochs stretches the decay instead of extending the run. "
            "It is part of the optimiser, so give runs that use it a new "
            "--variant rather than mixing them with runs that did not."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument(
        "--mask-ratio", type=float, default=0.5, help="Used by the 'study' plan."
    )
    parser.add_argument("--mask-span", type=int, default=2)
    parser.add_argument(
        "--ssl-epochs",
        type=int,
        default=0,
        help=(
            "Epochs for pretraining runs; 0 means --epochs. Pretraining and "
            "fine-tuning want budgets an order of magnitude apart -- 50 epochs "
            "is only 3,250 SSL steps at batch 256 -- and without this one "
            "number set both, so a long SSL budget also bought long "
            "fine-tunes, which are 60 of the study's 70 runs."
        ),
    )
    parser.add_argument(
        "--ssl-schedule-epochs",
        type=int,
        default=0,
        help="--schedule-epochs for pretraining runs; 0 anneals over --ssl-epochs.",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=1,
        help=(
            "Write the durable checkpoints every N epochs. At 600 epochs the "
            "per-epoch write is the run, not the instrumentation: 55 MB to "
            "Drive against an epoch of 65 steps. The best weights are held in "
            "memory and always flushed when the run ends, so raising this "
            "risks losing N epochs of progress to a disconnect, never the "
            "selected encoder."
        ),
    )
    parser.add_argument(
        "--snapshot-every",
        type=int,
        default=0,
        help=(
            "Also write a weights-only pretrain_epoch<N>.pt every N epochs; 0 "
            "writes none. Set automatically by '--plan ssl-budget "
            "--from-snapshots'."
        ),
    )
    parser.add_argument(
        "--ssl-budgets",
        type=int,
        nargs="+",
        default=list(DEFAULT_SSL_BUDGETS),
        help="Pretraining budgets compared by the 'ssl-budget' plan.",
    )
    parser.add_argument(
        "--from-snapshots",
        action="store_true",
        help=(
            "'ssl-budget' only: pretrain once at the longest budget and "
            "fine-tune from mid-run snapshots, instead of one properly "
            "annealed run per budget. Roughly half the cost, but the short "
            "rungs are taken at a learning rate still near peak and so "
            "understate what a real run of that length reaches. A first look, "
            "not a number for the write-up."
        ),
    )
    parser.add_argument(
        "--fractions",
        type=float,
        nargs="+",
        default=list(DEFAULT_FRACTIONS),
        help="Label fractions for the efficiency curve.",
    )
    parser.add_argument(
        "--mask-ratios",
        type=float,
        nargs="+",
        default=list(DEFAULT_MASK_RATIOS),
        help="Candidate ratios for the 'ablation' plan.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help=(
            f"Replicate seeds. Every experiment runs once per seed and is "
            f"reported as mean +/- sd, so this multiplies cost by its length "
            f"-- check the epoch total that --dry-run prints. Each seed drives "
            f"weights, batch order, masking AND which labelled records the "
            f"fraction draws. Defaults to {list(DEFAULT_SEEDS)} for --plan "
            f"study, and to [{DEFAULT_SEEDS[0]}] for the selection plans, "
            f"which choose a setting on validation rather than reporting a "
            f"result and do not need error bars to do it."
        ),
    )
    parser.add_argument(
        "--share-pretraining",
        action="store_true",
        help=(
            "Pretrain once per embedder instead of once per seed, and "
            "fine-tune every seed from it. Saves GPU hours; costs the SSL arm "
            "an interval that omits pretraining variance and so is narrower "
            "than the scratch arm's for a reason unrelated to SSL. Say so in "
            "the write-up if you use it."
        ),
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--patience",
        type=int,
        default=0,
        help=(
            "Stop a run after this many evaluations without improvement; 0 "
            "disables it. Applies to both loops: supervised runs watch val "
            "macro AUROC, pretraining runs watch the held-out reconstruction "
            "loss. Treat it as a safety net, not as a licence to set --epochs "
            "high -- the cosine schedule spans --epochs, so a big budget "
            "stretches the decay instead of merely capping the run, and a run "
            "that stops early never anneals. Both loops warn when that happens."
        ),
    )
    parser.add_argument("--no-amp", action="store_true", help="Disable bf16 autocast.")
    parser.add_argument("--no-track", action="store_true", help="Disable MLflow.")
    parser.add_argument(
        "--no-resume", action="store_true", help="Re-run runs that already finished."
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print the plan and exit."
    )
    return parser


#: Path fragments that indicate a network/FUSE mount where SQLite is unsafe.
_MOUNT_MARKERS: tuple[str, ...] = ("/drive/", "\\drive\\", "/gdrive/")


def warn_if_mounted(tracking_uri: str) -> None:
    """Warn when the tracking database would live on a FUSE mount.

    SQLite's locking assumes POSIX semantics that Google Drive's FUSE layer
    does not provide, and the failure mode is a corrupt database rather than an
    error at write time. The database belongs on local disk; a consistent copy
    reaches durable storage every epoch through
    :func:`ecg.training.checkpoints.sync_tracking`.

    Args:
        tracking_uri: The configured tracking path.
    """
    lowered = str(tracking_uri).lower()
    if any(marker in lowered for marker in _MOUNT_MARKERS):
        warnings.warn(
            f"tracking database {tracking_uri!r} looks like it is on a mounted "
            "drive. SQLite over FUSE can corrupt silently; point --tracking at "
            "local disk instead. It is copied to --output every epoch anyway.",
            RuntimeWarning,
            stacklevel=2,
        )


def base_config(args: argparse.Namespace) -> RunConfig:
    """Assemble the base configuration every run is derived from.

    Args:
        args: Parsed arguments.

    Returns:
        The base configuration.
    """
    output = Path(args.output)
    warn_if_mounted(args.tracking)
    return RunConfig(
        variant=args.variant,
        model=ModelConfig(
            d_model=args.d_model, n_layers=args.n_layers, n_heads=args.n_heads
        ),
        train=TrainConfig(
            epochs=args.epochs,
            schedule_epochs=args.schedule_epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            # Both seeds are overridden per replicate by the plan; the first is
            # only a placeholder, so a base config printed on its own is honest.
            seed=(args.seeds or DEFAULT_SEEDS)[0],
            amp=not args.no_amp,
            device=args.device,
            patience=args.patience,
            ssl_epochs=args.ssl_epochs,
            ssl_schedule_epochs=args.ssl_schedule_epochs,
            checkpoint_every=args.checkpoint_every,
            snapshot_every=args.snapshot_every,
        ),
        ssl=SslConfig(mask_ratio=args.mask_ratio, mask_span=args.mask_span),
        store_path=args.store,
        metadata_path=args.metadata,
        subset_seed=(args.seeds or DEFAULT_SEEDS)[0],
        experiment=args.experiment,
        tracking_uri=args.tracking,
        output_dir=str(output),
    )


def main(argv: list[str] | None = None) -> int:
    """Run the study.

    Args:
        argv: Argument list; defaults to ``sys.argv``.

    Returns:
        Process exit code.
    """
    args = build_parser().parse_args(argv)
    base = base_config(args)

    # A selection plan picks a setting on validation; it is not a reported
    # result, so it does not need five replicates and should not silently cost
    # what five replicates cost. An explicit --seeds still wins.
    seeds = tuple(args.seeds) if args.seeds else None
    if args.plan == "ssl-budget":
        specs = ssl_budget_plan(
            base,
            budgets=tuple(args.ssl_budgets),
            fraction=min(args.fractions),
            seeds=seeds or (DEFAULT_SEEDS[0],),
            from_snapshots=args.from_snapshots,
        )
    elif args.plan == "ablation":
        specs = ablation_plan(
            base, mask_ratios=tuple(args.mask_ratios), fraction=min(args.fractions),
            seeds=seeds or (DEFAULT_SEEDS[0],),
        )
    else:
        specs = experiment_plan(
            base,
            fractions=tuple(args.fractions),
            seeds=seeds or DEFAULT_SEEDS,
            share_pretraining=args.share_pretraining,
        )

    print(describe_plan(specs))
    if args.dry_run:
        return 0

    workspace = Workspace.load(base)
    outcomes = run_plan(
        specs,
        workspace,
        output_root=args.output,
        tracking_uri=base.tracking_uri,
        experiment=args.experiment,
        track=not args.no_track,
        resume=not args.no_resume,
    )

    frame = results_frame(outcomes)
    output = Path(args.output)
    frame.to_csv(output / "results.csv", index=False)

    # Two files, because they answer different questions: results.csv is the
    # per-run record, summary.csv is what a table in the write-up is built from.
    summary = aggregate_runs(frame)
    summary.to_csv(output / "summary.csv")
    print(f"\nwrote {output / 'results.csv'} and {output / 'summary.csv'}")

    supervised = frame[frame["kind"] == "supervised"]
    if not supervised.empty:
        seeds = sorted(supervised["seed"].unique())
        print(f"\nmean ± sd over {len(seeds)} seed(s) {seeds}")
        print("\nlabel efficiency (test macro AUROC)")
        print(label_efficiency_report(frame).to_string())
        print("\nwhat SSL bought (paired within seed)")
        print(ssl_benefit(frame).to_string())
        print("\nwhat the conv stem bought (paired within seed)")
        print(embedder_benefit(frame).to_string())
        if len(seeds) < 2:
            print(
                "\nnote: one seed, so every deviation is undefined and no "
                "contrast can be shown to beat noise. Pass --seeds 0 1 2 3 4."
            )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
