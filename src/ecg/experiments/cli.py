"""Command line entry point: ``ecg-run``.

One command runs the study, and ``--dry-run`` prints the grid without touching
a GPU -- worth doing before every cloud session, because the cheapest place to
notice a wrong fraction or a wrong mask ratio is here.

Resuming is the default. After a Colab disconnect, re-issuing the same command
skips every run that already wrote ``result.json`` and continues from the one
that was interrupted.
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

from ecg.experiments.plan import (
    DEFAULT_FRACTIONS,
    DEFAULT_MASK_RATIOS,
    ablation_plan,
    describe_plan,
    experiment_plan,
)
from ecg.experiments.runner import (
    Workspace,
    label_efficiency_table,
    results_frame,
    run_plan,
    ssl_benefit,
)
from ecg.models.config import ModelConfig, SslConfig
from ecg.training.config import RunConfig, TrainConfig


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
        choices=("study", "ablation"),
        default="study",
        help="'study' is the four arms; 'ablation' selects the mask ratio first.",
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
    parser.add_argument("--epochs", type=int, default=50)
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
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--patience", type=int, default=0, help="0 disables early stop.")
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
        model=ModelConfig(
            d_model=args.d_model, n_layers=args.n_layers, n_heads=args.n_heads
        ),
        train=TrainConfig(
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            seed=args.seed,
            amp=not args.no_amp,
            device=args.device,
            patience=args.patience,
        ),
        ssl=SslConfig(mask_ratio=args.mask_ratio, mask_span=args.mask_span),
        store_path=args.store,
        metadata_path=args.metadata,
        subset_seed=args.seed,
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

    if args.plan == "ablation":
        specs = ablation_plan(
            base, mask_ratios=tuple(args.mask_ratios), fraction=min(args.fractions),
            seed=args.seed,
        )
    else:
        specs = experiment_plan(
            base, fractions=tuple(args.fractions), seed=args.seed
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
    destination = Path(args.output) / "results.csv"
    frame.to_csv(destination, index=False)
    print(f"\nwrote {destination}")

    supervised = frame[frame["kind"] == "supervised"]
    if not supervised.empty:
        print("\nlabel efficiency (test macro AUROC)")
        print(label_efficiency_table(frame).to_string())
        print("\nwhat SSL bought")
        print(ssl_benefit(frame).to_string())
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
