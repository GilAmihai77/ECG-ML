"""The experiment grid, built as data before anything is trained.

A plan is a list of :class:`RunSpec`, each holding a complete
:class:`~ecg.training.config.RunConfig`. Building the grid separately from
executing it means the whole study can be printed, counted and checked before a
GPU is booked, and it makes the one property that matters structurally true
rather than merely intended: every run in the grid is derived from **one** base
configuration, so the arms cannot differ in anything except the embedder and
whether an encoder was pretrained.

Two plans are defined:

* :func:`experiment_plan` -- the study itself. Two pretraining runs, then four
  arms at each label fraction.
* :func:`ablation_plan` -- the mask-ratio selection. Pretraining runs at each
  candidate ratio, fine-tuned at the smallest label fraction only, where SSL's
  effect is largest and the runs are cheapest. Selection is on validation, and
  the winning ratio is then frozen for all four arms of the main study.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from ecg.models.config import Embedder
from ecg.training.config import DEFAULT_VARIANT, RunConfig

RunKind = Literal["pretrain", "supervised"]

#: Label fractions for the label-efficiency curve.
DEFAULT_FRACTIONS: tuple[float, ...] = (0.2, 0.5, 1.0)

#: Mask ratios compared before the main study.
DEFAULT_MASK_RATIOS: tuple[float, ...] = (0.3, 0.5, 0.7)

#: The two embedders under comparison.
EMBEDDERS: tuple[Embedder, ...] = ("linear", "conv")


@dataclass(frozen=True)
class RunSpec:
    """One run, with its dependency on a pretraining run if it has one.

    Attributes:
        name: Unique identifier; also the run's output subdirectory.
        kind: ``"pretrain"`` or ``"supervised"``.
        config: The complete run configuration.
        depends_on: Name of the pretraining run supplying the encoder, or
            ``None`` for a from-scratch arm.
    """

    name: str
    kind: RunKind
    config: RunConfig
    depends_on: str | None = None

    @property
    def arm(self) -> str:
        """Arm label, e.g. ``"conv+ssl"``."""
        return self.config.arm


def variant_name(variant: str, stem: str) -> str:
    """Prefix a run name with its architecture variant.

    A run name is also its output directory, and ``run_plan`` skips any
    directory that already holds a ``result.json``. So without this, changing
    the architecture and re-running would skip all fourteen runs and hand back
    the previous architecture's numbers -- silently, since nothing in a run
    name mentions the architecture.

    The default variant is deliberately left unprefixed: an existing study
    keeps the names it already has, and its finished runs still resume instead
    of being re-run under new ones.

    Args:
        variant: The variant slug, e.g. ``"deep6"``.
        stem: The name the run would have had, e.g. ``"sup-linear-ssl-f020"``.

    Returns:
        ``stem`` for the default variant, ``"<variant>-<stem>"`` otherwise.
    """
    return stem if variant == DEFAULT_VARIANT else f"{variant}-{stem}"


def pretrain_name(embedder: str, mask_ratio: float) -> str:
    """Name a pretraining run.

    Args:
        embedder: ``"linear"`` or ``"conv"``.
        mask_ratio: Masking ratio.

    Returns:
        A name like ``"pretrain-conv-m50"``.
    """
    return f"pretrain-{embedder}-m{int(round(mask_ratio * 100)):02d}"


def supervised_name(embedder: str, pretrained: bool, fraction: float) -> str:
    """Name a supervised run.

    Args:
        embedder: ``"linear"`` or ``"conv"``.
        pretrained: Whether the encoder starts from SSL weights.
        fraction: Label fraction.

    Returns:
        A name like ``"sup-linear-ssl-f020"``.
    """
    origin = "ssl" if pretrained else "scratch"
    return f"sup-{embedder}-{origin}-f{int(round(fraction * 100)):03d}"


def experiment_plan(
    base: RunConfig,
    *,
    fractions: tuple[float, ...] = DEFAULT_FRACTIONS,
    embedders: tuple[Embedder, ...] = EMBEDDERS,
    seed: int = 0,
) -> list[RunSpec]:
    """Build the full study: pretraining runs, then the four arms.

    Ordering matters -- pretraining runs come first because the SSL arms depend
    on their checkpoints -- and :func:`ecg.experiments.runner.run_plan` relies
    on it rather than resolving a graph.

    Every supervised run shares ``base.subset_seed``, so at a given fraction all
    four arms train on exactly the same records. Varying it per arm would make
    the label-efficiency curve confound sample size with which patients were
    drawn.

    Args:
        base: Base configuration. Its ``model.embedder`` is overridden per arm;
            everything else is inherited unchanged.
        fractions: Label fractions for the curve.
        embedders: Embedders to compare.
        seed: Training seed for every run.

    Returns:
        The runs, pretraining first.
    """
    specs: list[RunSpec] = []

    for embedder in embedders:
        name = variant_name(base.variant, pretrain_name(embedder, base.ssl.mask_ratio))
        specs.append(
            RunSpec(
                name=name,
                kind="pretrain",
                config=base.with_(
                    name=name,
                    model=base.model.for_arm(embedder),
                    train=_seeded(base, seed),
                    pretrained_from=None,
                ),
            )
        )

    for fraction in fractions:
        for embedder in embedders:
            for pretrained in (False, True):
                source = (
                    variant_name(
                        base.variant, pretrain_name(embedder, base.ssl.mask_ratio)
                    )
                    if pretrained
                    else None
                )
                name = variant_name(
                    base.variant, supervised_name(embedder, pretrained, fraction)
                )
                specs.append(
                    RunSpec(
                        name=name,
                        kind="supervised",
                        config=base.with_(
                            name=name,
                            model=base.model.for_arm(embedder),
                            train=_seeded(base, seed),
                            label_fraction=fraction,
                            # Filled in by the runner once the checkpoint path
                            # is known; kept as the run name here so the plan
                            # is printable without a filesystem.
                            pretrained_from=source,
                        ),
                        depends_on=source,
                    )
                )
    return specs


def ablation_plan(
    base: RunConfig,
    *,
    mask_ratios: tuple[float, ...] = DEFAULT_MASK_RATIOS,
    embedder: Embedder = "linear",
    fraction: float = 0.2,
    seed: int = 0,
) -> list[RunSpec]:
    """Build the mask-ratio selection.

    One embedder only, and one label fraction only. Tuning the mask separately
    per arm would fold the SSL setup into what is being compared, so the ratio
    chosen here is applied unchanged to all four arms.

    Args:
        base: Base configuration.
        mask_ratios: Candidate ratios.
        embedder: Which arm to select on.
        fraction: Label fraction to fine-tune at. The smallest one, where SSL's
            effect is largest and the runs are cheapest.
        seed: Training seed.

    Returns:
        The runs, each pretraining immediately followed by its fine-tune.
    """
    specs: list[RunSpec] = []
    for ratio in mask_ratios:
        ssl_config = base.ssl.__class__(mask_ratio=ratio, mask_span=base.ssl.mask_span)
        name = variant_name(base.variant, pretrain_name(embedder, ratio))
        specs.append(
            RunSpec(
                name=name,
                kind="pretrain",
                config=base.with_(
                    name=name,
                    model=base.model.for_arm(embedder),
                    train=_seeded(base, seed),
                    ssl=ssl_config,
                    pretrained_from=None,
                ),
            )
        )
        finetune = f"{name}-f{int(round(fraction * 100)):03d}"
        specs.append(
            RunSpec(
                name=finetune,
                kind="supervised",
                config=base.with_(
                    name=finetune,
                    model=base.model.for_arm(embedder),
                    train=_seeded(base, seed),
                    ssl=ssl_config,
                    label_fraction=fraction,
                    pretrained_from=name,
                ),
                depends_on=name,
            )
        )
    return specs


def describe_plan(specs: list[RunSpec]) -> str:
    """Render a plan as a table for review before spending GPU time.

    Args:
        specs: The runs.

    Returns:
        A printable multi-line string.
    """
    variants = sorted({spec.config.variant for spec in specs})
    lines = [f"{len(specs)} runs, variant {', '.join(variants)}", ""]
    header = f"{'run':40s} {'kind':11s} {'arm':12s} {'labels':>7s} {'mask':>5s}"
    lines += [header, "-" * len(header)]
    for spec in specs:
        labels = "-" if spec.kind == "pretrain" else f"{spec.config.label_fraction:.0%}"
        lines.append(
            f"{spec.name:40s} {spec.kind:11s} {spec.arm:12s} {labels:>7s} "
            f"{spec.config.ssl.mask_ratio:>5.2f}"
        )
    return "\n".join(lines)


def _seeded(base: RunConfig, seed: int):
    """Return the base training config with its seed replaced."""
    return base.train.__class__(**{**vars(base.train), "seed": seed})
