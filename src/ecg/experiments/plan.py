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

**Every experiment is replicated across seeds**, and the study reports mean and
standard deviation over them rather than one number per arm. A single run's
macro AUROC moves by more than the effects being compared, so a one-seed table
cannot distinguish "conv beats linear" from "this initialisation beat that one".

A replicate seed drives *both* sources of run-to-run variation at once:

* ``train.seed`` -- weight initialisation, batch order and the SSL mask draw.
* ``subset_seed`` -- **which** labelled records the label fraction happens to
  draw, and which records are held out of the SSL pool.

Including the subset draw is deliberate. In a label-scarcity study it is
typically the larger of the two, and leaving it fixed would report an interval
that says "if I re-initialised" when the question is "if I had a different 20%
of the labels". It costs nothing in the paired comparison: within one seed all
four arms still see exactly the same records, so the per-seed SSL-minus-scratch
difference is unaffected and only the marginal spread widens.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import reduce
from math import gcd
from typing import Literal, Sequence

from ecg.models.config import Embedder
from ecg.training.config import DEFAULT_VARIANT, RunConfig
from ecg.training.loops import snapshot_name

RunKind = Literal["pretrain", "supervised"]

#: Label fractions for the label-efficiency curve.
DEFAULT_FRACTIONS: tuple[float, ...] = (0.2, 0.5, 1.0)

#: Mask ratios compared before the main study.
DEFAULT_MASK_RATIOS: tuple[float, ...] = (0.3, 0.5, 0.7)

#: Pretraining budgets compared by :func:`ssl_budget_plan`, in epochs. The
#: study's default of 50 is 3,250 SSL steps at batch 256, which is very few for
#: masked reconstruction; these bracket the range worth asking about.
DEFAULT_SSL_BUDGETS: tuple[int, ...] = (100, 200, 400, 600)

#: Replicate seeds. Five is the smallest n for which a standard deviation is
#: worth printing and the paired interval is not dominated by its own error;
#: it also multiplies the GPU bill by five, so ``--dry-run`` prints the count.
DEFAULT_SEEDS: tuple[int, ...] = (0, 1, 2, 3, 4)

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
        checkpoint_name: File to load from inside ``depends_on``'s directory,
            instead of the selected ``pretrain_best.pt``. Only
            :func:`ssl_budget_plan` sets it, to fine-tune from a mid-run
            snapshot; leaving it ``None`` is what every arm of the study does.
    """

    name: str
    kind: RunKind
    config: RunConfig
    depends_on: str | None = None
    checkpoint_name: str | None = None

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


def seed_name(stem: str, seed: int, *, several: bool) -> str:
    """Suffix a run name with its replicate seed.

    Like :func:`variant_name`, this exists because a run name is also its
    output directory and ``run_plan`` skips any directory holding a
    ``result.json``. Without the suffix all five replicates would write to one
    directory, the first would be reported five times, and nothing would say so.

    A single-seed plan is left unsuffixed, so a study already on Drive keeps its
    names and its finished runs still resume. Asking for several seeds renames
    all of them, seed 0 included: one re-run, in exchange for a table in which
    no row can be mistaken for an average.

    Args:
        stem: The name the run would have had, e.g. ``"sup-linear-ssl-f020"``.
        seed: The replicate seed.
        several: Whether the plan holds more than one seed.

    Returns:
        ``stem`` when ``several`` is false, ``"<stem>-s<seed>"`` otherwise.
    """
    return f"{stem}-s{seed}" if several else stem


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
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
    share_pretraining: bool = False,
) -> list[RunSpec]:
    """Build the full study: pretraining runs, then the four arms, every seed.

    Ordering matters -- pretraining runs come first because the SSL arms depend
    on their checkpoints -- and :func:`ecg.experiments.runner.run_plan` relies
    on it rather than resolving a graph.

    Every supervised run at a given seed shares that seed's ``subset_seed``, so
    at a given fraction all four arms train on exactly the same records.
    Varying it *between arms* would make the label-efficiency curve confound
    sample size with which patients were drawn; varying it *between seeds* is
    the point, and is what puts the subset draw inside the reported spread.

    Pretraining is replicated per seed by default, so the SSL arm's interval
    covers the pretraining run as well as the fine-tune. Sharing one encoder
    across five fine-tunes would give the SSL arm a narrower interval than the
    scratch arm for a reason that has nothing to do with SSL -- the two error
    bars would no longer be measuring the same thing.

    Args:
        base: Base configuration. Its ``model.embedder`` is overridden per arm;
            everything else is inherited unchanged.
        fractions: Label fractions for the curve.
        embedders: Embedders to compare.
        seeds: Replicate seeds. Each drives weights, batch order, masking and
            the label-subset draw.
        share_pretraining: Pretrain once per embedder, at ``seeds[0]``, and
            fine-tune every seed from it. Cuts the study by one pretraining run
            per extra seed, at the cost of an SSL interval that understates its
            own uncertainty. Say so in the write-up if you use it.

    Returns:
        The runs, pretraining first; within an arm, one run per seed.

    Raises:
        ValueError: If ``seeds`` is empty or holds a repeat -- a repeated seed
            would produce two runs with one name, and the second would be
            skipped as already complete.
    """
    seeds = _checked_seeds(seeds)
    several = len(seeds) > 1
    pretrain_seeds = seeds[:1] if share_pretraining else seeds
    pretrain_several = len(pretrain_seeds) > 1

    def encoder_for(embedder: str, seed: int) -> str:
        """Name the pretraining run an SSL arm at this seed fine-tunes from."""
        source_seed = pretrain_seeds[0] if share_pretraining else seed
        return variant_name(
            base.variant,
            seed_name(
                pretrain_name(embedder, base.ssl.mask_ratio),
                source_seed,
                several=pretrain_several,
            ),
        )

    specs: list[RunSpec] = []

    for embedder in embedders:
        for seed in pretrain_seeds:
            name = encoder_for(embedder, seed)
            specs.append(
                RunSpec(
                    name=name,
                    kind="pretrain",
                    config=_replicate(base, seed).with_(
                        name=name,
                        model=base.model.for_arm(embedder),
                        pretrained_from=None,
                    ),
                )
            )

    for fraction in fractions:
        for embedder in embedders:
            for pretrained in (False, True):
                for seed in seeds:
                    source = encoder_for(embedder, seed) if pretrained else None
                    name = variant_name(
                        base.variant,
                        seed_name(
                            supervised_name(embedder, pretrained, fraction),
                            seed,
                            several=several,
                        ),
                    )
                    specs.append(
                        RunSpec(
                            name=name,
                            kind="supervised",
                            config=_replicate(base, seed).with_(
                                name=name,
                                model=base.model.for_arm(embedder),
                                label_fraction=fraction,
                                # Filled in by the runner once the checkpoint
                                # path is known; kept as the run name here so
                                # the plan is printable without a filesystem.
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
    seeds: tuple[int, ...] = (0,),
) -> list[RunSpec]:
    """Build the mask-ratio selection.

    One embedder only, and one label fraction only. Tuning the mask separately
    per arm would fold the SSL setup into what is being compared, so the ratio
    chosen here is applied unchanged to all four arms.

    Seeds default to one, unlike :func:`experiment_plan`. This is a selection
    step run on validation, not a reported result, and its cost is paid before
    the study rather than inside it. Pass several if the candidate ratios come
    out within noise of each other -- which is the case worth spending on,
    because picking on a single seed then is picking at random.

    Args:
        base: Base configuration.
        mask_ratios: Candidate ratios.
        embedder: Which arm to select on.
        fraction: Label fraction to fine-tune at. The smallest one, where SSL's
            effect is largest and the runs are cheapest.
        seeds: Replicate seeds.

    Returns:
        The runs, each pretraining immediately followed by its fine-tune.

    Raises:
        ValueError: If ``seeds`` is empty or holds a repeat.
    """
    seeds = _checked_seeds(seeds)
    several = len(seeds) > 1
    specs: list[RunSpec] = []
    for ratio in mask_ratios:
        ssl_config = base.ssl.__class__(mask_ratio=ratio, mask_span=base.ssl.mask_span)
        for seed in seeds:
            name = variant_name(
                base.variant,
                seed_name(pretrain_name(embedder, ratio), seed, several=several),
            )
            specs.append(
                RunSpec(
                    name=name,
                    kind="pretrain",
                    config=_replicate(base, seed).with_(
                        name=name,
                        model=base.model.for_arm(embedder),
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
                    config=_replicate(base, seed).with_(
                        name=finetune,
                        model=base.model.for_arm(embedder),
                        ssl=ssl_config,
                        label_fraction=fraction,
                        pretrained_from=name,
                    ),
                    depends_on=name,
                )
            )
    return specs


def budget_name(embedder: str, mask_ratio: float, epochs: int) -> str:
    """Name a pretraining run that is one rung of a budget ladder.

    Args:
        embedder: ``"linear"`` or ``"conv"``.
        mask_ratio: Masking ratio.
        epochs: The pretraining budget.

    Returns:
        A name like ``"pretrain-linear-m50-e0600"``.
    """
    return f"{pretrain_name(embedder, mask_ratio)}-e{epochs:04d}"


def ssl_budget_plan(
    base: RunConfig,
    *,
    budgets: tuple[int, ...] = DEFAULT_SSL_BUDGETS,
    embedder: Embedder = "linear",
    fraction: float = 0.2,
    seeds: tuple[int, ...] = (0,),
    from_snapshots: bool = False,
) -> list[RunSpec]:
    """Ask how long pretraining is worth running, before paying for it.

    The study's SSL budget multiplies the cost of the ten pretraining runs
    directly, and held-out reconstruction loss will not tell you where to set
    it: on 17,418 records it keeps falling long after the representation has
    stopped getting more useful, so ``pretrain_best.pt`` in a long run is
    essentially the last epoch. The only honest signal is downstream -- fine-
    tune from encoders of different ages and look at validation.

    Two ways to get that, and they answer slightly different questions.

    **Separate runs** (the default). One pretraining per budget, each annealed
    across its own budget, then one fine-tune each. This is the comparison you
    actually want -- "is a 600-epoch run better than a 150-epoch run?" -- with
    every rung a run you could really ship. It costs ``sum(budgets)`` epochs.

    **From snapshots** (``from_snapshots=True``). One pretraining at the
    longest budget, snapshotting as it goes, then one fine-tune per snapshot.
    Costs ``max(budgets)`` epochs, so roughly half. But the short rungs are
    taken mid-anneal, at a learning rate still near peak, so they understate
    what a real run of that length would achieve. It tells you where the curve
    flattens, not what a 150-epoch run is worth. Good for a first look; do not
    put its numbers in the write-up as if they were the other thing.

    Selection is on validation, and the chosen budget is then frozen for every
    arm of the study -- as with the mask ratio, tuning it per arm would fold
    the SSL setup into what is being compared.

    Args:
        base: Base configuration.
        budgets: Pretraining lengths to compare, in epochs.
        embedder: Which arm to select on.
        fraction: Label fraction to fine-tune at. The smallest one, where SSL's
            effect is largest and the fine-tunes are cheapest.
        seeds: Replicate seeds. One is usually enough to see the shape.
        from_snapshots: Take the cheap route described above.

    Returns:
        The runs, each pretraining before the fine-tunes that depend on it.

    Raises:
        ValueError: If ``budgets`` is empty or holds a value below one, or if
            ``seeds`` is empty or holds a repeat.
    """
    seeds = _checked_seeds(seeds)
    several = len(seeds) > 1
    ladder = tuple(sorted(set(budgets)))
    if not ladder:
        raise ValueError("at least one budget is required")
    if ladder[0] < 1:
        raise ValueError(f"budgets must be positive, got {sorted(budgets)}")

    ratio = base.ssl.mask_ratio
    specs: list[RunSpec] = []

    for seed in seeds:
        replicate = _replicate(base, seed)
        if from_snapshots:
            # Every rung must land on a snapshot, so snapshot at their gcd.
            every = reduce(gcd, ladder)
            longest = ladder[-1]
            source = variant_name(
                base.variant,
                seed_name(budget_name(embedder, ratio, longest), seed, several=several),
            )
            specs.append(
                RunSpec(
                    name=source,
                    kind="pretrain",
                    config=replicate.with_(
                        name=source,
                        model=base.model.for_arm(embedder),
                        train=_budgeted(replicate, longest, snapshot_every=every),
                        pretrained_from=None,
                    ),
                )
            )
            for rung in ladder:
                name = f"{source}-snap{rung:04d}-f{int(round(fraction * 100)):03d}"
                specs.append(
                    RunSpec(
                        name=name,
                        kind="supervised",
                        config=replicate.with_(
                            name=name,
                            model=base.model.for_arm(embedder),
                            label_fraction=fraction,
                            pretrained_from=source,
                        ),
                        depends_on=source,
                        checkpoint_name=snapshot_name(rung),
                    )
                )
            continue

        for rung in ladder:
            source = variant_name(
                base.variant,
                seed_name(budget_name(embedder, ratio, rung), seed, several=several),
            )
            specs.append(
                RunSpec(
                    name=source,
                    kind="pretrain",
                    config=replicate.with_(
                        name=source,
                        model=base.model.for_arm(embedder),
                        train=_budgeted(replicate, rung),
                        pretrained_from=None,
                    ),
                )
            )
            name = f"{source}-f{int(round(fraction * 100)):03d}"
            specs.append(
                RunSpec(
                    name=name,
                    kind="supervised",
                    config=replicate.with_(
                        name=name,
                        model=base.model.for_arm(embedder),
                        label_fraction=fraction,
                        pretrained_from=source,
                    ),
                    depends_on=source,
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
    seeds = sorted({spec.config.train.seed for spec in specs})
    # The seed count multiplies the bill, so it goes in the first line a
    # --dry-run prints rather than having to be counted off the table.
    lines = [
        f"{len(specs)} runs, variant {', '.join(variants)}, "
        f"{len(seeds)} seed(s) {seeds}",
        "",
    ]
    header = (
        f"{'run':52s} {'kind':11s} {'arm':12s} {'labels':>7s} {'mask':>5s} "
        f"{'seed':>5s} {'epochs':>7s}"
    )
    lines += [header, "-" * len(header)]
    total = 0
    for spec in specs:
        labels = "-" if spec.kind == "pretrain" else f"{spec.config.label_fraction:.0%}"
        budget = (
            spec.config.train.for_pretraining().epochs
            if spec.kind == "pretrain"
            else spec.config.train.epochs
        )
        total += budget
        lines.append(
            f"{spec.name:52s} {spec.kind:11s} {spec.arm:12s} {labels:>7s} "
            f"{spec.config.ssl.mask_ratio:>5.2f} {spec.config.train.seed:>5d} "
            f"{budget:>7d}"
        )
    # The number that actually sizes the bill, since pretraining and
    # fine-tuning no longer share a budget and a run count no longer implies one.
    lines += ["", f"{total:,} epochs total (ceiling; early stopping may cut it)"]
    return "\n".join(lines)


def _checked_seeds(seeds: Sequence[int]) -> tuple[int, ...]:
    """Validate a seed list.

    Args:
        seeds: Requested replicate seeds.

    Returns:
        The seeds as a tuple, order preserved.

    Raises:
        ValueError: If empty, or if a seed repeats. A repeat is worth an error
            rather than a silent de-duplication: the two runs would share a
            name, the second would be skipped as already complete, and the
            study would report ``n=5`` over four distinct runs.
    """
    seeds = tuple(seeds)
    if not seeds:
        raise ValueError("at least one seed is required")
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"seeds must be distinct, got {list(seeds)}")
    return seeds


def _budgeted(base: RunConfig, epochs: int, *, snapshot_every: int = 0):
    """Return a training config whose *pretraining* budget is ``epochs``.

    ``ssl_schedule_epochs`` is reset to 0, meaning "anneal across the whole SSL
    budget". Carrying the base value through would either stretch the decay
    past a short rung or fail validation outright, and every rung of a ladder
    has to be a run that annealed properly on its own terms.

    Args:
        base: Configuration to derive from.
        epochs: Pretraining epochs for this rung.
        snapshot_every: Snapshot period, or 0 for none.

    Returns:
        A new training config; the supervised ``epochs`` is left alone.
    """
    return base.train.__class__(
        **{
            **vars(base.train),
            "ssl_epochs": epochs,
            "ssl_schedule_epochs": 0,
            "snapshot_every": snapshot_every,
        }
    )


def _replicate(base: RunConfig, seed: int) -> RunConfig:
    """Return the base configuration set to one replicate seed.

    Both seeds move together: ``train.seed`` for weights, batch order and
    masking, ``subset_seed`` for the label-fraction draw and the SSL holdout.
    See the module docstring for why the subset draw belongs inside the
    replicate rather than being held fixed across it.

    Args:
        base: Base configuration.
        seed: The replicate seed.

    Returns:
        A copy with both seeds set.
    """
    return base.with_(
        train=base.train.__class__(**{**vars(base.train), "seed": seed}),
        subset_seed=seed,
    )
