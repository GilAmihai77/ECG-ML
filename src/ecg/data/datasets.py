"""Cohorts, label-efficiency subsets, and batches of normalised waveforms.

Three ideas, kept separate so each can be tested on its own:

* :class:`Cohort` names a set of records and the labels that go with them. The
  cohort definitions live in :mod:`ecg.data.ptbxl`; this module only assembles
  them into something a training loop can consume.
* :func:`nested_subsets` builds the label-efficiency ladder. The subsets are
  **nested** -- 20% is a subset of 50% is a subset of 100% -- so points on the
  curve differ only in how much data they have, not in which patients they drew.
* :class:`EcgBatches` yields batches. It holds the waveforms as ``int16`` on the
  target device and converts to normalised ``float32`` per batch, so the whole
  cohort costs a quarter of what ``float32`` would and no host-to-device copy
  happens during training.

Batches are **raw signal**, shaped ``(batch, 12, n_samples)`` -- not patches.
Patching belongs to the model, because the linear and convolutional embedders
consume the signal differently, and because SSL has to mask raw samples before
the convolutional stem rather than masking token embeddings after it.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from ecg.data.preprocess import SCALE, WaveformStore
from ecg.data.ptbxl import (
    SUPERCLASSES,
    apply_supervised_filter,
    label_matrix,
    ssl_pretraining_pool,
)

#: Fractions of the labelled training set used for the label-efficiency curve.
DEFAULT_FRACTIONS: tuple[float, ...] = (0.2, 0.5, 1.0)


@dataclass(frozen=True)
class Cohort:
    """A named set of records and their multi-label targets.

    Attributes:
        name: Human-readable identifier, used in logs and MLflow run names.
        ecg_ids: Record identifiers, in a fixed order.
        labels: ``float32`` array of shape ``(n_records, 5)``, columns ordered
            as :data:`ecg.data.ptbxl.SUPERCLASSES`.
    """

    name: str
    ecg_ids: np.ndarray
    labels: np.ndarray

    def __len__(self) -> int:
        """Number of records in the cohort."""
        return int(self.ecg_ids.shape[0])

    @property
    def positives(self) -> dict[str, int]:
        """Positive count per superclass, for reporting subset balance."""
        counts = self.labels.sum(axis=0).astype(int)
        return dict(zip(SUPERCLASSES, (int(c) for c in counts)))

    def subset(self, positions: np.ndarray, *, name: str) -> Cohort:
        """Take a subset by position.

        Args:
            positions: Indices into :attr:`ecg_ids`.
            name: Name for the new cohort.

        Returns:
            A cohort holding only the selected records.
        """
        return Cohort(name=name, ecg_ids=self.ecg_ids[positions], labels=self.labels[positions])


def build_cohorts(
    metadata: pd.DataFrame, *, policy: dict[str, Sequence[str]] | None = None
) -> dict[str, Cohort]:
    """Assemble every cohort the experiments need.

    Args:
        metadata: Metadata frame from :func:`ecg.data.ptbxl.load_metadata`.
        policy: Exclusion policy, passed through to
            :func:`ecg.data.ptbxl.apply_supervised_filter`. Defaults to the
            project's standard policy.

    Returns:
        Mapping with ``"train"``, ``"val"``, ``"test"`` (the filtered supervised
        cohorts) and ``"ssl"`` (training folds, unfiltered). The SSL cohort
        carries labels too, but pretraining must not read them.
    """
    kept, _ = apply_supervised_filter(metadata, policy=policy)
    cohorts: dict[str, Cohort] = {}
    for split in ("train", "val", "test"):
        frame = kept[kept["split"] == split]
        cohorts[split] = Cohort(
            name=split,
            ecg_ids=frame.index.to_numpy(dtype=np.int64),
            labels=label_matrix(frame).astype(np.float32),
        )

    pool = ssl_pretraining_pool(metadata)
    cohorts["ssl"] = Cohort(
        name="ssl",
        ecg_ids=pool.index.to_numpy(dtype=np.int64),
        labels=label_matrix(pool).astype(np.float32),
    )
    return cohorts


def nested_subsets(
    cohort: Cohort,
    fractions: Sequence[float] = DEFAULT_FRACTIONS,
    *,
    seed: int = 0,
) -> dict[float, Cohort]:
    """Build a nested ladder of labelled subsets for the label-efficiency curve.

    One permutation is drawn and every fraction takes a prefix of it, so smaller
    subsets are contained in larger ones. If each fraction were sampled
    independently, the points on the curve would differ in *which* patients they
    saw as well as how many, and the curve would confound the two.

    The training cohort already holds one record per patient, so record-disjoint
    and patient-disjoint are the same thing here; :func:`assert_patient_disjoint`
    checks that rather than assuming it.

    Args:
        cohort: Cohort to subsample, normally the supervised training cohort.
        fractions: Fractions to produce, in ``(0, 1]``.
        seed: Seed for the single permutation. Recorded in the config
            (integrity rule 4).

    Returns:
        Mapping from fraction to cohort, named ``"<cohort>@<pct>pct"``.

    Raises:
        ValueError: If a fraction is outside ``(0, 1]`` or rounds to zero
            records, which would silently produce an empty training run.
    """
    for fraction in fractions:
        if not 0.0 < fraction <= 1.0:
            raise ValueError(f"fraction must be in (0, 1], got {fraction}")

    order = np.random.default_rng(seed).permutation(len(cohort))
    subsets: dict[float, Cohort] = {}
    for fraction in sorted(fractions):
        count = int(round(len(cohort) * fraction))
        if count == 0:
            raise ValueError(f"fraction {fraction} selects 0 of {len(cohort)} records")
        subsets[fraction] = cohort.subset(
            order[:count], name=f"{cohort.name}@{fraction:.0%}"
        )
    return subsets


def holdout_split(
    cohort: Cohort, fraction: float, *, seed: int = 0
) -> tuple[Cohort, Cohort]:
    """Split a cohort into a larger part and a held-out part.

    Used to carve a reconstruction-validation slice out of the SSL pool. The
    slice is taken from the pool itself rather than from the supervised
    validation split, so pretraining never observes a cohort that is later used
    for model selection -- pretraining and fine-tuning would otherwise share an
    early-stopping signal.

    Args:
        cohort: Cohort to split.
        fraction: Fraction held out, in ``[0, 1)``.
        seed: Seed for the permutation (integrity rule 4).

    Returns:
        ``(kept, held_out)``. ``held_out`` is empty when ``fraction`` is zero.

    Raises:
        ValueError: If ``fraction`` is outside ``[0, 1)``.
    """
    if not 0.0 <= fraction < 1.0:
        raise ValueError(f"fraction must be in [0, 1), got {fraction}")

    order = np.random.default_rng(seed).permutation(len(cohort))
    n_held = int(round(len(cohort) * fraction))
    return (
        cohort.subset(order[n_held:], name=f"{cohort.name}-train"),
        cohort.subset(order[:n_held], name=f"{cohort.name}-holdout"),
    )


def assert_patient_disjoint(
    cohorts: dict[str, Cohort], metadata: pd.DataFrame, *, splits: Sequence[str] = ("train", "val", "test")
) -> None:
    """Fail if any patient appears in more than one split.

    Integrity rule 1, made executable at the point the model actually consumes
    the data rather than trusted from the published folds.

    Args:
        cohorts: Cohorts from :func:`build_cohorts`.
        metadata: Metadata frame carrying ``patient_id``.
        splits: Splits that must not share patients.

    Raises:
        AssertionError: If any patient spans two of ``splits``.
    """
    seen: dict[str, set[int]] = {}
    for split in splits:
        ids = cohorts[split].ecg_ids
        seen[split] = set(metadata.loc[ids, "patient_id"].to_numpy().tolist())

    for i, left in enumerate(splits):
        for right in splits[i + 1 :]:
            shared = seen[left] & seen[right]
            assert not shared, (
                f"patient leakage: {len(shared)} patient(s) in both {left} and "
                f"{right}, e.g. {sorted(shared)[:5]}"
            )


class EcgBatches:
    """Iterate over a cohort in batches of normalised waveforms.

    The whole cohort is held on the target device as ``int16`` -- 246 MB for the
    supervised training set at 100 Hz, 418 MB for the SSL pool -- so there is no
    DataLoader, no worker processes and no host-to-device copy in the training
    loop. That matters because a small model over a short sequence is bound by
    per-step overhead rather than by arithmetic.

    Args:
        store: Waveform store from :mod:`ecg.data.preprocess`.
        cohort: Records to iterate over.
        batch_size: Records per batch.
        shuffle: Shuffle record order each epoch. Use ``False`` for evaluation
            so predictions line up with :attr:`Cohort.ecg_ids`.
        device: Torch device for the resident tensors.
        seed: Seed for the shuffling generator (integrity rule 4).
        normalize: Apply per-record global normalisation. Off only for tests
            and for inspecting raw signal.
        drop_last: Drop a final short batch. Leave ``False`` for evaluation so
            every record is scored.
    """

    def __init__(
        self,
        store: WaveformStore,
        cohort: Cohort,
        *,
        batch_size: int = 256,
        shuffle: bool = True,
        device: str | torch.device = "cpu",
        seed: int = 0,
        normalize: bool = True,
        drop_last: bool = False,
    ) -> None:
        rows = store.rows_of(cohort.ecg_ids)
        block = np.ascontiguousarray(store.waveforms[rows])

        self.cohort = cohort
        self.scale = float(store.scale)
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.normalize = bool(normalize)
        self.drop_last = bool(drop_last)
        self.device = torch.device(device)
        self._generator = torch.Generator().manual_seed(int(seed))

        # (n, n_samples, 12) int16 -> kept int16 until a batch is drawn.
        self.waveforms = torch.from_numpy(block).to(self.device)
        self.labels = torch.from_numpy(cohort.labels).to(self.device)

    def __len__(self) -> int:
        """Number of batches per epoch."""
        n = len(self.cohort)
        if self.drop_last:
            return n // self.batch_size
        return (n + self.batch_size - 1) // self.batch_size

    @property
    def n_samples(self) -> int:
        """Samples per record."""
        return int(self.waveforms.shape[1])

    def _prepare(self, rows: torch.Tensor) -> torch.Tensor:
        """Convert stored counts to a normalised ``(batch, 12, n_samples)`` tensor."""
        signal = self.waveforms[rows].to(torch.float32) / self.scale
        signal = signal.permute(0, 2, 1)  # (batch, 12 leads, n_samples)
        if not self.normalize:
            return signal
        # One mean and std per record across all leads together, matching
        # ecg.data.preprocess.normalize_per_record. Biased std (unbiased=False)
        # so the torch and numpy paths agree exactly.
        flat = signal.reshape(signal.shape[0], -1)
        mean = flat.mean(dim=1).view(-1, 1, 1)
        std = flat.std(dim=1, unbiased=False).view(-1, 1, 1)
        return (signal - mean) / (std + 1e-6)

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        """Yield ``(signal, labels)`` batches for one epoch."""
        n = len(self.cohort)
        if self.shuffle:
            order = torch.randperm(n, generator=self._generator).to(self.device)
        else:
            order = torch.arange(n, device=self.device)

        limit = (n // self.batch_size) * self.batch_size if self.drop_last else n
        for start in range(0, limit, self.batch_size):
            rows = order[start : start + self.batch_size]
            yield self._prepare(rows), self.labels[rows]


def describe_cohorts(cohorts: dict[str, Cohort]) -> pd.DataFrame:
    """Tabulate cohort sizes and class balance.

    Args:
        cohorts: Cohorts from :func:`build_cohorts` or :func:`nested_subsets`.

    Returns:
        DataFrame indexed by cohort name with record counts and positives per
        superclass.
    """
    rows = []
    for name, cohort in cohorts.items():
        row: dict[str, object] = {"n_records": len(cohort)}
        row.update(cohort.positives)
        rows.append(pd.Series(row, name=str(name)))
    return pd.DataFrame(rows).rename_axis("cohort")
