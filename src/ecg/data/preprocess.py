"""Turn PTB-XL's WFDB records into one compact array for training.

Reading 21,799 pairs of WFDB files once per epoch is slow everywhere and
impossible to justify on an ephemeral cloud filesystem, so the waveforms are
converted once into a single array that loads in seconds and fits in memory.

The conversion is **lossless**. PTB-XL stores 16-bit samples with
``adc_gain=1000`` and ``baseline=0``, so an integer count is exactly one
microvolt and ``round(millivolts * 1000)`` reproduces the stored integers
bit for bit. :data:`SCALE` records that relationship; nothing here rounds away
information that the published dataset contains.

This module is the **only** place the training pipeline touches ``wfdb``. Once
the store is built, training reads the array and the metadata CSV alone, which
keeps the cloud environment small and fast to provision.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from ecg.data.download import DatasetLayout, resolve_layout
from ecg.data.ptbxl import SamplingRate, load_metadata, load_waveform

#: Integer counts per millivolt. PTB-XL's own ADC gain, so storing
#: ``round(mV * SCALE)`` as ``int16`` is exact rather than approximate.
SCALE: int = 1000

#: Lead order as ``wfdb`` returns it for PTB-XL. Stored so that a consumer of
#: the array never has to guess which column is which, and so a future dataset
#: with a different order fails loudly instead of silently mislabelling leads.
LEADS: tuple[str, ...] = (
    "I", "II", "III", "AVR", "AVL", "AVF",
    "V1", "V2", "V3", "V4", "V5", "V6",
)

#: Filenames inside a store directory.
ARRAY_NAME: str = "waveforms.npy"
IDS_NAME: str = "ecg_ids.npy"
META_NAME: str = "store.json"
FAILURES_NAME: str = "failures.csv"


@dataclass(frozen=True)
class WaveformStore:
    """A preprocessed waveform array and the identifiers that index it.

    Attributes:
        waveforms: ``int16`` array of shape ``(n_records, n_samples, 12)`` in
            microvolts.
        ecg_ids: ``ecg_id`` for each row, in row order.
        sampling_rate: Sampling rate of the stored waveforms, in Hz.
        scale: Integer counts per millivolt; see :data:`SCALE`.
        leads: Lead names in column order.
    """

    waveforms: np.ndarray
    ecg_ids: np.ndarray
    sampling_rate: int
    scale: int = SCALE
    leads: tuple[str, ...] = LEADS

    def __len__(self) -> int:
        """Number of records in the store."""
        return int(self.waveforms.shape[0])

    @property
    def n_samples(self) -> int:
        """Samples per record."""
        return int(self.waveforms.shape[1])

    def row_of(self, ecg_id: int) -> int:
        """Find the array row holding one record.

        Args:
            ecg_id: Record identifier.

        Returns:
            Row index into :attr:`waveforms`.

        Raises:
            KeyError: If the record is not in the store.
        """
        matches = np.flatnonzero(self.ecg_ids == ecg_id)
        if matches.size == 0:
            raise KeyError(f"ecg_id {ecg_id} is not in this store")
        return int(matches[0])

    def rows_of(self, ecg_ids: Sequence[int]) -> np.ndarray:
        """Map many record identifiers to array rows, preserving order.

        Args:
            ecg_ids: Record identifiers to look up.

        Returns:
            Integer array of row indices.

        Raises:
            KeyError: If any identifier is missing, naming the first few.
        """
        lookup = pd.Index(self.ecg_ids)
        positions = lookup.get_indexer(np.asarray(ecg_ids))
        missing = np.asarray(ecg_ids)[positions < 0]
        if missing.size:
            raise KeyError(f"{missing.size} ecg_id(s) not in store, e.g. {missing[:5].tolist()}")
        return positions

    def millivolts(self, rows: np.ndarray | int | None = None) -> np.ndarray:
        """Read records back as millivolts.

        Args:
            rows: Row index, array of row indices, or ``None`` for every record.

        Returns:
            ``float32`` array in millivolts, shaped ``(n_samples, 12)`` for a
            single row and ``(n, n_samples, 12)`` otherwise.
        """
        block = self.waveforms if rows is None else self.waveforms[rows]
        return block.astype(np.float32) / self.scale

    def save(self, dest: Path) -> Path:
        """Write the store to a directory.

        Args:
            dest: Directory to create and write into.

        Returns:
            The directory written to.
        """
        dest = Path(dest)
        dest.mkdir(parents=True, exist_ok=True)
        np.save(dest / ARRAY_NAME, self.waveforms)
        np.save(dest / IDS_NAME, self.ecg_ids)
        (dest / META_NAME).write_text(
            json.dumps(
                {
                    "sampling_rate": self.sampling_rate,
                    "scale": self.scale,
                    "leads": list(self.leads),
                    "n_records": len(self),
                    "n_samples": self.n_samples,
                    "dtype": str(self.waveforms.dtype),
                    "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return dest

    @classmethod
    def load(cls, source: Path, *, mmap: bool = True) -> WaveformStore:
        """Read a store written by :meth:`save`.

        Args:
            source: Directory containing the store.
            mmap: Memory-map the array instead of reading it into RAM. Keeps
                start-up fast; set ``False`` to force a resident copy, which is
                what you want once the array is small enough to hold.

        Returns:
            The loaded store.

        Raises:
            FileNotFoundError: If the directory is missing an expected file.
        """
        source = Path(source)
        for name in (ARRAY_NAME, IDS_NAME, META_NAME):
            if not (source / name).exists():
                raise FileNotFoundError(f"{source} is not a waveform store; missing {name}")

        meta = json.loads((source / META_NAME).read_text(encoding="utf-8"))
        return cls(
            waveforms=np.load(source / ARRAY_NAME, mmap_mode="r" if mmap else None),
            ecg_ids=np.load(source / IDS_NAME),
            sampling_rate=int(meta["sampling_rate"]),
            scale=int(meta["scale"]),
            leads=tuple(meta["leads"]),
        )


def to_counts(millivolts: np.ndarray, *, scale: int = SCALE) -> np.ndarray:
    """Convert a millivolt waveform to the stored integer representation.

    Args:
        millivolts: Waveform in mV, any shape.
        scale: Integer counts per millivolt.

    Returns:
        ``int16`` array of the same shape.

    Raises:
        ValueError: If any value would overflow ``int16``, which would corrupt
            the record silently rather than loudly.
    """
    counts = np.round(np.nan_to_num(millivolts, nan=0.0) * scale)
    limit = np.iinfo(np.int16)
    if counts.min() < limit.min or counts.max() > limit.max:
        raise ValueError(
            f"waveform exceeds int16 range at scale={scale}: "
            f"[{counts.min():.0f}, {counts.max():.0f}]"
        )
    return counts.astype(np.int16)


def normalize_per_record(waveform: np.ndarray, *, eps: float = 1e-6) -> np.ndarray:
    """Standardise each record using one mean and standard deviation.

    The statistics are taken over **all 12 leads together**, not per lead. Per-
    lead scaling would equalise the leads and destroy their relative amplitudes,
    which is how cardiac axis is read and how hypertrophy is diagnosed -- the
    Sokolow-Lyon criterion is a sum of amplitudes across V1 and V5/V6. One
    global factor conditions the reconstruction loss without touching those
    ratios.

    Statistics come from the record itself, so no information crosses between
    records and there is nothing for the test split to leak (integrity rule 2).

    Args:
        waveform: Array shaped ``(..., n_samples, n_leads)`` in any unit.
        eps: Guard added to the standard deviation, so a flat record returns
            zeros instead of ``inf``.

    Returns:
        ``float32`` array of the same shape, zero mean and unit variance per
        record.
    """
    values = np.asarray(waveform, dtype=np.float32)
    axes = (-2, -1)
    mean = values.mean(axis=axes, keepdims=True)
    std = values.std(axis=axes, keepdims=True)
    return ((values - mean) / (std + eps)).astype(np.float32)


def build_waveform_store(
    layout: DatasetLayout,
    metadata: pd.DataFrame,
    *,
    sampling_rate: SamplingRate = 100,
    loader: Callable[..., tuple[np.ndarray, list[str]]] = load_waveform,
    progress_every: int = 2000,
) -> tuple[WaveformStore, pd.DataFrame]:
    """Read every record in ``metadata`` into one array.

    Args:
        layout: Resolved dataset paths.
        metadata: Metadata frame from :func:`ecg.data.ptbxl.load_metadata`,
            already restricted to whatever cohort should be stored. Row order in
            the array follows this frame.
        sampling_rate: 100 or 500 Hz.
        loader: Waveform reader, injected so the build can be tested without
            WFDB files on disk.
        progress_every: Print a progress line this often; ``0`` disables it.

    Returns:
        Tuple of ``(store, failures)``. Failed records are reported rather than
        silently skipped (integrity rule 8) and are absent from the store.

    Raises:
        ValueError: If a record has an unexpected lead order or length, which
            would misalign the array without any visible error later.
    """
    expected_samples = int(sampling_rate * 10)
    buffer = np.zeros((len(metadata), expected_samples, len(LEADS)), dtype=np.int16)
    kept_ids: list[int] = []
    failures: list[dict] = []
    written = 0

    for position, (ecg_id, row) in enumerate(metadata.iterrows(), start=1):
        try:
            signal, lead_names = loader(layout, row, sampling_rate=sampling_rate)
            if tuple(lead_names) != LEADS:
                raise ValueError(f"unexpected lead order {tuple(lead_names)}")
            if signal.shape != (expected_samples, len(LEADS)):
                raise ValueError(
                    f"expected shape {(expected_samples, len(LEADS))}, got {signal.shape}"
                )
            n_missing = int((~np.isfinite(signal)).sum())
            buffer[written] = to_counts(signal)
        except Exception as error:  # reported, never silently dropped
            failures.append(
                {"ecg_id": int(ecg_id), "reason": f"{type(error).__name__}: {error}"}
            )
            continue

        if n_missing:
            failures.append(
                {"ecg_id": int(ecg_id), "reason": f"non-finite samples: {n_missing} (zeroed)"}
            )
        kept_ids.append(int(ecg_id))
        written += 1

        if progress_every and position % progress_every == 0:
            print(f"  {position:,}/{len(metadata):,} records")

    store = WaveformStore(
        waveforms=buffer[:written],
        ecg_ids=np.asarray(kept_ids, dtype=np.int64),
        sampling_rate=int(sampling_rate),
    )
    manifest = pd.DataFrame(failures, columns=["ecg_id", "reason"])
    return store, manifest


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(
        description="Convert PTB-XL WFDB records into one compact array."
    )
    parser.add_argument(
        "--dataset", type=Path, default=Path("data/ptbxl"), help="PTB-XL root"
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=None,
        help="store directory (default: data/store_<rate>hz)",
    )
    parser.add_argument(
        "--sampling-rate",
        type=int,
        choices=(100, 500),
        default=100,
        help="100 Hz keeps every frequency the 5 superclasses need and is 5x "
        "smaller; 500 Hz resolves QRS duration more finely (default: 100)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Build a waveform store from the command line.

    Args:
        argv: Argument vector, defaulting to ``sys.argv[1:]``.

    Returns:
        Process exit code.
    """
    args = build_parser().parse_args(argv)
    dest = args.dest or Path("data") / f"store_{args.sampling_rate}hz"

    layout = resolve_layout(args.dataset)
    metadata = load_metadata(layout)
    print(f"{len(metadata):,} records at {args.sampling_rate} Hz -> {dest}")

    store, failures = build_waveform_store(
        layout, metadata, sampling_rate=args.sampling_rate
    )
    store.save(dest)

    size_mb = store.waveforms.nbytes / 1e6
    print(f"stored {len(store):,} records, {store.n_samples} samples, {size_mb:.0f} MB")
    if len(failures):
        failures.to_csv(dest / FAILURES_NAME, index=False)
        print(f"{len(failures)} record(s) reported in {dest / FAILURES_NAME}")
    else:
        print("no failures")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
