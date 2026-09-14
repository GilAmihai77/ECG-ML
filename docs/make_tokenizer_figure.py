"""Regenerate the waveform data embedded in ``tokenizer_comparison.html``.

The figure explains the two tokenizers on real PTB-XL traces, so its data has to
come from the same detection pipeline the tokenizers use -- otherwise the
picture and the implementation can drift apart without anyone noticing.

This script rewrites only the ``<script id="ecg-data">`` block inside the HTML
and leaves the markup alone, so the page stays the source of truth for layout
while the numbers stay reproducible::

    python docs/make_tokenizer_figure.py --dataset data/ptbxl

Records are pinned by ``ecg_id`` (see :data:`CASES`) so regenerating the figure
is stable and reviewable rather than dependent on a search order.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ecg.data.download import DatasetLayout, resolve_layout
from ecg.data.ptbxl import apply_supervised_filter, load_metadata, load_waveform

#: Detection runs at 500 Hz always, on lead II, matching the frozen tokenizer
#: settings. Peak indices are reported in seconds so the figure is
#: resolution-independent.
DETECT_RATE: int = 500
LEAD_II: int = 1
DURATION_S: float = 10.0

#: Token geometry under test. 10 s / 12 gives the fixed arm 12 patches against a
#: median of 12 beats, so both arms land near 144 tokens once 12 leads are
#: stacked -- the point of choosing it. Note 10 s does not divide into 12 whole
#: samples at either resolution (416.67 at 500 Hz, 83.33 at 100 Hz), so the
#: implementation pads the final patch rather than silently resizing patches.
PATCH_MS: float = 10_000.0 / 12.0
HALF_MS: float = PATCH_MS / 2.0
N_FIXED: int = 12

#: Records pinned by ``ecg_id`` so the figure is stable across regenerations,
#: chosen to span the heart rates PTB-XL actually contains. The contrast is the
#: point: beat count, and therefore RR sequence length, moves with rate while
#: the fixed arm's token count never does -- and both extremes degrade the RR
#: tokenizer, in opposite ways.
#:
#: ``ecg_id 16753`` is close to the fastest record in the cleaned cohort; rates
#: above ~176 bpm do not occur in PTB-XL, so it is the honest upper bound rather
#: than a round number.
CASES: tuple[tuple[str, str, int], ...] = (
    ("brady", "Bradycardic", 2),
    ("normal", "Normal rate", 1),
    ("tachy", "Tachycardic", 108),
    ("extreme", "Extreme tachycardia", 16753),
)

DATA_TAG_OPEN = '<script id="ecg-data">'
DATA_TAG_CLOSE = "</script>"


def detect_peaks(signal: np.ndarray, *, rate: int = DETECT_RATE) -> np.ndarray:
    """Locate R peaks with the project's frozen detection settings.

    Args:
        signal: One lead of the waveform, in mV.
        rate: Sampling rate of ``signal`` in Hz.

    Returns:
        R-peak positions as sample indices.
    """
    import neurokit2 as nk

    cleaned = nk.ecg_clean(signal, sampling_rate=rate)
    _, info = nk.ecg_peaks(cleaned, sampling_rate=rate, method="neurokit")
    return np.asarray(info["ECG_R_Peaks"], dtype=float)


def rr_tokens(peaks: np.ndarray, n_samples: int, *, rate: int = DETECT_RATE) -> list[dict]:
    """Build the RR arm's token geometry for one record.

    Each token wants :data:`PATCH_MS` centred on its R peak, and is clipped to
    the midpoints between neighbouring beats so tokens tile the record exactly
    and never overlap. Overlapping tokens would leak a masked token's content
    into its neighbour during masked-reconstruction pretraining. The first and
    last tokens are clipped to the record edges instead, so no signal is lost.

    Args:
        peaks: R-peak sample indices, ascending.
        n_samples: Length of the record in samples.
        rate: Sampling rate in Hz.

    Returns:
        One dict per beat with times in seconds: the R peak, the kept span
        (``start``/``end``), the clip bounds, the unclipped span the token
        wanted, the fraction of the buffer filled, and whether it is an edge
        token.
    """
    half = HALF_MS * rate / 1000.0
    midpoints = [0.0]
    midpoints += [(peaks[i] + peaks[i + 1]) / 2.0 for i in range(len(peaks) - 1)]
    midpoints += [float(n_samples)]

    tokens: list[dict] = []
    for i, peak in enumerate(peaks):
        left_bound, right_bound = midpoints[i], midpoints[i + 1]
        left = max(peak - half, left_bound)
        right = min(peak + half, right_bound)
        tokens.append(
            {
                "r": round(float(peak) / rate, 4),
                "start": round(float(left) / rate, 4),
                "end": round(float(right) / rate, 4),
                "clipLeft": round(float(left_bound) / rate, 4),
                "clipRight": round(float(right_bound) / rate, 4),
                "wantStart": round(float(peak - half) / rate, 4),
                "wantEnd": round(float(peak + half) / rate, 4),
                "fill": round(float(right - left) / (2 * half), 4),
                "edge": i == 0 or i == len(peaks) - 1,
            }
        )
    return tokens


def load_record(
    layout: DatasetLayout, metadata: pd.DataFrame, ecg_id: int
) -> tuple[pd.Series, np.ndarray, np.ndarray, float]:
    """Load one record's lead II, its R peaks and its mean heart rate.

    Args:
        layout: Resolved dataset paths.
        metadata: Full metadata frame from :func:`load_metadata`.
        ecg_id: Record to load.

    Returns:
        Tuple of ``(row, lead_ii, peaks, heart_rate)``.

    Raises:
        ValueError: If the record carries non-finite samples or too few peaks to
            form RR intervals, which would make it unusable in the figure.
    """
    row = metadata.loc[ecg_id]
    signal, _ = load_waveform(layout, row, sampling_rate=DETECT_RATE)
    lead = signal[:, LEAD_II]
    if not np.isfinite(lead).all():
        raise ValueError(f"ecg_id {ecg_id} contains non-finite samples")
    peaks = detect_peaks(lead)
    if len(peaks) < 4:
        raise ValueError(f"ecg_id {ecg_id} yielded only {len(peaks)} R peaks")
    heart_rate = 60.0 / float(np.mean(np.diff(peaks) / DETECT_RATE))
    return row, lead, peaks, heart_rate


def build_payload(dataset: Path) -> dict:
    """Assemble everything the figure needs.

    Args:
        dataset: Path to the extracted PTB-XL dataset root.

    Returns:
        The JSON-serialisable payload assigned to ``window.ECG_DATA``.
    """
    layout = resolve_layout(dataset)
    metadata = load_metadata(layout)
    cohort, _ = apply_supervised_filter(metadata)

    payload: dict = {
        "rate": DETECT_RATE,
        "durationS": DURATION_S,
        "patchMs": round(PATCH_MS, 2),
        "halfMs": round(HALF_MS, 2),
        "nFixed": N_FIXED,
        "cases": {},
    }

    payload["order"] = [key for key, _, _ in CASES]
    for name, title, ecg_id in CASES:
        if ecg_id not in cohort.index:
            print(f"  {name}: ecg_id {ecg_id} is not in the cleaned cohort, skipped")
            continue
        try:
            row, lead, peaks, heart_rate = load_record(layout, metadata, ecg_id)
        except ValueError as err:
            print(f"  {name}: {err}, skipped")
            continue

        rr = np.diff(peaks) / DETECT_RATE
        tokens = rr_tokens(peaks, len(lead))
        # Coverage and fill are the two ways the geometry degrades at the
        # extremes: slow hearts leave signal in no token at all, fast hearts
        # leave most of every token as padding.
        covered = sum(t["end"] - t["start"] for t in tokens) / DURATION_S
        filled = float(np.mean([t["fill"] for t in tokens]))

        payload["cases"][name] = {
            "title": title,
            "ecgId": ecg_id,
            "labels": list(row["superclasses"]),
            "hr": round(float(heart_rate), 1),
            "age": None if not np.isfinite(row["age"]) else int(row["age"]),
            "nSamples": int(len(lead)),
            "coverage": round(float(covered), 4),
            "meanFill": round(filled, 4),
            "signal": [round(float(v), 3) for v in lead],
            "peaks": [round(float(p) / DETECT_RATE, 4) for p in peaks],
            "rrMs": [round(float(v) * 1000, 1) for v in rr],
            "rrTokens": tokens,
            "fixed": [
                {
                    "start": round(i * PATCH_MS / 1000, 4),
                    "end": round((i + 1) * PATCH_MS / 1000, 4),
                }
                for i in range(N_FIXED)
            ],
        }
        labels = ",".join(row["superclasses"]) or "-"
        print(
            f"  {name:<8} ecg_id {ecg_id:>6}  {heart_rate:5.1f} bpm  "
            f"{len(peaks):>2} beats  covered {100 * covered:4.0f}%  "
            f"filled {100 * filled:4.0f}%  {labels}"
        )

    if not payload["cases"]:
        raise SystemExit("no records selected; refusing to write an empty figure")
    return payload


def splice(html_path: Path, payload: dict) -> int:
    """Replace the figure's data block in place, leaving the markup untouched.

    Args:
        html_path: The figure to rewrite.
        payload: Data to assign to ``window.ECG_DATA``.

    Returns:
        Size of the written file in bytes.

    Raises:
        ValueError: If the data block is missing or ambiguous, which means the
            figure was edited in a way this script no longer understands.
    """
    html = html_path.read_text(encoding="utf-8")
    if html.count(DATA_TAG_OPEN) != 1:
        raise ValueError(f"expected exactly one {DATA_TAG_OPEN} in {html_path}")

    start = html.index(DATA_TAG_OPEN) + len(DATA_TAG_OPEN)
    end = html.index(DATA_TAG_CLOSE, start)
    block = "window.ECG_DATA=" + json.dumps(payload, separators=(",", ":")) + ";"
    html_path.write_text(html[:start] + block + html[end:], encoding="utf-8")
    return html_path.stat().st_size


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/ptbxl"),
        help="PTB-XL dataset root (default: data/ptbxl)",
    )
    parser.add_argument(
        "--html",
        type=Path,
        default=here / "tokenizer_comparison.html",
        help="figure to rewrite (default: the one beside this script)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point.

    Args:
        argv: Argument vector, defaulting to ``sys.argv[1:]``.

    Returns:
        Process exit code.
    """
    args = build_parser().parse_args(argv)
    print(f"reading {args.dataset}")
    payload = build_payload(args.dataset)
    size = splice(args.html, payload)
    print(f"wrote {args.html} ({size / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
