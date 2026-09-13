"""Download and extract the PTB-XL dataset from PhysioNet.

PTB-XL is a 12-lead, 10-second clinical ECG dataset of 21,799 records from
18,869 patients, published with a patient-stratified 10-fold assignment
(``strat_fold``) that this project uses verbatim for its splits.

The dataset ships as a single zip containing both the 100 Hz (``records100/``)
and 500 Hz (``records500/``) waveform sets. Both are required: R-peak detection
for the RR tokenizer always runs on the 500 Hz signal and maps the resulting
peak indices down to the model's working resolution, so the tokenizer does not
change behaviour when the resolution flag changes.

Typical use::

    ecg-download --dest data/ptbxl
    python -m ecg.data.download --dest data/ptbxl   # equivalent

The script is idempotent: a completed download is not re-fetched, an interrupted
one resumes via an HTTP Range request, and an already-extracted dataset is left
alone unless ``--force`` is given.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import threading
import time
import urllib.error
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

PTBXL_VERSION = "1.0.3"
PTBXL_ZIP_NAME = (
    f"ptb-xl-a-large-publicly-available-electrocardiography-dataset-{PTBXL_VERSION}.zip"
)

#: Canonical source. Correct but frequently throttled to tens of KiB/s, which
#: puts the 1.7 GiB archive well over ten hours.
PTBXL_ZIP_URL = f"https://physionet.org/static/published-projects/ptb-xl/{PTBXL_ZIP_NAME}"

#: AWS Open Data mirror of the same archive, public and unauthenticated. It is
#: byte-identical to the PhysioNet copy and serves ranged requests, so it can be
#: fetched in parallel. Verified with --sha256 either way.
PTBXL_S3_ZIP_URL = f"https://physionet-open.s3.amazonaws.com/ptb-xl/ptb-xl-{PTBXL_VERSION}.zip"

#: Named sources selectable from the command line.
MIRRORS: dict[str, str] = {"s3": PTBXL_S3_ZIP_URL, "physionet": PTBXL_ZIP_URL}

#: Entries that must exist for an extraction to count as complete.
REQUIRED_ENTRIES: tuple[str, ...] = (
    "ptbxl_database.csv",
    "scp_statements.csv",
    "records100",
    "records500",
)

_USER_AGENT = "ecg-research/0.1 (+https://physionet.org/content/ptb-xl/)"
_CHUNK_SIZE = 1 << 20  # 1 MiB


@dataclass(frozen=True)
class DatasetLayout:
    """Resolved paths inside an extracted PTB-XL directory.

    Attributes:
        root: Directory holding the metadata CSVs and record folders.
        database_csv: Per-record metadata, including ``patient_id`` and
            ``strat_fold``.
        scp_statements_csv: SCP-ECG statement definitions, including the
            ``diagnostic_class`` column used to derive the 5 superclasses.
        records100: Waveform directory at 100 Hz.
        records500: Waveform directory at 500 Hz.
    """

    root: Path
    database_csv: Path
    scp_statements_csv: Path
    records100: Path
    records500: Path


def human_bytes(n: float) -> str:
    """Format a byte count for human-readable output.

    Args:
        n: Number of bytes.

    Returns:
        A short string such as ``"1.42 GiB"``.
    """
    size = float(n)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(size) < 1024.0:
            return f"{size:,.2f} {unit}"
        size /= 1024.0
    return f"{size:,.2f} PiB"


def _report_progress(done: int, total: int | None, started: float) -> None:
    """Write a single-line transfer progress indicator to stderr.

    Args:
        done: Bytes transferred so far.
        total: Expected total bytes, or ``None`` if the server did not say.
        started: ``time.monotonic()`` reading from the start of the transfer.
    """
    elapsed = max(time.monotonic() - started, 1e-6)
    rate = done / elapsed
    if total:
        pct = 100.0 * done / total
        eta = (total - done) / rate if rate > 0 else float("inf")
        line = (
            f"\r  {pct:5.1f}%  {human_bytes(done)} / {human_bytes(total)}"
            f"  at {human_bytes(rate)}/s  ETA {eta / 60:5.1f} min   "
        )
    else:
        line = f"\r  {human_bytes(done)} at {human_bytes(rate)}/s   "
    sys.stderr.write(line)
    sys.stderr.flush()


def probe_size(url: str, *, timeout: float = 60.0) -> int | None:
    """Ask the server for the size of ``url`` without downloading it.

    Uses a one-byte ranged GET rather than HEAD, because PhysioNet's static file
    host does not reliably report ``Content-Length`` on HEAD requests.

    Args:
        url: URL to probe.
        timeout: Socket timeout in seconds.

    Returns:
        Size in bytes, or ``None`` if the server does not disclose it.
    """
    request = urllib.request.Request(
        url, headers={"User-Agent": _USER_AGENT, "Range": "bytes=0-0"}
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            content_range = response.headers.get("Content-Range")
            if content_range and "/" in content_range:
                total = content_range.rsplit("/", 1)[1].strip()
                if total.isdigit():
                    return int(total)
            length = response.headers.get("Content-Length")
            if length and int(response.status) == 200:
                return int(length)
    except (urllib.error.URLError, ValueError, OSError):
        return None
    return None


def download_file(
    url: str,
    dest: Path,
    *,
    resume: bool = True,
    timeout: float = 60.0,
    chunk_size: int = _CHUNK_SIZE,
) -> Path:
    """Download ``url`` to ``dest``, resuming a partial file when possible.

    If ``dest`` already exists at the server's reported size, nothing is
    transferred.

    Args:
        url: Source URL.
        dest: Destination file path; parent directories are created.
        resume: Attempt an HTTP Range request to continue a partial download.
            A server that ignores the header causes a clean restart.
        timeout: Socket timeout in seconds.
        chunk_size: Read buffer size in bytes.

    Returns:
        The path that was written.

    Raises:
        RuntimeError: On an unexpected HTTP status, or a short transfer.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    existing = dest.stat().st_size if dest.exists() else 0

    remote_size = probe_size(url, timeout=timeout)
    if remote_size is not None and existing == remote_size:
        print(f"  already complete ({human_bytes(existing)}), skipping download")
        return dest

    headers = {"User-Agent": _USER_AGENT}
    if resume and existing:
        headers["Range"] = f"bytes={existing}-"

    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        status = int(response.status)
        if status == 206:
            mode, done = "ab", existing
            print(f"  resuming from {human_bytes(existing)}")
        elif status == 200:
            if existing:
                print("  server ignored the Range request; restarting from zero")
            mode, done = "wb", 0
        else:  # pragma: no cover - defensive
            raise RuntimeError(f"Unexpected HTTP status {status} for {url}")

        length_header = response.headers.get("Content-Length")
        remaining = int(length_header) if length_header else None
        total = done + remaining if remaining is not None else remote_size

        started = time.monotonic()
        with open(dest, mode) as handle:
            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                handle.write(chunk)
                done += len(chunk)
                _report_progress(done, total, started)
    sys.stderr.write("\n")

    if total is not None and done != total:
        raise RuntimeError(
            f"Incomplete download: got {human_bytes(done)}, expected {human_bytes(total)}. "
            "Re-run to resume."
        )
    return dest


def _byte_ranges(total: int, parts: int) -> list[tuple[int, int]]:
    """Split a byte count into contiguous inclusive ``(start, end)`` ranges.

    Args:
        total: Size of the resource in bytes.
        parts: Number of ranges to produce.

    Returns:
        A list of at most ``parts`` non-overlapping ranges covering ``total``.
        Ranges are inclusive at both ends, matching HTTP Range semantics.
    """
    if total <= 0 or parts < 1:
        return []
    parts = min(parts, total)
    size, remainder = divmod(total, parts)
    ranges: list[tuple[int, int]] = []
    start = 0
    for index in range(parts):
        length = size + (1 if index < remainder else 0)
        ranges.append((start, start + length - 1))
        start += length
    return ranges


def _fetch_range(
    url: str, start: int, end: int, part_path: Path, timeout: float, on_bytes
) -> Path:
    """Download one byte range to its own part file, skipping completed parts.

    Args:
        url: Source URL.
        start: First byte of the range, inclusive.
        end: Last byte of the range, inclusive.
        part_path: Where to write this range.
        timeout: Socket timeout in seconds.
        on_bytes: Callback invoked with the number of bytes written per chunk,
            used for aggregate progress reporting.

    Returns:
        ``part_path``.

    Raises:
        RuntimeError: If the server does not honour the Range request, which
            would otherwise silently corrupt the reassembled file.
    """
    expected = end - start + 1
    if part_path.exists() and part_path.stat().st_size == expected:
        on_bytes(expected)
        return part_path

    request = urllib.request.Request(
        url, headers={"User-Agent": _USER_AGENT, "Range": f"bytes={start}-{end}"}
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        if int(response.status) != 206:
            raise RuntimeError(
                f"Mirror ignored the Range request (status {response.status}); "
                "cannot download in parallel from this source."
            )
        with open(part_path, "wb") as handle:
            while True:
                chunk = response.read(_CHUNK_SIZE)
                if not chunk:
                    break
                handle.write(chunk)
                on_bytes(len(chunk))

    written = part_path.stat().st_size
    if written != expected:
        raise RuntimeError(
            f"Range {start}-{end} returned {written} bytes, expected {expected}"
        )
    return part_path


def download_file_parallel(
    url: str,
    dest: Path,
    *,
    workers: int = 8,
    timeout: float = 60.0,
) -> Path:
    """Download ``url`` using several concurrent ranged requests.

    Each worker writes its own ``.partN`` file, which are concatenated on
    success. Because parts are only joined once every range has completed, an
    interrupted run leaves the finished parts on disk and re-running resumes
    from there instead of starting over.

    Falls back to :func:`download_file` when the server does not report a size
    or refuses ranged requests.

    Args:
        url: Source URL.
        dest: Destination file path.
        workers: Number of concurrent connections.
        timeout: Socket timeout in seconds.

    Returns:
        The path that was written.

    Raises:
        RuntimeError: If a range fails or the reassembled file is the wrong size.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    total = probe_size(url, timeout=timeout)

    if total is None:
        print("  server did not report a size; falling back to a single stream")
        return download_file(url, dest, timeout=timeout)
    if dest.exists() and dest.stat().st_size == total:
        print(f"  already complete ({human_bytes(total)}), skipping download")
        return dest

    ranges = _byte_ranges(total, workers)
    part_dir = dest.parent / f"{dest.name}.parts"
    part_dir.mkdir(exist_ok=True)

    done = 0
    lock = threading.Lock()
    started = time.monotonic()

    def on_bytes(n: int) -> None:
        nonlocal done
        with lock:
            done += n
            _report_progress(done, total, started)

    print(f"  {len(ranges)} parallel connections, {human_bytes(total)} total")
    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(
                    _fetch_range,
                    url,
                    start,
                    end,
                    part_dir / f"part{index:03d}",
                    timeout,
                    on_bytes,
                ): index
                for index, (start, end) in enumerate(ranges)
            }
            for future in as_completed(futures):
                future.result()  # re-raise the first failure
    except RuntimeError as error:
        sys.stderr.write("\n")
        if "Range request" in str(error):
            print(f"  {error}")
            print("  falling back to a single stream")
            shutil.rmtree(part_dir, ignore_errors=True)
            return download_file(url, dest, timeout=timeout)
        raise
    sys.stderr.write("\n")

    print("  joining parts ...")
    with open(dest, "wb") as out:
        for index in range(len(ranges)):
            part = part_dir / f"part{index:03d}"
            with open(part, "rb") as handle:
                shutil.copyfileobj(handle, out, _CHUNK_SIZE)

    final = dest.stat().st_size
    if final != total:
        raise RuntimeError(
            f"Reassembled file is {human_bytes(final)}, expected {human_bytes(total)}"
        )
    shutil.rmtree(part_dir, ignore_errors=True)
    return dest


def sha256sum(path: Path, *, chunk_size: int = _CHUNK_SIZE) -> str:
    """Compute the SHA-256 hex digest of a file.

    Args:
        path: File to hash.
        chunk_size: Read buffer size in bytes.

    Returns:
        Lowercase hex digest.
    """
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def extract_archive(zip_path: Path, dest: Path) -> Path:
    """Extract the PTB-XL zip into ``dest``, flattening its top-level folder.

    The published archive wraps everything in a single versioned directory. That
    directory is collapsed so ``dest/ptbxl_database.csv`` is valid regardless of
    release version. Extraction goes to a staging directory first, so an
    interrupted run cannot leave a half-populated dataset that
    :func:`is_already_extracted` would later accept.

    Args:
        zip_path: Path to the downloaded ``.zip``.
        dest: Directory to extract into; created if absent.

    Returns:
        ``dest``, now containing the dataset.

    Raises:
        zipfile.BadZipFile: If the archive is corrupt or truncated.
    """
    dest.mkdir(parents=True, exist_ok=True)
    staging = dest.parent / f"{dest.name}.extracting"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    with zipfile.ZipFile(zip_path) as archive:
        members = archive.infolist()
        total = len(members)
        started = time.monotonic()
        for index, member in enumerate(members, start=1):
            archive.extract(member, staging)
            if index % 500 == 0 or index == total:
                elapsed = max(time.monotonic() - started, 1e-6)
                sys.stderr.write(
                    f"\r  {index:,}/{total:,} entries  ({index / elapsed:,.0f}/s)   "
                )
                sys.stderr.flush()
    sys.stderr.write("\n")

    children = [p for p in staging.iterdir() if not p.name.startswith("__MACOSX")]
    source = children[0] if len(children) == 1 and children[0].is_dir() else staging
    for item in source.iterdir():
        target = dest / item.name
        if target.exists():
            shutil.rmtree(target) if target.is_dir() else target.unlink()
        shutil.move(str(item), str(target))
    shutil.rmtree(staging, ignore_errors=True)
    return dest


def resolve_layout(root: Path) -> DatasetLayout:
    """Validate an extracted PTB-XL directory and resolve its key paths.

    Args:
        root: Directory expected to contain ``ptbxl_database.csv``.

    Returns:
        The resolved :class:`DatasetLayout`.

    Raises:
        FileNotFoundError: If any required entry is missing. The message names
            every missing entry, so a partial extraction is diagnosable in one
            run rather than one failure at a time.
    """
    root = Path(root)
    missing = [name for name in REQUIRED_ENTRIES if not (root / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"PTB-XL at {root} is incomplete; missing: {', '.join(missing)}"
        )
    return DatasetLayout(
        root=root,
        database_csv=root / "ptbxl_database.csv",
        scp_statements_csv=root / "scp_statements.csv",
        records100=root / "records100",
        records500=root / "records500",
    )


def is_already_extracted(root: Path) -> bool:
    """Report whether ``root`` already holds a complete PTB-XL extraction.

    Args:
        root: Candidate dataset directory.

    Returns:
        ``True`` if every required entry is present.
    """
    root = Path(root)
    return root.exists() and all((root / name).exists() for name in REQUIRED_ENTRIES)


def summarise(layout: DatasetLayout) -> None:
    """Print a short inventory of an extracted dataset.

    Counts ``.dat`` signal files per resolution, which is the cheapest check
    that the waveform folders are populated rather than merely present.

    Args:
        layout: Resolved dataset paths.
    """
    print(f"\nPTB-XL ready at {layout.root}")
    for name, folder in (("100 Hz", layout.records100), ("500 Hz", layout.records500)):
        count = sum(1 for _ in folder.rglob("*.dat"))
        print(f"  {name:>7}: {count:,} signal files")
    size = sum(f.stat().st_size for f in layout.root.rglob("*") if f.is_file())
    print(f"  on disk: {human_bytes(size)}")


def download_ptbxl(
    dest: Path,
    *,
    url: str | None = None,
    source: str = "s3",
    archive_dir: Path | None = None,
    keep_archive: bool = False,
    force: bool = False,
    verify_sha256: str | None = None,
    workers: int = 8,
) -> DatasetLayout:
    """Download and extract PTB-XL, skipping work that is already done.

    Args:
        dest: Directory the dataset should end up in.
        url: Explicit archive URL. Overrides ``source`` when given.
        source: Named mirror, one of :data:`MIRRORS`. Defaults to the AWS
            mirror, which serves ranged requests and is typically far faster
            than physionet.org.
        archive_dir: Where to store the ``.zip``; defaults to ``dest.parent``.
        keep_archive: Retain the ``.zip`` after a successful extraction.
        force: Re-extract even if ``dest`` already looks complete.
        verify_sha256: Expected hex digest of the archive; checked before
            extraction when given.
        workers: Concurrent connections; 1 forces a single stream.

    Returns:
        The resolved :class:`DatasetLayout`.

    Raises:
        KeyError: If ``source`` is not a known mirror.
        ValueError: If ``verify_sha256`` does not match the downloaded archive.
    """
    dest = Path(dest).expanduser().resolve()
    if is_already_extracted(dest) and not force:
        print(f"Dataset already present at {dest} (use --force to re-extract)")
        layout = resolve_layout(dest)
        summarise(layout)
        return layout

    if url is None:
        if source not in MIRRORS:
            raise KeyError(f"Unknown source {source!r}; choose from {sorted(MIRRORS)}")
        url = MIRRORS[source]

    archive_dir = Path(archive_dir or dest.parent).expanduser().resolve()
    zip_path = archive_dir / PTBXL_ZIP_NAME

    print(f"Downloading {url}")
    print(f"  -> {zip_path}")
    if workers > 1:
        download_file_parallel(url, zip_path, workers=workers)
    else:
        download_file(url, zip_path)

    if verify_sha256:
        print("Verifying SHA-256 ...")
        actual = sha256sum(zip_path)
        if actual.lower() != verify_sha256.lower():
            raise ValueError(
                f"SHA-256 mismatch for {zip_path}: expected {verify_sha256}, got {actual}"
            )
        print("  digest OK")

    print(f"Extracting to {dest}")
    extract_archive(zip_path, dest)
    layout = resolve_layout(dest)

    if not keep_archive:
        zip_path.unlink(missing_ok=True)
        print(f"  removed {zip_path.name} (pass --keep-archive to retain it)")

    summarise(layout)
    return layout


def build_parser() -> argparse.ArgumentParser:
    """Construct the command-line parser.

    Returns:
        The configured parser.
    """
    parser = argparse.ArgumentParser(
        prog="ecg-download",
        description="Download and extract the PTB-XL ECG dataset from PhysioNet.",
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=Path("data/ptbxl"),
        help="Directory to extract the dataset into (default: %(default)s)",
    )
    parser.add_argument(
        "--source",
        choices=sorted(MIRRORS),
        default="s3",
        help=(
            "Which mirror to fetch from. 's3' is the AWS Open Data copy "
            "(supports parallel ranged requests); 'physionet' is the canonical "
            "but often heavily throttled origin. (default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--url", default=None, help="Explicit archive URL, overriding --source"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Concurrent connections; 1 forces a single stream (default: %(default)s)",
    )
    parser.add_argument(
        "--archive-dir",
        type=Path,
        default=None,
        help="Where to store the downloaded zip (default: parent of --dest)",
    )
    parser.add_argument(
        "--keep-archive",
        action="store_true",
        help="Keep the zip after extracting instead of deleting it",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-extract even if the destination already looks complete",
    )
    parser.add_argument(
        "--sha256",
        default=None,
        help="Expected SHA-256 of the archive; verified before extraction",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point for command-line use.

    Args:
        argv: Argument list; defaults to ``sys.argv[1:]``.

    Returns:
        Process exit code: 0 on success, 1 on a handled failure.
    """
    args = build_parser().parse_args(argv)
    try:
        download_ptbxl(
            args.dest,
            url=args.url,
            source=args.source,
            archive_dir=args.archive_dir,
            keep_archive=args.keep_archive,
            force=args.force,
            verify_sha256=args.sha256,
            workers=args.workers,
        )
    except (OSError, RuntimeError, ValueError, KeyError, zipfile.BadZipFile) as error:
        print(f"\nFAILED: {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nInterrupted. Re-run to resume the download.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
