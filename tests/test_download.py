"""Tests for the PTB-XL downloader.

These exercise the offline logic only -- archive handling, layout validation and
CLI wiring -- against a synthetic zip built to mirror the real PTB-XL structure.
Nothing here touches the network, so the suite stays runnable on the GPU box.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from ecg.data.download import (
    MIRRORS,
    PTBXL_VERSION,
    DatasetLayout,
    _byte_ranges,
    build_parser,
    extract_archive,
    human_bytes,
    is_already_extracted,
    resolve_layout,
    sha256sum,
)

#: Mirrors the real archive, which nests everything under one versioned folder.
_ARCHIVE_ROOT = f"ptb-xl-a-large-publicly-available-electrocardiography-dataset-{PTBXL_VERSION}"


def _make_archive(path: Path, *, nested: bool = True, omit: str = "") -> Path:
    """Build a miniature PTB-XL zip for testing.

    Args:
        path: Destination ``.zip`` path.
        nested: Wrap entries in a versioned top-level folder, as PhysioNet does.
        omit: Name of a required entry to leave out, to simulate a partial
            archive.

    Returns:
        ``path``.
    """
    prefix = f"{_ARCHIVE_ROOT}/" if nested else ""
    entries = {
        "ptbxl_database.csv": "ecg_id,patient_id,strat_fold\n1,100,1\n",
        "scp_statements.csv": "diagnostic_class\nNORM\n",
        "records100/00000/00001_lr.dat": "\x00\x01",
        "records500/00000/00001_hr.dat": "\x00\x01",
    }
    with zipfile.ZipFile(path, "w") as archive:
        for name, content in entries.items():
            if omit and name.startswith(omit):
                continue
            archive.writestr(prefix + name, content)
    return path


def test_human_bytes_scales_units() -> None:
    assert human_bytes(0) == "0.00 B"
    assert human_bytes(1536) == "1.50 KiB"
    assert human_bytes(3 * 1024**3) == "3.00 GiB"


def test_extract_archive_flattens_versioned_root(tmp_path: Path) -> None:
    """The versioned wrapper folder must be collapsed away."""
    archive = _make_archive(tmp_path / "ptbxl.zip")
    dest = tmp_path / "data" / "ptbxl"

    extract_archive(archive, dest)

    assert (dest / "ptbxl_database.csv").is_file()
    assert (dest / "records500" / "00000" / "00001_hr.dat").is_file()
    assert not (dest / _ARCHIVE_ROOT).exists()


def test_extract_archive_handles_flat_archive(tmp_path: Path) -> None:
    """An archive without a wrapper folder extracts just the same."""
    archive = _make_archive(tmp_path / "flat.zip", nested=False)
    dest = tmp_path / "flat-out"

    extract_archive(archive, dest)

    assert (dest / "ptbxl_database.csv").is_file()


def test_extract_archive_leaves_no_staging_directory(tmp_path: Path) -> None:
    """Staging must be cleaned up, or a later run would trip over it."""
    archive = _make_archive(tmp_path / "ptbxl.zip")
    dest = tmp_path / "ptbxl"

    extract_archive(archive, dest)

    assert not (dest.parent / f"{dest.name}.extracting").exists()


def test_resolve_layout_returns_expected_paths(tmp_path: Path) -> None:
    archive = _make_archive(tmp_path / "ptbxl.zip")
    dest = tmp_path / "ptbxl"
    extract_archive(archive, dest)

    layout = resolve_layout(dest)

    assert isinstance(layout, DatasetLayout)
    assert layout.database_csv == dest / "ptbxl_database.csv"
    assert layout.scp_statements_csv == dest / "scp_statements.csv"
    assert layout.records100.is_dir()
    assert layout.records500.is_dir()


def test_resolve_layout_names_every_missing_entry(tmp_path: Path) -> None:
    """A partial extraction should be diagnosable from one error message."""
    dest = tmp_path / "partial"
    dest.mkdir()
    (dest / "ptbxl_database.csv").write_text("ecg_id\n")

    with pytest.raises(FileNotFoundError) as excinfo:
        resolve_layout(dest)

    message = str(excinfo.value)
    assert "scp_statements.csv" in message
    assert "records100" in message
    assert "records500" in message


def test_missing_records500_is_rejected(tmp_path: Path) -> None:
    """records500 is mandatory: R-peak detection always runs at 500 Hz."""
    archive = _make_archive(tmp_path / "no500.zip", omit="records500")
    dest = tmp_path / "no500"
    extract_archive(archive, dest)

    assert is_already_extracted(dest) is False
    with pytest.raises(FileNotFoundError, match="records500"):
        resolve_layout(dest)


def test_is_already_extracted_on_complete_and_absent(tmp_path: Path) -> None:
    archive = _make_archive(tmp_path / "ptbxl.zip")
    dest = tmp_path / "ptbxl"
    extract_archive(archive, dest)

    assert is_already_extracted(dest) is True
    assert is_already_extracted(tmp_path / "nothing-here") is False


def test_sha256sum_matches_hashlib(tmp_path: Path) -> None:
    import hashlib

    payload = b"ecg" * 1000
    path = tmp_path / "blob.bin"
    path.write_bytes(payload)

    assert sha256sum(path) == hashlib.sha256(payload).hexdigest()


def test_parser_defaults_and_overrides() -> None:
    parser = build_parser()

    defaults = parser.parse_args([])
    assert defaults.dest == Path("data/ptbxl")
    assert defaults.force is False
    assert defaults.keep_archive is False
    assert defaults.source == "s3"
    assert defaults.workers == 8

    custom = parser.parse_args(
        ["--dest", "/tmp/x", "--force", "--keep-archive", "--source", "physionet", "--workers", "1"]
    )
    assert custom.dest == Path("/tmp/x")
    assert custom.force is True
    assert custom.keep_archive is True
    assert custom.source == "physionet"
    assert custom.workers == 1


def test_parser_rejects_unknown_mirror() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--source", "dropbox"])


def test_mirrors_are_https() -> None:
    """A plain-HTTP mirror would let the archive be tampered with in transit."""
    assert set(MIRRORS) == {"s3", "physionet"}
    for url in MIRRORS.values():
        assert url.startswith("https://")


class TestByteRanges:
    def test_ranges_tile_the_file_exactly(self) -> None:
        ranges = _byte_ranges(1000, 8)
        assert ranges[0][0] == 0
        assert ranges[-1][1] == 999
        for (_, prev_end), (next_start, _) in zip(ranges, ranges[1:]):
            assert next_start == prev_end + 1

    def test_total_length_is_preserved(self) -> None:
        for total, parts in ((1000, 8), (1001, 8), (7, 3), (1 << 30, 16)):
            ranges = _byte_ranges(total, parts)
            assert sum(end - start + 1 for start, end in ranges) == total

    def test_remainder_spread_across_leading_parts(self) -> None:
        """10 bytes over 4 parts must be 3,3,2,2 -- never a zero-length range."""
        lengths = [end - start + 1 for start, end in _byte_ranges(10, 4)]
        assert lengths == [3, 3, 2, 2]

    def test_never_more_parts_than_bytes(self) -> None:
        assert len(_byte_ranges(3, 8)) == 3

    def test_degenerate_inputs(self) -> None:
        assert _byte_ranges(0, 8) == []
        assert _byte_ranges(100, 0) == []
