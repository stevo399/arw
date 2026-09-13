from datetime import datetime, timezone

import pytest

import src.history.disk as disk
from src.history.compact_scan import CompactScan
from src.history.disk import (
    CompactScanLoadError,
    history_filename,
    load_compact_scan,
    save_compact_scan,
)
from tests.unit.test_history_compact_scan import _scan_with_content


def _compact() -> CompactScan:
    return CompactScan.from_buffered_scan(_scan_with_content())


def test_save_and_load_round_trip_exactly(tmp_path):
    compact = _compact()
    path = save_compact_scan(compact, tmp_path)
    assert path.name == "20260912_180000.arwscan"
    assert load_compact_scan(path) == compact


def test_filename_uses_utc():
    aware = datetime(2026, 9, 12, 18, 0, 5, tzinfo=timezone.utc)
    assert history_filename(aware) == "20260912_180005.arwscan"


@pytest.mark.parametrize("damage", ["truncate", "flip"])
def test_damaged_files_raise_load_error(tmp_path, damage):
    path = save_compact_scan(_compact(), tmp_path)
    data = bytearray(path.read_bytes())
    if damage == "truncate":
        data = data[: len(data) // 2]
    else:
        data[len(data) // 2] ^= 0xFF
    path.write_bytes(bytes(data))
    with pytest.raises(CompactScanLoadError):
        load_compact_scan(path)


def test_schema_mismatch_is_a_load_error(tmp_path, monkeypatch):
    path = save_compact_scan(_compact(), tmp_path)
    monkeypatch.setattr(disk, "SCHEMA_VERSION", 999)
    with pytest.raises(CompactScanLoadError, match="schema"):
        load_compact_scan(path)


def test_failed_atomic_write_leaves_previous_file_intact(tmp_path, monkeypatch):
    compact = _compact()
    path = save_compact_scan(compact, tmp_path)
    original_bytes = path.read_bytes()

    def failing_replace(src, dst):
        raise OSError("disk full")

    monkeypatch.setattr(disk.os, "replace", failing_replace)
    with pytest.raises(OSError):
        save_compact_scan(compact.with_tracking(None), tmp_path)
    assert path.read_bytes() == original_bytes
    assert not list(tmp_path.glob("*.tmp"))
