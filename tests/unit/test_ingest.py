import os
from unittest.mock import patch, MagicMock
from src.ingest import (
    get_cache_path,
    scan_is_cached,
    list_scans_for_date,
    list_latest_scans,
    download_scan,
    fetch_scan,
)


def test_get_cache_path():
    path = get_cache_path("KTLX", "KTLX20260408_183000_V06")
    assert "KTLX" in path
    assert "KTLX20260408_183000_V06" in path
    assert os.path.isabs(path)


def test_scan_is_cached_returns_false_when_missing(tmp_path):
    with patch("src.ingest.CACHE_DIR", str(tmp_path)):
        assert scan_is_cached("KTLX", "KTLX20260408_183000_V06") is False


def test_scan_is_cached_returns_true_when_present(tmp_path):
    site_dir = tmp_path / "KTLX"
    site_dir.mkdir()
    (site_dir / "KTLX20260408_183000_V06").write_bytes(b"fake data")
    with patch("src.ingest.CACHE_DIR", str(tmp_path)):
        assert scan_is_cached("KTLX", "KTLX20260408_183000_V06") is True


def test_list_scans_for_date():
    with patch("src.ingest._get_nexrad_conn") as mock_conn:
        mock_conn.return_value.get_avail_scans.return_value = [
            MagicMock(key="KTLX20260408_180000_V06", filename="KTLX20260408_180000_V06"),
            MagicMock(key="KTLX20260408_183000_V06", filename="KTLX20260408_183000_V06"),
        ]
        scans = list_scans_for_date("KTLX", "2026-04-08")
        assert len(scans) == 2


def test_download_scan(tmp_path):
    with patch("src.ingest.CACHE_DIR", str(tmp_path)), \
         patch("src.ingest._get_nexrad_conn") as mock_conn:
        mock_scan = MagicMock()
        mock_scan.filename = "KTLX20260408_183000_V06"
        mock_result = MagicMock()
        local_file = tmp_path / "KTLX" / "KTLX20260408_183000_V06"
        mock_result.filepath = str(local_file)
        mock_download_results = MagicMock()
        mock_download_results.success = [mock_result]
        mock_conn.return_value.download.return_value = mock_download_results
        path = download_scan("KTLX", mock_scan)
        assert path is not None
