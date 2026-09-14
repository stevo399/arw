from datetime import date
from unittest.mock import patch

from src import ingest


def test_fetch_spc_reports_downloads_once_into_the_cache(tmp_path):
    body = b"Time,F_Scale,Location,County,State,Lat,Lon,Comments\n"
    with patch.object(ingest, "CACHE_DIR", str(tmp_path)), patch("src.ingest.urllib.request.urlopen") as urlopen:
        urlopen.return_value.__enter__.return_value.read.return_value = body
        path = ingest.fetch_spc_reports(date(2026, 9, 12), "tornado")
        again = ingest.fetch_spc_reports(date(2026, 9, 12), "tornado")
    assert path == again
    assert path.endswith("260912_rpts_torn.csv")
    assert open(path, "rb").read() == body
    assert urlopen.call_count == 1
    assert urlopen.call_args[0][0] == "https://www.spc.noaa.gov/climo/reports/260912_rpts_torn.csv"


from datetime import datetime


def test_level3_keys_are_listed_per_date_and_filtered_to_the_window():
    pages = [{"Contents": [
        {"Key": "MVX_NMD_2026_09_11_23_58_00"},
        {"Key": "MVX_NMD_2026_09_12_00_05_32"},
        {"Key": "MVX_NMD_2026_09_12_00_40_00"},
    ]}]
    with patch("src.ingest._level3_client") as client:
        client.return_value.get_paginator.return_value.paginate.return_value = pages
        keys = ingest.list_level3_keys("KMVX", "NMD", datetime(2026, 9, 12, 0, 0), datetime(2026, 9, 12, 0, 30))
    assert keys == ["MVX_NMD_2026_09_12_00_05_32"]
    assert ingest.level3_key_time("MVX_NMD_2026_09_12_00_05_32") == datetime(2026, 9, 12, 0, 5, 32)


def test_download_level3_caches_by_site(tmp_path):
    with patch.object(ingest, "CACHE_DIR", str(tmp_path)), patch("src.ingest._level3_client") as client:
        path = ingest.download_level3("MVX_NMD_2026_09_12_00_05_32")
    assert path.replace("\\", "/").endswith("level3/MVX/MVX_NMD_2026_09_12_00_05_32")
    client.return_value.download_file.assert_called_once()
