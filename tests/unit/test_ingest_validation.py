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
