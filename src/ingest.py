import os
import urllib.request
from datetime import date, datetime, timedelta
from pathlib import Path
import nexradaws

CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "cache")


def _get_nexrad_conn() -> nexradaws.NexradAwsInterface:
    """Return a nexradaws connection. Isolated for mocking."""
    return nexradaws.NexradAwsInterface()


def get_cache_path(site_id: str, filename: str) -> str:
    """Return the absolute cache file path for a given scan."""
    return os.path.join(os.path.abspath(CACHE_DIR), site_id, filename)


def scan_is_cached(site_id: str, filename: str) -> bool:
    """Check if a scan file already exists in the local cache."""
    return os.path.isfile(get_cache_path(site_id, filename))


def list_scans_for_date(site_id: str, date_str: str) -> list:
    """List available scans for a site on a given date (YYYY-MM-DD).
    Filters out _MDM metadata files which are not real radar volume scans."""
    conn = _get_nexrad_conn()
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    scans = conn.get_avail_scans(dt.year, dt.month, dt.day, site_id)
    return [s for s in scans if not s.filename.endswith("_MDM")]


def list_latest_scans(site_id: str) -> list:
    """List the most recent scans for a site (today's date)."""
    now = datetime.utcnow()
    scans = list_scans_for_date(site_id, now.strftime("%Y-%m-%d"))
    if not scans:
        yesterday = now - timedelta(days=1)
        scans = list_scans_for_date(site_id, yesterday.strftime("%Y-%m-%d"))
    return scans


def download_scan(site_id: str, scan) -> str:
    """Download a single scan file to the cache. Returns the local file path."""
    filename = scan.filename
    if scan_is_cached(site_id, filename):
        return get_cache_path(site_id, filename)
    cache_dir = os.path.join(os.path.abspath(CACHE_DIR), site_id)
    os.makedirs(cache_dir, exist_ok=True)
    conn = _get_nexrad_conn()
    results = conn.download(scan, cache_dir)
    for result in results.success:
        return str(result.filepath)
    raise RuntimeError(f"Failed to download scan {filename} for {site_id}")


def fetch_scan(site_id: str, dt: datetime | None = None) -> str:
    """Fetch a scan for a site. If dt provided, find closest scan to that time.
    If dt is None, fetch the latest scan. Returns local file path."""
    if dt is None:
        scans = list_latest_scans(site_id)
        if not scans:
            raise RuntimeError(f"No scans available for {site_id}")
        scan = scans[-1]
    else:
        scans = list_scans_for_date(site_id, dt.strftime("%Y-%m-%d"))
        if not scans:
            raise RuntimeError(f"No scans available for {site_id} on {dt.date()}")
        scan = min(scans, key=lambda s: abs(
            datetime.strptime(s.filename.split("_V")[0][-15:], "%Y%m%d_%H%M%S") - dt
        ).total_seconds() if hasattr(s, 'filename') else float('inf'))
    return download_scan(site_id, scan)


SPC_REPORT_URL = "https://www.spc.noaa.gov/climo/reports/{yymmdd}_rpts_{name}.csv"
SPC_FILE_NAMES = {"tornado": "torn", "hail": "hail", "wind": "wind"}


def fetch_spc_reports(spc_day: date, kind: str) -> str:
    """Download one SPC daily report CSV into the cache (once). Returns the local path."""
    name = f"{spc_day:%y%m%d}_rpts_{SPC_FILE_NAMES[kind]}.csv"
    path = os.path.join(os.path.abspath(CACHE_DIR), "spc", name)
    if os.path.isfile(path):
        return path
    os.makedirs(os.path.dirname(path), exist_ok=True)
    url = SPC_REPORT_URL.format(yymmdd=f"{spc_day:%y%m%d}", name=SPC_FILE_NAMES[kind])
    with urllib.request.urlopen(url, timeout=60) as response:
        body = response.read()
    with open(path, "wb") as handle:
        handle.write(body)
    return path


LEVEL3_BUCKET = "unidata-nexrad-level3"


def _level3_client():
    """Unsigned S3 client for NWS Level III products. Isolated for mocking."""
    import boto3
    from botocore import UNSIGNED
    from botocore.config import Config

    return boto3.client("s3", config=Config(signature_version=UNSIGNED), region_name="us-east-1")


def level3_key_time(key: str) -> datetime:
    return datetime.strptime("_".join(key.split("_")[2:8]), "%Y_%m_%d_%H_%M_%S")


def list_level3_keys(site_id: str, product: str, start: datetime, end: datetime) -> list[str]:
    """Level III product keys for a site whose product time lies within [start, end]."""
    client = _level3_client()
    site = site_id[1:] if len(site_id) == 4 else site_id
    keys: list[str] = []
    day = start.date()
    while day <= end.date():
        prefix = f"{site}_{product}_{day:%Y_%m_%d}"
        for page in client.get_paginator("list_objects_v2").paginate(Bucket=LEVEL3_BUCKET, Prefix=prefix):
            for item in page.get("Contents", []):
                if start <= level3_key_time(item["Key"]) <= end:
                    keys.append(item["Key"])
        day += timedelta(days=1)
    return sorted(keys)


def download_level3(key: str) -> str:
    """Download one Level III product into the cache (once). Returns the local path."""
    site = key.split("_")[0]
    path = os.path.join(os.path.abspath(CACHE_DIR), "level3", site, key)
    if os.path.isfile(path):
        return path
    os.makedirs(os.path.dirname(path), exist_ok=True)
    _level3_client().download_file(LEVEL3_BUCKET, key, path)
    return path
