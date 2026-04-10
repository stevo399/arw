import os
from datetime import datetime
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
        from datetime import timedelta
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
