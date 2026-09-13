"""Proofs (compact scan history, Task 11) on real cached Level II volumes.

1. Identical tracking: with reacquisition disabled, a RadarHistory -- whose
   tracker reloads previous scans from compact records -- produces the same
   per-scan tracking snapshot, byte for byte, as a tracker that retains full
   scans (the previous pipeline).  With reacquisition enabled, snapshots are
   identical until the first reacquisition; every reacquisition is recorded
   for review.
2. Memory: after live-style ingest of five scans on two radars with all map
   layers rendered, no full-size grid is retained, and retained history and
   rendered-layer bytes are within limits derived from measured sizes.
3. Restart: reopening a radar's history mid-window continues identically.
4. Browsing: a historical request leaves live tracking unchanged.

Results are written to docs/test_reports/2026-09-12-compact-history-proof.json.
"""

from copy import deepcopy
from dataclasses import replace
from datetime import datetime
import gc
import json
from pathlib import Path
import tracemalloc

import numpy as np
import pytest

import src.server as server
from src.history.layer_cache import RenderedLayerCache
from src.history.records import dumps
from src.history.store import HistoryRegistry, RadarHistory
from src.tracker import StormTracker

KTLX_WINDOW = [
    f"cache/KTLX/KTLX20260410_{stamp}_V06"
    for stamp in ("224209", "224737", "225255", "225805", "230332", "230850", "231407")
]
KSOX_WINDOW = [
    f"cache/KSOX/KSOX20260410_{stamp}_V06"
    for stamp in ("223503", "224345", "225226", "230106", "230940")
]
KTLX_HOUR_EARLIER = "cache/KTLX/KTLX20260410_214929_V06"
REPORT = Path("docs/test_reports/2026-09-12-compact-history-proof.json")


def _analyze(path: str):
    return server._process_scan_file(Path(path).parent.name, str(Path(path).resolve()))


def _twin(buffered):
    """Same arrays (read-only), independent object records."""
    return replace(buffered, detected_objects=deepcopy(buffered.detected_objects), tracking=None)


def _record(section: str, payload) -> None:
    report = json.loads(REPORT.read_text()) if REPORT.exists() else {}
    report[section] = payload
    REPORT.write_text(json.dumps(report, indent=2, default=str))


class _NoReacquireHistory(RadarHistory):
    def _new_tracker(self):
        return StormTracker(scan_loader=self._load_for_tracker)

    def _restore_tracker(self, state):
        return StormTracker.from_state(state, self._load_for_tracker)


def test_history_tracking_is_identical_to_retained_scan_tracking(tmp_path):
    previous = StormTracker()
    plain = _NoReacquireHistory.open("KTLX", tmp_path / "plain")
    reacquiring = RadarHistory.open("KTLX", tmp_path / "reacquiring")
    rows, first_divergence = [], None

    for index, path in enumerate(KTLX_WINDOW):
        original = _analyze(path)
        old_input, reacquire_input = _twin(original), _twin(original)
        previous.update(old_input)
        for obj in old_input.detected_objects:
            obj.temporal_status = previous.temporal_status_for_current_object(obj.object_id)
        old_snapshot = dumps(previous.snapshot())

        plain.add_live_scan(original)
        plain_snapshot = dumps(plain.newest().to_buffered_scan().tracking)
        reacquiring.add_live_scan(reacquire_input)
        reacquired_tracking = reacquiring.newest().to_buffered_scan().tracking
        reacquired_events = [e for e in reacquired_tracking.recent_events if e["event_type"] == "reacquired"]

        assert plain_snapshot == old_snapshot, f"scan {index} ({path}) tracking differs"
        if first_divergence is None and dumps(reacquired_tracking) != old_snapshot:
            first_divergence = index
        rows.append({
            "scan": Path(path).name,
            "objects": len(original.detected_objects),
            "active_tracks": len(reacquired_tracking.active_tracks),
            "reacquired": reacquired_events,
        })
        del original, old_input, reacquire_input

    assert dumps(plain.tracker.export_state()) == dumps(previous.export_state())
    if first_divergence is not None:
        assert rows[first_divergence]["reacquired"], "reacquisition run diverged without a reacquisition"
    _record("identical_tracking", {"scans": rows, "first_reacquisition_divergence": first_divergence})


def test_retained_memory_is_bounded_and_no_full_grids_remain(tmp_path, monkeypatch):
    registry = HistoryRegistry(tmp_path / "history")
    monkeypatch.setattr(server, "_histories", registry)
    monkeypatch.setattr(server, "_map_layers", RenderedLayerCache(max_unpinned_scans=10))
    monkeypatch.setattr(server, "_schedule_live_refresh", lambda *a, **k: False)
    grid_gates = None
    old_pipeline_bytes = 0
    compact_sizes, rendered_sizes = [], []
    tracemalloc.start()

    for site_id, window in (("KTLX", KTLX_WINDOW[:5]), ("KSOX", KSOX_WINDOW)):
        paths = iter(str(Path(p).resolve()) for p in window)
        monkeypatch.setattr(server, "fetch_scan", lambda site, dt=None, paths=paths: next(paths))
        for _ in window:
            buffered = server._ingest_to_buffer(site_id)
            gates = buffered.reflectivity_data.reflectivity.size
            grid_gates = gates if grid_gates is None else min(grid_gates, gates)
            # What the previous pipeline retained for this one scan.
            old_pipeline_bytes = max(
                old_pipeline_bytes,
                buffered.reflectivity_data.reflectivity.nbytes
                + np.asarray(buffered.labeled_grid).nbytes
                + sum(mask.nbytes for mask in buffered.object_masks.values()),
            )
            layers = server._prepared_map_layers(buffered)
            precipitation = server._prepared_precipitation_layer(buffered)
            rendered_sizes.append(len(json.dumps(layers)) + len(json.dumps(precipitation)))
            del buffered, layers, precipitation
        with registry.use(site_id) as history:
            compact_sizes.extend(scan.nbytes for scan in history.scans())

    transient_peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    gc.collect()
    retained_grids = [
        (obj.shape, obj.dtype.str)
        for obj in gc.get_objects()
        if isinstance(obj, np.ndarray) and obj.size >= grid_gates
    ]
    retained_compact = sum(compact_sizes)
    retained_rendered = sum(
        len(json.dumps(server._map_layers.get(key)))
        for key in list(server._map_layers._entries)
    )
    compact_limit = 10 * max(compact_sizes) * 1.1
    rendered_limit = (2 + 10) * max(rendered_sizes) * 1.1

    _record("memory", {
        "compact_scan_bytes": compact_sizes,
        "retained_compact_bytes": retained_compact,
        "compact_limit_bytes": compact_limit,
        "rendered_scan_bytes": rendered_sizes,
        "retained_rendered_bytes": retained_rendered,
        "rendered_limit_bytes": rendered_limit,
        "old_pipeline_largest_scan_array_bytes": old_pipeline_bytes,
        "old_pipeline_two_scans_two_radars_bytes": 4 * old_pipeline_bytes,
        "transient_peak_traced_bytes": transient_peak,
        "retained_full_grids": retained_grids,
    })
    assert len(compact_sizes) == 10
    assert retained_grids == [], f"full-size grids still retained: {retained_grids}"
    assert retained_compact <= compact_limit
    assert retained_rendered <= rendered_limit


def test_restart_mid_window_continues_identically(tmp_path):
    uninterrupted = RadarHistory.open("KTLX", tmp_path / "uninterrupted")
    restarted = RadarHistory.open("KTLX", tmp_path / "restarted")
    for index, path in enumerate(KTLX_WINDOW[:5]):
        original = _analyze(path)
        twin = _twin(original)
        uninterrupted.add_live_scan(original)
        if index == 3:
            restarted = RadarHistory.open("KTLX", tmp_path / "restarted")
        restarted.add_live_scan(twin)
        del original, twin

    assert dumps(restarted.tracker.export_state()) == dumps(uninterrupted.tracker.export_state())
    assert not restarted.newest().to_buffered_scan().tracking.continuity_rebuilt
    _record("restart", {"restart_before_scan": 4, "identical": True})


def test_historical_request_leaves_live_tracking_unchanged(monkeypatch):
    monkeypatch.setattr(server, "_schedule_live_refresh", lambda *a, **k: False)
    live = iter(str(Path(p).resolve()) for p in KTLX_WINDOW[:3])
    monkeypatch.setattr(server, "fetch_scan", lambda site, dt=None: next(live))
    for _ in range(3):
        server._ingest_to_buffer("KTLX")
    with server._histories.use("KTLX") as history:
        before_state = dumps(history.tracker.export_state())
        before_ring = [scan.timestamp for scan in history.scans()]

    monkeypatch.setattr(server, "fetch_scan", lambda site, dt=None: str(Path(KTLX_HOUR_EARLIER).resolve()))
    earlier = server._ingest_to_buffer("KTLX", datetime(2026, 4, 10, 21, 49, 29))

    assert earlier.tracking is None
    with server._histories.use("KTLX") as history:
        assert dumps(history.tracker.export_state()) == before_state
        assert [scan.timestamp for scan in history.scans()] == before_ring
    _record("browsing", {"historical_volume": Path(KTLX_HOUR_EARLIER).name, "live_tracking_unchanged": True})
