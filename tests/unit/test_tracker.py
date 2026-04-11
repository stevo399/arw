# tests/unit/test_tracker.py
from datetime import datetime, timedelta
import numpy as np
from src.tracker import StormTracker, Track
from src.detection import DetectedObject
from src.buffer import BufferedScan
from src.parser import ReflectivityData
from src.preprocess import ScanQuality


def _make_object(obj_id: int, lat: float, lon: float, peak_dbz: float = 45.0) -> DetectedObject:
    return DetectedObject(
        object_id=obj_id,
        centroid_lat=lat,
        centroid_lon=lon,
        distance_km=40.0,
        bearing_deg=270.0,
        peak_dbz=peak_dbz,
        peak_label="heavy rain",
        area_km2=100.0,
        layers=[],
    )


def _make_scan(
    site_id: str,
    timestamp: datetime,
    objects: list[DetectedObject],
    grid_shape: tuple[int, int] = (360, 500),
    masks: dict[int, np.ndarray] | None = None,
) -> BufferedScan:
    ref_data = ReflectivityData(
        reflectivity=np.full(grid_shape, np.nan),
        azimuths=np.linspace(0, 359, grid_shape[0]),
        ranges_m=np.linspace(2000, 250000, grid_shape[1]),
        radar_lat=35.3331,
        radar_lon=-97.2778,
        elevation_angle=0.5,
        elevation_angles=[0.5],
        timestamp=timestamp.isoformat(),
    )
    labeled_grid = np.zeros(grid_shape, dtype=int)
    if masks is None:
        masks = {}
        for obj in objects:
            mask = np.zeros(grid_shape, dtype=bool)
            row = min(85 + (obj.object_id - 1) * 40, grid_shape[0] - 15)
            mask[row:row + 10, 195:205] = True
            labeled_grid[mask] = obj.object_id
            masks[obj.object_id] = mask
    else:
        for obj_id, mask in masks.items():
            labeled_grid[mask] = obj_id
    return BufferedScan(
        timestamp=timestamp,
        site_id=site_id,
        reflectivity_data=ref_data,
        detected_objects=objects,
        labeled_grid=labeled_grid,
        object_masks=masks,
    )


def test_tracker_first_scan_creates_tracks():
    tracker = StormTracker()
    t = datetime(2026, 4, 8, 18, 30)
    objects = [_make_object(1, 35.5, -97.3), _make_object(2, 35.8, -97.1)]
    scan = _make_scan("KTLX", t, objects)
    tracker.update(scan)
    assert len(tracker.active_tracks) == 2
    for track in tracker.active_tracks:
        assert track.status == "active"
        assert len(track.positions) == 1


def test_tracker_second_scan_updates_tracks():
    tracker = StormTracker()
    t1 = datetime(2026, 4, 8, 18, 30)
    t2 = datetime(2026, 4, 8, 18, 35)
    obj1 = _make_object(1, 35.5, -97.3)
    obj2 = _make_object(1, 35.51, -97.29)  # Slightly moved
    mask1 = np.zeros((360, 500), dtype=bool)
    mask1[85:95, 195:205] = True
    mask2 = np.zeros((360, 500), dtype=bool)
    mask2[86:96, 196:206] = True  # Overlaps with mask1
    scan1 = _make_scan("KTLX", t1, [obj1], masks={1: mask1})
    scan2 = _make_scan("KTLX", t2, [obj2], masks={1: mask2})
    tracker.update(scan1)
    tracker.update(scan2)
    assert len(tracker.active_tracks) == 1
    track = tracker.active_tracks[0]
    assert len(track.positions) == 2


def test_tracker_unmatched_object_creates_new_track():
    tracker = StormTracker()
    t1 = datetime(2026, 4, 8, 18, 30)
    t2 = datetime(2026, 4, 8, 18, 35)
    obj1 = _make_object(1, 35.5, -97.3)
    obj2 = _make_object(1, 36.5, -96.0)  # Far away, no overlap
    mask1 = np.zeros((360, 500), dtype=bool)
    mask1[85:95, 195:205] = True
    mask2 = np.zeros((360, 500), dtype=bool)
    mask2[300:310, 400:410] = True  # No overlap with mask1
    scan1 = _make_scan("KTLX", t1, [obj1], masks={1: mask1})
    scan2 = _make_scan("KTLX", t2, [obj2], masks={1: mask2})
    tracker.update(scan1)
    tracker.update(scan2)
    # Original track should be active (1 missed scan), new track created
    all_tracks = tracker.all_tracks
    assert len(all_tracks) >= 2


def test_tracker_lost_after_missed_scans():
    tracker = StormTracker()
    t1 = datetime(2026, 4, 8, 18, 30)
    obj1 = _make_object(1, 35.5, -97.3)
    mask1 = np.zeros((360, 500), dtype=bool)
    mask1[85:95, 195:205] = True
    scan1 = _make_scan("KTLX", t1, [obj1], masks={1: mask1})
    tracker.update(scan1)
    # Two empty scans = lost
    for i in range(2):
        t = t1 + timedelta(minutes=(i + 1) * 5)
        empty_scan = _make_scan("KTLX", t, [])
        tracker.update(empty_scan)
    lost = [t for t in tracker.all_tracks if t.status == "lost"]
    assert len(lost) == 1


def test_tracker_merge_detection():
    tracker = StormTracker()
    t1 = datetime(2026, 4, 8, 18, 30)
    t2 = datetime(2026, 4, 8, 18, 35)
    # Two separate objects in scan 1
    obj_a = _make_object(1, 35.5, -97.3, peak_dbz=50.0)
    obj_b = _make_object(2, 35.5, -97.28, peak_dbz=40.0)
    mask_a = np.zeros((360, 500), dtype=bool)
    mask_a[85:95, 195:205] = True
    mask_b = np.zeros((360, 500), dtype=bool)
    mask_b[85:95, 205:215] = True
    scan1 = _make_scan("KTLX", t1, [obj_a, obj_b], masks={1: mask_a, 2: mask_b})
    # One merged object in scan 2, overlapping both mask_a and mask_b
    obj_merged = _make_object(1, 35.5, -97.29, peak_dbz=52.0)
    mask_merged = np.zeros((360, 500), dtype=bool)
    mask_merged[85:95, 195:215] = True  # Covers both previous masks
    scan2 = _make_scan("KTLX", t2, [obj_merged], masks={1: mask_merged})
    tracker.update(scan1)
    tracker.update(scan2)
    merged = [t for t in tracker.all_tracks if t.status == "merged"]
    assert len(merged) >= 1
    events = tracker.recent_events
    merge_events = [e for e in events if e["event_type"] == "merge"]
    assert len(merge_events) >= 1


def test_tracker_split_detection():
    tracker = StormTracker()
    t1 = datetime(2026, 4, 8, 18, 30)
    t2 = datetime(2026, 4, 8, 18, 35)
    # One object in scan 1
    obj1 = _make_object(1, 35.5, -97.3, peak_dbz=50.0)
    mask1 = np.zeros((360, 500), dtype=bool)
    mask1[85:95, 195:215] = True
    scan1 = _make_scan("KTLX", t1, [obj1], masks={1: mask1})
    # Two objects in scan 2, each overlapping the original
    obj_a = _make_object(1, 35.5, -97.31, peak_dbz=48.0)
    obj_b = _make_object(2, 35.5, -97.28, peak_dbz=35.0)
    mask_a = np.zeros((360, 500), dtype=bool)
    mask_a[85:95, 195:205] = True  # Left half
    mask_b = np.zeros((360, 500), dtype=bool)
    mask_b[85:95, 205:215] = True  # Right half
    scan2 = _make_scan("KTLX", t2, [obj_a, obj_b], masks={1: mask_a, 2: mask_b})
    tracker.update(scan1)
    tracker.update(scan2)
    active = tracker.active_tracks
    assert len(active) >= 2
    events = tracker.recent_events
    split_events = [e for e in events if e["event_type"] == "split"]
    assert len(split_events) >= 1
    parent = next(track for track in tracker.all_tracks if track.track_id == 1)
    children = [track for track in tracker.all_tracks if track.split_from == 1]
    assert children
    for child in children:
        assert 1 in child.parent_track_ids
        assert child.track_id in parent.child_track_ids


def test_tracker_get_track_by_id():
    tracker = StormTracker()
    t = datetime(2026, 4, 8, 18, 30)
    scan = _make_scan("KTLX", t, [_make_object(1, 35.5, -97.3)])
    tracker.update(scan)
    track = tracker.get_track(1)
    assert track is not None
    assert track.track_id == 1
    assert tracker.get_track(999) is None


def test_tracker_resets_on_site_change():
    """Switching radar sites must clear prior state so cross-site masks
    don't collide and produce phantom matches."""
    tracker = StormTracker()
    t = datetime(2026, 4, 8, 18, 30)
    ktlx_scan = _make_scan("KTLX", t, [_make_object(1, 35.5, -97.3)])
    tracker.update(ktlx_scan)
    assert len(tracker.active_tracks) == 1

    # Now switch to a different site — previous tracks should be gone
    kfws_scan = _make_scan("KFWS", t + timedelta(minutes=5), [_make_object(1, 32.5, -97.3)])
    tracker.update(kfws_scan)
    # Only the KFWS track should be active; KTLX track should not exist
    active = tracker.active_tracks
    assert len(active) == 1
    # Track ids should restart at 1 for the new site
    assert active[0].track_id == 1
    # And there should be exactly one position (no lingering KTLX history)
    assert len(active[0].positions) == 1


def test_tracker_merge_event_excludes_surviving_and_dedupes():
    """A merge event's involved_track_ids must not contain the surviving
    track in the 'merged' list, and must not contain duplicates."""
    tracker = StormTracker()
    t1 = datetime(2026, 4, 8, 18, 30)
    t2 = datetime(2026, 4, 8, 18, 35)
    # Three separate objects in scan 1
    obj_a = _make_object(1, 35.5, -97.30, peak_dbz=50.0)
    obj_b = _make_object(2, 35.5, -97.28, peak_dbz=45.0)
    obj_c = _make_object(3, 35.5, -97.26, peak_dbz=42.0)
    mask_a = np.zeros((360, 500), dtype=bool)
    mask_a[85:95, 190:200] = True
    mask_b = np.zeros((360, 500), dtype=bool)
    mask_b[85:95, 200:210] = True
    mask_c = np.zeros((360, 500), dtype=bool)
    mask_c[85:95, 210:220] = True
    scan1 = _make_scan("KTLX", t1, [obj_a, obj_b, obj_c], masks={1: mask_a, 2: mask_b, 3: mask_c})
    # One big merged object in scan 2 covering all three
    obj_merged = _make_object(1, 35.5, -97.28, peak_dbz=52.0)
    mask_merged = np.zeros((360, 500), dtype=bool)
    mask_merged[85:95, 190:220] = True
    scan2 = _make_scan("KTLX", t2, [obj_merged], masks={1: mask_merged})
    tracker.update(scan1)
    tracker.update(scan2)

    merge_events = [e for e in tracker.recent_events if e["event_type"] == "merge"]
    assert len(merge_events) == 1
    event = merge_events[0]
    involved = event["involved_track_ids"]
    # Should be exactly [surviving, merged_1, merged_2] with no duplicates
    assert len(involved) == len(set(involved)), f"duplicates in involved: {involved}"
    # Surviving track should appear exactly once (at the start)
    surviving = involved[0]
    assert involved.count(surviving) == 1
    # And the description should not name the surviving track in the merged list
    assert f"merged into track {surviving}" in event["description"]
    merged_part = event["description"].split("merged into")[0]
    assert str(surviving) not in merged_part
    surviving_track = tracker.get_track(surviving)
    assert surviving_track is not None
    for merged_track_id in involved[1:]:
        merged_track = tracker.get_track(merged_track_id)
        assert merged_track is not None
        assert merged_track.merged_into == surviving
        assert surviving in merged_track.parent_track_ids
        assert merged_track_id in surviving_track.absorbed_track_ids


def test_tracker_leftover_after_merge_becomes_new_track():
    """If a new object 1:1-matches a previous object whose track was already
    consumed by a merge, the leftover must become a new track — not reuse
    the merged survivor."""
    tracker = StormTracker()
    t1 = datetime(2026, 4, 8, 18, 30)
    t2 = datetime(2026, 4, 8, 18, 35)

    # Scan 1: two objects A and B
    obj_a = _make_object(1, 35.5, -97.30, peak_dbz=50.0)
    obj_b = _make_object(2, 35.5, -97.28, peak_dbz=40.0)
    mask_a = np.zeros((360, 500), dtype=bool)
    mask_a[85:95, 195:205] = True
    mask_b = np.zeros((360, 500), dtype=bool)
    mask_b[85:95, 205:215] = True
    scan1 = _make_scan("KTLX", t1, [obj_a, obj_b], masks={1: mask_a, 2: mask_b})

    # Scan 2: X merges A and B, Y only overlaps A
    obj_x = _make_object(1, 35.5, -97.29, peak_dbz=52.0)
    obj_y = _make_object(2, 35.5, -97.31, peak_dbz=45.0)
    mask_x = np.zeros((360, 500), dtype=bool)
    mask_x[85:95, 200:210] = True  # overlaps both A (cols 200-205) and B (cols 205-210)
    mask_y = np.zeros((360, 500), dtype=bool)
    mask_y[85:95, 197:203] = True  # overlaps only A
    scan2 = _make_scan("KTLX", t2, [obj_x, obj_y], masks={1: mask_x, 2: mask_y})

    tracker.update(scan1)
    tracker.update(scan2)

    # After scan 2: X should be in the surviving merged track (1),
    # Y should be in its own new track (not track 1).
    # Total tracks: 1 surviving (from A), 1 merged-out (B), 1 new (Y) = 3.
    all_tracks = tracker.all_tracks
    assert len(all_tracks) == 3, f"expected 3 tracks, got {len(all_tracks)}"
    # No active track should have duplicate positions at the same timestamp
    for track in all_tracks:
        timestamps = [p.timestamp for p in track.positions]
        assert len(timestamps) == len(set(timestamps)), (
            f"track {track.track_id} has duplicate position timestamps: {timestamps}"
        )
    # Exactly two active tracks: the merged survivor and the new track for Y
    active = tracker.active_tracks
    assert len(active) == 2
    # Merge events should reference real merges (track IDs must be distinct)
    merge_events = [e for e in tracker.recent_events if e["event_type"] == "merge"]
    assert len(merge_events) == 1
    involved = merge_events[0]["involved_track_ids"]
    assert len(involved) == len(set(involved))


def test_tracker_two_merges_sharing_prev_do_not_duplicate_track_mapping():
    """If two new objects both try to merge using the same previous track as
    their strongest overlap, the tracker must not end up with both new objects
    mapped to the same surviving track. That duplicate mapping would cause
    phantom 'track N merged into track N' events on the next scan — the
    symptom observed in live KEYX data."""
    tracker = StormTracker()
    t1 = datetime(2026, 4, 8, 18, 30)
    t2 = datetime(2026, 4, 8, 18, 35)
    t3 = datetime(2026, 4, 8, 18, 40)

    # Scan 1: three previous objects. A is big (wide mask); B and C are
    # small satellites on either side of A.
    obj_a = _make_object(1, 35.5, -97.30, peak_dbz=50.0)
    obj_b = _make_object(2, 35.5, -97.28, peak_dbz=42.0)
    obj_c = _make_object(3, 35.5, -97.32, peak_dbz=42.0)
    mask_a = np.zeros((360, 500), dtype=bool)
    mask_a[85:95, 190:210] = True  # area 200
    mask_b = np.zeros((360, 500), dtype=bool)
    mask_b[85:95, 210:214] = True  # area 40
    mask_c = np.zeros((360, 500), dtype=bool)
    mask_c[85:95, 176:180] = True  # area 40
    scan1 = _make_scan("KTLX", t1, [obj_a, obj_b, obj_c],
                       masks={1: mask_a, 2: mask_b, 3: mask_c})

    # Scan 2: X and Y both have their STRONGEST overlap with A's mask,
    # so A's track would be picked as each merge's surviving candidate.
    # Only one merge can legitimately claim track A.
    obj_x = _make_object(1, 35.5, -97.29, peak_dbz=52.0)
    obj_y = _make_object(2, 35.5, -97.31, peak_dbz=51.0)
    mask_x = np.zeros((360, 500), dtype=bool)
    mask_x[85:95, 195:212] = True
    # A∩X: cols 195-209 = 15 wide  => overlap(A,X)=150/200=0.75
    # B∩X: cols 210-211 = 2 wide   => overlap(B,X)=20/40=0.5
    mask_y = np.zeros((360, 500), dtype=bool)
    mask_y[85:95, 178:199] = True
    # A∩Y: cols 190-198 = 9 wide   => overlap(A,Y)=90/200=0.45
    # C∩Y: cols 178-179 = 2 wide   => overlap(C,Y)=20/40=0.5
    # ^ A still outranks C in Y's match list because 0.45>0 beats no-match,
    #   and importantly Y has A in its match_candidates list.
    scan2 = _make_scan("KTLX", t2, [obj_x, obj_y],
                       masks={1: mask_x, 2: mask_y})

    tracker.update(scan1)
    tracker.update(scan2)

    # Invariant: no two new objects may map to the same track this scan.
    # If they do, the NEXT scan's overlap matching will produce phantom
    # "Track N merged into track N" events.
    obj_to_track = tracker._obj_to_track
    assigned_tracks = list(obj_to_track.values())
    assert len(assigned_tracks) == len(set(assigned_tracks)), (
        f"two new objects mapped to the same track: {obj_to_track}"
    )

    # Now push a third scan where Z overlaps both X's and Y's masks.
    # If the mapping is clean, this produces a real merge. If the previous
    # scan produced duplicate mappings, this produces a self-merge event.
    obj_z = _make_object(1, 35.5, -97.30, peak_dbz=53.0)
    mask_z = np.zeros((360, 500), dtype=bool)
    mask_z[85:95, 178:212] = True  # covers both mask_x and mask_y
    scan3 = _make_scan("KTLX", t3, [obj_z], masks={1: mask_z})
    tracker.update(scan3)

    # No merge event should list the same track id twice.
    merge_events = [e for e in tracker.recent_events if e["event_type"] == "merge"]
    for event in merge_events:
        involved = event["involved_track_ids"]
        assert len(involved) == len(set(involved)), (
            f"merge event has duplicate track ids: {event}"
        )
        # Surviving track should not also appear in the merged list
        surviving = involved[0]
        assert involved.count(surviving) == 1, (
            f"surviving track {surviving} appears more than once: {event}"
        )


def test_tracker_initial_confidence_is_penalized_by_scan_quality():
    tracker = StormTracker()
    t = datetime(2026, 4, 8, 18, 30)
    scan = _make_scan("KTLX", t, [_make_object(1, 35.5, -97.3)])
    scan.scan_quality = ScanQuality(
        score=0.5,
        finite_fraction=0.7,
        removed_speckle_pixels=0,
        removed_speckle_fraction=0.0,
        flags=["high_missing_fraction"],
    )
    tracker.update(scan)
    assert len(tracker.active_tracks) == 1
    assert tracker.active_tracks[0].identity_confidence == 0.62
    assert tracker.active_tracks[0].identity_diagnostics is not None
    assert tracker.active_tracks[0].identity_diagnostics.scan_quality == 0.5


def test_tracker_identity_diagnostics_capture_ambiguous_match():
    tracker = StormTracker()
    t1 = datetime(2026, 4, 8, 18, 30)
    t2 = datetime(2026, 4, 8, 18, 35)

    obj_a = _make_object(1, 35.5, -97.30, peak_dbz=50.0)
    obj_b = _make_object(2, 35.5, -97.28, peak_dbz=48.0)
    mask_a = np.zeros((360, 500), dtype=bool)
    mask_a[85:95, 195:205] = True
    mask_b = np.zeros((360, 500), dtype=bool)
    mask_b[85:95, 205:215] = True
    scan1 = _make_scan("KTLX", t1, [obj_a, obj_b], masks={1: mask_a, 2: mask_b})

    obj_follow = _make_object(1, 35.5, -97.29, peak_dbz=49.0)
    mask_follow = np.zeros((360, 500), dtype=bool)
    mask_follow[85:95, 200:210] = True
    scan2 = _make_scan("KTLX", t2, [obj_follow], masks={1: mask_follow})

    tracker.update(scan1)
    tracker.update(scan2)

    active = tracker.active_tracks
    assert active
    survivor = next(track for track in active if track.current_object is not None)
    assert survivor.identity_diagnostics is not None
    assert survivor.identity_diagnostics.event_context in {"merge_survivor", "matched"}
    assert survivor.identity_diagnostics.ambiguity_margin is not None
    assert survivor.identity_diagnostics.reason is not None


def test_tracker_lineage_persists_after_followup_scan():
    tracker = StormTracker()
    t1 = datetime(2026, 4, 8, 18, 30)
    t2 = datetime(2026, 4, 8, 18, 35)
    t3 = datetime(2026, 4, 8, 18, 40)

    obj1 = _make_object(1, 35.5, -97.3, peak_dbz=50.0)
    mask1 = np.zeros((360, 500), dtype=bool)
    mask1[85:95, 195:215] = True
    scan1 = _make_scan("KTLX", t1, [obj1], masks={1: mask1})

    obj_a = _make_object(1, 35.5, -97.31, peak_dbz=48.0)
    obj_b = _make_object(2, 35.5, -97.28, peak_dbz=35.0)
    mask_a = np.zeros((360, 500), dtype=bool)
    mask_a[85:95, 195:205] = True
    mask_b = np.zeros((360, 500), dtype=bool)
    mask_b[85:95, 205:215] = True
    scan2 = _make_scan("KTLX", t2, [obj_a, obj_b], masks={1: mask_a, 2: mask_b})

    obj_follow = _make_object(1, 35.5, -97.305, peak_dbz=47.0)
    obj_child = _make_object(2, 35.5, -97.275, peak_dbz=34.0)
    mask_follow = np.zeros((360, 500), dtype=bool)
    mask_follow[85:95, 195:205] = True
    mask_child = np.zeros((360, 500), dtype=bool)
    mask_child[85:95, 205:215] = True
    scan3 = _make_scan("KTLX", t3, [obj_follow, obj_child], masks={1: mask_follow, 2: mask_child})

    tracker.update(scan1)
    tracker.update(scan2)
    tracker.update(scan3)

    parent = tracker.get_track(1)
    assert parent is not None
    assert parent.child_track_ids
    child = tracker.get_track(parent.child_track_ids[0])
    assert child is not None
    assert parent.track_id in child.parent_track_ids
