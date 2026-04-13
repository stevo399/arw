from scripts.evaluate_tracking import BenchmarkResult, ReplaySnapshot, render_markdown


def test_render_markdown_includes_heading_flip_metric():
    result = BenchmarkResult(
        benchmark_id="dense_cached_quick",
        category="dense",
        site_id="KTLX",
        date="2026-04-10",
        local_only=True,
        scan_count=3,
        mean_objects=50.0,
        mean_active_tracks=55.0,
        max_speed_mph=20,
        total_merges=10,
        total_splits=8,
        mean_uncertain_tracks=0.0,
        mean_focus_identity_confidence=0.52,
        mean_focus_continuity=0.38,
        mean_focus_motion_confidence=0.41,
        focus_low_identity_scans=1,
        focus_low_continuity_scans=3,
        focus_low_motion_scans=2,
        focus_switches=1,
        focus_heading_flips_ge_90=2,
        focus_flips_with_low_motion_confidence=1,
        focus_track_distance_changes=2,
        summary_tracking_uncertain_count=3,
        summary_motion_published_count=0,
        summary_stationaryish_count=1,
        total_new_tracks_after_first_scan=6,
        merged_tracks_total=3,
        lost_tracks_total=0,
        split_children_total=4,
        absorbed_links_total=3,
        fragmentation_proxy=0.04,
        snapshots=[
            ReplaySnapshot(
                timestamp="2026-04-10T23:46:05Z",
                object_count=48,
                active_count=48,
                uncertain_tracks=0,
                max_speed_mph=0,
                merge_count=0,
                split_count=0,
                focus_track_id=1,
                focus_identity_label="medium",
                focus_identity_score=0.52,
                focus_continuity_label="low",
                focus_continuity_score=0.38,
                focus_heading_deg=None,
                focus_heading_label="stationary",
                focus_speed_mph=0,
                focus_motion_confidence_label="low",
                focus_motion_confidence_score=0.41,
                new_tracks=48,
                total_tracks_seen=48,
                summary="Example summary",
            )
        ],
    )
    markdown = render_markdown([result])
    assert "focus heading flips >=90 deg" in markdown
    assert "focus flips with low motion confidence" in markdown
    assert "focus low-continuity scans" in markdown
    assert "summary tracking-uncertain count" in markdown
    assert "merged tracks total" in markdown
    assert "focus_continuity=low:0.38" in markdown
    assert "focus_identity=medium:0.52" in markdown
    assert "`3`" in markdown
