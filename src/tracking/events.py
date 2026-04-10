from datetime import datetime


def _dedupe_preserve_order(track_ids: list[int]) -> list[int]:
    return list(dict.fromkeys(track_ids))


def normalize_merge_event(
    timestamp: datetime,
    surviving_track_id: int,
    merged_track_ids: list[int],
) -> dict | None:
    merged_track_ids = [
        track_id
        for track_id in _dedupe_preserve_order(merged_track_ids)
        if track_id != surviving_track_id
    ]
    if not merged_track_ids:
        return None
    return {
        "event_type": "merge",
        "timestamp": timestamp.isoformat(),
        "description": f"Tracks {', '.join(str(t) for t in merged_track_ids)} merged into track {surviving_track_id}",
        "involved_track_ids": [surviving_track_id] + merged_track_ids,
    }


def normalize_split_event(
    timestamp: datetime,
    parent_track_id: int,
    child_track_ids: list[int],
) -> dict | None:
    child_track_ids = [
        track_id
        for track_id in _dedupe_preserve_order(child_track_ids)
        if track_id != parent_track_id
    ]
    if not child_track_ids:
        return None
    return {
        "event_type": "split",
        "timestamp": timestamp.isoformat(),
        "description": f"Track {parent_track_id} split into tracks {', '.join(str(c) for c in child_track_ids)}",
        "involved_track_ids": [parent_track_id] + child_track_ids,
    }

