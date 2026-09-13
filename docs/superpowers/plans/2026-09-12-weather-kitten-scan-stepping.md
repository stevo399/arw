# Weather Kitten Recent Scan Stepping Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let a Weather Kitten user step backward and forward through the radar scans ARW retains (up to 5 per radar) on any of the four radar layers, with keyboard- and screen-reader-friendly links and a spoken position.

**Architecture:** After a radar map is ready, Weather Kitten asks ARW's `GET /map/history` (same location parameters as the map) for the retained scans. A pure function turns them into a navigation model: position text, previous and next scan, and a list with the shown scan marked. A small Jinja partial renders it as plain links that reload the page in the panel's existing specific-time mode, focused on the position line. No new JavaScript.

**Tech Stack:** Python 3.11+, Flask, Jinja2, `requests`, `unittest` (Weather Kitten repository at `C:\Users\steve\Dropbox\PC\Documents\weatherKitten`).

**Spec:** `C:\Users\steve\Dropbox\PC\Documents\arw\docs\superpowers\specs\2026-09-12-compact-scan-history-design.md`, section 3 "Weather Kitten" (and Amendment 1).

## Global Constraints

- **Prerequisites:**
  - ARW's `docs/superpowers/plans/2026-09-12-compact-scan-history.md` is implemented. `GET /map/history` exists, and `/map/status?datetime=<retained timestamp>` is ready immediately.
  - Weather Kitten's uncommitted changes (`.env.example`, `app.py`, `templates/index.html`, `tests/test_perf.py`) have been committed or set aside by the owner.
- All commands run from the Weather Kitten repository root with its venv: `.venv/Scripts/python.exe`.
- Tests: `.venv/Scripts/python.exe -m unittest discover -s tests -v` must pass before every commit.
- Accessibility first:
  - every control is a real link or text
  - no custom keyboard shortcuts (Alt+Left is browser Back)
  - the shown scan is marked with `aria-current="true"`
  - the position line is a polite status region that links target, so focus lands on it after the reload
- Wording must be accurate. ARW's objects are detected radar echoes, not confirmed storms, so labels say "detected echoes". A scan without tracking context says "not tracked".
- Automatic radar refresh stays limited to "latest" mode (unchanged). Stepping pins a specific time, so a reviewer is never pulled forward.
- Commit messages end with:
  ```
  Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_011VMHfi1nbuux7187WtgQTt
  ```

## File Structure

- `app.py`: adds `arw_scan_history`, `build_recent_scan_navigation` and helpers; `format_radar_snapshot` attaches `recent_scans` to an available snapshot.
- `templates/_radar_recent_scans.html` (new): the "Recent scans" section.
- `templates/index.html`: includes the partial inside the available-radar block.
- `tests/test_radar_recent_scans.py` (new): model, snapshot integration and template rendering tests.

---

### Task 1: Recent-scan navigation model and ARW history client

**Files:**
- Modify: `app.py` (near `arw_map_status` and `format_radar_snapshot`)
- Test: `tests/test_radar_recent_scans.py` (create)

**Interfaces:**
- Consumes: ARW `GET /map/history`, which returns `{"site_id", "location", "scans": [{"timestamp": str, "object_count": int, "tracked": bool}]}`, newest first.
- Produces:
  - `arw_scan_history(params: dict[str, str]) -> list[dict]`
  - `build_recent_scan_navigation(scans: list[dict], current_timestamp: str | None, radar_layer: str) -> dict`, with keys:
    - `scans` (newest first; each item has `label`, `query`, `current`)
    - `message`, `position_text`, `previous`, `next`, `at_oldest`, `at_newest`
  - `format_radar_snapshot(...)["recent_scans"]` on available snapshots

- [ ] **Step 1: Write the failing tests**

Create `tests/test_radar_recent_scans.py`:

```python
"""Recent radar scan stepping (ARW compact scan history)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import app as A  # noqa: E402

LOCATION = {
    "name": "Moore", "city": "Moore", "state": "OK",
    "latitude": 35.3395, "longitude": -97.4867,
}
# ARW lists retained scans newest first.
SCANS = [
    {"timestamp": "2026-09-12T18:20:00Z", "object_count": 14, "tracked": True},
    {"timestamp": "2026-09-12T18:15:00Z", "object_count": 12, "tracked": True},
    {"timestamp": "2026-09-12T18:10:00Z", "object_count": 14, "tracked": True},
    {"timestamp": "2026-09-12T18:05:00Z", "object_count": 1, "tracked": True},
    {"timestamp": "2026-09-12T18:00:00Z", "object_count": 9, "tracked": False},
]


class FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class NavigationModelTest(unittest.TestCase):
    def test_middle_scan_has_position_previous_and_next(self):
        nav = A.build_recent_scan_navigation(SCANS, "2026-09-12T18:10:00Z", "precipitation")

        self.assertEqual(
            nav["position_text"],
            "Scan 3 of 5 retained scans, 18:10 UTC, 14 detected echoes, tracked.",
        )
        self.assertEqual(nav["previous"]["label"], "18:05 UTC, 1 detected echo, tracked")
        self.assertEqual(nav["next"]["label"], "18:15 UTC, 12 detected echoes, tracked")
        self.assertEqual(nav["previous"]["query"], {
            "radar_layer": "precipitation",
            "radar_time_mode": "timestamp",
            "radar_datetime": "2026-09-12T18:05:00",
            "radar_load": "1",
        })
        self.assertFalse(nav["at_oldest"])
        self.assertFalse(nav["at_newest"])

    def test_list_is_newest_first_with_only_the_shown_scan_current(self):
        nav = A.build_recent_scan_navigation(SCANS, "2026-09-12T18:10:00Z", "storms")
        self.assertEqual([item["label"][:5] for item in nav["scans"]], ["18:20", "18:15", "18:10", "18:05", "18:00"])
        self.assertEqual([item["current"] for item in nav["scans"]], [False, False, True, False, False])

    def test_oldest_scan_has_no_previous(self):
        nav = A.build_recent_scan_navigation(SCANS, "2026-09-12T18:00:00Z", "storms")
        self.assertIsNone(nav["previous"])
        self.assertTrue(nav["at_oldest"])
        self.assertEqual(nav["position_text"], "Scan 1 of 5 retained scans, 18:00 UTC, 9 detected echoes, not tracked.")

    def test_newest_scan_has_no_next(self):
        nav = A.build_recent_scan_navigation(SCANS, "2026-09-12T18:20:00Z", "storms")
        self.assertIsNone(nav["next"])
        self.assertTrue(nav["at_newest"])

    def test_current_scan_matches_across_timestamp_formats(self):
        nav = A.build_recent_scan_navigation(SCANS, "2026-09-12T18:15:00", "storms")
        self.assertEqual(nav["position_text"][:9], "Scan 4 of")

    def test_shown_scan_outside_the_retained_scans_is_said_plainly(self):
        nav = A.build_recent_scan_navigation(SCANS, "2026-09-12T16:00:00Z", "storms")
        self.assertEqual(nav["position_text"], "The scan shown is not one of the 5 recent scans ARW retains.")
        self.assertIsNone(nav["previous"])
        self.assertIsNone(nav["next"])
        self.assertFalse(any(item["current"] for item in nav["scans"]))

    def test_empty_history_explains_itself(self):
        nav = A.build_recent_scan_navigation([], "2026-09-12T18:00:00Z", "storms")
        self.assertEqual(nav["scans"], [])
        self.assertEqual(nav["message"], "ARW has no recent scans retained for this radar yet.")

    def test_scans_spanning_midnight_include_the_date(self):
        scans = [
            {"timestamp": "2026-09-13T00:02:00Z", "object_count": 2, "tracked": True},
            {"timestamp": "2026-09-12T23:57:00Z", "object_count": 2, "tracked": True},
        ]
        nav = A.build_recent_scan_navigation(scans, "2026-09-13T00:02:00Z", "storms")
        self.assertEqual(nav["previous"]["label"], "2026-09-12 23:57 UTC, 2 detected echoes, tracked")

    def test_unparseable_entries_are_skipped(self):
        nav = A.build_recent_scan_navigation(
            [{"timestamp": "not a time", "object_count": 1, "tracked": True}] + SCANS[:1],
            "2026-09-12T18:20:00Z",
            "storms",
        )
        self.assertEqual(len(nav["scans"]), 1)


class SnapshotIntegrationTest(unittest.TestCase):
    def setUp(self):
        self._real_get = A.requests.get
        self.urls: list[str] = []

    def tearDown(self):
        A.requests.get = self._real_get

    def _route(self, history):
        def fake_get(url, **kwargs):
            self.urls.append(url)
            if url.endswith("/map/history"):
                if isinstance(history, Exception):
                    raise history
                return FakeResponse({"site_id": "KTLX", "scans": history})
            return FakeResponse({
                "available": True, "refresh_state": "ready",
                "scan_timestamp": "2026-09-12T18:10:00Z",
            })
        A.requests.get = fake_get

    def test_available_snapshot_carries_recent_scans(self):
        self._route(SCANS)
        snapshot = A.format_radar_snapshot(
            LOCATION, "intensity", "timestamp", "2026-09-12T18:10:00", requested=True
        )
        self.assertTrue(snapshot["available"])
        self.assertTrue(any(url.endswith("/map/history") for url in self.urls))
        self.assertEqual(snapshot["recent_scans"]["position_text"][:9], "Scan 3 of")
        self.assertEqual(snapshot["recent_scans"]["next"]["query"]["radar_layer"], "intensity")

    def test_history_failure_keeps_the_map_and_says_so(self):
        self._route(A.requests.ConnectionError("refused"))
        snapshot = A.format_radar_snapshot(LOCATION, "storms", "latest", requested=True)
        self.assertTrue(snapshot["available"])
        self.assertEqual(snapshot["recent_scans"]["message"], "Recent scans could not be loaded from ARW.")

    def test_unavailable_snapshot_does_not_request_history(self):
        A.requests.get = lambda url, **kwargs: self.urls.append(url) or FakeResponse(
            {"available": False, "poll_after_seconds": 3}
        )
        snapshot = A.format_radar_snapshot(LOCATION, "storms", "latest", requested=True)
        self.assertNotIn("recent_scans", snapshot)
        self.assertFalse(any(url.endswith("/map/history") for url in self.urls))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run them to verify they fail**

Run: `.venv/Scripts/python.exe -m unittest tests.test_radar_recent_scans -v`
Expected: FAIL with `AttributeError: module 'app' has no attribute 'build_recent_scan_navigation'`.

- [ ] **Step 3: Implement the history client and navigation model**

In `app.py`, below `ARW_STATUS_TIMEOUT_SECONDS = 3` add `ARW_HISTORY_TIMEOUT_SECONDS = 3`. After `arw_map_status`, add:

```python
def arw_scan_history(params: dict[str, str]) -> list[dict[str, Any]]:
    """Ask ARW which recent scans it retains for the radar at this location.

    ARW lists them newest first.  Each timestamp selects exactly that scan
    when passed back as a map request's datetime.
    """
    response = requests.get(
        f"{ARW_BASE_URL}/map/history",
        params=params,
        timeout=ARW_HISTORY_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    return response.json().get("scans", [])


def _scan_instant(timestamp: Any) -> datetime | None:
    """A scan time as naive UTC, whatever ISO-8601 form ARW or a form used."""
    if not isinstance(timestamp, str):
        return None
    try:
        parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is not None:
        parsed = parsed.astimezone(timezone.utc).replace(tzinfo=None)
    return parsed


def build_recent_scan_navigation(
    scans: list[dict[str, Any]], current_timestamp: str | None, radar_layer: str
) -> dict[str, Any]:
    """Previous/next links and a list for the scans ARW retains.

    "Previous" is the older scan and "next" the newer one.  Labels say
    "detected echoes" because ARW's objects are radar echoes, not confirmed
    storms, and "not tracked" when ARW has no tracking context for a scan.
    """
    entries = []
    for scan in scans:
        instant = _scan_instant(scan.get("timestamp"))
        if instant is None:
            continue
        entries.append({
            "instant": instant,
            "object_count": int(scan.get("object_count") or 0),
            "tracked": bool(scan.get("tracked")),
        })
    navigation: dict[str, Any] = {
        "scans": [], "message": None, "position_text": None,
        "previous": None, "next": None, "at_oldest": False, "at_newest": False,
    }
    if not entries:
        navigation["message"] = "ARW has no recent scans retained for this radar yet."
        return navigation

    entries.sort(key=lambda entry: entry["instant"])  # oldest first
    show_date = len({entry["instant"].date() for entry in entries}) > 1
    current = _scan_instant(current_timestamp)
    current_index = next(
        (index for index, entry in enumerate(entries) if entry["instant"] == current), None
    )

    items = []
    for index, entry in enumerate(entries):
        time_text = entry["instant"].strftime("%Y-%m-%d %H:%M UTC" if show_date else "%H:%M UTC")
        count = entry["object_count"]
        echoes = "detected echo" if count == 1 else "detected echoes"
        tracking = "tracked" if entry["tracked"] else "not tracked"
        items.append({
            "label": f"{time_text}, {count} {echoes}, {tracking}",
            "query": {
                "radar_layer": radar_layer,
                "radar_time_mode": "timestamp",
                "radar_datetime": entry["instant"].strftime("%Y-%m-%dT%H:%M:%S"),
                "radar_load": "1",
            },
            "current": index == current_index,
        })
    navigation["scans"] = list(reversed(items))  # newest first, as ARW lists them

    if current_index is None:
        navigation["position_text"] = (
            f"The scan shown is not one of the {len(items)} recent scans ARW retains."
        )
        return navigation
    navigation["position_text"] = (
        f"Scan {current_index + 1} of {len(items)} retained scans, {items[current_index]['label']}."
    )
    navigation["previous"] = items[current_index - 1] if current_index > 0 else None
    navigation["next"] = items[current_index + 1] if current_index + 1 < len(items) else None
    navigation["at_oldest"] = current_index == 0
    navigation["at_newest"] = current_index == len(items) - 1
    return navigation
```

In `format_radar_snapshot`, directly before the final `return {**base_snapshot, "available": True, ...}`, add:

```python
    try:
        recent_scans = build_recent_scan_navigation(
            arw_scan_history(params), scan_timestamp or api_datetime, radar_layer
        )
    except (requests.RequestException, ValueError):
        recent_scans = {
            "scans": [], "message": "Recent scans could not be loaded from ARW.",
            "position_text": None, "previous": None, "next": None,
            "at_oldest": False, "at_newest": False,
        }
```

and add `"recent_scans": recent_scans,` to that returned dict.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/Scripts/python.exe -m unittest tests.test_radar_recent_scans -v`
Expected: all PASS.

- [ ] **Step 5: Run the whole suite**

Run: `.venv/Scripts/python.exe -m unittest discover -s tests -v`
Expected: all pass. The existing radar tests stub `requests.get` with one payload for every URL. That payload has no `scans` key, so the history comes back empty and those snapshots stay available. If a performance test's call-count or timing expectation now fails because an available snapshot makes one more ARW request, update only that expectation, and note the extra call in the commit message.

- [ ] **Step 6: Commit**

```bash
git add app.py tests/test_radar_recent_scans.py
git commit -m "$(cat <<'EOF'
Build recent radar scan navigation from ARW history

An available radar snapshot now asks ARW which scans it retains for the
location and builds previous/next scan links, a newest-first list with
the shown scan marked, and a position line. Labels say detected echoes
and whether ARW tracked each scan. A history failure keeps the map and
says recent scans could not be loaded.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_011VMHfi1nbuux7187WtgQTt
EOF
)"
```

---

### Task 2: Accessible "Recent scans" section, and a live check

**Files:**
- Create: `templates/_radar_recent_scans.html`
- Modify: `templates/index.html` (inside `{% if radar_snapshot.available %}`, after the "Actual radar scan time" paragraph)
- Test: `tests/test_radar_recent_scans.py`

**Interfaces:**
- Consumes: `radar_snapshot.recent_scans` from Task 1.

- [ ] **Step 1: Write the failing rendering tests**

Append to `tests/test_radar_recent_scans.py`, above the `if __name__ == "__main__":` block:

```python
class RecentScansTemplateTest(unittest.TestCase):
    def _render(self, current_timestamp, scans=SCANS):
        navigation = A.build_recent_scan_navigation(scans, current_timestamp, "footprints")
        with A.app.test_request_context("/"):
            return A.render_template("_radar_recent_scans.html", radar_snapshot={"recent_scans": navigation})

    def test_middle_scan_renders_heading_status_links_and_current_marker(self):
        html = self._render("2026-09-12T18:10:00Z")
        self.assertIn('<h3 id="radar-recent-scans-heading">Recent scans</h3>', html)
        self.assertIn('id="radar-scan-position" role="status" aria-live="polite" tabindex="-1"', html)
        self.assertIn("Scan 3 of 5 retained scans, 18:10 UTC, 14 detected echoes, tracked.", html)
        self.assertIn("Previous scan: 18:05 UTC, 1 detected echo, tracked</a>", html)
        self.assertIn("Next scan: 18:15 UTC, 12 detected echoes, tracked</a>", html)
        self.assertEqual(html.count('aria-current="true"'), 1)
        self.assertIn("radar_time_mode=timestamp", html)
        self.assertIn("radar_layer=footprints", html)
        self.assertIn("#radar-scan-position", html)

    def test_oldest_scan_says_so_instead_of_a_dead_previous_link(self):
        html = self._render("2026-09-12T18:00:00Z")
        self.assertIn("This is the oldest retained scan.", html)
        self.assertNotIn("Previous scan:", html)

    def test_newest_scan_says_so_instead_of_a_dead_next_link(self):
        html = self._render("2026-09-12T18:20:00Z")
        self.assertIn("This is the newest retained scan.", html)
        self.assertNotIn("Next scan:", html)

    def test_empty_history_shows_only_the_explanation(self):
        html = self._render("2026-09-12T18:00:00Z", scans=[])
        self.assertIn("ARW has no recent scans retained for this radar yet.", html)
        self.assertNotIn("<a ", html)
```

- [ ] **Step 2: Run them to verify they fail**

Run: `.venv/Scripts/python.exe -m unittest tests.test_radar_recent_scans.RecentScansTemplateTest -v`
Expected: FAIL with `jinja2.exceptions.TemplateNotFound: _radar_recent_scans.html`.

- [ ] **Step 3: Create the partial and include it**

Create `templates/_radar_recent_scans.html`:

```html
{% set recent = radar_snapshot.recent_scans %}
{% if recent %}
  <section class="radar-recent-scans" aria-labelledby="radar-recent-scans-heading">
    <h3 id="radar-recent-scans-heading">Recent scans</h3>
    {% if recent.message %}
      <p>{{ recent.message }}</p>
    {% else %}
      {% if recent.position_text %}
        <p id="radar-scan-position" role="status" aria-live="polite" tabindex="-1">{{ recent.position_text }}</p>
      {% endif %}
      {% if recent.previous %}
        <p><a href="{{ url_for('index', **recent.previous.query) }}#radar-scan-position">Previous scan: {{ recent.previous.label }}</a></p>
      {% elif recent.at_oldest %}
        <p>This is the oldest retained scan.</p>
      {% endif %}
      {% if recent.next %}
        <p><a href="{{ url_for('index', **recent.next.query) }}#radar-scan-position">Next scan: {{ recent.next.label }}</a></p>
      {% elif recent.at_newest %}
        <p>This is the newest retained scan.</p>
      {% endif %}
      <ul>
        {% for scan in recent.scans %}
          <li>
            <a href="{{ url_for('index', **scan.query) }}#radar-scan-position"{% if scan.current %} aria-current="true"{% endif %}>{{ scan.label }}{% if scan.current %} (shown){% endif %}</a>
          </li>
        {% endfor %}
      </ul>
    {% endif %}
  </section>
{% endif %}
```

In `templates/index.html`, directly after the `{% if radar_snapshot.scan_timestamp %} ... {% endif %}` block that prints "Actual radar scan time (UTC)", add:

```html
          {% include "_radar_recent_scans.html" %}
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/Scripts/python.exe -m unittest discover -s tests -v`
Expected: all PASS.

- [ ] **Step 5: Live check with a running ARW and a screen reader**

1. Start ARW (`.venv/Scripts/python.exe -m uvicorn src.server:app --port 8000` in the ARW repository) and let a radar with current precipitation retain at least three live scans (check ARW's `/map/history`).
2. Start Weather Kitten (`uv run flask --app app run --port 5000`) with a saved location in that radar's area.
3. Load radar on the latest scan, then use the "Recent scans" links with the keyboard only, with a screen reader running (NVDA).

For each of the four layers (storm interpretation, footprints, intensity bands, precipitation field), step from the newest retained scan to the oldest and back. Confirm:

- after each step, focus lands on the position line and it is read ("Scan N of M retained scans, ...")
- the map and its GeoJSON link show that scan's time (the "Actual radar scan time" line matches the position line)
- the newest and oldest scans read "This is the newest/oldest retained scan." with no dead link
- the list marks exactly one scan as shown
- automatic refresh (if enabled in settings) does not move off a stepped-to scan

Record the radar, times and any issue found.

- [ ] **Step 6: Commit**

```bash
git add templates/_radar_recent_scans.html templates/index.html tests/test_radar_recent_scans.py
git commit -m "$(cat <<'EOF'
Add accessible recent radar scan stepping to the radar panel

A Recent scans section under the radar map offers previous and next
scan links, a newest-first list with the shown scan marked
aria-current, and a polite status line that the links land on after
reloading. The ends of the retained scans are stated in text rather
than shown as dead links. Verified with a live ARW and a screen reader
across all four layers.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_011VMHfi1nbuux7187WtgQTt
EOF
)"
```

- [ ] **Step 7: Record progress in ARW**

In the ARW repository's `PROGRESS.md`, add the Weather Kitten scan stepping result and the live check notes. Then commit there with the same attribution lines.
