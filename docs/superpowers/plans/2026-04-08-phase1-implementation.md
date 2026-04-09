# ARW Phase 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the foundational pipeline — user enters city/state, picks a radar, gets a speech summary of detected rain objects.

**Architecture:** Python FastAPI backend exposes a local REST API. NVGT frontend (not built in this phase) will call these endpoints. Each pipeline stage is a separate module with pure functions (except ingest, which does I/O).

**Tech Stack:** Python 3.11+, FastAPI, uvicorn, Py-ART, nexradaws, boto3, scipy, numpy, geopy, pytest

---

## File Structure

```
arw/
├── requirements.txt               # Python dependencies
├── src/
│   ├── __init__.py
│   ├── models.py                  # Pydantic response models
│   ├── sites.py                   # NEXRAD site database, geocoding, beam height ranking
│   ├── ingest.py                  # NEXRAD Level II download + caching (only network module)
│   ├── parser.py                  # Py-ART parsing, reflectivity extraction
│   ├── detection.py               # Connected component labeling, nested layers, object properties
│   ├── summary.py                 # Speech text generation
│   └── server.py                  # FastAPI app, all endpoint definitions
├── tests/
│   ├── __init__.py
│   ├── conftest.py                # Shared fixtures (sample radar data, mock ingest)
│   ├── unit/
│   │   ├── __init__.py
│   │   ├── test_models.py         # Pydantic model validation
│   │   ├── test_sites.py          # Geocoding, distance, beam height, ranking
│   │   ├── test_parser.py         # Reflectivity extraction from Py-ART radar objects
│   │   ├── test_detection.py      # Object detection, layering, properties, filtering
│   │   └── test_summary.py        # Speech text formatting
│   ├── smoke/
│   │   ├── __init__.py
│   │   └── test_server_smoke.py   # Server starts, all endpoints return 200
│   └── e2e/
│       ├── __init__.py
│       └── test_full_pipeline.py  # City → sites → scan → objects → summary
```

---

### Task 1: Project Setup & Dependencies

**Files:**
- Create: `requirements.txt`
- Create: `src/__init__.py`
- Create: `tests/__init__.py`
- Create: `tests/unit/__init__.py`
- Create: `tests/smoke/__init__.py`
- Create: `tests/e2e/__init__.py`

- [ ] **Step 1: Create requirements.txt**

```
fastapi==0.115.0
uvicorn==0.30.6
arm-pyart==1.19.1
nexradaws==1.1.0
boto3==1.35.0
scipy==1.14.0
numpy==1.26.4
geopy==2.4.1
pytest==8.3.2
httpx==0.27.0
```

- [ ] **Step 2: Create __init__.py files**

Create empty `src/__init__.py`, `tests/__init__.py`, `tests/unit/__init__.py`, `tests/smoke/__init__.py`, `tests/e2e/__init__.py`.

- [ ] **Step 3: Install dependencies**

Run: `pip install -r requirements.txt`
Expected: All packages install successfully.

- [ ] **Step 4: Verify imports work**

Run: `python -c "import fastapi; import pyart; import nexradaws; import scipy; import geopy; print('All imports OK')"`
Expected: `All imports OK`

- [ ] **Step 5: Commit**

```bash
git add requirements.txt src/__init__.py tests/__init__.py tests/unit/__init__.py tests/smoke/__init__.py tests/e2e/__init__.py
git commit -m "feat: project setup with dependencies and package structure"
```

---

### Task 2: Pydantic Response Models

**Files:**
- Create: `src/models.py`
- Create: `tests/unit/test_models.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_models.py
from src.models import RadarSite, ScanMeta, IntensityLayer, RainObject, ObjectsResponse, SummaryResponse


def test_radar_site_model():
    site = RadarSite(
        site_id="KTLX",
        name="Oklahoma City",
        latitude=35.3331,
        longitude=-97.2778,
        elevation_m=370.0,
        distance_km=50.0,
        beam_height_m=1200.0,
    )
    assert site.site_id == "KTLX"
    assert site.distance_km == 50.0


def test_scan_meta_model():
    meta = ScanMeta(
        site_id="KTLX",
        timestamp="2026-04-08T18:30:00Z",
        elevation_angles=[0.5, 1.5, 2.4],
    )
    assert meta.site_id == "KTLX"
    assert len(meta.elevation_angles) == 3


def test_intensity_layer_model():
    layer = IntensityLayer(
        label="heavy rain",
        min_dbz=40.0,
        max_dbz=50.0,
        area_km2=12.5,
    )
    assert layer.label == "heavy rain"


def test_rain_object_model():
    obj = RainObject(
        object_id=1,
        centroid_lat=35.5,
        centroid_lon=-97.3,
        distance_km=25.0,
        bearing_deg=270.0,
        peak_dbz=55.0,
        peak_label="intense rain",
        area_km2=80.0,
        layers=[
            IntensityLayer(label="light rain", min_dbz=20.0, max_dbz=30.0, area_km2=80.0),
            IntensityLayer(label="moderate rain", min_dbz=30.0, max_dbz=40.0, area_km2=40.0),
            IntensityLayer(label="heavy rain", min_dbz=40.0, max_dbz=50.0, area_km2=15.0),
            IntensityLayer(label="intense rain", min_dbz=50.0, max_dbz=60.0, area_km2=5.0),
        ],
    )
    assert obj.peak_label == "intense rain"
    assert len(obj.layers) == 4


def test_objects_response_model():
    resp = ObjectsResponse(
        site_id="KTLX",
        timestamp="2026-04-08T18:30:00Z",
        object_count=0,
        objects=[],
    )
    assert resp.object_count == 0


def test_summary_response_model():
    resp = SummaryResponse(
        site_id="KTLX",
        timestamp="2026-04-08T18:30:00Z",
        text="KTLX: No significant precipitation detected.",
    )
    assert "No significant" in resp.text
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_models.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.models'`

- [ ] **Step 3: Write the implementation**

```python
# src/models.py
from pydantic import BaseModel


class RadarSite(BaseModel):
    site_id: str
    name: str
    latitude: float
    longitude: float
    elevation_m: float
    distance_km: float
    beam_height_m: float


class ScanMeta(BaseModel):
    site_id: str
    timestamp: str
    elevation_angles: list[float]


class IntensityLayer(BaseModel):
    label: str
    min_dbz: float
    max_dbz: float
    area_km2: float


class RainObject(BaseModel):
    object_id: int
    centroid_lat: float
    centroid_lon: float
    distance_km: float
    bearing_deg: float
    peak_dbz: float
    peak_label: str
    area_km2: float
    layers: list[IntensityLayer]


class ObjectsResponse(BaseModel):
    site_id: str
    timestamp: str
    object_count: int
    objects: list[RainObject]


class SummaryResponse(BaseModel):
    site_id: str
    timestamp: str
    text: str
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_models.py -v`
Expected: All 6 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/models.py tests/unit/test_models.py
git commit -m "feat: add Pydantic response models for API"
```

---

### Task 3: Site Database & Geocoding

**Files:**
- Create: `src/sites.py`
- Create: `tests/unit/test_sites.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/test_sites.py
import math
from src.sites import (
    NEXRAD_SITES,
    geocode_city_state,
    haversine_distance_km,
    compute_beam_height_m,
    rank_sites,
)


def test_nexrad_sites_has_entries():
    assert len(NEXRAD_SITES) > 150
    ktlx = next(s for s in NEXRAD_SITES if s["site_id"] == "KTLX")
    assert ktlx["name"] == "Oklahoma City"
    assert abs(ktlx["latitude"] - 35.3331) < 0.01
    assert abs(ktlx["longitude"] - (-97.2778)) < 0.01
    assert ktlx["elevation_m"] > 0


def test_haversine_distance_km():
    # OKC to Dallas is roughly 300 km
    dist = haversine_distance_km(35.4676, -97.5164, 32.7767, -96.7970)
    assert 290 < dist < 320


def test_compute_beam_height_m():
    # At 100 km distance, 0.5° elevation, ~0m radar elevation
    height = compute_beam_height_m(distance_km=100.0, radar_elevation_m=0.0)
    # Should be roughly 1600-1700 m (geometric + earth curvature)
    assert 1400 < height < 2000


def test_compute_beam_height_m_increases_with_distance():
    h1 = compute_beam_height_m(distance_km=50.0, radar_elevation_m=0.0)
    h2 = compute_beam_height_m(distance_km=200.0, radar_elevation_m=0.0)
    assert h2 > h1


def test_compute_beam_height_m_includes_radar_elevation():
    h1 = compute_beam_height_m(distance_km=100.0, radar_elevation_m=0.0)
    h2 = compute_beam_height_m(distance_km=100.0, radar_elevation_m=500.0)
    assert h2 - h1 == 500.0


def test_rank_sites_returns_sorted_by_beam_height():
    results = rank_sites(lat=35.4676, lon=-97.5164)
    assert len(results) > 0
    # Check sorted ascending by beam_height_m
    for i in range(len(results) - 1):
        assert results[i]["beam_height_m"] <= results[i + 1]["beam_height_m"]


def test_rank_sites_filters_high_beam_height():
    results = rank_sites(lat=35.4676, lon=-97.5164)
    for r in results:
        assert r["beam_height_m"] <= 10000.0


def test_rank_sites_includes_distance():
    results = rank_sites(lat=35.4676, lon=-97.5164)
    for r in results:
        assert r["distance_km"] > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_sites.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.sites'`

- [ ] **Step 3: Write the implementation**

```python
# src/sites.py
import math
from geopy.geocoders import Nominatim

EARTH_RADIUS_KM = 6371.0
LOWEST_ELEVATION_DEG = 0.5
MAX_BEAM_HEIGHT_M = 10000.0

# Full NEXRAD WSR-88D site list.
# Each entry: site_id, name, latitude, longitude, elevation_m
NEXRAD_SITES = [
    {"site_id": "KABR", "name": "Aberdeen", "latitude": 45.4558, "longitude": -98.4131, "elevation_m": 397},
    {"site_id": "KABX", "name": "Albuquerque", "latitude": 35.1497, "longitude": -106.8239, "elevation_m": 1789},
    {"site_id": "KAKQ", "name": "Wakefield", "latitude": 36.9839, "longitude": -77.0075, "elevation_m": 34},
    {"site_id": "KAMA", "name": "Amarillo", "latitude": 35.2334, "longitude": -101.7092, "elevation_m": 1093},
    {"site_id": "KAMX", "name": "Miami", "latitude": 25.6111, "longitude": -80.4128, "elevation_m": 4},
    {"site_id": "KAPX", "name": "Gaylord", "latitude": 44.9072, "longitude": -84.7197, "elevation_m": 446},
    {"site_id": "KARX", "name": "La Crosse", "latitude": 43.8228, "longitude": -91.1911, "elevation_m": 389},
    {"site_id": "KATX", "name": "Seattle", "latitude": 48.1946, "longitude": -122.4957, "elevation_m": 151},
    {"site_id": "KBBX", "name": "Beale AFB", "latitude": 39.4961, "longitude": -121.6317, "elevation_m": 53},
    {"site_id": "KBGM", "name": "Binghamton", "latitude": 42.1997, "longitude": -75.9847, "elevation_m": 490},
    {"site_id": "KBHX", "name": "Eureka", "latitude": 40.4986, "longitude": -124.2919, "elevation_m": 732},
    {"site_id": "KBIS", "name": "Bismarck", "latitude": 46.7708, "longitude": -100.7606, "elevation_m": 505},
    {"site_id": "KBLX", "name": "Billings", "latitude": 45.8536, "longitude": -108.6069, "elevation_m": 1097},
    {"site_id": "KBMX", "name": "Birmingham", "latitude": 33.1722, "longitude": -86.7697, "elevation_m": 197},
    {"site_id": "KBOX", "name": "Boston", "latitude": 41.9558, "longitude": -71.1369, "elevation_m": 36},
    {"site_id": "KBRO", "name": "Brownsville", "latitude": 25.9161, "longitude": -97.4189, "elevation_m": 7},
    {"site_id": "KBUF", "name": "Buffalo", "latitude": 42.9486, "longitude": -78.7369, "elevation_m": 211},
    {"site_id": "KBYX", "name": "Key West", "latitude": 24.5975, "longitude": -81.7031, "elevation_m": 3},
    {"site_id": "KCAE", "name": "Columbia", "latitude": 33.9486, "longitude": -81.1186, "elevation_m": 70},
    {"site_id": "KCBW", "name": "Caribou", "latitude": 46.0392, "longitude": -67.8067, "elevation_m": 227},
    {"site_id": "KCBX", "name": "Boise", "latitude": 43.4908, "longitude": -116.2356, "elevation_m": 933},
    {"site_id": "KCCX", "name": "State College", "latitude": 40.9228, "longitude": -78.0039, "elevation_m": 733},
    {"site_id": "KCLE", "name": "Cleveland", "latitude": 41.4131, "longitude": -81.8597, "elevation_m": 233},
    {"site_id": "KCLX", "name": "Charleston", "latitude": 32.6556, "longitude": -81.0422, "elevation_m": 30},
    {"site_id": "KCRP", "name": "Corpus Christi", "latitude": 27.7842, "longitude": -97.5108, "elevation_m": 14},
    {"site_id": "KCXX", "name": "Burlington", "latitude": 44.5111, "longitude": -73.1669, "elevation_m": 97},
    {"site_id": "KCYS", "name": "Cheyenne", "latitude": 41.1519, "longitude": -104.8061, "elevation_m": 1868},
    {"site_id": "KDAX", "name": "Sacramento", "latitude": 38.5011, "longitude": -121.6778, "elevation_m": 9},
    {"site_id": "KDDC", "name": "Dodge City", "latitude": 37.7608, "longitude": -99.9686, "elevation_m": 790},
    {"site_id": "KDFX", "name": "Laughlin AFB", "latitude": 29.2725, "longitude": -100.2803, "elevation_m": 345},
    {"site_id": "KDGX", "name": "Jackson", "latitude": 32.2800, "longitude": -89.9844, "elevation_m": 153},
    {"site_id": "KDIX", "name": "Philadelphia", "latitude": 39.9469, "longitude": -74.4108, "elevation_m": 45},
    {"site_id": "KDLH", "name": "Duluth", "latitude": 46.8369, "longitude": -92.2097, "elevation_m": 435},
    {"site_id": "KDMX", "name": "Des Moines", "latitude": 41.7311, "longitude": -93.7228, "elevation_m": 299},
    {"site_id": "KDOX", "name": "Dover AFB", "latitude": 38.8256, "longitude": -75.4400, "elevation_m": 15},
    {"site_id": "KDTX", "name": "Detroit", "latitude": 42.6997, "longitude": -83.4717, "elevation_m": 327},
    {"site_id": "KDVN", "name": "Davenport", "latitude": 41.6117, "longitude": -90.5808, "elevation_m": 230},
    {"site_id": "KDYX", "name": "Dyess AFB", "latitude": 32.5386, "longitude": -99.2542, "elevation_m": 463},
    {"site_id": "KEAX", "name": "Kansas City", "latitude": 38.8103, "longitude": -94.2644, "elevation_m": 303},
    {"site_id": "KEMX", "name": "Tucson", "latitude": 31.8936, "longitude": -110.6303, "elevation_m": 1587},
    {"site_id": "KENX", "name": "Albany", "latitude": 42.5864, "longitude": -74.0639, "elevation_m": 557},
    {"site_id": "KEOX", "name": "Fort Rucker", "latitude": 31.4606, "longitude": -85.4592, "elevation_m": 132},
    {"site_id": "KEPZ", "name": "El Paso", "latitude": 31.8731, "longitude": -106.6981, "elevation_m": 1251},
    {"site_id": "KESX", "name": "Las Vegas", "latitude": 35.7011, "longitude": -114.8917, "elevation_m": 1483},
    {"site_id": "KEVX", "name": "Eglin AFB", "latitude": 30.5644, "longitude": -85.9214, "elevation_m": 43},
    {"site_id": "KEWX", "name": "Austin/San Antonio", "latitude": 29.7039, "longitude": -98.0286, "elevation_m": 193},
    {"site_id": "KEYX", "name": "Edwards AFB", "latitude": 35.0978, "longitude": -117.5608, "elevation_m": 840},
    {"site_id": "KFCX", "name": "Roanoke", "latitude": 37.0242, "longitude": -80.2742, "elevation_m": 874},
    {"site_id": "KFDR", "name": "Frederick", "latitude": 34.3622, "longitude": -98.9764, "elevation_m": 386},
    {"site_id": "KFDX", "name": "Cannon AFB", "latitude": 34.6356, "longitude": -103.6292, "elevation_m": 1417},
    {"site_id": "KFFC", "name": "Atlanta", "latitude": 33.3636, "longitude": -84.5658, "elevation_m": 262},
    {"site_id": "KFSD", "name": "Sioux Falls", "latitude": 43.5878, "longitude": -96.7292, "elevation_m": 436},
    {"site_id": "KFSX", "name": "Flagstaff", "latitude": 34.5744, "longitude": -111.1983, "elevation_m": 2261},
    {"site_id": "KFTG", "name": "Denver", "latitude": 39.7867, "longitude": -104.5458, "elevation_m": 1675},
    {"site_id": "KFWS", "name": "Dallas/Fort Worth", "latitude": 32.5731, "longitude": -97.3031, "elevation_m": 208},
    {"site_id": "KGGW", "name": "Glasgow", "latitude": 48.2064, "longitude": -106.6253, "elevation_m": 694},
    {"site_id": "KGJX", "name": "Grand Junction", "latitude": 39.0622, "longitude": -108.2139, "elevation_m": 3045},
    {"site_id": "KGLD", "name": "Goodland", "latitude": 39.3669, "longitude": -101.7003, "elevation_m": 1113},
    {"site_id": "KGRB", "name": "Green Bay", "latitude": 44.4986, "longitude": -88.1111, "elevation_m": 208},
    {"site_id": "KGRK", "name": "Fort Hood", "latitude": 30.7217, "longitude": -97.3828, "elevation_m": 164},
    {"site_id": "KGRR", "name": "Grand Rapids", "latitude": 42.8939, "longitude": -85.5447, "elevation_m": 237},
    {"site_id": "KGSP", "name": "Greenville/Spartanburg", "latitude": 34.8833, "longitude": -82.2200, "elevation_m": 296},
    {"site_id": "KGWX", "name": "Columbus AFB", "latitude": 33.8967, "longitude": -88.3289, "elevation_m": 145},
    {"site_id": "KGYX", "name": "Portland", "latitude": 43.8914, "longitude": -70.2564, "elevation_m": 125},
    {"site_id": "KHDX", "name": "Holloman AFB", "latitude": 33.0769, "longitude": -106.1222, "elevation_m": 1287},
    {"site_id": "KHGX", "name": "Houston", "latitude": 29.4719, "longitude": -95.0792, "elevation_m": 5},
    {"site_id": "KHNX", "name": "Hanford", "latitude": 36.3142, "longitude": -119.6319, "elevation_m": 74},
    {"site_id": "KHPX", "name": "Fort Campbell", "latitude": 36.7369, "longitude": -87.2847, "elevation_m": 176},
    {"site_id": "KHTX", "name": "Huntsville", "latitude": 34.9306, "longitude": -86.0833, "elevation_m": 536},
    {"site_id": "KICT", "name": "Wichita", "latitude": 37.6544, "longitude": -97.4428, "elevation_m": 407},
    {"site_id": "KICX", "name": "Cedar City", "latitude": 37.5911, "longitude": -112.8622, "elevation_m": 3231},
    {"site_id": "KILN", "name": "Cincinnati", "latitude": 39.4203, "longitude": -83.8217, "elevation_m": 322},
    {"site_id": "KILX", "name": "Lincoln", "latitude": 40.1506, "longitude": -89.3369, "elevation_m": 177},
    {"site_id": "KIND", "name": "Indianapolis", "latitude": 39.7075, "longitude": -86.2803, "elevation_m": 241},
    {"site_id": "KINX", "name": "Tulsa", "latitude": 36.1750, "longitude": -95.5644, "elevation_m": 204},
    {"site_id": "KIWA", "name": "Phoenix", "latitude": 33.2892, "longitude": -111.6700, "elevation_m": 412},
    {"site_id": "KJAX", "name": "Jacksonville", "latitude": 30.4847, "longitude": -81.7019, "elevation_m": 10},
    {"site_id": "KJGX", "name": "Robins AFB", "latitude": 32.6753, "longitude": -83.3511, "elevation_m": 159},
    {"site_id": "KJKL", "name": "Jackson", "latitude": 37.5908, "longitude": -83.3131, "elevation_m": 415},
    {"site_id": "KLBB", "name": "Lubbock", "latitude": 33.6539, "longitude": -101.8142, "elevation_m": 993},
    {"site_id": "KLCH", "name": "Lake Charles", "latitude": 30.1253, "longitude": -93.2158, "elevation_m": 4},
    {"site_id": "KLIX", "name": "New Orleans", "latitude": 30.3367, "longitude": -89.8256, "elevation_m": 7},
    {"site_id": "KLNX", "name": "North Platte", "latitude": 41.9578, "longitude": -100.5764, "elevation_m": 905},
    {"site_id": "KLOT", "name": "Chicago", "latitude": 41.6044, "longitude": -88.0847, "elevation_m": 202},
    {"site_id": "KLRX", "name": "Elko", "latitude": 40.7400, "longitude": -116.8025, "elevation_m": 2056},
    {"site_id": "KLSX", "name": "St. Louis", "latitude": 38.6986, "longitude": -90.6828, "elevation_m": 185},
    {"site_id": "KLTX", "name": "Wilmington", "latitude": 33.9892, "longitude": -78.4292, "elevation_m": 19},
    {"site_id": "KLVX", "name": "Louisville", "latitude": 37.9753, "longitude": -85.9436, "elevation_m": 219},
    {"site_id": "KLWX", "name": "Sterling", "latitude": 38.9753, "longitude": -77.4778, "elevation_m": 83},
    {"site_id": "KLZK", "name": "Little Rock", "latitude": 34.8364, "longitude": -92.2622, "elevation_m": 173},
    {"site_id": "KMAF", "name": "Midland/Odessa", "latitude": 31.9433, "longitude": -102.1892, "elevation_m": 874},
    {"site_id": "KMAX", "name": "Medford", "latitude": 42.0811, "longitude": -122.7167, "elevation_m": 2290},
    {"site_id": "KMBX", "name": "Minot AFB", "latitude": 48.3925, "longitude": -100.8644, "elevation_m": 455},
    {"site_id": "KMHX", "name": "Morehead City", "latitude": 34.7758, "longitude": -76.8764, "elevation_m": 9},
    {"site_id": "KMKX", "name": "Milwaukee", "latitude": 42.9678, "longitude": -88.5506, "elevation_m": 292},
    {"site_id": "KMLB", "name": "Melbourne", "latitude": 28.1133, "longitude": -80.6542, "elevation_m": 11},
    {"site_id": "KMOB", "name": "Mobile", "latitude": 30.6794, "longitude": -88.2397, "elevation_m": 63},
    {"site_id": "KMPX", "name": "Minneapolis", "latitude": 44.8489, "longitude": -93.5653, "elevation_m": 288},
    {"site_id": "KMQT", "name": "Marquette", "latitude": 46.5314, "longitude": -87.5486, "elevation_m": 430},
    {"site_id": "KMRX", "name": "Knoxville", "latitude": 36.1686, "longitude": -83.4017, "elevation_m": 408},
    {"site_id": "KMSX", "name": "Missoula", "latitude": 47.0411, "longitude": -113.9861, "elevation_m": 2394},
    {"site_id": "KMTX", "name": "Salt Lake City", "latitude": 41.2628, "longitude": -112.4478, "elevation_m": 1969},
    {"site_id": "KMUX", "name": "San Francisco", "latitude": 37.1553, "longitude": -121.8983, "elevation_m": 1057},
    {"site_id": "KMVX", "name": "Grand Forks", "latitude": 47.5278, "longitude": -97.3256, "elevation_m": 301},
    {"site_id": "KMXX", "name": "Maxwell AFB", "latitude": 32.5367, "longitude": -85.7897, "elevation_m": 122},
    {"site_id": "KNKX", "name": "San Diego", "latitude": 32.9189, "longitude": -117.0419, "elevation_m": 291},
    {"site_id": "KNQA", "name": "Memphis", "latitude": 35.3447, "longitude": -89.8733, "elevation_m": 86},
    {"site_id": "KOAX", "name": "Omaha", "latitude": 41.3203, "longitude": -96.3667, "elevation_m": 350},
    {"site_id": "KOHX", "name": "Nashville", "latitude": 36.2472, "longitude": -86.5625, "elevation_m": 177},
    {"site_id": "KOKX", "name": "New York City", "latitude": 40.8656, "longitude": -72.8639, "elevation_m": 26},
    {"site_id": "KOTX", "name": "Spokane", "latitude": 47.6803, "longitude": -117.6267, "elevation_m": 728},
    {"site_id": "KPAH", "name": "Paducah", "latitude": 37.0683, "longitude": -88.7719, "elevation_m": 119},
    {"site_id": "KPBZ", "name": "Pittsburgh", "latitude": 40.5317, "longitude": -80.2181, "elevation_m": 361},
    {"site_id": "KPDT", "name": "Pendleton", "latitude": 45.6906, "longitude": -118.8531, "elevation_m": 462},
    {"site_id": "KPOE", "name": "Fort Polk", "latitude": 31.1556, "longitude": -92.9756, "elevation_m": 124},
    {"site_id": "KPUX", "name": "Pueblo", "latitude": 38.4597, "longitude": -104.1814, "elevation_m": 1600},
    {"site_id": "KRAX", "name": "Raleigh", "latitude": 35.6656, "longitude": -78.4903, "elevation_m": 106},
    {"site_id": "KRGX", "name": "Reno", "latitude": 39.7542, "longitude": -119.4614, "elevation_m": 2530},
    {"site_id": "KRIW", "name": "Riverton", "latitude": 43.0661, "longitude": -108.4772, "elevation_m": 1697},
    {"site_id": "KRLX", "name": "Charleston", "latitude": 38.3111, "longitude": -81.7231, "elevation_m": 329},
    {"site_id": "KRMX", "name": "Griffiss AFB", "latitude": 43.4678, "longitude": -75.4581, "elevation_m": 462},
    {"site_id": "KRTX", "name": "Portland", "latitude": 45.7150, "longitude": -122.9653, "elevation_m": 479},
    {"site_id": "KSFX", "name": "Pocatello", "latitude": 43.1058, "longitude": -112.6861, "elevation_m": 1364},
    {"site_id": "KSGF", "name": "Springfield", "latitude": 37.2353, "longitude": -93.4006, "elevation_m": 390},
    {"site_id": "KSHV", "name": "Shreveport", "latitude": 32.4508, "longitude": -93.8414, "elevation_m": 83},
    {"site_id": "KSJT", "name": "San Angelo", "latitude": 31.3714, "longitude": -100.4925, "elevation_m": 576},
    {"site_id": "KSOX", "name": "Santa Ana Mountains", "latitude": 33.8178, "longitude": -117.6358, "elevation_m": 923},
    {"site_id": "KSRX", "name": "Fort Smith", "latitude": 35.2903, "longitude": -94.3619, "elevation_m": 195},
    {"site_id": "KTBW", "name": "Tampa Bay", "latitude": 27.7056, "longitude": -82.4017, "elevation_m": 13},
    {"site_id": "KTFX", "name": "Great Falls", "latitude": 47.4597, "longitude": -111.3856, "elevation_m": 1132},
    {"site_id": "KTLH", "name": "Tallahassee", "latitude": 30.3975, "longitude": -84.3289, "elevation_m": 19},
    {"site_id": "KTLX", "name": "Oklahoma City", "latitude": 35.3331, "longitude": -97.2778, "elevation_m": 370},
    {"site_id": "KTWX", "name": "Topeka", "latitude": 38.9969, "longitude": -96.2325, "elevation_m": 417},
    {"site_id": "KTYX", "name": "Montague", "latitude": 43.7558, "longitude": -75.6800, "elevation_m": 562},
    {"site_id": "KUDX", "name": "Rapid City", "latitude": 44.1253, "longitude": -102.8297, "elevation_m": 919},
    {"site_id": "KUEX", "name": "Hastings", "latitude": 40.3208, "longitude": -98.4417, "elevation_m": 602},
    {"site_id": "KVAX", "name": "Moody AFB", "latitude": 30.8900, "longitude": -83.0017, "elevation_m": 54},
    {"site_id": "KVBX", "name": "Vandenberg AFB", "latitude": 34.8383, "longitude": -120.3975, "elevation_m": 376},
    {"site_id": "KVNX", "name": "Vance AFB", "latitude": 36.7408, "longitude": -98.1278, "elevation_m": 369},
    {"site_id": "KVTX", "name": "Los Angeles", "latitude": 34.4114, "longitude": -119.1794, "elevation_m": 831},
    {"site_id": "KVWX", "name": "Evansville", "latitude": 38.2603, "longitude": -87.7247, "elevation_m": 190},
    {"site_id": "KYUX", "name": "Yuma", "latitude": 32.4953, "longitude": -114.6567, "elevation_m": 53},
    {"site_id": "PABC", "name": "Bethel", "latitude": 60.7919, "longitude": -161.8764, "elevation_m": 49},
    {"site_id": "PACG", "name": "Juneau", "latitude": 56.8525, "longitude": -135.5292, "elevation_m": 84},
    {"site_id": "PAEC", "name": "Nome", "latitude": 64.5114, "longitude": -165.2950, "elevation_m": 16},
    {"site_id": "PAHG", "name": "Anchorage", "latitude": 60.7258, "longitude": -151.3514, "elevation_m": 74},
    {"site_id": "PAIH", "name": "Middleton Island", "latitude": 59.4614, "longitude": -146.3031, "elevation_m": 20},
    {"site_id": "PAKC", "name": "King Salmon", "latitude": 58.6794, "longitude": -156.6292, "elevation_m": 19},
    {"site_id": "PAPD", "name": "Fairbanks", "latitude": 65.0350, "longitude": -147.5014, "elevation_m": 790},
    {"site_id": "PGUA", "name": "Guam", "latitude": 13.4544, "longitude": 144.8111, "elevation_m": 110},
    {"site_id": "PHKI", "name": "South Kauai", "latitude": 21.8942, "longitude": -159.5522, "elevation_m": 55},
    {"site_id": "PHKM", "name": "Kamuela", "latitude": 20.1256, "longitude": -155.7781, "elevation_m": 1162},
    {"site_id": "PHMO", "name": "Molokai", "latitude": 21.1328, "longitude": -157.1803, "elevation_m": 416},
    {"site_id": "PHWA", "name": "South Shore", "latitude": 19.0950, "longitude": -155.5686, "elevation_m": 421},
    {"site_id": "TJUA", "name": "San Juan", "latitude": 18.1156, "longitude": -66.0781, "elevation_m": 863},
]


def geocode_city_state(city: str, state: str) -> tuple[float, float]:
    """Convert city/state to (latitude, longitude) using Nominatim."""
    geolocator = Nominatim(user_agent="arw-radar-workstation")
    location = geolocator.geocode(f"{city}, {state}, United States")
    if location is None:
        raise ValueError(f"Could not geocode '{city}, {state}'")
    return (location.latitude, location.longitude)


def haversine_distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Compute great-circle distance in km between two lat/lon points."""
    lat1_r, lon1_r = math.radians(lat1), math.radians(lon1)
    lat2_r, lon2_r = math.radians(lat2), math.radians(lon2)
    dlat = lat2_r - lat1_r
    dlon = lon2_r - lon1_r
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1_r) * math.cos(lat2_r) * math.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(a))


def compute_beam_height_m(distance_km: float, radar_elevation_m: float) -> float:
    """Compute radar beam height at a given distance using the beam height formula.

    beam_height = distance * tan(elevation_angle) + (distance^2 / (2 * earth_radius)) + radar_elevation
    """
    distance_m = distance_km * 1000.0
    elevation_rad = math.radians(LOWEST_ELEVATION_DEG)
    earth_radius_m = EARTH_RADIUS_KM * 1000.0
    height = (
        distance_m * math.tan(elevation_rad)
        + (distance_m ** 2) / (2 * earth_radius_m)
        + radar_elevation_m
    )
    return height


def rank_sites(lat: float, lon: float) -> list[dict]:
    """Rank NEXRAD sites by beam height at the given location.

    Returns list of dicts with site info plus distance_km and beam_height_m,
    sorted by beam_height_m ascending. Filters out sites with beam height > 10 km.
    """
    results = []
    for site in NEXRAD_SITES:
        dist = haversine_distance_km(lat, lon, site["latitude"], site["longitude"])
        beam_h = compute_beam_height_m(dist, site["elevation_m"])
        if beam_h <= MAX_BEAM_HEIGHT_M:
            results.append({
                "site_id": site["site_id"],
                "name": site["name"],
                "latitude": site["latitude"],
                "longitude": site["longitude"],
                "elevation_m": site["elevation_m"],
                "distance_km": round(dist, 1),
                "beam_height_m": round(beam_h, 1),
            })
    results.sort(key=lambda x: x["beam_height_m"])
    return results
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_sites.py -v`
Expected: All 8 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/sites.py tests/unit/test_sites.py
git commit -m "feat: add NEXRAD site database, geocoding, and beam height ranking"
```

---

### Task 4: NEXRAD Ingest Manager

**Files:**
- Create: `src/ingest.py`
- Create: `tests/unit/test_ingest.py`
- Create: `tests/conftest.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/test_ingest.py
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
        mock_result.success = True
        local_file = tmp_path / "KTLX" / "KTLX20260408_183000_V06"
        mock_result.filepath = str(local_file)
        mock_conn.return_value.download.return_value = [mock_result]
        path = download_scan("KTLX", mock_scan)
        assert path is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_ingest.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.ingest'`

- [ ] **Step 3: Write the implementation**

```python
# src/ingest.py
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

    Returns a list of nexradaws scan objects.
    """
    conn = _get_nexrad_conn()
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    scans = conn.get_avail_scans(dt.year, dt.month, dt.day, site_id)
    return scans


def list_latest_scans(site_id: str) -> list:
    """List the most recent scans for a site (today's date).

    Returns a list of nexradaws scan objects.
    """
    now = datetime.utcnow()
    scans = list_scans_for_date(site_id, now.strftime("%Y-%m-%d"))
    if not scans:
        # Try yesterday in case of UTC date boundary
        from datetime import timedelta
        yesterday = now - timedelta(days=1)
        scans = list_scans_for_date(site_id, yesterday.strftime("%Y-%m-%d"))
    return scans


def download_scan(site_id: str, scan) -> str:
    """Download a single scan file to the cache. Returns the local file path.

    Skips download if already cached.
    """
    filename = scan.filename
    if scan_is_cached(site_id, filename):
        return get_cache_path(site_id, filename)
    cache_dir = os.path.join(os.path.abspath(CACHE_DIR), site_id)
    os.makedirs(cache_dir, exist_ok=True)
    conn = _get_nexrad_conn()
    results = conn.download(scan, cache_dir)
    for result in results:
        if result.success:
            return str(result.filepath)
    raise RuntimeError(f"Failed to download scan {filename} for {site_id}")


def fetch_scan(site_id: str, dt: datetime | None = None) -> str:
    """Fetch a scan for a site. If dt is provided, find the closest scan to that time.
    If dt is None, fetch the latest scan.

    Returns the local file path to the downloaded scan.
    """
    if dt is None:
        scans = list_latest_scans(site_id)
        if not scans:
            raise RuntimeError(f"No scans available for {site_id}")
        scan = scans[-1]  # Most recent
    else:
        scans = list_scans_for_date(site_id, dt.strftime("%Y-%m-%d"))
        if not scans:
            raise RuntimeError(f"No scans available for {site_id} on {dt.date()}")
        # Find closest scan to requested datetime
        scan = min(scans, key=lambda s: abs(
            datetime.strptime(s.filename.split("_V")[0][-15:], "%Y%m%d_%H%M%S") - dt
        ).total_seconds() if hasattr(s, 'filename') else float('inf'))
    return download_scan(site_id, scan)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_ingest.py -v`
Expected: All 5 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/ingest.py tests/unit/test_ingest.py
git commit -m "feat: add NEXRAD ingest manager with caching"
```

---

### Task 5: Reflectivity Parser

**Files:**
- Create: `src/parser.py`
- Create: `tests/unit/test_parser.py`
- Create: `tests/conftest.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/test_parser.py
import numpy as np
from unittest.mock import patch, MagicMock
from src.parser import extract_reflectivity, ReflectivityData


def _make_mock_radar():
    """Create a mock Py-ART radar object with reflectivity data."""
    radar = MagicMock()
    radar.nsweeps = 3
    radar.fixed_angle = {"data": np.array([0.5, 1.5, 2.4])}
    radar.latitude = {"data": np.array([35.3331])}
    radar.longitude = {"data": np.array([-97.2778])}

    # Sweep 0 (0.5 deg): 360 azimuths x 1832 range bins
    radar.get_start_end.return_value = (0, 359)
    sweep_data = np.random.uniform(-10, 60, (360, 1832)).astype(np.float32)
    radar.fields = {
        "reflectivity": {"data": np.ma.array(sweep_data, mask=False)}
    }
    radar.azimuth = {"data": np.linspace(0, 359, 360)}
    radar.range = {"data": np.linspace(0, 459750, 1832)}
    return radar


def test_extract_reflectivity_returns_dataclass():
    mock_radar = _make_mock_radar()
    with patch("src.parser.pyart.io.read_nexrad_archive", return_value=mock_radar):
        result = extract_reflectivity("/fake/path.V06")
    assert isinstance(result, ReflectivityData)
    assert result.reflectivity.shape[0] == 360
    assert result.reflectivity.shape[1] == 1832
    assert result.radar_lat == 35.3331
    assert result.radar_lon == -97.2778


def test_extract_reflectivity_uses_lowest_sweep():
    mock_radar = _make_mock_radar()
    with patch("src.parser.pyart.io.read_nexrad_archive", return_value=mock_radar):
        result = extract_reflectivity("/fake/path.V06")
    assert result.elevation_angle == 0.5


def test_extract_reflectivity_returns_azimuth_and_range():
    mock_radar = _make_mock_radar()
    with patch("src.parser.pyart.io.read_nexrad_archive", return_value=mock_radar):
        result = extract_reflectivity("/fake/path.V06")
    assert len(result.azimuths) == 360
    assert len(result.ranges_m) == 1832
    assert result.ranges_m[-1] > 400000  # Max range > 400km


def test_extract_reflectivity_elevation_angles():
    mock_radar = _make_mock_radar()
    with patch("src.parser.pyart.io.read_nexrad_archive", return_value=mock_radar):
        result = extract_reflectivity("/fake/path.V06")
    assert result.elevation_angles == [0.5, 1.5, 2.4]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_parser.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.parser'`

- [ ] **Step 3: Write the implementation**

```python
# src/parser.py
from dataclasses import dataclass
import numpy as np
import pyart


@dataclass
class ReflectivityData:
    """Parsed reflectivity data from a single radar sweep."""
    reflectivity: np.ndarray  # 2D array (azimuth x range), dBZ values
    azimuths: np.ndarray      # 1D array of azimuth angles in degrees
    ranges_m: np.ndarray      # 1D array of range bin distances in meters
    radar_lat: float
    radar_lon: float
    elevation_angle: float
    elevation_angles: list[float]  # All available elevation angles
    timestamp: str


def extract_reflectivity(filepath: str) -> ReflectivityData:
    """Read a NEXRAD Level II file and extract reflectivity from the lowest sweep.

    Args:
        filepath: Path to the NEXRAD Level II file.

    Returns:
        ReflectivityData with the reflectivity grid and metadata.
    """
    radar = pyart.io.read_nexrad_archive(filepath)

    elevation_angles = sorted(set(np.round(radar.fixed_angle["data"], 1)))

    # Use the lowest elevation sweep (index 0)
    sweep_start, sweep_end = radar.get_start_end(0)
    reflectivity = radar.fields["reflectivity"]["data"][sweep_start:sweep_end + 1]
    azimuths = radar.azimuth["data"][sweep_start:sweep_end + 1]
    ranges_m = radar.range["data"]

    # Fill masked values with NaN for easier downstream processing
    if hasattr(reflectivity, "filled"):
        reflectivity = reflectivity.filled(np.nan)

    return ReflectivityData(
        reflectivity=reflectivity,
        azimuths=azimuths,
        ranges_m=ranges_m,
        radar_lat=float(radar.latitude["data"][0]),
        radar_lon=float(radar.longitude["data"][0]),
        elevation_angle=float(radar.fixed_angle["data"][0]),
        elevation_angles=[float(a) for a in elevation_angles],
        timestamp=str(radar.time["units"]).replace("seconds since ", ""),
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_parser.py -v`
Expected: All 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/parser.py tests/unit/test_parser.py
git commit -m "feat: add reflectivity parser using Py-ART"
```

---

### Task 6: Rain Object Detection

**Files:**
- Create: `src/detection.py`
- Create: `tests/unit/test_detection.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/test_detection.py
import math
import numpy as np
from src.detection import (
    detect_objects,
    classify_intensity,
    compute_object_properties,
    degrees_to_bearing,
    polar_to_latlon,
    MIN_OBJECT_AREA_KM2,
    INTENSITY_THRESHOLDS,
    DetectedObject,
)


def test_intensity_thresholds_defined():
    assert len(INTENSITY_THRESHOLDS) == 5
    assert INTENSITY_THRESHOLDS[0] == (20, 30, "light rain")
    assert INTENSITY_THRESHOLDS[-1] == (60, float("inf"), "severe core")


def test_classify_intensity():
    assert classify_intensity(15.0) == "drizzle"
    assert classify_intensity(25.0) == "light rain"
    assert classify_intensity(35.0) == "moderate rain"
    assert classify_intensity(45.0) == "heavy rain"
    assert classify_intensity(55.0) == "intense rain"
    assert classify_intensity(65.0) == "severe core"


def test_degrees_to_bearing():
    assert degrees_to_bearing(0) == "N"
    assert degrees_to_bearing(90) == "E"
    assert degrees_to_bearing(180) == "S"
    assert degrees_to_bearing(270) == "W"
    assert degrees_to_bearing(45) == "NE"
    assert degrees_to_bearing(22.5) == "NNE"


def test_polar_to_latlon():
    # From a radar at 35.0, -97.0, a point 100km due north should be ~35.9, -97.0
    lat, lon = polar_to_latlon(
        radar_lat=35.0, radar_lon=-97.0,
        azimuth_deg=0.0, range_m=100000.0,
    )
    assert abs(lat - 35.9) < 0.1
    assert abs(lon - (-97.0)) < 0.1


def test_detect_objects_empty_grid():
    """No reflectivity above threshold should return no objects."""
    reflectivity = np.full((360, 500), 10.0)  # All below 20 dBZ
    azimuths = np.linspace(0, 359, 360)
    ranges_m = np.linspace(0, 250000, 500)
    objects = detect_objects(
        reflectivity=reflectivity,
        azimuths=azimuths,
        ranges_m=ranges_m,
        radar_lat=35.0,
        radar_lon=-97.0,
    )
    assert len(objects) == 0


def test_detect_objects_single_blob():
    """A single blob of high reflectivity should produce one object."""
    reflectivity = np.full((360, 500), np.nan)
    # Place a 10x10 blob of 45 dBZ at azimuth 90, range bin 200
    reflectivity[85:95, 195:205] = 45.0
    azimuths = np.linspace(0, 359, 360)
    ranges_m = np.linspace(2000, 250000, 500)  # Start at 2km to avoid zero range
    objects = detect_objects(
        reflectivity=reflectivity,
        azimuths=azimuths,
        ranges_m=ranges_m,
        radar_lat=35.0,
        radar_lon=-97.0,
    )
    assert len(objects) == 1
    assert objects[0].peak_dbz == 45.0
    assert objects[0].peak_label == "heavy rain"


def test_detect_objects_two_separate_blobs():
    """Two separated blobs should produce two objects."""
    reflectivity = np.full((360, 500), np.nan)
    reflectivity[10:20, 50:60] = 35.0  # Blob 1
    reflectivity[200:210, 300:310] = 55.0  # Blob 2
    azimuths = np.linspace(0, 359, 360)
    ranges_m = np.linspace(2000, 250000, 500)
    objects = detect_objects(
        reflectivity=reflectivity,
        azimuths=azimuths,
        ranges_m=ranges_m,
        radar_lat=35.0,
        radar_lon=-97.0,
    )
    assert len(objects) == 2


def test_detect_objects_nested_layers():
    """An object with varying intensities should have nested layers."""
    reflectivity = np.full((360, 500), np.nan)
    # Outer ring: light rain (25 dBZ)
    reflectivity[80:100, 190:210] = 25.0
    # Inner ring: moderate rain (35 dBZ)
    reflectivity[85:95, 195:205] = 35.0
    # Core: heavy rain (48 dBZ)
    reflectivity[88:92, 198:202] = 48.0
    azimuths = np.linspace(0, 359, 360)
    ranges_m = np.linspace(2000, 250000, 500)
    objects = detect_objects(
        reflectivity=reflectivity,
        azimuths=azimuths,
        ranges_m=ranges_m,
        radar_lat=35.0,
        radar_lon=-97.0,
    )
    assert len(objects) == 1
    obj = objects[0]
    assert obj.peak_dbz == 48.0
    layer_labels = [layer.label for layer in obj.layers]
    assert "light rain" in layer_labels
    assert "moderate rain" in layer_labels
    assert "heavy rain" in layer_labels


def test_detect_objects_filters_small_objects():
    """Very small objects below MIN_OBJECT_AREA_KM2 should be filtered out."""
    reflectivity = np.full((360, 500), np.nan)
    # Place a tiny 2x2 blob — should be smaller than 4 km²
    reflectivity[90:92, 100:102] = 40.0
    azimuths = np.linspace(0, 359, 360)
    # Use tight range spacing so 2x2 pixels are < 4 km²
    ranges_m = np.linspace(2000, 50000, 500)  # ~96m per bin
    objects = detect_objects(
        reflectivity=reflectivity,
        azimuths=azimuths,
        ranges_m=ranges_m,
        radar_lat=35.0,
        radar_lon=-97.0,
    )
    assert len(objects) == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_detection.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.detection'`

- [ ] **Step 3: Write the implementation**

```python
# src/detection.py
import math
from dataclasses import dataclass, field
import numpy as np
from scipy.ndimage import label

MIN_OBJECT_AREA_KM2 = 4.0
MIN_DBZ_THRESHOLD = 20.0

INTENSITY_THRESHOLDS = [
    (20, 30, "light rain"),
    (30, 40, "moderate rain"),
    (40, 50, "heavy rain"),
    (50, 60, "intense rain"),
    (60, float("inf"), "severe core"),
]

BEARING_LABELS = [
    "N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
    "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW",
]


@dataclass
class IntensityLayerData:
    label: str
    min_dbz: float
    max_dbz: float
    area_km2: float


@dataclass
class DetectedObject:
    object_id: int
    centroid_lat: float
    centroid_lon: float
    distance_km: float
    bearing_deg: float
    peak_dbz: float
    peak_label: str
    area_km2: float
    layers: list[IntensityLayerData] = field(default_factory=list)


def classify_intensity(dbz: float) -> str:
    """Classify a dBZ value into an intensity label."""
    if dbz < 20:
        return "drizzle"
    for min_dbz, max_dbz, label_str in INTENSITY_THRESHOLDS:
        if min_dbz <= dbz < max_dbz:
            return label_str
    return "severe core"


def degrees_to_bearing(deg: float) -> str:
    """Convert compass degrees (0=N, 90=E) to a 16-point cardinal direction."""
    idx = round(deg / 22.5) % 16
    return BEARING_LABELS[idx]


def polar_to_latlon(
    radar_lat: float, radar_lon: float,
    azimuth_deg: float, range_m: float,
) -> tuple[float, float]:
    """Convert a polar coordinate (azimuth, range) relative to a radar to lat/lon."""
    earth_radius_m = 6371000.0
    az_rad = math.radians(azimuth_deg)
    lat1 = math.radians(radar_lat)
    lon1 = math.radians(radar_lon)
    angular_dist = range_m / earth_radius_m

    lat2 = math.asin(
        math.sin(lat1) * math.cos(angular_dist)
        + math.cos(lat1) * math.sin(angular_dist) * math.cos(az_rad)
    )
    lon2 = lon1 + math.atan2(
        math.sin(az_rad) * math.sin(angular_dist) * math.cos(lat1),
        math.cos(angular_dist) - math.sin(lat1) * math.sin(lat2),
    )
    return (math.degrees(lat2), math.degrees(lon2))


def _compute_pixel_area_km2(
    azimuths: np.ndarray, ranges_m: np.ndarray,
    az_idx: int, rng_idx: int,
) -> float:
    """Approximate the area of a single polar pixel in km²."""
    if len(ranges_m) < 2 or len(azimuths) < 2:
        return 0.0
    range_spacing_m = abs(ranges_m[1] - ranges_m[0])
    az_spacing_deg = abs(azimuths[1] - azimuths[0]) if len(azimuths) > 1 else 1.0
    az_spacing_rad = math.radians(az_spacing_deg)
    r = ranges_m[rng_idx]
    area_m2 = r * az_spacing_rad * range_spacing_m
    return area_m2 / 1e6


def compute_object_properties(
    obj_mask: np.ndarray,
    reflectivity: np.ndarray,
    azimuths: np.ndarray,
    ranges_m: np.ndarray,
    radar_lat: float,
    radar_lon: float,
    object_id: int,
) -> DetectedObject | None:
    """Compute properties for a single detected object.

    Args:
        obj_mask: Boolean mask of pixels belonging to this object.
        reflectivity: Full reflectivity grid.
        azimuths: Array of azimuth angles.
        ranges_m: Array of range distances.
        radar_lat: Radar latitude.
        radar_lon: Radar longitude.
        object_id: ID for this object.

    Returns:
        DetectedObject or None if object is too small.
    """
    az_indices, rng_indices = np.where(obj_mask)
    if len(az_indices) == 0:
        return None

    # Compute total area
    total_area_km2 = sum(
        _compute_pixel_area_km2(azimuths, ranges_m, int(az), int(rng))
        for az, rng in zip(az_indices, rng_indices)
    )

    if total_area_km2 < MIN_OBJECT_AREA_KM2:
        return None

    # Centroid in polar coordinates (weighted by reflectivity)
    obj_dbz = reflectivity[obj_mask]
    valid = ~np.isnan(obj_dbz)
    if not np.any(valid):
        return None

    weights = np.where(valid, obj_dbz, 0)
    weight_sum = weights.sum()
    if weight_sum == 0:
        return None

    centroid_az_idx = np.average(az_indices[valid], weights=weights[valid])
    centroid_rng_idx = np.average(rng_indices[valid], weights=weights[valid])
    centroid_az = float(np.interp(centroid_az_idx, range(len(azimuths)), azimuths))
    centroid_range = float(np.interp(centroid_rng_idx, range(len(ranges_m)), ranges_m))

    centroid_lat, centroid_lon = polar_to_latlon(
        radar_lat, radar_lon, centroid_az, centroid_range,
    )
    distance_km = centroid_range / 1000.0
    bearing_deg = centroid_az % 360

    peak_dbz = float(np.nanmax(obj_dbz))
    peak_label = classify_intensity(peak_dbz)

    # Nested intensity layers
    layers = []
    for min_dbz, max_dbz, layer_label in INTENSITY_THRESHOLDS:
        layer_mask = obj_mask & (reflectivity >= min_dbz)
        if max_dbz != float("inf"):
            layer_mask = layer_mask & (reflectivity < max_dbz)
        layer_pixels = np.where(layer_mask)
        if len(layer_pixels[0]) == 0:
            continue
        layer_area = sum(
            _compute_pixel_area_km2(azimuths, ranges_m, int(az), int(rng))
            for az, rng in zip(layer_pixels[0], layer_pixels[1])
        )
        if layer_area > 0:
            layers.append(IntensityLayerData(
                label=layer_label,
                min_dbz=min_dbz,
                max_dbz=max_dbz,
                area_km2=round(layer_area, 2),
            ))

    return DetectedObject(
        object_id=object_id,
        centroid_lat=round(centroid_lat, 4),
        centroid_lon=round(centroid_lon, 4),
        distance_km=round(distance_km, 1),
        bearing_deg=round(bearing_deg, 1),
        peak_dbz=round(peak_dbz, 1),
        peak_label=peak_label,
        area_km2=round(total_area_km2, 2),
        layers=layers,
    )


def detect_objects(
    reflectivity: np.ndarray,
    azimuths: np.ndarray,
    ranges_m: np.ndarray,
    radar_lat: float,
    radar_lon: float,
) -> list[DetectedObject]:
    """Detect rain objects in a reflectivity grid.

    1. Threshold at MIN_DBZ_THRESHOLD
    2. Connected component labeling
    3. Compute properties and nested layers for each object
    4. Filter by minimum area

    Returns list of DetectedObject sorted by peak_dbz descending.
    """
    # Mask: True where reflectivity >= threshold and not NaN
    valid = ~np.isnan(reflectivity) & (reflectivity >= MIN_DBZ_THRESHOLD)
    labeled, num_features = label(valid)

    objects = []
    for i in range(1, num_features + 1):
        obj_mask = labeled == i
        obj = compute_object_properties(
            obj_mask=obj_mask,
            reflectivity=reflectivity,
            azimuths=azimuths,
            ranges_m=ranges_m,
            radar_lat=radar_lat,
            radar_lon=radar_lon,
            object_id=i,
        )
        if obj is not None:
            objects.append(obj)

    objects.sort(key=lambda o: o.peak_dbz, reverse=True)
    return objects
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_detection.py -v`
Expected: All 10 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/detection.py tests/unit/test_detection.py
git commit -m "feat: add rain object detection with nested intensity layers"
```

---

### Task 7: Speech Summary Generator

**Files:**
- Create: `src/summary.py`
- Create: `tests/unit/test_summary.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/test_summary.py
from src.summary import generate_summary, km_to_miles
from src.detection import DetectedObject, IntensityLayerData, degrees_to_bearing


def test_km_to_miles():
    assert km_to_miles(1.60934) == 1
    assert km_to_miles(0.0) == 0
    assert km_to_miles(100.0) == 62


def test_generate_summary_no_objects():
    text = generate_summary(
        site_id="KTLX",
        site_name="Oklahoma City",
        timestamp="2026-04-08T18:30:00Z",
        objects=[],
    )
    assert text == "Oklahoma City: No significant precipitation detected."


def test_generate_summary_single_object():
    obj = DetectedObject(
        object_id=1,
        centroid_lat=35.5,
        centroid_lon=-97.3,
        distance_km=40.2,
        bearing_deg=270.0,
        peak_dbz=45.0,
        peak_label="heavy rain",
        area_km2=120.5,
        layers=[
            IntensityLayerData(label="light rain", min_dbz=20, max_dbz=30, area_km2=120.5),
            IntensityLayerData(label="moderate rain", min_dbz=30, max_dbz=40, area_km2=60.0),
            IntensityLayerData(label="heavy rain", min_dbz=40, max_dbz=50, area_km2=15.0),
        ],
    )
    text = generate_summary(
        site_id="KTLX",
        site_name="Oklahoma City",
        timestamp="2026-04-08T18:30:00Z",
        objects=[obj],
    )
    assert "Oklahoma City" in text
    assert "1 rain object" in text
    assert "heavy rain" in text
    assert "25 miles" in text
    assert "W" in text
    assert "75 square miles" in text


def test_generate_summary_multiple_objects():
    obj1 = DetectedObject(
        object_id=1,
        centroid_lat=35.5, centroid_lon=-97.3,
        distance_km=40.2, bearing_deg=270.0,
        peak_dbz=55.0, peak_label="intense rain",
        area_km2=200.0, layers=[],
    )
    obj2 = DetectedObject(
        object_id=2,
        centroid_lat=35.8, centroid_lon=-97.1,
        distance_km=60.0, bearing_deg=0.0,
        peak_dbz=30.0, peak_label="moderate rain",
        area_km2=50.0, layers=[],
    )
    text = generate_summary(
        site_id="KTLX",
        site_name="Oklahoma City",
        timestamp="2026-04-08T18:30:00Z",
        objects=[obj1, obj2],
    )
    assert "2 rain objects" in text
    assert "intense rain" in text  # Strongest object
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_summary.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.summary'`

- [ ] **Step 3: Write the implementation**

```python
# src/summary.py
from src.detection import DetectedObject, degrees_to_bearing

KM_PER_MILE = 1.60934


def km_to_miles(km: float) -> int:
    """Convert kilometers to miles, rounded to nearest whole number."""
    return round(km / KM_PER_MILE)


def km2_to_mi2(km2: float) -> int:
    """Convert square kilometers to square miles, rounded to nearest whole number."""
    return round(km2 / (KM_PER_MILE ** 2))


def generate_summary(
    site_id: str,
    site_name: str,
    timestamp: str,
    objects: list[DetectedObject],
) -> str:
    """Generate a speech-ready text summary of detected rain objects.

    Args:
        site_id: Radar site ID (e.g., "KTLX").
        site_name: Radar site name (e.g., "Oklahoma City").
        timestamp: Scan timestamp string.
        objects: List of DetectedObject, sorted by peak_dbz descending.

    Returns:
        Plain-text summary string.
    """
    if not objects:
        return f"{site_name}: No significant precipitation detected."

    count = len(objects)
    obj_word = "rain object" if count == 1 else "rain objects"
    strongest = objects[0]
    distance_mi = km_to_miles(strongest.distance_km)
    bearing = degrees_to_bearing(strongest.bearing_deg)
    area_mi2 = km2_to_mi2(strongest.area_km2)

    return (
        f"{site_name}: {count} {obj_word} detected. "
        f"Strongest: {strongest.peak_label}, "
        f"{distance_mi} miles {bearing} of the radar. "
        f"Covering approximately {area_mi2} square miles."
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_summary.py -v`
Expected: All 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/summary.py tests/unit/test_summary.py
git commit -m "feat: add speech summary generator"
```

---

### Task 8: FastAPI Server & Endpoints

**Files:**
- Create: `src/server.py`
- Create: `tests/smoke/test_server_smoke.py`

- [ ] **Step 1: Write the failing smoke tests**

```python
# tests/smoke/test_server_smoke.py
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import numpy as np
from src.server import app

client = TestClient(app)


def test_root_returns_200():
    resp = client.get("/")
    assert resp.status_code == 200
    assert "ARW" in resp.json()["name"]


def test_sites_endpoint_returns_200():
    with patch("src.server.geocode_city_state", return_value=(35.4676, -97.5164)):
        resp = client.get("/sites?city=Oklahoma+City&state=OK")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list)
    assert len(data) > 0
    assert "site_id" in data[0]


def test_sites_endpoint_missing_params_returns_422():
    resp = client.get("/sites")
    assert resp.status_code == 422


def test_scan_endpoint_returns_200():
    mock_ref = MagicMock()
    mock_ref.reflectivity = np.zeros((360, 500))
    mock_ref.elevation_angle = 0.5
    mock_ref.elevation_angles = [0.5, 1.5]
    mock_ref.timestamp = "2026-04-08T18:30:00Z"
    mock_ref.radar_lat = 35.3331
    mock_ref.radar_lon = -97.2778
    with patch("src.server.fetch_scan", return_value="/fake/path"), \
         patch("src.server.extract_reflectivity", return_value=mock_ref):
        resp = client.get("/scan/KTLX")
    assert resp.status_code == 200
    data = resp.json()
    assert data["site_id"] == "KTLX"
    assert "elevation_angles" in data


def test_objects_endpoint_returns_200():
    mock_ref = MagicMock()
    mock_ref.reflectivity = np.full((360, 500), np.nan)
    mock_ref.azimuths = np.linspace(0, 359, 360)
    mock_ref.ranges_m = np.linspace(2000, 250000, 500)
    mock_ref.radar_lat = 35.3331
    mock_ref.radar_lon = -97.2778
    mock_ref.timestamp = "2026-04-08T18:30:00Z"
    with patch("src.server.fetch_scan", return_value="/fake/path"), \
         patch("src.server.extract_reflectivity", return_value=mock_ref):
        resp = client.get("/objects/KTLX")
    assert resp.status_code == 200
    data = resp.json()
    assert "objects" in data
    assert data["object_count"] == 0


def test_summary_endpoint_returns_200():
    mock_ref = MagicMock()
    mock_ref.reflectivity = np.full((360, 500), np.nan)
    mock_ref.azimuths = np.linspace(0, 359, 360)
    mock_ref.ranges_m = np.linspace(2000, 250000, 500)
    mock_ref.radar_lat = 35.3331
    mock_ref.radar_lon = -97.2778
    mock_ref.timestamp = "2026-04-08T18:30:00Z"
    with patch("src.server.fetch_scan", return_value="/fake/path"), \
         patch("src.server.extract_reflectivity", return_value=mock_ref):
        resp = client.get("/summary/KTLX")
    assert resp.status_code == 200
    data = resp.json()
    assert "text" in data
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/smoke/test_server_smoke.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.server'`

- [ ] **Step 3: Write the implementation**

```python
# src/server.py
from datetime import datetime
from fastapi import FastAPI, Query
from src.models import RadarSite, ScanMeta, ObjectsResponse, SummaryResponse, RainObject, IntensityLayer
from src.sites import geocode_city_state, rank_sites, NEXRAD_SITES
from src.ingest import fetch_scan
from src.parser import extract_reflectivity
from src.detection import detect_objects
from src.summary import generate_summary

app = FastAPI(title="ARW - Accessible Radar Workstation", version="0.1.0")


def _find_site_name(site_id: str) -> str:
    """Look up the display name for a NEXRAD site."""
    for site in NEXRAD_SITES:
        if site["site_id"] == site_id.upper():
            return site["name"]
    return site_id


def _parse_datetime(dt_str: str | None) -> datetime | None:
    """Parse an optional datetime query parameter."""
    if dt_str is None:
        return None
    return datetime.fromisoformat(dt_str)


@app.get("/")
def root():
    return {"name": "ARW - Accessible Radar Workstation", "version": "0.1.0"}


@app.get("/sites", response_model=list[RadarSite])
def get_sites(city: str = Query(...), state: str = Query(...)):
    lat, lon = geocode_city_state(city, state)
    ranked = rank_sites(lat, lon)
    return [RadarSite(**site) for site in ranked]


@app.get("/scan/{site_id}", response_model=ScanMeta)
def get_scan(site_id: str, datetime: str | None = Query(None)):
    dt = _parse_datetime(datetime)
    filepath = fetch_scan(site_id.upper(), dt)
    ref_data = extract_reflectivity(filepath)
    return ScanMeta(
        site_id=site_id.upper(),
        timestamp=ref_data.timestamp,
        elevation_angles=ref_data.elevation_angles,
    )


@app.get("/objects/{site_id}", response_model=ObjectsResponse)
def get_objects(site_id: str, datetime: str | None = Query(None)):
    dt = _parse_datetime(datetime)
    filepath = fetch_scan(site_id.upper(), dt)
    ref_data = extract_reflectivity(filepath)
    detected = detect_objects(
        reflectivity=ref_data.reflectivity,
        azimuths=ref_data.azimuths,
        ranges_m=ref_data.ranges_m,
        radar_lat=ref_data.radar_lat,
        radar_lon=ref_data.radar_lon,
    )
    rain_objects = [
        RainObject(
            object_id=obj.object_id,
            centroid_lat=obj.centroid_lat,
            centroid_lon=obj.centroid_lon,
            distance_km=obj.distance_km,
            bearing_deg=obj.bearing_deg,
            peak_dbz=obj.peak_dbz,
            peak_label=obj.peak_label,
            area_km2=obj.area_km2,
            layers=[
                IntensityLayer(
                    label=layer.label,
                    min_dbz=layer.min_dbz,
                    max_dbz=layer.max_dbz,
                    area_km2=layer.area_km2,
                )
                for layer in obj.layers
            ],
        )
        for obj in detected
    ]
    return ObjectsResponse(
        site_id=site_id.upper(),
        timestamp=ref_data.timestamp,
        object_count=len(rain_objects),
        objects=rain_objects,
    )


@app.get("/summary/{site_id}", response_model=SummaryResponse)
def get_summary(site_id: str, datetime: str | None = Query(None)):
    dt = _parse_datetime(datetime)
    filepath = fetch_scan(site_id.upper(), dt)
    ref_data = extract_reflectivity(filepath)
    detected = detect_objects(
        reflectivity=ref_data.reflectivity,
        azimuths=ref_data.azimuths,
        ranges_m=ref_data.ranges_m,
        radar_lat=ref_data.radar_lat,
        radar_lon=ref_data.radar_lon,
    )
    site_name = _find_site_name(site_id)
    text = generate_summary(
        site_id=site_id.upper(),
        site_name=site_name,
        timestamp=ref_data.timestamp,
        objects=detected,
    )
    return SummaryResponse(
        site_id=site_id.upper(),
        timestamp=ref_data.timestamp,
        text=text,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/smoke/test_server_smoke.py -v`
Expected: All 6 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/server.py tests/smoke/test_server_smoke.py
git commit -m "feat: add FastAPI server with all Phase 1 endpoints"
```

---

### Task 9: End-to-End Tests

**Files:**
- Create: `tests/e2e/test_full_pipeline.py`

- [ ] **Step 1: Write the e2e tests**

```python
# tests/e2e/test_full_pipeline.py
"""End-to-end tests for the full ARW pipeline.

These tests mock only the network boundary (NEXRAD download and geocoding)
and exercise the entire pipeline: sites → scan → objects → summary.
"""
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import numpy as np
from src.server import app
from src.parser import ReflectivityData

client = TestClient(app)


def _make_reflectivity_data_with_storm() -> ReflectivityData:
    """Create a ReflectivityData with a synthetic storm."""
    reflectivity = np.full((360, 500), np.nan)
    # Storm 1: large, intense — centered at azimuth 90, range bin 200
    reflectivity[80:110, 180:220] = 25.0   # light rain shell
    reflectivity[85:105, 190:210] = 35.0   # moderate core
    reflectivity[90:100, 195:205] = 50.0   # intense core
    # Storm 2: smaller, moderate — centered at azimuth 270, range bin 350
    reflectivity[265:280, 340:360] = 30.0  # moderate rain
    reflectivity[268:278, 345:355] = 42.0  # heavy rain core

    return ReflectivityData(
        reflectivity=reflectivity,
        azimuths=np.linspace(0, 359, 360),
        ranges_m=np.linspace(2000, 250000, 500),
        radar_lat=35.3331,
        radar_lon=-97.2778,
        elevation_angle=0.5,
        elevation_angles=[0.5, 1.5, 2.4],
        timestamp="2026-04-08T18:30:00Z",
    )


def test_full_pipeline_sites_to_summary():
    """Test the complete flow: get sites → get objects → get summary."""
    # Step 1: Get sites for Oklahoma City
    with patch("src.server.geocode_city_state", return_value=(35.4676, -97.5164)):
        resp = client.get("/sites?city=Oklahoma+City&state=OK")
    assert resp.status_code == 200
    sites = resp.json()
    assert len(sites) > 0
    # KTLX should be in the list (it's the OKC radar)
    site_ids = [s["site_id"] for s in sites]
    assert "KTLX" in site_ids

    # Step 2: Get objects for KTLX with synthetic storm data
    ref_data = _make_reflectivity_data_with_storm()
    with patch("src.server.fetch_scan", return_value="/fake/path"), \
         patch("src.server.extract_reflectivity", return_value=ref_data):
        resp = client.get("/objects/KTLX")
    assert resp.status_code == 200
    data = resp.json()
    assert data["object_count"] == 2
    # Strongest object should be first
    assert data["objects"][0]["peak_dbz"] >= data["objects"][1]["peak_dbz"]
    # Check nested layers on the strongest object
    strongest = data["objects"][0]
    layer_labels = [l["label"] for l in strongest["layers"]]
    assert "light rain" in layer_labels
    assert "moderate rain" in layer_labels
    assert "intense rain" in layer_labels

    # Step 3: Get summary for KTLX
    with patch("src.server.fetch_scan", return_value="/fake/path"), \
         patch("src.server.extract_reflectivity", return_value=ref_data):
        resp = client.get("/summary/KTLX")
    assert resp.status_code == 200
    summary = resp.json()
    assert "2 rain objects" in summary["text"]
    assert "intense rain" in summary["text"]
    assert "Oklahoma City" in summary["text"]


def test_full_pipeline_no_precipitation():
    """Test the pipeline when there is no precipitation."""
    ref_data = ReflectivityData(
        reflectivity=np.full((360, 500), np.nan),
        azimuths=np.linspace(0, 359, 360),
        ranges_m=np.linspace(2000, 250000, 500),
        radar_lat=35.3331,
        radar_lon=-97.2778,
        elevation_angle=0.5,
        elevation_angles=[0.5],
        timestamp="2026-04-08T18:30:00Z",
    )
    with patch("src.server.fetch_scan", return_value="/fake/path"), \
         patch("src.server.extract_reflectivity", return_value=ref_data):
        resp = client.get("/summary/KTLX")
    assert resp.status_code == 200
    assert "No significant precipitation" in resp.json()["text"]


def test_full_pipeline_scan_metadata():
    """Test getting scan metadata."""
    ref_data = _make_reflectivity_data_with_storm()
    with patch("src.server.fetch_scan", return_value="/fake/path"), \
         patch("src.server.extract_reflectivity", return_value=ref_data):
        resp = client.get("/scan/KTLX")
    assert resp.status_code == 200
    data = resp.json()
    assert data["site_id"] == "KTLX"
    assert data["elevation_angles"] == [0.5, 1.5, 2.4]
    assert data["timestamp"] == "2026-04-08T18:30:00Z"
```

- [ ] **Step 2: Run tests to verify they pass**

Run: `pytest tests/e2e/test_full_pipeline.py -v`
Expected: All 3 tests PASS. (These depend on all prior tasks being complete.)

- [ ] **Step 3: Run the full test suite**

Run: `pytest tests/ -v`
Expected: All tests across smoke/, unit/, and e2e/ PASS.

- [ ] **Step 4: Commit**

```bash
git add tests/e2e/test_full_pipeline.py
git commit -m "feat: add end-to-end pipeline tests"
```

---

### Task 10: Final Verification & Push

- [ ] **Step 1: Run the complete test suite**

Run: `pytest tests/ -v --tb=short`
Expected: All tests PASS.

- [ ] **Step 2: Verify the server starts**

Run: `python -c "from src.server import app; print('Server module loads OK')"`
Expected: `Server module loads OK`

- [ ] **Step 3: Push to remote**

```bash
git push
```

- [ ] **Step 4: Update PROGRESS.md**

```markdown
# ARW Progress

## Completed
- Phase 1 implementation: site database, reflectivity ingest, rain object detection, speech summaries
- FastAPI server with 4 endpoints: /sites, /scan/{site_id}, /objects/{site_id}, /summary/{site_id}
- Full test suite: unit tests, smoke tests, end-to-end tests

## In Progress
- None

## Next
- Phase 2: Motion tracking, replay buffer
- NVGT frontend integration with the REST API

## Blockers / Decisions
- None
```
