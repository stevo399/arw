"""NEXRAD site database, geocoding, and beam height ranking."""

import math
from typing import Optional

from geopy.geocoders import Nominatim

# Constants
LOWEST_ELEVATION_DEG = 0.5
EARTH_RADIUS_KM = 6371.0
MAX_BEAM_HEIGHT_M = 10000.0

# Full NEXRAD WSR-88D site list
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
    """Convert city/state name to (latitude, longitude) using Nominatim."""
    geolocator = Nominatim(user_agent="arw_nexrad_app")
    location = geolocator.geocode(f"{city}, {state}, USA")
    if location is None:
        raise ValueError(f"Could not geocode '{city}, {state}'")
    return location.latitude, location.longitude


def haversine_distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate great-circle distance between two points in kilometers."""
    r = EARTH_RADIUS_KM
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return r * c


def compute_beam_height_m(distance_km: float, radar_elevation_m: float) -> float:
    """
    Compute the height of the lowest radar beam above ground at a given distance.

    Uses the formula:
        beam_height = distance * tan(elevation_angle) + (distance^2 / (2 * earth_radius)) + radar_elevation

    Args:
        distance_km: Distance from the radar in km
        radar_elevation_m: Elevation of the radar antenna in meters

    Returns:
        Beam height in meters
    """
    elevation_rad = math.radians(LOWEST_ELEVATION_DEG)
    # Convert distance to meters for height calculation
    distance_m = distance_km * 1000.0
    earth_radius_m = EARTH_RADIUS_KM * 1000.0

    beam_height = (
        distance_m * math.tan(elevation_rad)
        + (distance_m ** 2) / (2 * earth_radius_m)
        + radar_elevation_m
    )
    return beam_height


def rank_sites(lat: float, lon: float) -> list[dict]:
    """
    Rank NEXRAD sites by beam height at the given location.

    Returns a list of site dicts augmented with:
        - distance_km: great-circle distance from (lat, lon) to site
        - beam_height_m: computed beam height at that distance

    Sites with beam_height_m > MAX_BEAM_HEIGHT_M are filtered out.
    Results are sorted ascending by beam_height_m.
    """
    results = []
    for site in NEXRAD_SITES:
        distance_km = haversine_distance_km(lat, lon, site["latitude"], site["longitude"])
        beam_height_m = compute_beam_height_m(
            distance_km=distance_km,
            radar_elevation_m=site["elevation_m"],
        )
        if beam_height_m <= MAX_BEAM_HEIGHT_M:
            entry = dict(site)
            entry["distance_km"] = distance_km
            entry["beam_height_m"] = beam_height_m
            results.append(entry)

    results.sort(key=lambda x: x["beam_height_m"])
    return results
