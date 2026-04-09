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
    """Generate a speech-ready text summary of detected rain objects."""
    if not objects:
        return f"{site_name}: No significant precipitation detected."

    count = len(objects)
    obj_word = "rain object" if count == 1 else "rain objects"
    strongest = objects[0]
    distance_mi = km_to_miles(strongest.distance_km)
    bearing = degrees_to_bearing(strongest.bearing_deg)
    area_mi2 = km_to_miles(strongest.area_km2)

    return (
        f"{site_name}: {count} {obj_word} detected. "
        f"Strongest: {strongest.peak_label}, "
        f"{distance_mi} miles {bearing} of the radar. "
        f"Covering approximately {area_mi2} square miles."
    )
