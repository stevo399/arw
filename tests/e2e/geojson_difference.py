"""Fast, readable comparison of two GeoJSON layers for real-data proofs."""

import json


def _features_by_id(layer: dict) -> dict:
    # Point layers may have no feature id; fall back to position.
    return {feature.get("id", index): feature for index, feature in enumerate(layer["features"])}


def first_difference(actual: dict, expected: dict) -> str | None:
    """Name the first differing feature property, or None when identical.

    Deliberately avoids `assert a == b` on the serialized layers: on failure
    pytest would diff two multi-megabyte strings, which takes many minutes.
    """
    if json.dumps(actual, sort_keys=True) == json.dumps(expected, sort_keys=True):
        return None
    actual_features = _features_by_id(actual)
    expected_features = _features_by_id(expected)
    if actual_features.keys() != expected_features.keys():
        return f"feature ids differ: {sorted(actual_features.keys() ^ expected_features.keys())}"
    for feature_id, expected_feature in expected_features.items():
        actual_feature = actual_features[feature_id]
        for key, value in expected_feature["properties"].items():
            if actual_feature["properties"].get(key) != value:
                return f"{feature_id}.{key}: {actual_feature['properties'].get(key)!r} != {value!r}"
        if actual_feature["geometry"] != expected_feature["geometry"]:
            return f"{feature_id} geometry differs"
    return "layer metadata differs"
