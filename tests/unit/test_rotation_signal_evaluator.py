from scripts.evaluate_rotation_signal_corpus import _haversine_km


def test_evaluator_haversine_is_zero_at_same_location():
    assert _haversine_km(35.0, -97.0, 35.0, -97.0) == 0.0


def test_evaluator_haversine_is_symmetric():
    forward = _haversine_km(35.0, -97.0, 35.1, -96.9)
    backward = _haversine_km(35.1, -96.9, 35.0, -97.0)
    assert forward > 0.0
    assert abs(forward - backward) < 1e-9
