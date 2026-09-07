"""Guard the documented evaluation population used by rotation work."""

import json
from pathlib import Path


CORPUS_PATH = Path("docs/validation/rotation-signal-corpus.json")


def test_rotation_signal_corpus_has_null_and_confirmed_contexts():
    corpus = json.loads(CORPUS_PATH.read_text(encoding="utf-8"))
    assert corpus["schema_version"] == 1
    labels = {case["label"] for case in corpus["cases"]}
    assert "null_clear_air" in labels
    assert "confirmed_tornado_context" in labels
    assert "confirmed_hail_context" in labels


def test_rotation_signal_corpus_records_evidence_and_disallowed_inferences():
    corpus = json.loads(CORPUS_PATH.read_text(encoding="utf-8"))
    seen_volumes: set[str] = set()
    for case in corpus["cases"]:
        assert case["evidence"]
        assert case["use_for"]
        assert case["must_not_use_for"]
        volumes = case["volumes"] if "volumes" in case else [case["volume"]]
        for volume in volumes:
            assert volume not in seen_volumes
            seen_volumes.add(volume)
