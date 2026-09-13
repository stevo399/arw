import pytest

import src.server as server
from src.history.store import HistoryRegistry


@pytest.fixture(autouse=True)
def _in_memory_radar_history(monkeypatch):
    """No test may write radar history into the repository's cache/ directory.

    Tests that need persistence construct their own registry or RadarHistory
    on tmp_path.
    """
    monkeypatch.setattr(server, "_histories", HistoryRegistry(None))
