from src.history.layer_cache import RenderedLayerCache


def test_unpinned_scans_are_evicted_least_recently_used():
    cache = RenderedLayerCache(max_unpinned_scans=2)
    cache.put(("KTLX", "a"), {"audiom": {"a": 1}}, pinned=False)
    cache.put(("KTLX", "b"), {"audiom": {}}, pinned=False)
    assert cache.get(("KTLX", "a")) == {"audiom": {"a": 1}}  # refreshes "a"
    cache.put(("KTLX", "c"), {"audiom": {}}, pinned=False)
    assert ("KTLX", "b") not in cache
    assert ("KTLX", "a") in cache and ("KTLX", "c") in cache


def test_newest_scan_per_site_is_never_evicted():
    cache = RenderedLayerCache(max_unpinned_scans=1)
    cache.put(("KTLX", "newest"), {"audiom": {}}, pinned=True)
    cache.put(("KEMX", "newest"), {"audiom": {}}, pinned=True)
    for name in ("x", "y", "z"):
        cache.put(("KIWA", name), {"audiom": {}}, pinned=False)
    assert ("KTLX", "newest") in cache and ("KEMX", "newest") in cache
    assert len(cache) == 3


def test_pinning_a_newer_scan_unpins_the_previous_one():
    cache = RenderedLayerCache(max_unpinned_scans=0)
    cache.put(("KTLX", "first"), {"audiom": {}}, pinned=True)
    cache.put(("KTLX", "second"), {"audiom": {}}, pinned=True)
    assert ("KTLX", "first") not in cache
    assert ("KTLX", "second") in cache
