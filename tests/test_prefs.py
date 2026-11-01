import json
from dataclasses import dataclass

import pytest

from face_reframe import prefs


@pytest.fixture(autouse=True)
def tmp_prefs(tmp_path, monkeypatch):
    path = str(tmp_path / "prefs.json")
    monkeypatch.setattr(prefs, "prefs_path", lambda: path)
    return path


def test_section_creates_file_with_defaults(tmp_prefs):
    s = prefs.section("enhance")
    assert s == {"scope": "all", "track_index": 1, "mode": "fast"}
    with open(tmp_prefs) as fh:
        saved = json.load(fh)
    assert saved["enhance"]["mode"] == "fast"
    assert "_comment" in saved


def test_section_merges_and_drops_unknown(tmp_prefs):
    with open(tmp_prefs, "w") as fh:
        json.dump({"enhance": {"mode": "ai", "bogus": 1}}, fh)
    s = prefs.section("enhance")
    assert s["mode"] == "ai" and "bogus" not in s and s["scope"] == "all"


def test_corrupt_file_falls_back(tmp_prefs):
    with open(tmp_prefs, "w") as fh:
        fh.write("{not json")
    assert prefs.section("enhance")["mode"] == "fast"


def test_save_section_roundtrip(tmp_prefs):
    prefs.save_section("markers", {"scope": "track", "track_index": 2,
                                   "keywords": "a,b", "topics": False,
                                   "min_pause": 6.0})
    assert prefs.section("markers")["keywords"] == "a,b"
    assert prefs.section("markers")["topics"] is False


def test_as_options_filters_fields():
    @dataclass
    class Opts:
        scope: str = "current"
        mode: str = "x"
    prefs.save_section("enhance", {"scope": "all", "mode": "ai", "extra": 9})
    o = prefs.as_options("enhance", Opts)
    assert o.scope == "all" and o.mode == "ai"


class _FakeFusion:
    UIManager = None


class _FakeApp:
    def __init__(self, ver="19.1", ui=None):
        self._ver = ver
        self._ui = ui

    def GetVersionString(self):
        return self._ver

    def Fusion(self):
        return self._ui or _FakeFusion()


def test_ui_probe_negative_is_cached():
    assert prefs.ui_available(_FakeApp()) is False
    # cached: a new app that would have a UI is still reported unavailable
    class GoodFusion:
        UIManager = object()
    app = _FakeApp(ui=GoodFusion())
    assert prefs.ui_available(app) is False
    # version change re-probes and finds the UI
    app._ver = "20.0"
    assert prefs.ui_available(app) is True
    assert prefs.load().get("_ui_ok") is None


def test_ui_probe_positive_when_ui_present():
    class GoodFusion:
        UIManager = object()
    assert prefs.ui_available(_FakeApp(ui=GoodFusion())) is True


def test_reframe_options_from_prefs():
    p = {"aspect": "4:5", "scope": "track", "track_index": 2,
         "mode": "render", "enhance": "ai", "zoom": 1.3,
         "follow_vertical": True, "smoothing": 0.7, "dead_zone": 0.1,
         "max_speed": 1.2, "wide_mode": "keep", "wide_threshold": 0.2,
         "max_zoom": 3.0, "set_timeline_resolution": False}
    prefs.save_section("face_reframe", p)
    o = prefs.reframe_options()
    assert (o.scope, o.mode, o.enhance) == ("track", "render", "ai")
    assert o.reframe.aspect == "4:5" and o.reframe.track.smoothing == 0.7
    assert o.reframe.shots.wide_mode == "keep" and o.set_timeline_resolution is False
    o2 = prefs.reframe_options(scope="all")
    assert o2.scope == "all"
