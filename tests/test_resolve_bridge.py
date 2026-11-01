"""Exercise the Resolve/Fusion integration against an in-memory fake API."""

import numpy as np
import pytest

from face_reframe import resolve_bridge as rb
from face_reframe.detect import Analysis
from face_reframe.geometry import plan_crops
from face_reframe.pipeline import Result
from face_reframe.track import CameraPath


class Output:
    def __init__(self, tool):
        self.tool = tool


class Input:
    def __init__(self):
        self.src = None

    def GetConnectedOutput(self):
        return self.src


class Tool:
    def __init__(self, kind, name):
        self.__dict__.update(kind=kind, name=name, inputs={}, keys={}, modifiers={})
        self.__dict__["Output"] = Output(self)
        self.__dict__["Input"] = Input()

    def __setattr__(self, key, value):
        if key == "Input":
            self.__dict__["Input"].src = value
        else:
            self.inputs[key] = value

    def __getattr__(self, key):
        return self.inputs.get(key)

    def SetAttrs(self, attrs):
        self.__dict__["name"] = attrs.get("TOOLS_Name", self.name)

    def AddModifier(self, inp, kind):
        self.modifiers[inp] = kind
        return True

    def SetInput(self, inp, value, time):
        self.keys.setdefault(inp, {})[time] = value


class Comp:
    def __init__(self):
        self.mi, self.mo = Tool("MediaIn", "MediaIn1"), Tool("MediaOut", "MediaOut1")
        self.mo.Input = self.mi.Output
        self.tools = [self.mi, self.mo]

    def Lock(self): pass
    def Unlock(self): pass
    def StartUndo(self, name): pass
    def EndUndo(self, keep): pass
    def GetAttrs(self): return {"COMPN_RenderStart": 0.0}

    def GetToolList(self, selected, kind):
        found = [t for t in self.tools if t.kind == kind]
        return {i + 1: t for i, t in enumerate(found)}

    def FindTool(self, name):
        return next((t for t in self.tools if t.name == name), None)

    def AddTool(self, kind, x, y):
        t = Tool(kind, "%s%d" % (kind, len(self.tools)))
        self.tools.append(t)
        return t


class Item:
    def __init__(self):
        self.comp = None

    def GetName(self): return "clip"
    def GetFusionCompCount(self): return 1 if self.comp else 0
    def GetFusionCompByIndex(self, i): return self.comp

    def AddFusionComp(self):
        self.comp = Comp()
        return self.comp


def fake_result(cx, zoom=1.0, segments=None):
    n = len(cx)
    zoom = np.broadcast_to(np.asarray(zoom, float), (n,))
    plan = plan_crops(1920, 1080, np.asarray(cx, float), np.full(n, 540.0), 9 / 16, zoom=zoom)
    path = CameraPath(plan.cx, plan.cy, np.zeros(n), np.ones(n, bool), segments or [(0, n)], zoom, [])
    return Result(Analysis(1920, 1080, 25.0, 0, n, []), path, plan)


def test_comp_values_centre_the_crop():
    res = fake_result([960.0, 303.75, 1616.25])
    xs, ys, size = rb.comp_values(res, 1080, 1920)
    assert np.allclose(size, 1080 / 607.5)
    assert xs[0] == pytest.approx(0.5)
    # crop at the left edge -> foreground shifted right so its left edge meets the canvas edge
    fg_w = 1920 * size[0] / 1080
    assert xs[1] - fg_w / 2 == pytest.approx(0.0)
    assert xs[2] + fg_w / 2 == pytest.approx(1.0)
    assert np.allclose(ys, 0.5)


def test_apply_fusion_builds_graph_and_keys():
    item = Item()
    res = fake_result(np.linspace(400, 1500, 50))
    rb.apply_fusion(item, res, 1080, 1920, rb.ResolveOptions(), lambda m: None)
    comp = item.comp
    merge, bg = comp.FindTool(rb.MERGE_NAME), comp.FindTool(rb.BG_NAME)
    assert comp.mo.Input.src is merge.Output
    assert merge.inputs["Foreground"] is comp.mi.Output
    assert merge.inputs["Background"] is bg.Output
    assert (bg.inputs["Width"], bg.inputs["Height"]) == (1080, 1920)
    assert merge.modifiers["Center"] == "XYPath"
    keys = merge.keys["Center"]
    assert 0 in keys and 49 in keys
    # linear pan -> only a couple of keyframes needed
    assert len(keys) <= 4

    assert merge.inputs["Size"] == pytest.approx(1080 / 607.5)
    assert "Size" not in merge.keys

    # re-running reuses the same nodes
    rb.apply_fusion(item, res, 1080, 1920, rb.ResolveOptions(), lambda m: None)
    assert len(comp.tools) == 4


def test_make_timeline_vertical():
    class TL:
        def __init__(self):
            self.s = {"timelineResolutionWidth": "1920", "timelineResolutionHeight": "1080"}

        def GetSetting(self, k): return self.s.get(k)

        def SetSetting(self, k, v):
            self.s[k] = v
            return True

    tl = TL()
    out = rb.make_timeline_vertical(None, tl, 9 / 16, 1920, lambda m: None)
    assert out is tl
    assert rb.timeline_size(tl) == (1080, 1920)


def test_make_timeline_vertical_falls_back_to_duplicate():
    class TL:
        def __init__(self, ok=True):
            self.ok = ok
            self.s = {"timelineResolutionWidth": "1920", "timelineResolutionHeight": "1080"}

        def GetSetting(self, k): return self.s.get(k)

        def SetSetting(self, k, v):
            if not self.ok:
                return False
            self.s[k] = v
            return True

        def GetName(self): return "TL"

    class Pool:
        def __init__(self): self.dup = TL(ok=True)

        def DuplicateTimeline(self, tl): return self.dup

    class Proj:
        def __init__(self):
            self.pool = Pool()
            self.current = None

        def GetMediaPool(self): return self.pool

        def SetCurrentTimeline(self, tl): self.current = tl

    src, proj = TL(ok=False), Proj()
    out = rb.make_timeline_vertical(proj, src, 9 / 16, 1920, lambda m: None)
    assert out is proj.pool.dup
    assert rb.timeline_size(out) == (1080, 1920)
    assert proj.current is out                    # dup became current


def test_zoomed_wide_shot_keys_size_with_hard_step():
    item = Item()
    zoom = np.r_[np.ones(30), np.full(30, 2.0)]
    res = fake_result(np.full(60, 960.0), zoom=zoom, segments=[(0, 30), (30, 60)])
    rb.apply_fusion(item, res, 1080, 1920, rb.ResolveOptions(), lambda m: None)
    merge = item.comp.FindTool(rb.MERGE_NAME)
    assert merge.modifiers["Size"] == "BezierSpline"
    sizes = merge.keys["Size"]
    base = 1080 / 607.5
    assert sizes[29] == pytest.approx(base) and sizes[30] == pytest.approx(2 * base)
    assert sizes[0] == pytest.approx(base) and sizes[59] == pytest.approx(2 * base)
