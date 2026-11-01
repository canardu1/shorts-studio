"""Resolve integration for color normalization: a per-clip auto LUT
(gray-world white balance + exposure) applied via the clip's LUT slot."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass

from .color import measure_gains, write_lut
from .resolve_bridge import (Log, ResolveOptions, _int, _mark_comp_dirty,
                             source_range, target_items)

LUT_NAME = "FaceReframeLUT"


def _lut_dir() -> str:
    """Resolve's own LUT folder (shows up in Effects > LUT), ours nested inside."""
    here = os.path.dirname(os.path.abspath(__file__))
    # installed layout: <Fusion>/Scripts/Utility/face_reframe_lib/face_reframe
    parts = here.split(os.sep)
    if "face_reframe_lib" in parts:
        i = parts.index("face_reframe_lib")
        fusion = os.sep.join(parts[:i - 2])  # <...>/Fusion
        if os.path.isdir(fusion):
            return os.path.join(fusion, "LUTs", "face_reframe")
    return os.path.join(here, "luts")


LUT_DIR = _lut_dir()


def _apply_lut(item, lut_path: str, log: Log) -> bool:
    """Apply the LUT: timeline-item LUT slot first, Fusion LUT node as fallback."""
    try:
        if item.SetLUT(1, lut_path):
            return True
    except Exception:
        pass
    comp = item.GetFusionCompByIndex(1) if _int(item.GetFusionCompCount()) > 0 else item.AddFusionComp()
    if comp is None:
        return False
    comp.Lock()
    comp.StartUndo("Normalize Color")
    node = None
    try:
        media_out = (comp.GetToolList(False, "MediaOut") or {}).get(1)
        if media_out is None:
            return False
        node = comp.FindTool(LUT_NAME)
        if node is None:
            upstream = media_out.Input.GetConnectedOutput()
            if upstream is None:
                return False
            node = None
            for tool_name in ("LUT", "FileLUT"):
                try:
                    node = comp.AddTool(tool_name, -32768, -32768)
                except Exception:
                    node = None
                if node is not None:
                    break
            if node is None:
                return False
            node.SetAttrs({"TOOLS_Name": LUT_NAME})
            node.Input = upstream
            media_out.Input = node.Output
        for key in ("LUTFile", "LUT", "Filename"):
            try:
                node[key] = lut_path
                break
            except Exception:
                try:
                    node.SetInput(key, lut_path)
                    break
                except Exception:
                    continue
        else:
            return False
        return True
    finally:
        comp.EndUndo(True)
        comp.Unlock()
        if node is not None:
            _mark_comp_dirty(comp, node, log)


@dataclass
class ColorOptions:
    scope: str = "current"
    track_index: int = 1
    samples: int = 9


def _lut_path(item) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", item.GetName() or "clip")
    return os.path.join(LUT_DIR, "%s_%s.cube" % (safe, item.GetUniqueId() or "0"))


def run_color(resolve, opts: ColorOptions, log: Log = print) -> int:
    project = resolve.GetProjectManager().GetCurrentProject()
    timeline = project.GetCurrentTimeline() if project else None
    if timeline is None:
        log("Open a timeline first.")
        return 0
    items = target_items(timeline, ResolveOptions(scope=opts.scope, track_index=opts.track_index))
    if not items:
        log("No clip found.")
        return 0
    done = 0
    for item in items:
        pool_item = item.GetMediaPoolItem()
        path = pool_item.GetClipProperty("File Path") if pool_item else ""
        if not path or not os.path.exists(path):
            log("Skipping %s: source file not found" % item.GetName())
            continue
        start, count = source_range(item)
        try:
            gains = measure_gains(path, start, count, opts.samples)
        except Exception as exc:
            log("  colour analysis failed on %s (%s); skipped" % (item.GetName(), exc))
            continue
        if gains is None:
            log("  could not read frames of %s; skipped" % item.GetName())
            continue
        lut = write_lut(gains, _lut_path(item))
        ok = _apply_lut(item, lut, log)
        if ok:
            log("  %s: gains r=%.2f g=%.2f b=%.2f -> %s" % (item.GetName(), gains[0], gains[1], gains[2], os.path.basename(lut)))
            done += 1
        else:
            log("  %s: LUT saved to %s (it's in Effects > LUT > face_reframe — drag it onto the clip)" % (item.GetName(), lut))
    log("Done: %d clip(s) normalized." % done)
    return done
