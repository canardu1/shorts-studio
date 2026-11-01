"""Compute side of the Lua bridge: runs outside Resolve on the system Python.

Reads a plain-text job file (written by the .lua launcher inside Resolve),
computes, and writes results as a Lua table the launcher loads with dofile().
Keeps Resolve's Python-gating irrelevant: only the heavy lifting is Python;
all Resolve API calls happen in the .lua script.
"""

from __future__ import annotations

import os
import sys

from . import silence, transcribe
from .cut_bridge import _filler_holes, _subtract
from .geometry import parse_aspect
from .pipeline import Options, reframe, simplify_keys
from .resolve_bridge import comp_values


def _parse_job(path):
    opts = {}
    items = []
    op = None
    for line in open(path, encoding="utf-8"):
        line = line.rstrip("\n")
        if not line:
            continue
        p = line.split("\t")
        if p[0] == "OP":
            op = p[1]
        elif p[0] == "ITEM":
            # id, path, src_start, src_count, clip_fps, timeline_start_frame
            items.append(dict(id=p[1], path=p[2], start=int(p[3]), count=int(p[4]),
                              clip_fps=float(p[5]), tl_start=float(p[6])))
        elif len(p) == 2:
            opts[p[0]] = p[1]
    return op, opts, items


def _f(opts, name, default):
    v = opts.get(name)
    if v is None:
        return default
    try:
        return float(v)
    except ValueError:
        return default


def _lua_value(v):
    if v is None:
        return "nil"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return repr(v)
    if isinstance(v, str):
        return '"' + v.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n") + '"'
    if isinstance(v, (list, tuple)):
        return "{" + ",".join(_lua_value(x) for x in v) + "}"
    if isinstance(v, dict):
        return "{" + ",".join("[%s]=%s" % (_lua_value(str(k)), _lua_value(val))
                              for k, val in v.items()) + "}"
    raise TypeError(type(v))


def _write_out(path, table):
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("return %s\n" % _lua_value(table))


def _op_reframe(item, opts, logs):
    options = Options(
        aspect=opts.get("ASPECT", "9:16"),
        zoom=_f(opts, "ZOOM", 1.0),
        follow_vertical=opts.get("VERTICAL", "1") == "1",
    )
    out_w = int(_f(opts, "OUTW", 1080))
    out_h = int(_f(opts, "OUTH", 1920))
    tol = _f(opts, "KEYTOL", 0.5)
    start, count = item["start"], item["count"]
    logs.append("Analysing %s (frames %d-%d)" % (os.path.basename(item["path"]), start, start + count - 1))
    result = reframe(item["path"], start, count, options)
    found = int(result.path.has_face.sum())
    logs.append("  face on %d/%d frames, %d shot(s)" % (found, result.analysis.frame_count, len(result.path.segments)))
    xs, ys, size = comp_values(result, out_w, out_h)
    cuts = tuple(a for a, _ in result.path.segments[1:])
    frames = simplify_keys(xs * out_w, ys * out_h, tol, cuts)
    entry = {
        "op": "reframe",
        "keys": [{"f": int(f), "x": float(xs[f]), "y": float(ys[f])} for f in frames],
        "src_w": int(result.plan.src_w), "src_h": int(result.plan.src_h),
        "wide": [{"a": int(start + s.start), "b": int(start + s.end - 1), "zoom": float(s.zoom)}
                 for s in result.path.shots if s.kind == "wide"],
    }
    if float(size.max() - size.min()) < 1e-6:
        entry["size_const"] = float(size[0])
    else:
        entry["size_keys"] = [{"f": int(f), "v": float(size[f])} for f in frames]
    return entry


def _speech_params(opts):
    return silence.SpeechParams(
        min_pause=_f(opts, "MINPAUSE", 0.35),
        padding=_f(opts, "PADDING", 0.16),
        margin_db=_f(opts, "MARGINDB", 8.0),
        release_db=_f(opts, "RELEASEDB", 4.0),
        min_speech=_f(opts, "MINSPEECH", 0.15),
        floor_percentile=_f(opts, "FLOORPCT", 25.0),
    )


def _op_cut(item, opts, logs):
    start, count = item["start"], item["count"]
    fps = item["clip_fps"] or _f(opts, "TLFPS", 25.0)
    logs.append("Analysing %s" % os.path.basename(item["path"]))
    try:
        spans = silence.clip_intervals(item["path"], start, count, fps, _speech_params(opts))
    except Exception as exc:
        logs.append("  audio analysis failed (%s); keeping it whole" % exc)
        spans = []
    if not spans:
        logs.append("  no speech; kept whole")
        spans = [(start, start + count)]
    if opts.get("FILLERS") == "1":
        holes = _filler_holes(item["path"], start, count, fps, logs.append)
        spans = _subtract(spans, holes)
    logs.append("  %d segment(s) kept" % len(spans))
    return {"op": "cut", "spans": [[int(a), int(b)] for a, b in spans]}


def _op_subtitles(item, opts, logs):
    start, count = item["start"], item["count"]
    fps = item["clip_fps"] or _f(opts, "TLFPS", 25.0)
    tl_fps = _f(opts, "TLFPS", 25.0)
    logs.append("Transcribing %s" % os.path.basename(item["path"]))
    pcm = silence.extract_audio(item["path"], start / fps, count / fps)
    words, lang = transcribe.transcribe(pcm, opts.get("LANG") or None)
    logs.append("  %d words (%s)" % (len(words), lang or "?"))
    shift = (item["tl_start"] / tl_fps) - (start / fps)
    for w in words:
        w.start += shift
        w.end += shift
    return {"op": "subtitles", "words": [{"start": w.start, "end": w.end, "text": w.text} for w in words]}


def run_job(job_path):
    op, opts, items = _parse_job(job_path)
    logs = []
    out_items = {}
    all_words = []
    for item in items:
        try:
            if op == "reframe":
                out_items[item["id"]] = _op_reframe(item, opts, logs)
            elif op == "cut":
                out_items[item["id"]] = _op_cut(item, opts, logs)
            elif op == "subtitles":
                res = _op_subtitles(item, opts, logs)
                all_words += [transcribe.Word(w["start"], w["end"], w["text"]) for w in res["words"]]
        except Exception as exc:
            logs.append("  failed: %s" % exc)
    table = {"ok": True, "log": logs, "items": out_items}
    if op == "subtitles":
        all_words.sort(key=lambda w: w.start)
        events = transcribe.group_captions(all_words, int(_f(opts, "MAXWORDS", 3)), _f(opts, "MAXDUR", 1.6))
        srt_path = os.path.join(os.path.dirname(job_path), "subtitles.srt")
        with open(srt_path, "w", encoding="utf-8") as fh:
            fh.write(transcribe.to_srt(events))
        table["srt"] = srt_path
        table["captions"] = len(events)
    _write_out(job_path + ".out.lua", table)


def main(argv=None):
    argv = argv or sys.argv[1:]
    if not argv:
        print("usage: python -m face_reframe.exec_ops <job.txt>")
        return 2
    run_job(argv[0])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
