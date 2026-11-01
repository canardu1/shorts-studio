"""Command line: ``python -m face_reframe input.mp4 -o short.mp4``."""

from __future__ import annotations

import argparse
import json
import os
import sys

from .pipeline import Options, reframe
from .render import render
from .track import ShotSettings, TrackSettings


def _size(text: str):
    w, h = text.lower().split("x")
    return int(w), int(h)


def _progress(label):
    def cb(p):
        sys.stderr.write("\r%s %3d%%" % (label, int(p * 100)))
        if p >= 1.0:
            sys.stderr.write("\n")
        sys.stderr.flush()
    return cb


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="face_reframe", description="Reframe a video so the main face stays centred.")
    p.add_argument("input")
    p.add_argument("-o", "--output", help="output video (default: <input>_reframed.mp4)")
    p.add_argument("--aspect", default="9:16", help="target aspect ratio, e.g. 9:16, 4:5, 1:1")
    p.add_argument("--size", type=_size, default=None, help="output size WxH (default 1080x1920 for 9:16)")
    p.add_argument("--zoom", type=float, default=1.0, help="extra punch-in (>=1)")
    p.add_argument("--follow-vertical", action="store_true", help="also follow the face vertically (useful with --zoom)")
    p.add_argument("--face-pos", type=float, default=0.4, help="vertical face position in frame, 0=top 0.5=middle")
    p.add_argument("--subject", choices=["main", "speaker"], default="main",
                   help="several people in shot: follow the main face, or cut to whoever is speaking")
    p.add_argument("--smoothing", type=float, default=0.5, help="camera smoothing in seconds")
    p.add_argument("--dead-zone", type=float, default=0.05, help="fraction of the crop width the face can drift before panning")
    p.add_argument("--max-speed", type=float, default=0.6, help="max pan speed in frame widths per second (0=off)")
    p.add_argument("--wide", choices=["zoom", "keep"], default="zoom",
                   help="wide shots (small face): zoom in on the speaker or keep full height")
    p.add_argument("--wide-threshold", type=float, default=0.12,
                   help="shot counts as wide when face height < this fraction of the frame height")
    p.add_argument("--wide-face", type=float, default=0.18, help="face height to aim for when zooming wide shots")
    p.add_argument("--max-zoom", type=float, default=2.0, help="max wide-shot zoom (image quality limit)")
    p.add_argument("--stride", type=int, default=2, help="detect faces every N frames")
    p.add_argument("--start", type=int, default=0, help="first source frame")
    p.add_argument("--frames", type=int, default=0, help="number of frames (0 = to the end)")
    p.add_argument("--enhance", choices=["off", "fast", "ai"], default="off",
                   help="fast = unsharp + face sharpening; ai = Real-ESRGAN compact (needs onnxruntime)")
    p.add_argument("--json", help="write the per-frame crop track to this JSON file")
    p.add_argument("--debug", action="store_true", help="draw the tracked face line on the output")
    p.add_argument("--no-render", action="store_true", help="only analyse (use with --json)")
    return p


def default_size(aspect: float):
    if aspect <= 1.0:
        return int(round(1920 * aspect / 2.0) * 2), 1920
    return 1920, int(round(1920 / aspect / 2.0) * 2)


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    opts = Options(
        aspect=args.aspect, zoom=args.zoom, follow_vertical=args.follow_vertical,
        face_height_pos=args.face_pos, stride=max(1, args.stride),
        track=TrackSettings(smoothing=args.smoothing, dead_zone=args.dead_zone, max_speed=args.max_speed,
                            subject=args.subject),
        shots=ShotSettings(wide_mode=args.wide, wide_threshold=args.wide_threshold,
                           target_face=args.wide_face, max_zoom=args.max_zoom),
    )
    res = reframe(args.input, args.start, args.frames or None, opts, progress=_progress("analysing"))
    a, plan = res.analysis, res.plan
    detected = int(res.path.has_face.sum())
    sys.stderr.write("%d frames, face found on %d, %d shot(s)\n" % (a.frame_count, detected, len(res.path.segments)))
    for shot in res.path.shots:
        sys.stderr.write("  frames %d-%d: %s (face %.0f%% of height), zoom %.2fx\n"
                         % (shot.start, shot.end - 1, shot.kind, shot.face_frac * 100, shot.zoom))

    if args.json:
        with open(args.json, "w") as fh:
            json.dump({
                "source": os.path.abspath(args.input), "width": a.width, "height": a.height, "fps": a.fps,
                "start_frame": args.start,
                "cuts": [s[0] for s in res.path.segments[1:]],
                "shots": [{"start": s.start, "end": s.end, "kind": s.kind, "face_frac": round(s.face_frac, 4),
                           "zoom": round(s.zoom, 3)} for s in res.path.shots],
                "frames": [{"cx": round(float(plan.cx[i]), 2), "cy": round(float(plan.cy[i]), 2),
                            "w": round(float(plan.crop_w[i]), 2), "h": round(float(plan.crop_h[i]), 2)}
                           for i in range(len(plan.cx))],
            }, fh)
    if args.no_render:
        return 0

    from .geometry import parse_aspect
    size = args.size or default_size(parse_aspect(args.aspect))
    out = args.output or os.path.splitext(args.input)[0] + "_reframed.mp4"
    enhancer = None
    if args.enhance != "off":
        from .enhance import EnhanceOptions, Enhancer
        enhancer = Enhancer(EnhanceOptions(mode=args.enhance))
    render(args.input, plan, out, size, a.fps, args.start,
           debug_overlay=res.path.x if args.debug else None, progress=_progress("rendering"),
           enhancer=enhancer)
    sys.stderr.write("wrote %s\n" % out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
