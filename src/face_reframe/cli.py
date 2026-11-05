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


def _find_name(pool, name: str):
    """Case/space-insensitive lookup in a preset dict; returns the real key."""
    for k in pool:
        if k.lower() == name.lower() or k.lower().replace(" ", "") == name.lower().replace(" ", ""):
            return k
    return None


def _render_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="face_reframe render",
        description="Full Shorts Studio pipeline, headless: same presets, templates, "
                    "cuts, captions and moments as the app.")
    p.add_argument("inputs", nargs="*", help="source video(s)")
    p.add_argument("-o", "--out", help="output folder (default: next to each source)")
    p.add_argument("--preset", help="content preset name, e.g. \"Interview\" "
                                    "(see the app preset menu)")
    p.add_argument("--style", action="append", metavar="GROUP:NAME",
                   help="per-tab template: camera|voice|grade|export|caption, "
                        "e.g. --style \"camera:Tight podcast\"")
    p.add_argument("--set", dest="sets", action="append", metavar="KEY=VALUE",
                   help="override any option widget, e.g. --set zoom=1.3 "
                        "--set cut_fillers=1 --set subtitles=burn")
    p.add_argument("--aspect", help="output aspect (9:16, 4:5, 1:1, 16:9)")
    p.add_argument("--subs", choices=["off", "srt", "burn"], help="subtitles mode")
    p.add_argument("--lang", help="transcription language (auto-detect if unset)")
    p.add_argument("--moments", type=int, metavar="N",
                   help="also render the N best moments as separate clips")
    p.add_argument("--suffix", help="output suffix (default _short)")
    p.add_argument("--list", action="store_true",
                   help="list preset/template names and exit")
    return p


def _cli_window():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([sys.argv[0]])
    from .app.window import MainWindow
    return app, MainWindow()


def _list_presets():
    from .app import presets
    print("Presets:", ", ".join(sorted(presets.BUILTIN)))
    for grp in ("CAMERA", "AUDIO", "GRADE", "EXPORT", "CAPTION"):
        pool = getattr(presets, grp + "_BUILTIN")
        print("%s templates: %s" % (grp.lower(), ", ".join(pool)))
    return 0


def _apply_styles(win, styles):
    from .app import presets
    pools = {"camera": presets.CAMERA_BUILTIN, "voice": presets.AUDIO_BUILTIN,
             "audio": presets.AUDIO_BUILTIN, "grade": presets.GRADE_BUILTIN,
             "export": presets.EXPORT_BUILTIN, "caption": presets.CAPTION_BUILTIN,
             "captions": presets.CAPTION_BUILTIN}
    for spec in styles or []:
        grp, _, name = spec.partition(":")
        pool = pools.get(grp.strip().lower())
        key = _find_name(pool or {}, name.strip())
        if pool is None or key is None:
            sys.stderr.write("warning: unknown template %r (see --list)\n" % spec)
            continue
        for k, v in pool[key].items():
            win._set(k, v)
        sys.stderr.write("template %s:%s applied\n" % (grp, key))


def _coerce(text: str):
    low = text.lower()
    if low in ("1", "true", "on", "yes"):
        return True
    if low in ("0", "false", "off", "no"):
        return False
    try:
        return int(text)
    except ValueError:
        try:
            return float(text)
        except ValueError:
            return text


def render_cmd(argv=None) -> int:
    args = _render_parser().parse_args(argv)
    if args.list:
        return _list_presets()
    if not args.inputs:
        _render_parser().error("the following arguments are required: inputs")
    app, win = _cli_window()
    from .app import presets
    from .app.jobs import run_job, AnalysisCache

    if args.preset:
        key = _find_name(presets.BUILTIN, args.preset)
        if key is None:
            sys.stderr.write("unknown preset %r (see --list)\n" % args.preset)
            return 2
        for k, v in presets.BUILTIN[key].items():
            win._set(k, v)
        sys.stderr.write("preset %r applied\n" % key)
    _apply_styles(win, args.style)
    for kv in args.sets or []:
        k, _, v = kv.partition("=")
        win._set(k.strip(), _coerce(v.strip()))
    if args.aspect:
        win._set("aspect", args.aspect)
    if args.subs:
        win._set("subtitles", args.subs)
    if args.lang:
        win._set("language", args.lang)
    if args.out:
        os.makedirs(args.out, exist_ok=True)
        win.out_dir.setText(os.path.abspath(args.out))
    if args.suffix:
        win.suffix.setText(args.suffix)

    settings = win.job_settings()
    try:
        from .app.captions_qt import make_renderer, make_thumb_text_renderer, make_title_renderer
        renderer = make_renderer(win._val("caption_color"), win._val("caption_upper"),
                                 win._val("caption_hl"), win._val("caption_size"),
                                 win.caption_font.currentFont().family(),
                                 win._val("caption_box"), sweep=win._val("caption_sweep"))
        title_r, thumb_r = make_title_renderer(), make_thumb_text_renderer()
    except Exception as exc:
        sys.stderr.write("Qt captions unavailable (%s), using OpenCV\n" % exc)
        renderer = title_r = thumb_r = None

    cache = win.cache
    ok = 0
    for path in args.inputs:
        if not os.path.exists(path):
            sys.stderr.write("not found: %s\n" % path)
            continue
        clips = [None]
        if args.moments:
            from .app.jobs import transcribe_file
            from .silence import extract_audio
            from .app.highlights import find_highlights, dedupe_highlights
            def log(msg):
                sys.stderr.write("%s\n" % msg)
            pcm = extract_audio(path)
            words = cache.words(path, settings.language) or \
                transcribe_file(path, settings.language, cache,
                                lambda f: sys.stderr.write("\rtranscribing %d%%" % int(f * 100)),
                                log, pcm)
            sys.stderr.write("\n")
            dur = words[-1].end if words else 0
            emb = None
            if win._val("hl_semantic"):
                try:
                    from .app.sembed import get
                    emb = get(lambda f: None, log)
                except Exception:
                    pass
            n = args.moments
            hls = find_highlights(words, dur, count=n * 2, embedder=emb)
            if win._val("hl_dedup") and hls:
                dropped = len(hls) - len(dedupe_highlights(hls, words, emb))
                hls = dedupe_highlights(hls, words, emb)[:n]
                if dropped:
                    log("skipped %d near-duplicate moment(s)" % dropped)
            else:
                hls = hls[:n]
            if not hls:
                log("no moments found; rendering the whole file")
            else:
                if win._val("hl_loop"):
                    from .loopmatch import suggest_loop_end
                    for h in hls:
                        h.end = suggest_loop_end(path, h.start, h.end)
                clips = [(h.start, h.end) for h in hls]
                for i, h in enumerate(hls, 1):
                    log("moment %d: %.0f–%.0fs (score %.2f) %s"
                        % (i, h.start, h.end, h.score, h.title))
        for i, clip in enumerate(clips, 1):
            stem = None if clip is None else \
                "%s_clip%d" % (os.path.splitext(os.path.basename(path))[0], i)
            def progress(stage, f):
                sys.stderr.write("\r%s %3d%%" % (stage, int(f * 100)))
                if f >= 1.0:
                    sys.stderr.write("\n")
            try:
                outs = run_job(path, settings, progress,
                               lambda m: sys.stderr.write("%s\n" % m),
                               cache=cache, caption_renderer=renderer,
                               clip=clip, stem=stem,
                               title_renderer=title_r, thumb_renderer=thumb_r)
                ok += 1
                for o in outs:
                    print(o)
            except Exception as exc:
                sys.stderr.write("render failed for %s: %s\n" % (path, exc))
    return 0 if ok else 1


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else list(argv)
    if argv and argv[0] == "render":
        return render_cmd(argv[1:])
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
