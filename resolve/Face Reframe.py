"""Face Reframe - keep the speaker's face centred for vertical shorts.

Run from DaVinci Resolve: Workspace > Scripts > Face Reframe.
Installed by install.py, which fills in the library path below.
"""

import os
import platform
import sys

FACE_REFRAME_LIB = r"__FACE_REFRAME_LIB__"
# deps live outside Scripts/ so Resolve's script scan doesn't crawl them
_FUSION_DIR = os.path.dirname(os.path.dirname(os.path.dirname(FACE_REFRAME_LIB)))
_DEPS_DIRS = [os.path.join(_FUSION_DIR, "face_reframe_deps", platform.machine()),
              os.path.join(FACE_REFRAME_LIB, "deps", platform.machine())]

for _p in _DEPS_DIRS + [FACE_REFRAME_LIB, os.environ.get("FACE_REFRAME_LIB", "")]:
    if _p and os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

try:
    from face_reframe.pipeline import Options  # noqa: E402
    from face_reframe.prefs import (no_ui_notice, reframe_options,  # noqa: E402
                                    save_section, ui_available)
    from face_reframe.resolve_bridge import ResolveOptions, get_resolve, run  # noqa: E402
    from face_reframe.track import ShotSettings, TrackSettings  # noqa: E402
except ImportError as exc:
    print("[Face Reframe] missing dependency: %s" % exc)
    print("[Face Reframe] run once with your system Python, in the repo folder:")
    print("[Face Reframe]   python install.py --deps --arch %s --python-version %d.%d"
          % (platform.machine(), sys.version_info[0], sys.version_info[1]))
    print("[Face Reframe] then run install.py here again.")
    raise SystemExit(1)

ASPECTS = ["9:16", "4:5", "1:1", "16:9"]
SCOPES = ["Clip under playhead", "All clips on track", "All video tracks"]
MODES = ["Fusion keyframes (non-destructive)", "Render new file + timeline"]
ENHANCES = ["Off", "Fast (sharpen)", "AI (Real-ESRGAN, slower)"]
WIDE_MODES = ["Zoom in on speaker", "Keep full height"]
SUBJECTS = ["Main person", "Whoever is speaking"]


def _resolve():
    app = globals().get("resolve")
    return app if app is not None else get_resolve()


def _bmd():
    mod = globals().get("bmd")
    if mod is not None:
        return mod
    import DaVinciResolveScript as dvr  # noqa: F401
    return dvr


def show_dialog(app):
    fu = app.Fusion()
    ui = fu.UIManager
    disp = _bmd().UIDispatcher(ui)

    def spin(id_, lo, hi, step, value):
        return ui.DoubleSpinBox({"ID": id_, "Minimum": lo, "Maximum": hi, "SingleStep": step, "Value": value, "Weight": 0.6})

    def row(label, widget):
        return ui.HGroup([ui.Label({"Text": label, "Weight": 0.4}), widget])

    win = disp.AddWindow(
        {"ID": "FaceReframe", "WindowTitle": "Face Reframe", "Geometry": [300, 200, 460, 590]},
        ui.VGroup([
            row("Aspect", ui.ComboBox({"ID": "Aspect", "Weight": 0.6})),
            row("Apply to", ui.ComboBox({"ID": "Scope", "Weight": 0.6})),
            row("Track (V#)", ui.SpinBox({"ID": "Track", "Minimum": 1, "Maximum": 50, "Value": 1, "Weight": 0.6})),
            row("Mode", ui.ComboBox({"ID": "Mode", "Weight": 0.6})),
            row("Enhance (render)", ui.ComboBox({"ID": "Enhance", "Weight": 0.6})),
            row("Follow", ui.ComboBox({"ID": "Subject", "Weight": 0.6})),
            row("Smoothing (s)", spin("Smooth", 0, 3, 0.1, 0.5)),
            row("Dead zone", spin("Dead", 0, 0.5, 0.01, 0.05)),
            row("Max pan speed", spin("Speed", 0, 5, 0.1, 0.6)),
            row("Zoom", spin("Zoom", 1, 3, 0.05, 1.0)),
            row("Wide shots", ui.ComboBox({"ID": "Wide", "Weight": 0.6})),
            row("Wide if face <", spin("WideThr", 0.02, 0.5, 0.01, 0.12)),
            row("Max wide zoom", spin("MaxZoom", 1, 4, 0.1, 2.0)),
            ui.CheckBox({"ID": "Vertical", "Text": "Follow face vertically (with zoom)"}),
            ui.CheckBox({"ID": "SetRes", "Text": "Set timeline to vertical resolution", "Checked": True}),
            ui.Label({"ID": "Status", "Text": "Ready", "WordWrap": True}),
            ui.HGroup([ui.Button({"ID": "Run", "Text": "Reframe"}), ui.Button({"ID": "Close", "Text": "Close"})]),
        ]),
    )
    items = win.GetItems()
    items["Aspect"].AddItems(ASPECTS)
    items["Scope"].AddItems(SCOPES)
    items["Mode"].AddItems(MODES)
    items["Enhance"].AddItems(ENHANCES)
    items["Wide"].AddItems(WIDE_MODES)
    items["Subject"].AddItems(SUBJECTS)

    def status(text):
        print("[Face Reframe] " + text)
        items["Status"].Text = text

    def on_run(ev):
        opts = ResolveOptions(
            reframe=Options(
                aspect=ASPECTS[int(items["Aspect"].CurrentIndex)],
                zoom=float(items["Zoom"].Value),
                follow_vertical=bool(items["Vertical"].Checked),
                track=TrackSettings(
                    smoothing=float(items["Smooth"].Value),
                    dead_zone=float(items["Dead"].Value),
                    max_speed=float(items["Speed"].Value),
                    subject="speaker" if int(items["Subject"].CurrentIndex) == 1 else "main",
                ),
                shots=ShotSettings(
                    wide_mode="keep" if int(items["Wide"].CurrentIndex) == 1 else "zoom",
                    wide_threshold=float(items["WideThr"].Value),
                    max_zoom=float(items["MaxZoom"].Value),
                ),
            ),
            scope=("current", "track", "all")[int(items["Scope"].CurrentIndex)],
            track_index=int(items["Track"].Value),
            mode="render" if int(items["Mode"].CurrentIndex) == 1 else "fusion",
            enhance=("off", "fast", "ai")[int(items["Enhance"].CurrentIndex)],
            set_timeline_resolution=bool(items["SetRes"].Checked),
        )
        save_section("face_reframe", {
            "aspect": opts.reframe.aspect, "scope": opts.scope,
            "track_index": opts.track_index, "mode": opts.mode,
            "enhance": opts.enhance, "zoom": opts.reframe.zoom,
            "follow_vertical": opts.reframe.follow_vertical,
            "smoothing": opts.reframe.track.smoothing,
            "dead_zone": opts.reframe.track.dead_zone,
            "max_speed": opts.reframe.track.max_speed,
            "wide_mode": opts.reframe.shots.wide_mode,
            "wide_threshold": opts.reframe.shots.wide_threshold,
            "max_zoom": opts.reframe.shots.max_zoom,
            "set_timeline_resolution": opts.set_timeline_resolution,
        })
        status("Working... (see Workspace > Console for progress)")
        try:
            run(app, opts, status)
        except Exception as exc:  # surface errors in the dialog instead of failing silently
            status("Error: %s" % exc)
            raise

    def on_close(ev):
        disp.ExitLoop()

    win.On.Run.Clicked = on_run
    win.On.Close.Clicked = on_close
    win.On.FaceReframe.Close = on_close
    win.Show()
    disp.RunLoop()
    win.Hide()


def main():
    app = _resolve()
    if app is None:
        print("Face Reframe: run this from DaVinci Resolve (Workspace > Scripts).")
        return
    if ui_available(app):
        show_dialog(app)
    else:
        no_ui_notice("Face Reframe")
        opts = reframe_options()
        print("[Face Reframe] scope=%s mode=%s enhance=%s aspect=%s"
              % (opts.scope, opts.mode, opts.enhance, opts.reframe.aspect))
        run(app, opts)


main()
