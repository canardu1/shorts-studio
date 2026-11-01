"""Auto Markers - timeline markers from the transcript (cue words, topic breaks).

Run from DaVinci Resolve: Workspace > Scripts > Auto Markers.
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
    from face_reframe.markers_bridge import MarkerOptions, run_markers  # noqa: E402
    from face_reframe.prefs import (as_options, no_ui_notice,  # noqa: E402
                                    save_section, ui_available)
    from face_reframe.resolve_bridge import get_resolve  # noqa: E402
except ImportError as exc:
    print("[Auto Markers] missing dependency: %s" % exc)
    print("[Auto Markers] run once with your system Python, in the repo folder:")
    print("[Auto Markers]   python install.py --deps --arch %s --python-version %d.%d"
          % (platform.machine(), sys.version_info[0], sys.version_info[1]))
    raise SystemExit(1)

SCOPES = ["Clip under playhead", "All clips on track", "All video tracks"]


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

    def row(label, widget):
        return ui.HGroup([ui.Label({"Text": label, "Weight": 0.4}), widget])

    win = disp.AddWindow(
        {"ID": "AutoMarkers", "WindowTitle": "Auto Markers",
         "Geometry": [300, 200, 440, 240]},
        ui.VGroup([
            row("Apply to", ui.ComboBox({"ID": "Scope", "Weight": 0.6})),
            row("Track (V#)", ui.SpinBox({"ID": "Track", "Minimum": 1, "Maximum": 50,
                                         "Value": 1, "Weight": 0.6})),
            row("Cue words",
                ui.LineEdit({"ID": "Keys", "Weight": 0.6,
                             "Text": "taglia,rifacciamo,retake,daccapo,ricominciamo"})),
            ui.CheckBox({"ID": "Topics", "Text": "Mark topic breaks (after a long pause)",
                         "Checked": True}),
            row("Topic pause (s)", ui.DoubleSpinBox({"ID": "Pause", "Minimum": 1.0,
                                                    "Maximum": 30.0, "SingleStep": 0.5,
                                                    "Value": 4.0, "Weight": 0.6})),
            ui.Label({"ID": "Status", "Text": "Ready", "WordWrap": True}),
            ui.HGroup([ui.Button({"ID": "Run", "Text": "Mark"}),
                       ui.Button({"ID": "Close", "Text": "Close"})]),
        ]),
    )
    items = win.GetItems()
    items["Scope"].AddItems(SCOPES)

    def status(text):
        print("[Auto Markers] " + text)
        items["Status"].Text = text

    def on_run(ev):
        scope = ("current", "track", "all")[int(items["Scope"].CurrentIndex)]
        opts = MarkerOptions(
            scope=scope,
            track_index=int(items["Track"].Value),
            keywords=str(items["Keys"].Text),
            topics=bool(items["Topics"].Checked),
            min_pause=float(items["Pause"].Value),
        )
        save_section("markers", {"scope": opts.scope,
                               "track_index": opts.track_index,
                               "keywords": opts.keywords,
                               "topics": opts.topics,
                               "min_pause": opts.min_pause})
        status("Working... (see Workspace > Console for progress)")
        try:
            run_markers(app, opts, status)
        except Exception as exc:
            status("Error: %s" % exc)
            raise

    def on_close(ev):
        disp.ExitLoop()

    win.On.Run.Clicked = on_run
    win.On.Close.Clicked = on_close
    win.On.AutoMarkers.Close = on_close
    win.Show()
    disp.RunLoop()
    win.Hide()


def main():
    app = _resolve()
    if app is None:
        print("Auto Markers: run this from DaVinci Resolve (Workspace > Scripts).")
        return
    if ui_available(app):
        show_dialog(app)
    else:
        no_ui_notice("Auto Markers")
        run_markers(app, as_options("markers", MarkerOptions))


main()
