"""Enhance - sharpen or AI-restore the clips on the timeline as they are.

Run from DaVinci Resolve: Workspace > Scripts > Enhance.
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
    from face_reframe.enhance_bridge import EnhanceRunOptions, run_enhance  # noqa: E402
    from face_reframe.prefs import (as_options, no_ui_notice,  # noqa: E402
                                    save_section, ui_available)
    from face_reframe.resolve_bridge import get_resolve  # noqa: E402
except ImportError as exc:
    print("[Enhance] missing dependency: %s" % exc)
    print("[Enhance] run once with your system Python, in the repo folder:")
    print("[Enhance]   python install.py --deps --arch %s --python-version %d.%d"
          % (platform.machine(), sys.version_info[0], sys.version_info[1]))
    raise SystemExit(1)

SCOPES = ["Clip under playhead", "All clips on track", "All video tracks"]
MODES = ["Fast (sharpen)", "AI fast (Real-ESRGAN, half res)", "AI full (Real-ESRGAN, slow)"]


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
        {"ID": "EnhanceTool", "WindowTitle": "Enhance",
         "Geometry": [300, 200, 420, 190]},
        ui.VGroup([
            row("Apply to", ui.ComboBox({"ID": "Scope", "Weight": 0.6})),
            row("Track (V#)", ui.SpinBox({"ID": "Track", "Minimum": 1, "Maximum": 50,
                                         "Value": 1, "Weight": 0.6})),
            row("Quality", ui.ComboBox({"ID": "Mode", "Weight": 0.6})),
            ui.Label({"ID": "Status", "Text": "Ready", "WordWrap": True}),
            ui.HGroup([ui.Button({"ID": "Run", "Text": "Enhance"}),
                       ui.Button({"ID": "Close", "Text": "Close"})]),
        ]),
    )
    items = win.GetItems()
    items["Scope"].AddItems(SCOPES)
    items["Mode"].AddItems(MODES)

    def status(text):
        print("[Enhance] " + text)
        items["Status"].Text = text

    def on_run(ev):
        opts = EnhanceRunOptions(
            scope=("current", "track", "all")[int(items["Scope"].CurrentIndex)],
            track_index=int(items["Track"].Value),
            mode=("fast", "ai", "ai-hq")[int(items["Mode"].CurrentIndex)],
        )
        save_section("enhance", {"scope": opts.scope,
                               "track_index": opts.track_index,
                               "mode": opts.mode})
        status("Working... (see Workspace > Console for progress)")
        try:
            run_enhance(app, opts, status)
        except Exception as exc:
            status("Error: %s" % exc)
            raise

    def on_close(ev):
        disp.ExitLoop()

    win.On.Run.Clicked = on_run
    win.On.Close.Clicked = on_close
    win.On.EnhanceTool.Close = on_close
    win.Show()
    disp.RunLoop()
    win.Hide()


def main():
    app = _resolve()
    if app is None:
        print("Enhance: run this from DaVinci Resolve (Workspace > Scripts).")
        return
    if ui_available(app):
        show_dialog(app)
    else:
        no_ui_notice("Enhance")
        opts = as_options("enhance", EnhanceRunOptions)
        print("[Enhance] scope=%s track=%d mode=%s"
              % (opts.scope, opts.track_index, opts.mode))
        run_enhance(app, opts)


main()
