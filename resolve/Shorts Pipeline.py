"""Shorts Pipeline - run the whole toolkit in the right order on all tracks.

Run from DaVinci Resolve: Workspace > Scripts > Shorts Pipeline.
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
    from face_reframe.cinematic_bridge import CinematicOptions, run_cinematic  # noqa: E402
    from face_reframe.cut_bridge import CutOptions, run_cut  # noqa: E402
    from face_reframe.pipeline import Options  # noqa: E402
    from face_reframe.prefs import (no_ui_notice, save_section,  # noqa: E402
                                    section, ui_available)
    from face_reframe.punch_bridge import PunchOptions, run_punch  # noqa: E402
    from face_reframe.resolve_bridge import ResolveOptions, get_resolve, run  # noqa: E402
    from face_reframe.subtitles_bridge import SubtitleOptions, run_subtitles  # noqa: E402
except ImportError as exc:
    print("[Shorts Pipeline] missing dependency: %s" % exc)
    print("[Shorts Pipeline] run once with your system Python, in the repo folder:")
    print("[Shorts Pipeline]   python install.py --deps --arch %s --python-version %d.%d"
          % (platform.machine(), sys.version_info[0], sys.version_info[1]))
    raise SystemExit(1)

ASPECTS = ["9:16", "4:5", "1:1"]


def _resolve():
    app = globals().get("resolve")
    return app if app is not None else get_resolve()


def _bmd():
    mod = globals().get("bmd")
    if mod is not None:
        return mod
    import DaVinciResolveScript as dvr  # noqa: F401
    return dvr


def run_pipeline(app, steps, aspect, log):
    """Checked steps run in fixed order; each bake step swaps the current
    timeline, so the next tool always works on the right timeline."""
    if "cut" in steps:
        run_cut(app, CutOptions(scope="all"), log)
    if "cinematic" in steps:
        run_cinematic(app, CinematicOptions(scope="all"), log)
    if "reframe" in steps:
        run(app, ResolveOptions(scope="all", reframe=Options(aspect=aspect)), log)
    if "punch" in steps:
        run_punch(app, PunchOptions(scope="all"), log)
    if "subs" in steps:
        run_subtitles(app, SubtitleOptions(scope="all"), log)
    log("Pipeline done.")


def show_dialog(app):
    fu = app.Fusion()
    ui = fu.UIManager
    disp = _bmd().UIDispatcher(ui)

    def row(label, widget):
        return ui.HGroup([ui.Label({"Text": label, "Weight": 0.4}), widget])

    win = disp.AddWindow(
        {"ID": "ShortsPipeline", "WindowTitle": "Shorts Pipeline",
         "Geometry": [300, 200, 420, 340]},
        ui.VGroup([
            ui.Label({"Text": "Run in order on ALL tracks:", "Weight": 0.2}),
            ui.CheckBox({"ID": "Cut", "Text": "1. Cut Pauses  (new '(no pauses)' timeline)",
                         "Checked": True}),
            ui.CheckBox({"ID": "Cine", "Text": "2. Cinematic Audio  (new '(cinematic)' timeline)",
                         "Checked": True}),
            ui.CheckBox({"ID": "Reframe", "Text": "3. Face Reframe  (Fusion comps)", "Checked": True}),
            ui.CheckBox({"ID": "Punch", "Text": "4. Punch In", "Checked": False}),
            ui.CheckBox({"ID": "Subs", "Text": "5. Auto Subtitles", "Checked": False}),
            row("Aspect", ui.ComboBox({"ID": "Aspect", "Weight": 0.6})),
            ui.Label({"ID": "Status", "Text": "Ready", "WordWrap": True}),
            ui.HGroup([ui.Button({"ID": "Run", "Text": "Run pipeline"}),
                       ui.Button({"ID": "Close", "Text": "Close"})]),
        ]),
    )
    items = win.GetItems()
    items["Aspect"].AddItems(ASPECTS)

    def status(text):
        print("[Shorts Pipeline] " + text)
        items["Status"].Text = text

    def on_run(ev):
        steps = set()
        if items["Cut"].Checked:
            steps.add("cut")
        if items["Cine"].Checked:
            steps.add("cinematic")
        if items["Reframe"].Checked:
            steps.add("reframe")
        if items["Punch"].Checked:
            steps.add("punch")
        if items["Subs"].Checked:
            steps.add("subs")
        save_section("pipeline", {
            "steps": sorted(steps),
            "aspect": ASPECTS[int(items["Aspect"].CurrentIndex)]})
        status("Running... (see Workspace > Console for progress)")
        try:
            run_pipeline(app, steps, ASPECTS[int(items["Aspect"].CurrentIndex)], status)
        except Exception as exc:
            status("Error: %s" % exc)
            raise

    def on_close(ev):
        disp.ExitLoop()

    win.On.Run.Clicked = on_run
    win.On.Close.Clicked = on_close
    win.On.ShortsPipeline.Close = on_close
    win.Show()
    disp.RunLoop()
    win.Hide()


def main():
    app = _resolve()
    if app is None:
        print("Shorts Pipeline: run this from DaVinci Resolve (Workspace > Scripts).")
        return
    if ui_available(app):
        show_dialog(app)
    else:
        no_ui_notice("Shorts Pipeline")
        p = section("pipeline")
        print("[Shorts Pipeline] steps=%s aspect=%s"
              % (",".join(p["steps"]) or "none", p["aspect"]))
        run_pipeline(app, set(p["steps"]), str(p["aspect"]), print)


main()
