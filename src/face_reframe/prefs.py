"""Persistent settings for Resolve builds without the script UI.

DaVinci Resolve Free 19.1+ disables ``fusion.UIManager``, so launchers
cannot show their dialog and would silently run with defaults. In that
case options come from a JSON file instead: ``~/.face_reframe_prefs.json``.
The 'Face Reframe Settings' script creates and opens it in an editor.

On builds where the UI exists, dialogs write their choices back to the
same file, so no-dialog launchers ('All Tracks' variants) reuse the last
used settings.
"""

from __future__ import annotations

import json
import os

DEFAULTS = {
    "face_reframe": {
        "aspect": "9:16",
        "scope": "all",
        "track_index": 1,
        "mode": "fusion",
        "enhance": "off",
        "zoom": 1.0,
        "follow_vertical": False,
        "smoothing": 0.5,
        "dead_zone": 0.05,
        "max_speed": 0.6,
        "wide_mode": "zoom",
        "wide_threshold": 0.12,
        "max_zoom": 2.0,
        "set_timeline_resolution": True,
    },
    "enhance": {
        "scope": "all",
        "track_index": 1,
        "mode": "fast",
    },
    "markers": {
        "scope": "all",
        "track_index": 1,
        "keywords": "taglia,rifacciamo,retake,daccapo,ricominciamo",
        "topics": True,
        "min_pause": 4.0,
    },
    "pipeline": {
        "steps": ["cut", "cinematic", "reframe"],
        "aspect": "9:16",
    },
}

_COMMENT = (
    "Options for the Face Reframe toolkit. On DaVinci Resolve FREE 19.1+ "
    "script dialogs are disabled, so every tool reads its section here; "
    "edit, save, re-run the script. On Studio the dialogs update this file."
)


def prefs_path() -> str:
    return os.path.join(os.path.expanduser("~"), ".face_reframe_prefs.json")


def load() -> dict:
    try:
        with open(prefs_path()) as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def save(data: dict) -> None:
    try:
        with open(prefs_path(), "w") as fh:
            json.dump(data, fh, indent=2, sort_keys=True)
            fh.write("\n")
    except OSError:
        pass


def section(name: str) -> dict:
    """The defaults for *name* merged with the file's overrides."""
    ensure()
    merged = dict(DEFAULTS.get(name, {}))
    overrides = load().get(name)
    if isinstance(overrides, dict):
        merged.update({k: v for k, v in overrides.items() if k in merged})
    return merged


def reframe_options(scope=None):
    """Build ``ResolveOptions`` from the ``face_reframe`` prefs section."""
    from .pipeline import Options
    from .resolve_bridge import ResolveOptions
    from .track import ShotSettings, TrackSettings
    p = section("face_reframe")
    return ResolveOptions(
        reframe=Options(
            aspect=str(p["aspect"]),
            zoom=float(p["zoom"]),
            follow_vertical=bool(p["follow_vertical"]),
            track=TrackSettings(smoothing=float(p["smoothing"]),
                                dead_zone=float(p["dead_zone"]),
                                max_speed=float(p["max_speed"])),
            shots=ShotSettings(wide_mode=str(p["wide_mode"]),
                               wide_threshold=float(p["wide_threshold"]),
                               max_zoom=float(p["max_zoom"]))),
        scope=scope or str(p["scope"]),
        track_index=int(p["track_index"]),
        mode=str(p["mode"]),
        enhance=str(p["enhance"]),
        set_timeline_resolution=bool(p["set_timeline_resolution"]))


def save_section(name: str, values: dict) -> None:
    data = load()
    data.pop("_comment", None)
    data[name] = values
    save(data)


def as_options(name: str, cls):
    """Instantiate an options dataclass from a prefs section."""
    import dataclasses
    names = {f.name for f in dataclasses.fields(cls)}
    return cls(**{k: v for k, v in section(name).items() if k in names})


def ensure() -> None:
    """Write the prefs file (with a help header) if it doesn't exist."""
    if os.path.exists(prefs_path()):
        return
    data = {"_comment": _COMMENT}
    data.update(DEFAULTS)
    save(data)


def open_in_editor(log=print) -> None:
    """Create the prefs file and open it in the platform's text editor."""
    ensure()
    path = prefs_path()
    log("Settings file: %s" % path)
    try:
        import subprocess, sys
        if sys.platform == "darwin":
            subprocess.Popen(["open", "-t", path])
        elif os.name == "nt":
            os.startfile(path)  # type: ignore[attr-defined]
        else:
            subprocess.Popen(["xdg-open", path])
        log("Opened in the default editor - edit, save, then re-run the tool.")
    except Exception:
        log("Open it manually in any text editor.")


def no_ui_notice(tool: str, log=print) -> None:
    ensure()
    log("[%s] no script UI on this Resolve (Free 19.1+ disables it)" % tool)
    log("[%s] options come from %s" % (tool, prefs_path()))
    log("[%s] run 'Face Reframe Settings' to edit them" % tool)


def ui_available(app, log=print) -> bool:
    """Probe for UIManager, caching a negative result per Resolve version.

    Free 19.1+ has no script UI; if Resolve is later upgraded (e.g. to
    Studio) the cached verdict is discarded when the version string
    changes and the probe runs again.
    """
    try:
        ver = str(app.GetVersionString() or "")
    except Exception:
        ver = ""
    data = load()
    if data.get("_ui_ok") is False and data.get("_ui_ver") == ver:
        return False
    try:
        ok = app.Fusion().UIManager is not None
    except Exception:
        ok = False
    if not ok:
        data["_ui_ok"] = False
        data["_ui_ver"] = ver
        save(data)
    else:
        if data.pop("_ui_ok", None) is not None or data.pop("_ui_ver", None):
            save(data)
    return ok
