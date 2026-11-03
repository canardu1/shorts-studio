"""Shared pytest fixtures."""

import pytest


@pytest.fixture(autouse=True)
def _isolated_qsettings(tmp_path, monkeypatch):
    """Redirect every ``QSettings(...)`` in the app module to a throwaway
    INI file for the duration of the test.

    ``MainWindow`` persists UI state (logo, closing CTA, output dir, every
    option) via ``QSettings("face_reframe", "ShortsStudio")``.  Without this,
    GUI tests silently wrote test artefacts — like a solid-red test logo —
    into the user's real preferences, where the actual app then picked them
    up and burned them into previews and exports.  (On macOS the documented
    ``setDefaultFormat``/``setPath`` redirection is ignored once the native
    plist exists, so the symbol itself is patched.)
    """
    try:
        from PySide6.QtCore import QSettings
        import face_reframe.app.window as win_mod
    except ImportError:
        yield
        return
    ini = str(tmp_path / "settings.ini")
    monkeypatch.setattr(win_mod, "QSettings",
                        lambda *a, **k: QSettings(ini, QSettings.IniFormat))
    yield
