# PyInstaller spec for Shorts Studio.app — run packaging/build_mac.sh
import os

from PyInstaller.utils.hooks import collect_all

ROOT = os.path.abspath(os.path.join(SPECPATH, ".."))
datas = [(os.path.join(ROOT, "src", "face_reframe", "models"), "face_reframe/models")]
binaries, hidden = [], ["face_reframe.app.window", "face_reframe.app.captions_qt"]
# packages with native libs / data files PyInstaller doesn't find on its own;
# imageio_ffmpeg carries the ffmpeg binary the app falls back to
for pkg in ("faster_whisper", "ctranslate2", "tokenizers", "onnxruntime", "imageio_ffmpeg", "av"):
    try:
        d, b, h = collect_all(pkg)
    except Exception:
        continue
    datas += d
    binaries += b
    hidden += h

a = Analysis(
    [os.path.join(SPECPATH, "launch_app.py")],
    pathex=[os.path.join(ROOT, "src")],
    binaries=binaries,
    datas=datas,
    hiddenimports=hidden,
    excludes=["tkinter", "matplotlib", "PySide6.QtWebEngineCore", "PySide6.Qt3DCore"],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name="Shorts Studio",
          console=False, argv_emulation=False)
coll = COLLECT(exe, a.binaries, a.datas, name="Shorts Studio")
icon = os.path.join(SPECPATH, "ShortsStudio.icns")
app = BUNDLE(
    coll,
    name="Shorts Studio.app",
    icon=icon if os.path.exists(icon) else None,
    bundle_identifier="local.face-reframe.shorts-studio",
    info_plist={
        "CFBundleShortVersionString": "0.7.0",
        "NSHighResolutionCapable": True,
        "LSMinimumSystemVersion": "11.0",
    },
)
