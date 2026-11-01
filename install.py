#!/usr/bin/env python3
"""Install the Face Reframe script into DaVinci Resolve.

Two-step install:

    python install.py --deps                       # vendor numpy + opencv for THIS machine's arch
    python install.py --deps --arch x86_64         # vendor for an x86_64 Resolve (Rosetta)
    python install.py --deps --python-version 3.13 # vendor wheels for Resolve's embedded Python
    python install.py                              # inside Resolve: Workspace > Scripts > install.py

Wheels are vendored inside the script's own folder under
``face_reframe_lib/deps/<arch>``, so they don't depend on which Python
Resolve uses internally, and arm64/x86_64 copies can coexist.

Run inside Resolve, this script installs itself next to its own location and
reports the exact ``--deps`` command for Resolve's architecture.
"""

import argparse
import os
import platform
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

def _mac_platforms(arch: str) -> list:
    tags = ["macosx_10_%d_%s" % (v, arch) for v in range(9, 16)]
    tags += ["macosx_%d_0_%s" % (v, arch) for v in range(11, 27)]
    for uarch in ({arch, "universal2"} if arch != "universal2" else {"universal2"}):
        tags += ["macosx_%d_%d_%s" % (maj, mnr, uarch)
                 for maj in (10, 11) for mnr in range(9, 16)]
    return tags


PLATFORMS = {"x86_64": _mac_platforms("x86_64"), "arm64": _mac_platforms("arm64")}


def in_resolve() -> bool:
    if any(g in globals() for g in ("resolve", "bmd", "fusion")):
        return True
    try:
        import DaVinciResolveScript  # noqa: F401
        return True
    except ImportError:
        return False


def scripts_dir() -> str:
    system = platform.system()
    if system == "Windows":
        base = os.path.join(os.environ["APPDATA"], "Blackmagic Design", "DaVinci Resolve", "Support", "Fusion", "Scripts")
    elif system == "Darwin":
        base = os.path.expanduser("~/Library/Application Support/Blackmagic Design/DaVinci Resolve/Fusion/Scripts")
    else:
        base = os.path.expanduser("~/.local/share/DaVinciResolve/Fusion/Scripts")
    return os.path.join(base, "Utility")


def _clean(path):
    if os.path.isdir(path):
        shutil.rmtree(path)


def _deps_root(target: str) -> str:
    """<Fusion>/face_reframe_deps — outside Scripts/, next to face_reframe_lib's level."""
    return os.path.join(os.path.dirname(os.path.dirname(target)), "face_reframe_deps")


def install(target: str) -> str:
    lib = os.path.join(target, "face_reframe_lib")
    old_deps = os.path.join(lib, "deps")
    if os.path.isdir(old_deps):
        _clean(old_deps)  # deps inside Scripts/Utility freeze Resolve's script scan
        print("Removed deps/ from the scripts folder (it was slowing Resolve down).")
    pkg_dst = os.path.join(lib, "face_reframe")
    _clean(pkg_dst)
    os.makedirs(lib, exist_ok=True)
    shutil.copytree(os.path.join(HERE, "src", "face_reframe"), pkg_dst,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "__main__.py"))
    # shared Lua bridge inside the lib folder
    shutil.copyfile(os.path.join(HERE, "lua", "fdr.lua"), os.path.join(lib, "fdr.lua"))
    script = None
    for name in ("Face Reframe.py", "Face Reframe (All Tracks).py",
                 "Cut Pauses.py", "Cut Pauses (All Tracks).py",
                 "Cut Fillers.py", "Cut Fillers (All Tracks).py",
                 "Auto Subtitles.py",
                 "Punch In.py", "Punch In (All Tracks).py",
                 "Normalize Audio.py", "Normalize Audio (All Tracks).py",
                 "Cinematic Audio.py", "Cinematic Audio (All Tracks).py",
                 "Cut Breaths.py", "Cut Breaths (All Tracks).py",
                 "Cut Retakes.py", "Cut Retakes (All Tracks).py",
                 "Auto Markers.py",
                 "Sync Audio.py", "Duck Music.py", "Enhance.py",
                 "Face Reframe Settings.py",
                 "Shorts Pipeline.py",
                 "Split At Cuts.py", "Split At Cuts (All Tracks).py",
                 "Normalize Color.py", "Normalize Color (All Tracks).py"):
        with open(os.path.join(HERE, "resolve", name)) as fh:
            launcher = fh.read().replace("__FACE_REFRAME_LIB__", lib)
        script = os.path.join(target, name)
        with open(script, "w") as fh:
            fh.write(launcher)
    # Lua launchers are skipped: they're only needed on Resolve 21.1+ free
    # (Python is gated there), and on 21.0.x they just duplicate the menu.
    return script


def vendor_deps(target: str, arch: str, py_version: str) -> str:
    deps_dst = os.path.join(_deps_root(target), arch)
    _clean(deps_dst)
    os.makedirs(deps_dst, exist_ok=True)
    cmd = [sys.executable, "-m", "pip", "install", "--target", deps_dst,
           "--only-binary=:all:", "--implementation", "cp",
           "--python-version", py_version, "--upgrade", "--force-reinstall"]
    for plat in PLATFORMS[arch]:
        cmd += ["--platform", plat]
    cmd += ["numpy>=2,<3", "opencv-python>=4.12,<5", "imageio-ffmpeg", "faster-whisper",
            "onnxruntime>=1.16,<2"]
    print("$ %s" % " ".join(cmd))
    subprocess.check_call(cmd)
    return deps_dst


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--deps", action="store_true", help="vendor numpy and opencv into the script folder")
    p.add_argument("--arch", choices=["x86_64", "arm64"], default=None,
                   help="CPU arch of the Resolve app (default: this machine's)")
    p.add_argument("--python-version", default=None, metavar="X.Y",
                   help="Python version embedded in Resolve (default: this interpreter's)")
    p.add_argument("--target", default=None, help="override the Resolve Scripts/Utility folder")
    args = p.parse_args()

    in_res = in_resolve()
    machine = platform.machine() or "unknown"
    target = args.target or (HERE if in_res else scripts_dir())
    os.makedirs(target, exist_ok=True)

    if args.deps:
        arch = args.arch or machine
        pyver = args.python_version or "%d.%d" % sys.version_info[:2]
        dst = vendor_deps(target, arch, pyver)
        print("Vendored numpy + opencv (%s, py%s) into %s" % (arch, pyver, dst))
        print("Note: Resolve's Python must also be %s — check the traceback's 'The Python version is' line." % pyver)

    script = install(target)
    if in_res:
        deps = os.path.join(_deps_root(target), machine)
        print("Face Reframe installed next to this file.")
        print("Restart Resolve, then use Workspace > Scripts > Face Reframe (or 'Face Reframe (All Tracks)').")
        if not os.path.isdir(deps):
            print("Dependencies for %s are missing. Once, on your system's Python, run:" % machine)
            print("  python install.py --deps --arch %s --python-version %d.%d"
                  % (machine, sys.version_info[0], sys.version_info[1]))
    else:
        print("Installed: %s" % script)
        print("Restart Resolve, then use Workspace > Scripts > Face Reframe (or 'Face Reframe (All Tracks)').")
        if not args.deps:
            print("Run 'python install.py --deps' if Resolve says it can't find numpy or cv2.")


if __name__ == "__main__":
    main()
