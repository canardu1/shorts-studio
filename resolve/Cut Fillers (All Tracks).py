"""Cut filler words (uhm, ehm, ...) from every clip on every video track."""

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
    from face_reframe.cut_bridge import CutOptions, run_cut  # noqa: E402
    from face_reframe.resolve_bridge import get_resolve  # noqa: E402
except ImportError as exc:
    print("[Cut Fillers] missing dependency: %s" % exc)
    print("[Cut Fillers] run once with your system Python, in the repo folder:")
    print("[Cut Fillers]   python install.py --deps --arch %s --python-version %d.%d"
          % (platform.machine(), sys.version_info[0], sys.version_info[1]))
    raise SystemExit(1)


def _resolve():
    app = globals().get("resolve")
    return app if app is not None else get_resolve()


def main():
    app = _resolve()
    if app is None:
        print("Cut Fillers: run this from DaVinci Resolve (Workspace > Scripts).")
        return
    run_cut(app, CutOptions(scope="all", remove_fillers=True))


main()
