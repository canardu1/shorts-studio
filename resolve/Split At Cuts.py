"""Split At Cuts (no dialog)."""

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
    from face_reframe.split_bridge import SplitOptions, run_split  # noqa: E402
    from face_reframe.resolve_bridge import get_resolve  # noqa: E402
except ImportError as exc:
    print("[Split At Cuts] missing dependency: %s" % exc)
    print("[Split At Cuts] run once with your system Python, in the repo folder:")
    print("[Split At Cuts]   python install.py --deps --arch %s --python-version %d.%d"
          % (platform.machine(), sys.version_info[0], sys.version_info[1]))
    raise SystemExit(1)


def _resolve():
    app = globals().get("resolve")
    return app if app is not None else get_resolve()


def main():
    app = _resolve()
    if app is None:
        print("Split At Cuts: run this from DaVinci Resolve (Workspace > Scripts).")
        return
    run_split(app, SplitOptions(scope="current"))


main()
