"""Face Reframe Settings - open the toolkit's options file in a text editor.

DaVinci Resolve Free 19.1+ disables the script UI, so the tools read their
options from ~/.face_reframe_prefs.json instead of showing a dialog.
Run this to create (if needed) and open that file; edit it, save, then
re-run the tool.
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
    from face_reframe.prefs import open_in_editor, prefs_path  # noqa: E402
except ImportError as exc:
    print("[Settings] missing dependency: %s" % exc)
    raise SystemExit(1)


def main():
    open_in_editor(print)
    try:
        with open(prefs_path()) as fh:
            print(fh.read())
    except OSError:
        pass


main()
