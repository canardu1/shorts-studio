"""Light preview proxies for heavy camera files.

4K / HEVC 10-bit 4:2:2 footage (e.g. Sony XAVC HS) decodes slowly in
software and — worse — every seek re-decodes from a keyframe that can be
seconds back (1.7 s per seek measured on an M1). The preview then stalls at
every skipped cut. Like an editing app, we transcode a small proxy once
(540p, keyframe every 12 frames) and play from it; stills and the export
still use the original file.
"""

from __future__ import annotations

import hashlib
import os
import re
import subprocess

from ..i18n import tr
import sys
import time
from typing import Callable, Optional

from . import diskcache

PROXY_HEIGHT = 540


def proxy_path(src: str) -> Optional[str]:
    try:
        st = os.stat(src)
    except OSError:
        return None
    raw = "%s|%d|%d|%d" % (os.path.abspath(src), st.st_size, int(st.st_mtime), PROXY_HEIGHT)
    return os.path.join(diskcache.cache_dir(), "proxy_%s.mp4" % hashlib.sha1(raw.encode()).hexdigest()[:20])


def existing(src: str) -> Optional[str]:
    p = proxy_path(src)
    return p if p and os.path.exists(p) and os.path.getsize(p) > 1000 else None


def needs_proxy(reader) -> bool:
    """Heavy source? (above 1080p, or slow to decode even sequentially)."""
    if reader.width * reader.height > 2_200_000:
        return True
    cap = reader.cap
    t = time.perf_counter()
    n = 0
    for _ in range(8):
        if not cap.grab():
            break
        n += 1
    reader._next += n  # the reader's position moved
    return n > 0 and (time.perf_counter() - t) / n > 0.008


def build(src: str, out: str, progress: Optional[Callable[[float], None]] = None,
          duration: float = 0.0) -> str:
    """Transcode the proxy (hardware decode on macOS when available)."""
    from ..silence import find_ffmpeg
    tmp = out + ".part.mp4"
    base = ["-i", src, "-map", "0:v:0", "-vf", "scale=-2:'min(%d,ih)'" % PROXY_HEIGHT,
            "-c:v", "libx264", "-preset", "ultrafast", "-crf", "23", "-g", "12", "-bf", "0",
            "-pix_fmt", "yuv420p", "-an"]
    attempts = []
    # keep every frame (1:1 with the source's frame numbers); -fps_mode is
    # ffmpeg >= 5, -vsync 0 the older spelling
    for sync in (["-fps_mode", "passthrough"], ["-vsync", "0"]):
        if sys.platform == "darwin":
            attempts.append((["-hwaccel", "videotoolbox"], sync))
        attempts.append(([], sync))
    err = ""
    for hw, sync in attempts:
        cmd = ([find_ffmpeg(), "-y", "-loglevel", "error", "-nostats", "-progress", "pipe:1"]
               + hw + base + sync + [tmp])
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        try:
            for line in proc.stdout:
                m = re.match(r"out_time_us=(\d+)", line.strip())
                if m and progress and duration > 0:
                    progress(min(1.0, int(m.group(1)) / 1e6 / duration))
        except BaseException:
            proc.kill()
            proc.wait()
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
        proc.wait()
        err = proc.stderr.read() if proc.stderr else ""
        if proc.returncode == 0:
            os.replace(tmp, out)
            return out
    raise RuntimeError(tr("proxy not created: %s") % err[-300:])
