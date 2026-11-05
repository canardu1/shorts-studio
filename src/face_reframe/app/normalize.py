"""One-time source normalisation for clips that would break the frame-index
timeline: **interlaced** footage (combing survives the reframe) and **VFR**
(frame i is not at i/fps s, so cuts/captions/audio drift — the lesson tapewake
calls "fix the timeline first").

Detected once via PyAV, fixed with a single ffmpeg pass, cached on disk —
every later step (analysis, render, preview) sees a plain CFR progressive
file. ``bwdif=mode=0`` keeps the frame count so indices still match the
original timeline.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
from typing import Callable, Optional

from . import diskcache
from ..i18n import tr
from ..silence import find_ffmpeg


def probe(path: str) -> dict:
    """``{'interlaced': bool, 'vfr': bool, 'fps': float}`` via PyAV.

    VFR = the container's average rate differs from its nominal base rate;
    interlaced = the decoded frames carry the interlace flag."""
    info = {"interlaced": False, "vfr": False, "fps": 0.0}
    try:
        import av
        c = av.open(path)
        vs = c.streams.video[0]
        info["fps"] = float(vs.average_rate or vs.base_rate or 0)
        for f in c.decode(vs):
            info["interlaced"] = bool(getattr(f, "interlaced_frame", False))
            if f is not None:
                break
        c.close()
        # VFR = uneven presentation intervals (avg/base rates often still
        # match, and packet.duration is unset on e.g. mkv) — look at sorted
        # timestamps of the first packets instead
        c = av.open(path)
        pts = []
        for pkt in c.demux(c.streams.video[0]):
            if pkt.pts is not None:
                pts.append(int(pkt.pts))
            if len(pts) >= 60:
                break
        c.close()
        pts.sort()
        ds = {b - a for a, b in zip(pts, pts[1:]) if b > a}
        # ±1-tick timestamp rounding isn't VFR — a real VFR (mixed rates,
        # dropped frames) swings the interval by far more
        info["vfr"] = bool(ds) and max(ds) > 1.25 * min(ds)
    except Exception:
        pass
    return info


def normalised_path(src: str) -> str:
    st = os.stat(src)
    raw = "%s|%d|%d|norm1" % (os.path.abspath(src), st.st_size, int(st.st_mtime))
    return os.path.join(diskcache.cache_dir(),
                        "norm_%s.mkv" % hashlib.sha1(raw.encode()).hexdigest()[:20])


def normalise(src: str, check: Callable[[], None], log) -> str:
    """The path to use in place of ``src`` — ``src`` itself when already clean.

    Near-lossless intermediate (crf 12 + pcm audio) cached next to the
    proxies, so a second render of the same file skips the pass."""
    pr = probe(src)
    if not (pr["interlaced"] or pr["vfr"]):
        return src
    out = normalised_path(src)
    if os.path.exists(out) and os.path.getsize(out) > 10_000:
        log(tr("  normalised source reused"))
        return out
    what = "interlaced" if pr["interlaced"] else "VFR"
    log(tr("  %s source: normalising once (deinterlace + CFR)…") % what)
    cmd = [find_ffmpeg(), "-y", "-loglevel", "error", "-i", src]
    if pr["interlaced"]:
        cmd += ["-vf", "bwdif=mode=0"]
    if pr["vfr"]:
        cmd += ["-fps_mode", "cfr"]
        if pr["fps"] > 0:
            cmd += ["-r", repr(round(pr["fps"], 5))]
    tmp = out + ".part"
    cmd += ["-map", "0:v:0", "-map", "0:a:0?", "-c:v", "libx264", "-crf", "12",
            "-preset", "veryfast", "-pix_fmt", "yuv420p", "-c:a", "pcm_s16le",
            "-f", "matroska", tmp]
    proc = subprocess.Popen(cmd, stderr=subprocess.PIPE)
    while True:
        try:
            proc.wait(timeout=0.3)
            break
        except subprocess.TimeoutExpired:
            try:
                check()
            except Exception:
                proc.kill()
                proc.wait()
                raise
    if proc.returncode != 0:
        err = proc.stderr.read().decode(errors="replace") if proc.stderr else ""
        log(tr("  normalise failed (%s) — using the source as is") % err[-160:])
        try:
            os.unlink(tmp)
        except OSError:
            pass
        return src
    os.replace(tmp, out)
    return out
