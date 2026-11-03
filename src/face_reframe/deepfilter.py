"""DeepFilterNet3 via the official self-contained ``deep-filter`` binary.

The binary (model embedded, ~15 MB, MIT) is downloaded on first use into
``models_dir`` and run as a subprocess — a third denoise engine next to
RNNoise and the spectral filter, clearly better on non-stationary noise.
``denoise_db`` maps onto the binary's attenuation limit; ``-D`` compensates
the STFT lookahead so the audio stays in sync with the video.
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import tempfile
from typing import Callable, Optional

from .i18n import tr
from .restore import fetch, models_dir

_VERSION = "0.5.6"
_BASE = "https://github.com/Rikorose/DeepFilterNet/releases/download/v%s/" % _VERSION


def _asset() -> Optional[str]:
    m, sysname = platform.machine().lower(), platform.system()
    if sysname == "Darwin":
        return "deep-filter-%s-%s-apple-darwin" % (
            _VERSION, "aarch64" if m in ("arm64", "aarch64") else "x86_64")
    if sysname == "Linux":
        return "deep-filter-%s-x86_64-unknown-linux-gnu" % _VERSION if m == "x86_64" else None
    if sysname == "Windows":
        return "deep-filter-%s-x86_64-pc-windows-msvc.exe" % _VERSION if m == "amd64" else None
    return None


def binary_path() -> str:
    name = "deep-filter.exe" if os.name == "nt" else "deep-filter"
    return os.path.join(models_dir(), name)


def available() -> bool:
    p = binary_path()
    return os.path.exists(p) and os.path.getsize(p) > 5_000_000


def download(progress: Optional[Callable[[float], None]] = None) -> Optional[str]:
    if available():
        return binary_path()
    asset = _asset()
    if asset is None:
        return None
    path = binary_path()
    fetch(_BASE + asset, path, progress)
    if os.name != "nt":
        os.chmod(path, 0o755)
    return path


def clean_to_wav(src: str, work_dir: Optional[str], atten_db: float = 100.0,
                 log: Callable[[str], None] = print,
                 progress: Optional[Callable[[float], None]] = None,
                 start: float = 0.0, duration: Optional[float] = None) -> Optional[str]:
    """Extract the audio and run DeepFilterNet on it.

    Returns the denoised WAV path, or None when the binary can't run —
    callers then fall back to the ffmpeg denoisers."""
    from .silence import find_ffmpeg
    exe = download(progress)
    if exe is None:
        log(tr("DeepFilterNet unavailable on this platform: RNNoise/spectral stay"))
        return None
    work = work_dir or tempfile.mkdtemp(prefix="dfn_")
    raw = os.path.join(work, "dfn_in.wav")
    try:
        cmd = [find_ffmpeg(), "-y", "-v", "error"]
        if start > 0:
            cmd += ["-ss", "%.6f" % start]
        cmd += ["-i", src, "-vn"]
        if duration and duration > 0:
            cmd += ["-t", "%.6f" % duration]
        cmd += ["-ac", "2", "-ar", "48000", raw]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if r.returncode != 0 or not os.path.exists(raw):
            log(tr("DeepFilterNet: audio extraction failed"))
            return None
        outdir = os.path.join(work, "dfn")
        r = subprocess.run([exe, "-D", "-a", "%.0f" % min(100.0, max(0.0, atten_db)),
                            "-o", outdir, raw],
                           capture_output=True, text=True, timeout=3600)
        out = os.path.join(outdir, "dfn_in.wav")
        if r.returncode != 0 or not os.path.exists(out):
            log(tr("DeepFilterNet failed (%s): RNNoise/spectral stay")
                % (r.stderr.strip()[-200:] or r.returncode))
            return None
        return out
    except Exception as exc:
        log(tr("DeepFilterNet failed (%s): RNNoise/spectral stay") % exc)
        return None
