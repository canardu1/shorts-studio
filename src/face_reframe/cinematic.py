"""Clean + "cinematic" voice: denoise, de-ess, EQ, compress, normalize.

Everything runs through ffmpeg's audio filters — Resolve exposes no
per-clip audio-FX scripting API, so the bridge bakes a processed copy of
each clip (video stream-copied, lossless) and rebuilds the timeline.
The denoise floor adapts to the clip's own noise floor, measured the same
way Cut Pauses does.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import numpy as np

from .loudness import TARGET_I, gain_db
from .silence import SAMPLE_RATE, extract_audio, find_ffmpeg


@dataclass
class CinematicParams:
    denoise_db: float = 12.0
    """Noise-reduction depth (dB). 0 disables denoising; with the ``rnn``
    engine the value is only an on/off switch — RNNoise has no depth knob."""
    denoise_engine: str = "rnn"
    """``rnn`` = arnndn (RNNoise GRU net) when the ffmpeg build and the model
    support it, falling back to ``fft``; ``fft`` = afftdn spectral subtraction;
    ``dfn`` = DeepFilterNet3 (external binary, best on non-stationary noise)."""
    highpass: float = 85.0
    """Rumble/handling-noise cut below this frequency (Hz)."""
    deess: float = 0.3
    """De-esser trigger intensity (0-1). 0 disables it."""
    dereverb_db: float = 12.0
    """Max attenuation (dB) for the room tail between words — a downward
    expander tuned just above the noise floor. It tightens the room; it
    doesn't deconvolve true echo. 0 disables it."""
    mud_cut_db: float = -1.2
    """Peaking-EQ cut at 320 Hz to reduce boxiness (dB)."""
    comp_threshold: float = -20.0
    """Compressor threshold (dBFS)."""
    comp_ratio: float = 2.5
    """Compressor ratio; 1 disables compression."""
    comp_makeup_db: float = 6.0
    """Compressor makeup gain (dB)."""
    warmth_db: float = 1.6
    """Low shelf at 110 Hz — body of the voice (dB)."""
    presence_db: float = 2.2
    """Peaking-EQ boost at 4.2 kHz — clarity/intelligibility (dB)."""
    air_db: float = 1.2
    """High shelf at 10.5 kHz — "expensive mic" sheen (dB)."""
    limiter: float = 0.95
    """Brick-wall ceiling, as a fraction of full scale."""


def _window_rms_db(pcm: np.ndarray, window_ms: float = 25.0) -> np.ndarray:
    win = max(1, int(round(window_ms * SAMPLE_RATE / 1000.0)))
    n = len(pcm) // win
    if n == 0:
        return np.zeros(0)
    rms = np.sqrt(np.mean(pcm[: n * win].reshape(n, win) ** 2, axis=1) + 1e-12)
    return 20.0 * np.log10(rms)


def noise_floor_db(pcm: np.ndarray, percentile: float = 20.0) -> float:
    """Estimated noise floor of the clip, in dBFS (windowed-RMS percentile)."""
    db = _window_rms_db(pcm)
    return float(np.percentile(db, percentile)) if len(db) else -60.0


def speech_level_db(pcm: np.ndarray, percentile: float = 90.0) -> float:
    """Level of the loud (spoken) parts of the clip, in dBFS."""
    db = _window_rms_db(pcm)
    return float(np.percentile(db, percentile)) if len(db) else -20.0


_GS_OK = None


def afftdn_gs_supported() -> bool:
    """Whether the resolved ffmpeg understands afftdn's ``gs`` (gain smooth).

    ``gs`` only exists in ffmpeg >= 5; on older binaries the whole chain
    fails to build, so the option is probed once and dropped when missing.
    """
    global _GS_OK
    if _GS_OK is None:
        try:
            raw = subprocess.run(
                [find_ffmpeg(), "-v", "error", "-f", "lavfi", "-i",
                 "anullsrc=d=0.1", "-af", "afftdn=gs=1", "-f", "null", "-"],
                capture_output=True, check=False)
            _GS_OK = raw.returncode == 0
        except Exception:
            _GS_OK = False
    return _GS_OK


RNN_URL = "https://github.com/richardpl/arnndn-models/raw/master/sh.rnnn"
"""somnolent-hogwash: trained for speech over recording/room noise (~300 kB)."""
RNN_FILE = "sh.rnnn"


def rnn_model_path() -> str:
    from .restore import models_dir
    return os.path.join(models_dir(), RNN_FILE)


def download_rnn() -> Optional[str]:
    """The RNNoise model file for arnndn, or None if it can't be fetched."""
    path = rnn_model_path()
    if os.path.exists(path) and os.path.getsize(path) > 50_000:
        return path
    try:
        from .restore import fetch
        return fetch(RNN_URL, path, None)
    except Exception:
        return None


_ARNNDN_OK = None


def arnndn_supported() -> bool:
    """Whether the resolved ffmpeg ships the arnndn (RNNoise) filter."""
    global _ARNNDN_OK
    if _ARNNDN_OK is None:
        try:
            raw = subprocess.run([find_ffmpeg(), "-hide_banner", "-filters"],
                                 capture_output=True, check=False)
            _ARNNDN_OK = b"arnndn" in raw.stdout
        except Exception:
            _ARNNDN_OK = False
    return _ARNNDN_OK


def _rnnoise() -> Optional[str]:
    """``arnndn=`` filter term when both the filter and the model exist."""
    if not arnndn_supported():
        return None
    path = download_rnn()
    if path is None:
        return None
    return "arnndn=m='%s'" % path.replace("'", "\\'")


def _denoiser(p: CinematicParams, floor_db: float) -> Optional[str]:
    """Denoise stage. ``rnn`` uses the RNNoise net (arnndn) and falls back to
    the afftdn stage below when the model or the filter can't run.

    afftdn stage: nf tracks the measured floor; nr scales a little with it.
    ``gs`` smooths the per-band gain decision — without it the residual
    comes out as isolated spectral bins (the "metallic"/watery artifact).
    nr stays capped on purpose: deep FFT suppression is what makes that
    residue audible."""
    if p.denoise_db <= 0:
        return None
    if p.denoise_engine == "rnn":
        rnn = _rnnoise()
        if rnn is not None:
            return rnn
    nr = p.denoise_db * (1.15 if floor_db > -40 else 0.8 if floor_db < -55 else 1.0)
    nf = min(-20.0, max(-80.0, floor_db))
    return "afftdn=nr=%.1f:nf=%.1f%s" % (
        min(97.0, max(0.0, nr)), nf, ":gs=10" if afftdn_gs_supported() else "")


def output_dir_for(path: str) -> str:
    """Where baked files may be written: beside the source, else a temp dir.

    Sources can live on read-only mounts (external drives, NAS)."""
    d = os.path.dirname(os.path.abspath(path))
    if os.access(d, os.W_OK):
        return d
    return tempfile.mkdtemp(prefix="face_reframe_")


def combine_loudness(parts: List[Tuple[float, float]]) -> float:
    """Energy-weighted integrated loudness (LUFS) of ``[(lufs, seconds)]``.

    LUFS is an energy measure, so the loudness of several ranges played back
    to back approximates the energy mean of their individual loudnesses."""
    total = sum(d for _, d in parts)
    if total <= 0:
        return -70.0
    e = sum(d * 10.0 ** (lufs / 10.0) for lufs, d in parts) / total
    return 10.0 * float(np.log10(max(e, 1e-12)))


def audio_chain(p: CinematicParams, floor_db: float,
                speech_db: Optional[float] = None) -> str:
    """ffmpeg -af string: clean first, then dynamics, then the tonal polish.

    With dereverberation on, the graph splits after the cleaning stage: the
    sweetened branch is gated by the *un-boosted* cleaned signal, so the
    compressor's makeup gain can't re-amplify the room tail it just
    suppressed. ``speech_db`` anchors the gate threshold to the spoken level.
    """
    clean = []
    if p.highpass > 0:
        clean.append("highpass=f=%.0f" % p.highpass)
    denoise = _denoiser(p, floor_db)
    if denoise:
        clean.append(denoise)
    if p.deess > 0:
        clean.append("deesser=i=%.2f:m=0.5:f=0.5" % min(1.0, p.deess))
    clean = ",".join(clean)

    sweet = []
    if p.mud_cut_db:
        sweet.append("equalizer=f=320:t=q:w=1.2:g=%.1f" % p.mud_cut_db)
    if p.comp_ratio > 1.0:
        sweet.append(
            "acompressor=threshold=%.4f:ratio=%.1f:attack=8:release=150:makeup=%.2f"
            % (10.0 ** (p.comp_threshold / 20.0), p.comp_ratio, 10.0 ** (p.comp_makeup_db / 20.0)))
    if p.warmth_db:
        sweet.append("bass=g=%.1f:f=110" % p.warmth_db)
    if p.presence_db:
        sweet.append("equalizer=f=4200:t=q:w=1.0:g=%.1f" % p.presence_db)
    if p.air_db:
        sweet.append("treble=g=%.1f:f=10500" % p.air_db)
    sweet = ",".join(sweet)

    if p.dereverb_db <= 0 or not sweet:
        if p.dereverb_db > 0:
            # nothing boosts the tail afterwards — a plain inline gate suffices
            clean = _append(clean, _gate(p, floor_db, speech_db, "agate"))
        return _append(clean, sweet)
    # [sweetened | clean] sidechaingate: duck the boosted signal by its
    # pre-boost level; threshold sits 16 dB under the spoken level so speech
    # stays open and the (lower) room tail gets ducked by up to dereverb_db
    # both gate inputs pinned to one sample format: older ffmpeg (4.x) can't
    # negotiate sidechaingate's formats on its own
    pin = "aformat=sample_fmts=dbl:channel_layouts=mono|stereo"
    return "%s%sasplit=2[k0][m];[k0]%s[k];[m]%s,%s[s];[s][k]%s" % (
        clean, "," if clean else "", pin, sweet, pin, _gate(p, floor_db, speech_db, "sidechaingate"))


def _append(chain: str, stage: str) -> str:
    return "%s,%s" % (chain, stage) if chain and stage else chain or stage


def _gate(p: CinematicParams, floor_db: float, speech_db: Optional[float], name: str) -> str:
    """Downward expander spec: ducks whatever sits below the threshold by up
    to ``dereverb_db`` — the room tail between words. The threshold anchors
    to the spoken level (the tail is quiet by definition); with no speech
    estimate it floats 14 dB above the noise floor instead."""
    loud = speech_db if speech_db is not None else floor_db + 16.0
    thr_db = loud - 16.0 if speech_db is not None else floor_db + 14.0
    thr = min(0.3, max(0.005, 10.0 ** (thr_db / 20.0)))
    rng = min(1.0, max(0.01, 10.0 ** (-p.dereverb_db / 20.0)))
    return ("%s=mode=downward:threshold=%.4f:ratio=2.2:attack=15:release=200"
            ":range=%.3f:knee=2.8" % (name, thr, rng))


def file_duration(path: str) -> float:
    """Container duration in seconds (0.0 when ffmpeg can't tell)."""
    raw = subprocess.run([find_ffmpeg(), "-hide_banner", "-i", path],
                         capture_output=True, check=False)
    m = re.search(r"Duration: (\d+):(\d+):(\d+\.\d+)", raw.stderr.decode(errors="replace"))
    if not m:
        return 0.0
    return int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))


def _run_ffmpeg(cmd, total: float = 0.0,
                progress: Optional[Callable[[float], None]] = None) -> str:
    """Run ffmpeg, calling ``progress`` with out_time/total fractions.

    ``-progress pipe:2`` muxes machine-readable progress into stderr, which is
    also where loudnorm writes its JSON — the full stderr text is returned.
    """
    proc = subprocess.Popen(cmd + ["-nostats", "-progress", "pipe:2"],
                            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                            text=True, errors="replace")
    lines, seen = [], set()
    for line in proc.stderr:
        lines.append(line)
        if progress is None or total <= 0:
            continue
        m = re.match(r"out_time=(\d+):(\d+):(\d+\.\d+)", line.strip())
        if m:
            t = int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))
            frac = min(1.0, t / total)
            if int(frac * 100) not in seen:
                seen.add(int(frac * 100))
                progress(frac)
    proc.wait()
    text = "".join(lines)
    if proc.returncode != 0:
        raise RuntimeError("ffmpeg failed: %s" % text[-400:])
    return text


def measure_processed(path: str, start: float, duration: Optional[float],
                      p: CinematicParams, floor_db: float,
                      speech_db: Optional[float] = None,
                      progress: Optional[Callable[[float], None]] = None,
                      audio_src: Optional[str] = None) -> float:
    """Integrated loudness (LUFS) the clip will have after the chain runs.

    ``audio_src`` measures the chain on a different audio file (e.g. the
    DeepFilterNet output) while ``path`` only marks the position."""
    chain = audio_chain(p, floor_db, speech_db)
    cmd = [find_ffmpeg(), "-v", "info"]
    if start > 0:
        cmd += ["-ss", "%.6f" % start]
    cmd += ["-i", audio_src or path, "-map", "a:0"]
    if duration and duration > 0:
        cmd += ["-t", "%.6f" % duration]
    cmd += ["-af", "%sloudnorm=I=%.1f:print_format=json"
            % ((chain + ",") if chain else "", TARGET_I), "-f", "null", "-"]
    text = _run_ffmpeg(cmd, duration or 0.0, progress)
    m = re.search(r"\{[^{}]*\"input_i\"[^{}]*\}", text)
    if not m:
        raise RuntimeError("ffmpeg loudnorm failed: %s" % text[-300:])
    return float(json.loads(m.group(0))["input_i"])


def bake_cinematic(path: str, out_path: str, p: CinematicParams,
                   floor_db: float, gain: float,
                   speech_db: Optional[float] = None,
                   progress: Optional[Callable[[float], None]] = None,
                   duration: float = 0.0, audio_src: Optional[str] = None) -> str:
    """Write ``out_path``: video stream-copied, audio cleaned + sweetened.

    Mirrors :func:`face_reframe.loudness.bake_normalized` — the copy keeps
    the source's frame layout, so timeline source ranges carry over.
    ``audio_src`` swaps the audio input (DeepFilterNet's cleaned WAV)."""
    chain = audio_chain(p, floor_db, speech_db)
    tail = "volume=%.2fdB,alimiter=limit=%.2f" % (gain, min(1.0, max(0.06, p.limiter)))
    chain = "%s,%s" % (chain, tail) if chain else tail
    cmd = [find_ffmpeg(), "-y", "-v", "error", "-i", path]
    if audio_src is not None:
        cmd += ["-i", audio_src, "-map", "0:v:0?", "-map", "1:a:0"]
    else:
        cmd += ["-map", "0:v:0?", "-map", "0:a?"]
    cmd += ["-c:v", "copy", "-af", chain,
            "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", out_path]
    _run_ffmpeg(cmd, duration or file_duration(path), progress)
    return out_path


def enhance(path: str, out_path: str, p: Optional[CinematicParams] = None,
            target: float = TARGET_I, start: float = 0.0,
            duration: Optional[float] = None,
            progress: Optional[Callable[[float], None]] = None) -> float:
    """Clean + sweeten ``path`` into ``out_path``. Returns the applied gain."""
    p = p or CinematicParams()
    pcm = extract_audio(path, start, duration)
    floor = noise_floor_db(pcm)
    loud = speech_level_db(pcm)
    dfn_wav = None
    # start>0 sections are left to the ffmpeg denoisers: the baked video is
    # whole-file, a sectioned wav would go out of sync
    if p.denoise_engine == "dfn" and p.denoise_db > 0 and start <= 0:
        from .deepfilter import clean_to_wav
        import dataclasses as _dc
        dfn_wav = clean_to_wav(path, None, atten_db=min(100.0, p.denoise_db * 2.5),
                               log=print, progress=progress,
                               start=start, duration=duration)
        if dfn_wav is not None:
            # already denoised: the ffmpeg chain skips its own denoise stage
            p = _dc.replace(p, denoise_db=0)
    measure_src = dfn_wav or path
    measured = measure_processed(measure_src, 0.0 if dfn_wav else start, duration,
                                 p, floor, speech_db=loud, progress=progress)
    gain = gain_db(measured, target)
    bake_cinematic(path, out_path, p, floor, gain, speech_db=loud, progress=progress,
                   audio_src=dfn_wav)
    return gain


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="face_reframe.cinematic",
        description="Clean dialogue audio and give it a cinematic polish "
                    "(denoise, de-ess, EQ, compression, loudness).")
    p.add_argument("input")
    p.add_argument("-o", "--output")
    p.add_argument("--denoise", type=float, default=12.0, help="noise reduction depth in dB (0=off)")
    p.add_argument("--warmth", type=float, default=1.6, help="low-shelf boost at 110 Hz (dB)")
    p.add_argument("--presence", type=float, default=2.2, help="presence boost at 4.2 kHz (dB)")
    p.add_argument("--air", type=float, default=1.2, help="high-shelf boost at 10.5 kHz (dB)")
    p.add_argument("--dereverb", type=float, default=12.0,
                   help="max dB the room tail is ducked between words (0=off)")
    p.add_argument("--target", type=float, default=TARGET_I, help="loudness target in LUFS")
    return p


def _progress(label):
    def cb(p):
        sys.stderr.write("\r%s %3d%%" % (label, int(p * 100)))
        if p >= 1.0:
            sys.stderr.write("\n")
        sys.stderr.flush()
    return cb


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    params = CinematicParams(denoise_db=args.denoise, warmth_db=args.warmth,
                             presence_db=args.presence, air_db=args.air,
                             dereverb_db=args.dereverb)
    out = args.output or os.path.splitext(args.input)[0] + "_cinematic.mp4"
    gain = enhance(args.input, out, params, target=args.target,
                   progress=_progress("processing"))
    sys.stderr.write("wrote %s (%+.1f dB trim)\n" % (out, gain))
    return 0


if __name__ == "__main__":
    sys.exit(main())
