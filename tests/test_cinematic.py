"""Tests for the Cinematic Audio tool (clean + sweeten chain)."""

import os
import shutil
import subprocess
import wave

import numpy as np
import pytest

from face_reframe.cinematic import (CinematicParams, afftdn_gs_supported,
                                    audio_chain, combine_loudness, enhance,
                                    noise_floor_db, speech_level_db)
from face_reframe.loudness import TARGET_I, measure_integrated
from face_reframe.silence import SAMPLE_RATE


def _pcm(db_profile, sr=SAMPLE_RATE):
    """Build PCM from a per-100ms dB profile (sine tone at that level)."""
    pcm = np.zeros(int(len(db_profile) * 0.1 * sr), dtype=np.float32)
    for i, db in enumerate(db_profile):
        t = np.arange(int(0.1 * sr)) / sr
        amp = 10 ** (db / 20.0)
        pcm[i * len(t):(i + 1) * len(t)] = amp * np.sin(2 * np.pi * 300 * t)
    return pcm


def test_noise_floor_db():
    # sine at peak -45dB has RMS -48.01dB (peak/sqrt(2)); bursts don't move the floor
    profile = [-45.0] * 10 + [-20.0] * 15 + [-45.0] * 10
    floor = noise_floor_db(_pcm(profile))
    assert floor == pytest.approx(-48.0, abs=1.5)


def test_audio_chain_has_all_stages():
    chain = audio_chain(CinematicParams(), -52.0)
    for stage in ("highpass", "afftdn", "deesser", "sidechaingate",
                  "acompressor", "bass", "treble"):
        assert stage in chain
    assert "nf=-52.0" in chain
    assert "nr=12.0" in chain


def test_audio_chain_adapts_to_noisy_floor():
    chain = audio_chain(CinematicParams(), -35.0)
    assert "nf=-35.0" in chain
    assert "nr=13.8" in chain  # noisier clip -> deeper reduction, capped vs artifacts
    if afftdn_gs_supported():
        assert ":gs=10" in chain   # gain smoothing = no metallic residue
    else:
        assert ":gs=" not in chain  # ffmpeg < 5: probed and dropped, not fatal


def test_combine_loudness():
    # energy-weighted: identical parts return themselves; the loud part
    # dominates a mix, scaled by how long each part runs
    assert combine_loudness([(-20.0, 1.0), (-20.0, 3.0)]) == pytest.approx(-20.0)
    mix = combine_loudness([(-20.0, 9.0), (-40.0, 1.0)])
    assert -21.0 < mix < -20.0
    assert combine_loudness([]) == -70.0


def test_audio_chain_stages_can_be_disabled():
    p = CinematicParams(denoise_db=0.0, deess=0.0, dereverb_db=0.0, comp_ratio=1.0)
    chain = audio_chain(p, -50.0)
    assert "afftdn" not in chain
    assert "deesser" not in chain
    assert "sidechaingate" not in chain
    assert "acompressor" not in chain
    assert "bass" in chain  # tonal polish stays


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")
def test_dereverb_stage_widens_speech_tail_gap(tmp_path):
    """The gap between speech and the reverb tail widens after the chain."""
    import wave
    sr, out = SAMPLE_RATE, str(tmp_path / "rv.wav")
    rng = np.random.default_rng(1)
    # real studio clips always have a noise bed — digital silence would make
    # the measured floor unrealistically low and the denoiser timid
    pcm = 0.01 * rng.standard_normal(int(4 * sr))                   # -40dB bed
    t = np.arange(int(sr)) / sr
    pcm[:len(t)] += 0.4 * np.sin(2 * np.pi * 300 * t)               # speech 1s
    tl = np.arange(int(1.8 * sr)) / sr
    pcm[int(sr):int(sr) + len(tl)] += 0.06 * np.exp(-3 * tl) * rng.standard_normal(len(tl))
    with wave.open(out, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes((pcm * 32767).astype(np.int16).tobytes())
    # floor + speech level are measured on the clip, exactly like the bridge
    # does — on a reverb-y recording the "noise floor" IS the tail itself
    chain = audio_chain(CinematicParams(), noise_floor_db(pcm), speech_level_db(pcm))
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", out, "-af", chain,
                    "-ar", "16000", "-ac", "1", "-f", "f32le",
                    str(tmp_path / "rv.raw")], check=True)
    done = np.fromfile(str(tmp_path / "rv.raw"), dtype=np.float32)

    def level(sig, a, b):
        return 20 * np.log10(np.sqrt(np.mean(sig[int(a * sr):int(b * sr)] ** 2)) + 1e-12)

    gap_in = level(pcm, 0.2, 0.9) - level(pcm, 1.4, 2.4)
    gap_out = level(done, 0.2, 0.9) - level(done, 1.4, 2.4)
    assert gap_out - gap_in > 3.0                                     # tail ducked vs speech


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")
def test_enhance_e2e(tmp_path):
    """12s clip: tone bursts over a -40dB noise bed -> lands near -14 LUFS."""
    sr, dur = SAMPLE_RATE, 12.0
    rng = np.random.default_rng(0)
    pcm = (0.01 * rng.standard_normal(int(dur * sr))).astype(np.float64)
    for a, d in [(1.0, 2.0), (5.0, 2.0), (9.0, 2.0)]:
        t = np.arange(int(d * sr)) / sr
        pcm[int(a * sr):int(a * sr) + len(t)] += 0.4 * np.sin(2 * np.pi * 440 * t)
    wav_path = str(tmp_path / "talk.wav")
    with wave.open(wav_path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes((pcm * 32767).astype(np.int16).tobytes())
    src = str(tmp_path / "talk.mp4")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i",
                    "color=c=black:s=640x360:d=12:r=25", "-i", wav_path,
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", src], check=True)
    out = str(tmp_path / "talk_cine.mp4")
    enhance(src, out)
    assert os.path.exists(out)
    loud = measure_integrated(out)
    assert loud == pytest.approx(TARGET_I, abs=2.0)
