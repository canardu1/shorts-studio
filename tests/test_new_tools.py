"""Tests for punch-in, breaths, loudness, split-at-cuts and color normalize."""

import os

import numpy as np
import pytest

from face_reframe import silence
from face_reframe.punch import PunchParams, curve_keys, punch_curve
from face_reframe.loudness import gain_db
from face_reframe.color import measure_gains, write_lut
from face_reframe.split_bridge import SplitOptions, shot_boundaries


def _pcm(db_profile, sr=silence.SAMPLE_RATE):
    """Build PCM from a per-100ms dB profile (sine tone at that level)."""
    pcm = np.zeros(int(len(db_profile) * 0.1 * sr), dtype=np.float32)
    for i, db in enumerate(db_profile):
        t = np.arange(int(0.1 * sr)) / sr
        amp = 10 ** (db / 20.0)
        pcm[i * len(t):(i + 1) * len(t)] = amp * np.sin(2 * np.pi * 300 * t)
    return pcm


def test_punch_curve_alternates():
    curve = punch_curve(int(8 * 25), 25.0, PunchParams(interval=2.0, zoom=1.2, ease=0.2))
    assert curve[0] == pytest.approx(1.0)
    assert curve[3 * 25] == pytest.approx(1.2, abs=1e-3)   # zoomed during 2-4s
    assert curve[5 * 25] == pytest.approx(1.0, abs=1e-3)   # back out during 4-6s
    assert curve[7 * 25] == pytest.approx(1.2, abs=1e-3)
    keys = curve_keys(curve)
    assert keys[0][0] == 0 and keys[-1][0] == len(curve) - 1
    assert all(1.0 - 1e-6 <= v <= 1.2 + 1e-6 for _, v in keys)


def test_breath_intervals_finds_inhales():
    # floor noise -45dB; speech bursts at -20dB; a -40dB "breath" between them
    # (above floor+2dB, below the speech-entry threshold floor+8dB)
    profile = ([-45.0] * 10 + [-20.0] * 15 + [-45.0] * 3 +
               [-40.0] * 3 + [-45.0] * 3 + [-20.0] * 15 + [-45.0] * 10)
    pcm = _pcm(profile)
    breaths = silence.breath_intervals(pcm)
    assert len(breaths) == 1
    assert breaths[0][0] == pytest.approx(2.8, abs=0.1)
    assert breaths[0][1] == pytest.approx(3.1, abs=0.1)


def test_breath_intervals_ignores_speech_and_silence():
    profile = [-45.0] * 10 + [-20.0] * 20 + [-45.0] * 10
    assert silence.breath_intervals(_pcm(profile)) == []


def test_gain_db():
    assert gain_db(-20.0) == pytest.approx(6.0)
    assert gain_db(-8.0) == pytest.approx(-6.0)
    assert gain_db(-60.0) == 20.0   # clamped


def test_write_lut(tmp_path):
    lut = write_lut((1.2, 1.0, 0.8), str(tmp_path / "x.cube"))
    text = open(lut).read()
    assert "LUT_3D_SIZE 17" in text
    lines = [l for l in text.splitlines() if l and l[0].isdigit() or l.startswith("0.")]
    assert len(lines) == 17 ** 3
    # last grid point (1,1,1) scaled by gains
    r, g, b = lines[-1].split()
    assert float(r) == pytest.approx(1.2)
    assert float(b) == pytest.approx(0.8)


def test_measure_gains_and_split(tmp_path):
    import cv2
    video = str(tmp_path / "v.mp4")
    out = cv2.VideoWriter(video, cv2.VideoWriter_fourcc(*"mp4v"), 25, (160, 90))
    for i in range(60):
        color = (30, 30, 200) if i < 30 else (200, 30, 30)  # hard cut at 30
        frame = np.full((90, 160, 3), color, np.uint8)
        out.write(frame)
    out.release()
    ranges = shot_boundaries(video, 0, 60, SplitOptions(min_shot=0.1))
    assert len(ranges) == 2
    assert ranges[0][0] == 0 and abs(ranges[1][0] - 30) <= 1
    r, g, b = measure_gains(video, 0, 30)
    assert r < 1.0 < b  # reddish clip gains red down, blue up
