import os
import shutil
import subprocess
import wave

import numpy as np
import pytest

from face_reframe.cut import main
from face_reframe.silence import SAMPLE_RATE, clip_intervals

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")


def make_talk(path):
    """12s clip: tone bursts at 1-3s, 5-7s, 9-11s over black video."""
    sr, dur = SAMPLE_RATE, 12.0
    pcm = np.zeros(int(dur * sr), np.int16)
    for a, d in [(1.0, 2.0), (5.0, 2.0), (9.0, 2.0)]:
        t = np.arange(int(d * sr)) / sr
        pcm[int(a * sr):int(a * sr) + len(t)] = (0.5 * np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)
    wav_path = os.path.splitext(path)[0] + ".wav"
    with wave.open(wav_path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i",
                    "color=c=black:s=640x360:d=%s:r=25" % int(dur), "-i", wav_path,
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", path], check=True)


def test_clip_intervals_maps_frames(tmp_path):
    src = str(tmp_path / "talk.mp4")
    make_talk(src)
    spans = clip_intervals(src, 0, 300, 25.0)
    starts = [round(a / 25.0, 1) for a, _ in spans]
    assert starts == pytest.approx([0.9, 4.9, 8.9], abs=0.2)


def test_cli_jump_cut(tmp_path):
    src, out = str(tmp_path / "talk.mp4"), str(tmp_path / "cut.mp4")
    make_talk(src)
    assert main([src, "-o", out]) == 0
    probe = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                            "-of", "csv=p=0", out], capture_output=True, text=True)
    assert 5.0 < float(probe.stdout) < 8.0  # ~6s of speech kept from 12s
