
import numpy as np

from face_reframe.silence import SAMPLE_RATE, SpeechParams, speech_intervals


def synth_pcm(segments, sr=SAMPLE_RATE):
    """segments: list of (start_s, dur_s) tone bursts; returns float32 pcm + total s."""
    end = max(a + d for a, d in segments) + 1.0
    pcm = np.random.RandomState(0).normal(0, 0.001, int(end * sr)).astype(np.float32)
    for a, d in segments:
        t = np.arange(int(d * sr)) / sr
        pcm[int(a * sr):int(a * sr) + len(t)] += 0.3 * np.sin(2 * np.pi * 440 * t)
    return pcm


def test_finds_bursts_and_keeps_short_pauses():
    pcm = synth_pcm([(1.0, 2.0), (3.2, 2.0), (6.0, 2.0)])
    spans = speech_intervals(pcm, SpeechParams(min_pause=0.35))
    # the 0.2s pause at 3.0s is kept -> first two bursts merge
    assert len(spans) == 2
    assert spans[0][0] < 1.1 and spans[0][1] > 5.0
    assert spans[1][0] < 6.1 and spans[1][1] > 7.9


def test_long_pauses_are_cut():
    pcm = synth_pcm([(0.5, 1.0), (5.0, 1.0)])
    spans = speech_intervals(pcm, SpeechParams(min_pause=0.35))
    assert len(spans) == 2


def test_all_silence_returns_nothing():
    pcm = np.zeros(SAMPLE_RATE * 5, np.float32)
    assert speech_intervals(pcm) == []


def test_tone_only_returns_whole_clip():
    t = np.arange(SAMPLE_RATE * 5) / SAMPLE_RATE
    pcm = 0.3 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
    spans = speech_intervals(pcm)
    assert len(spans) == 1 and spans[0][1] > 4.9


def test_hysteresis_keeps_soft_word_tails():
    from face_reframe.silence import SAMPLE_RATE
    win = int(0.025 * SAMPLE_RATE)
    profile = np.concatenate([np.full(100, -40.0), np.full(40, -20.0),
                              np.full(20, -35.0), np.full(40, -40.0)])
    pcm = np.concatenate([np.full(win, 10 ** (db / 20)) for db in profile]).astype(np.float32)
    spans = speech_intervals(pcm, SpeechParams(margin_db=8.0, release_db=4.0, padding=0.16))
    assert spans and spans[0][1] > 4.0  # tail at -35 dB kept, cut after it
