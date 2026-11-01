"""Tests for the shorts batch: Auto Markers, Cut Retakes, Sync Audio, Duck Music."""

import numpy as np
import pytest

from face_reframe.duck import build_voice_key, slice_key
from face_reframe.markers import keyword_hits, topic_marks
from face_reframe.retakes import retake_spans, utterances
from face_reframe.silence import SAMPLE_RATE
from face_reframe.syncaudio import find_offset
from face_reframe.transcribe import Word


def _w(t, text, dur=0.4):
    return Word(start=t, end=t + dur, text=text)


def _phrase(words, t0, text, step=0.4):
    for i, tok in enumerate(text.split()):
        words.append(_w(t0 + i * step, tok))


# ---------- markers ----------------------------------------------------

def test_keyword_hits_normalizes():
    words = [_w(1.0, "ok"), _w(2.0, "TAGLIA!"), _w(3.0, "rifacciamo")]
    hits = keyword_hits(words, ["taglia", "rifacciamo"])
    assert [k for _, k in hits] == ["taglia", "rifacciamo"]
    assert hits[0][0] == pytest.approx(2.0)


def test_keyword_hits_empty_keys():
    assert keyword_hits([_w(0, "taglia")], []) == []


def test_topic_marks_split_on_long_pause():
    words = []
    _phrase(words, 0.0, "primo blocco di discorso qui")
    _phrase(words, 10.0, "secondo argomento del video")
    marks = topic_marks(words, min_pause=4.0, title_words=3)
    assert len(marks) == 2
    assert marks[0][0] == pytest.approx(0.0)
    assert marks[1][0] == pytest.approx(10.0)
    assert marks[1][1] == "secondo argomento del"


def test_topic_marks_single_block_no_pause():
    words = []
    _phrase(words, 0.0, "tutto attaccato senza pause")
    assert len(topic_marks(words, min_pause=4.0)) == 1


# ---------- retakes ----------------------------------------------------

def test_retakes_drops_earlier_take():
    words = []
    _phrase(words, 0.0, "ciao oggi parliamo di montaggio video")
    _phrase(words, 8.0, "ok")                                   # short filler rides along
    _phrase(words, 10.0, "ciao oggi parliamo di montaggio video")  # keeper
    _phrase(words, 20.0, "e poi continuiamo con altro")
    holes = retake_spans(words)
    assert holes == [pytest.approx((0.0, 10.0), abs=0.5)]


def test_retakes_keeps_unrelated_content():
    words = []
    _phrase(words, 0.0, "primo argomento completamente diverso")
    _phrase(words, 10.0, "secondo argomento su tutt'altro tema")
    _phrase(words, 20.0, "terza parte senza ripetizioni")
    assert retake_spans(words) == []


def test_retakes_ignores_short_blocks():
    words = []
    _phrase(words, 0.0, "ciao")
    _phrase(words, 3.0, "ciao")
    assert retake_spans(words) == []              # under min_words -> never cut


def test_utterances_groups_on_gap():
    words = []
    _phrase(words, 0.0, "uno due", step=0.4)
    _phrase(words, 5.0, "tre quattro", step=0.4)
    blocks = utterances(words, gap=1.2)
    assert len(blocks) == 2
    assert blocks[0][2] == ["uno", "due"]


# ---------- sync -------------------------------------------------------

def _noise(seed, dur_s):
    return (0.1 * np.random.default_rng(seed).standard_normal(int(dur_s * SAMPLE_RATE))
            ).astype(np.float32)


def test_find_offset_positive():
    ext = _noise(3, 60)
    ref = ext[int(23.5 * SAMPLE_RATE):int(33.5 * SAMPLE_RATE)] + _noise(4, 10) * 0.05
    lag, score = find_offset(ref, ext)
    assert lag == pytest.approx(23.5, abs=0.05)
    assert score > 0.05


def test_find_offset_negative():
    ext = _noise(3, 60)
    ref = np.concatenate([_noise(7, 5), ext[:10 * SAMPLE_RATE]])
    lag, score = find_offset(ref, ext)
    assert lag == pytest.approx(-5.0, abs=0.5)
    assert score > 0.03


def test_find_offset_no_match_scores_low():
    ext = _noise(3, 60)
    _lag, score = find_offset(_noise(9, 10), ext)
    assert score < 0.03


# ---------- duck -------------------------------------------------------

def test_slice_key_windows_and_pads():
    key = np.arange(4 * SAMPLE_RATE, dtype=np.float32)
    out = slice_key(key, 1.0, 2.0)
    assert len(out) == 2 * SAMPLE_RATE
    assert np.array_equal(out, key[SAMPLE_RATE:3 * SAMPLE_RATE])
    padded = slice_key(key, -1.0, 2.0)              # off-axis start -> zeros first
    assert padded[0] == 0.0
    assert padded[-1] == key[SAMPLE_RATE - 1] or padded[-1] == 0.0


def test_build_voice_key_places_segments(tmp_path):
    import wave, os
    path = str(tmp_path / "src.wav")
    tone = np.ones(2 * SAMPLE_RATE, np.float32) * 0.5
    with wave.open(path, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(SAMPLE_RATE)
        w.writeframes((tone * 32767).astype(np.int16).tobytes())
    key = build_voice_key([(path, 0.5, 1.0, 3.0)], 6.0, log=lambda *_: None)
    # source 0.5s-1.5s placed at timeline 3s-4s
    assert key[3 * SAMPLE_RATE] != 0
    assert key[SAMPLE_RATE] == 0
    assert abs(key[3 * SAMPLE_RATE + 100]) > 0.1
