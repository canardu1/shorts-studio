import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from face_reframe import transcribe
from face_reframe.cut_bridge import _subtract


def w(a, b, t):
    return transcribe.Word(a, b, t)


def test_filler_spans_flags_isolated_fillers():
    words = [w(0.0, 0.5, "Hello"), w(1.0, 1.4, "ehm,"), w(1.5, 2.0, "there"),
             w(3.0, 3.3, "uhm"), w(4.0, 6.0, "ehmmmmmmmmmmmmm")]
    spans = transcribe.filler_spans(words, pad=0.05)
    assert len(spans) == 2  # the 2s filler exceeds max_dur
    assert abs(spans[0][0] - 0.95) < 1e-6 and abs(spans[0][1] - 1.45) < 1e-6


def test_is_filler_punctuation_and_case():
    assert transcribe.is_filler("Ehm...")
    assert transcribe.is_filler("UH,")
    assert not transcribe.is_filler("scheme")


def test_group_captions_packs_words():
    words = [w(0.0, 0.4, "One"), w(0.4, 0.8, "two"), w(0.8, 1.2, "three"),
             w(1.2, 1.6, "four"), w(3.0, 3.4, "five")]
    ev = transcribe.group_captions(words, max_words=3, max_dur=1.6)
    assert ev == [(0.0, 1.2, "One two three"), (1.2, 1.6, "four"), (3.0, 3.4, "five")]


def test_to_srt_format():
    srt = transcribe.to_srt([(0.0, 1.5, "ciao mondo")])
    assert "1\n00:00:00,000 --> 00:00:01,500\nciao mondo" in srt


def test_subtract_splits_spans():
    assert _subtract([(0, 100)], [(40, 60)]) == [(0, 40), (60, 100)]
    assert _subtract([(0, 100)], [(0, 50)]) == [(50, 100)]
    assert _subtract([(0, 100)], [(200, 300)]) == [(0, 100)]
    assert _subtract([(0, 10)], [(0, 20)]) == []


def test_group_captions_long_single_word_no_hang():
    """A word longer than max_dur must still emit an event (was an infinite loop)."""
    from face_reframe.transcribe import Word, group_captions
    words = [Word(0.0, 0.5, "hello"), Word(0.5, 5.0, "stretchy"), Word(5.0, 5.3, "world")]
    events = group_captions(words)
    assert len(events) == 3
    assert events[1][2] == "stretchy"
