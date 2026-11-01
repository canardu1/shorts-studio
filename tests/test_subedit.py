"""Editing the caption transcript before burning it in."""

import pytest

from face_reframe.app import subedit
from face_reframe.transcribe import Word


def _words():
    toks = "oggi parliamo di fischi per fiaschi con il dottor rossi".split()
    return [Word(0.5 * i, 0.5 * i + 0.4, t) for i, t in enumerate(toks)]


def test_lines_group_like_the_burner():
    ls = subedit.lines(_words(), 3, 1.6)
    assert [l[2] for l in ls][:2] == ["oggi parliamo di", "fischi per fiaschi"]
    assert ls[1][3] == 3 and ls[1][4] == 3


def test_edit_line_retimes_inside_the_same_span():
    w = _words()
    ls = subedit.lines(w, 3, 1.6)
    a, b, _, first, n = ls[1]
    new = subedit.edit_line(w, first, n, "fiaschi per fiaschi davvero")
    grp = new[first:first + 4]
    assert [x.text for x in grp] == ["fiaschi", "per", "fiaschi", "davvero"]
    assert grp[0].start == pytest.approx(a) and grp[-1].end <= b + 1e-6
    assert all(x.start < y.start for x, y in zip(grp, grp[1:]))
    assert len(new) == len(w) + 1


def test_empty_text_deletes_the_line():
    w = _words()
    first, n = subedit.lines(w, 3, 1.6)[0][3:]
    assert [x.text for x in subedit.edit_line(w, first, n, "")][:1] == ["fischi"]


def test_replace_all_whole_word_case_insensitive():
    w = _words() + [Word(6.0, 6.4, "Rossi.")]
    new, n = subedit.replace_all(w, "rossi", "Russo", 3, 1.6)
    texts = [x.text for x in new]
    assert "Russo" in texts and "Russo." in texts and n == 2
    new2, n2 = subedit.replace_all(w, "fisc", "x", 3, 1.6, whole_word=True)
    assert n2 == 0
