"""Overlays, social texts, colour looks, stabilisation, cover, HW encode args."""

import os
import subprocess

import numpy as np
import pytest

from face_reframe.app import look, overlays, social
from face_reframe.app.jobs import JobSettings, TimeMap, source_frames, video_args
from face_reframe.transcribe import Word


def _w(text, t0=0.0, step=0.4):
    return [Word(t0 + i * step, t0 + i * step + 0.3, tok) for i, tok in enumerate(text.split())]


def test_first_sentence_and_social_texts():
    words = _w("Perché nessuno ti dice questo trucco? La cucina italiana è semplice. Basta poco olio.")
    assert overlays.first_sentence(words) == "Perché nessuno ti dice questo trucco?"
    t = social.make_texts(words, lang="it")
    assert t["title"].startswith("Perché") and "#shorts" in t["hashtags"] and "#cucina" in t["hashtags"]
    assert "Seguimi" in t["description"]
    assert "TITLE" in social.to_text(t)


def test_overlays_hook_logo_cta(tmp_path):
    import cv2
    logo = np.zeros((40, 80, 4), np.uint8)
    logo[..., 2] = 255
    logo[..., 3] = 255
    lp = str(tmp_path / "logo.png")
    cv2.imwrite(lp, logo)
    o = overlays.OverlaySettings(hook="auto", hook_dur=1.0, cta="Seguimi!", cta_dur=1.0, logo_path=lp)
    parts = overlays.build(o, "Il trucco che nessuno ti dice", 100, 25.0, 360, 640, None)
    chain = overlays.Chain(parts)
    blank = np.zeros((640, 360, 3), np.uint8)
    first, mid, last = chain(5, blank.copy()), chain(50, blank.copy()), chain(95, blank.copy())
    top = slice(0, 220)
    assert first[top].any() and not mid[top][:, :200].any()        # hook only at the start
    assert mid[..., 2].max() > 200                                 # logo always (red, 85% opacity)
    centre = slice(250, 390)
    assert last[centre].any() and not mid[centre].any()            # CTA only at the end


def test_overlays_progress_and_cards(tmp_path):
    import cv2
    card = np.full((100, 100, 3), (0, 0, 255), np.uint8)   # red card
    cp = str(tmp_path / "card.png")
    cv2.imwrite(cp, card)
    o = overlays.OverlaySettings(progress="bottom", card_in_path=cp, card_in_dur=1.0,
                                 card_out_path=cp, card_out_dur=1.0)
    parts = overlays.build(o, "", 100, 25.0, 360, 640, None)
    chain = overlays.Chain(parts)
    blank = np.zeros((640, 360, 3), np.uint8)
    # card windows: intro [0,25), outro [75,100); past the 8-frame fades
    first, mid, last = chain(12, blank.copy()), chain(50, blank.copy()), chain(88, blank.copy())
    assert first[..., 2].mean() > 200          # intro card covers the frame early on
    assert mid[..., 2].mean() < 20             # ...but not in the middle
    assert last[..., 2].mean() > 200           # outro card covers the end
    # edge frames hold the card at full opacity: feeds stills show it properly
    assert chain(0, blank.copy())[..., 2].mean() > 200
    assert chain(99, blank.copy())[..., 2].mean() > 200
    # progress bar: the accent-coloured bar on the bottom edge grows with time
    early = chain(30, blank.copy())[636:, :, 2].sum()
    late = chain(60, blank.copy())[636:, :, 2].sum()
    assert early > 0 and late > early


def test_caption_animations():
    import cv2
    from face_reframe.app.jobs import CaptionBurner
    from face_reframe.transcribe import Word

    def renderer(words, active, out_w, emojis=()):  # fake: solid white 40x8 px
        return np.full((8, 40, 4), (255, 255, 255, 255), np.uint8)

    fps, W, H = 25.0, 360, 640
    blank = np.zeros((H, W, 3), np.uint8)

    words = [Word(1.0, 1.4, "ciao"), Word(1.4, 1.8, "mondo"), Word(1.8, 2.4, "bello")]
    ev_fade = [(1.0, 2.4, words)]
    b = CaptionBurner(ev_fade, fps, W, H, renderer, 0.5, "word", anim="fade")
    at_start = b(int(1.0 * fps), blank.copy())     # t=1.0: alpha ~0 -> nothing drawn
    mid = b(int(1.7 * fps), blank.copy())          # inside the event: fully drawn
    assert not at_start.any() and mid.any()

    words = [Word(1.0, 1.5, "ciao"), Word(1.5, 2.5, "mondo"), Word(2.5, 3.0, "bello")]
    ev_pop = [(1.0, 3.0, words)]
    b = CaptionBurner(ev_pop, fps, W, H, renderer, 0.5, "word", anim="pop")
    f_pop = b(int(1.56 * fps), blank.copy())       # just after word 2 starts: scaled up
    f_settled = b(int(2.3 * fps), blank.copy())    # 0.8 s into the word: settled
    assert f_pop.any() and f_settled.any()
    # the pulsed caption covers more pixels than the settled one
    assert (f_pop > 0).sum() > (f_settled > 0).sum()

    b = CaptionBurner(ev_pop, fps, W, H, renderer, 0.5, "line", anim="slide")
    early = b(int(1.06 * fps), blank.copy())       # slides up from below its slot
    late = b(int(1.6 * fps), blank.copy())
    ys_e = np.nonzero(early.any(axis=2).any(axis=1))[0]
    ys_l = np.nonzero(late.any(axis=2).any(axis=1))[0]
    assert len(ys_e) and len(ys_l) and ys_e.mean() > ys_l.mean()   # earlier = lower


def test_fill_background_modes(tmp_path):
    import cv2
    from face_reframe.app import rvm

    class FakeMatter:                      # person mask = left half
        def reset(self):
            pass

        def __call__(self, img):
            m = np.zeros(img.shape[:2], np.float32)
            m[:, :img.shape[1] // 2] = 1.0
            return m

    frame = np.zeros((100, 200, 3), np.uint8)

    fill = rvm.ColorFill("#ff0000", 200, 100)                 # green-screen style colour
    fb = rvm.FillBackground(FakeMatter(), fill, refine=False)
    out = fb(0, frame)
    assert out[:, :100, 2].mean() < 20       # person side: the frame shows through
    assert out[:, 150:, 2].mean() > 200      # backdrop side: solid red

    img = np.full((50, 400, 3), (0, 255, 0), np.uint8)         # wide image -> cover-cropped
    ip = str(tmp_path / "bg.png")
    cv2.imwrite(ip, img)
    fb = rvm.FillBackground(FakeMatter(), rvm.ImageFill(ip, 200, 100), refine=False)
    out = fb(0, frame)
    assert out[:, 150:, 1].mean() > 200 and out[:, 150:, 2].mean() < 30

    vw = cv2.VideoWriter(str(tmp_path / "bg.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 25, (64, 64))
    for k in range(6):
        vw.write(np.full((64, 64, 3), (0, 0, 40 * k), np.uint8))
    vw.release()
    vf = rvm.VideoFill(str(tmp_path / "bg.mp4"), 200, 100, 25.0)
    a, b = vf(0, frame), vf(5, frame)                         # sequential reads advance
    assert a.shape == (100, 200, 3) and not np.array_equal(a, b)
    vf(2, frame)                                              # backward jump -> seek, no crash

    vf = rvm.VideoFill(str(tmp_path / "bg.mp4"), 200, 100, 25.0)
    vf.rate = 0.5                                             # bg slower than out: repeats
    class SpyCap:
        def __init__(self, cap):
            self._cap, self.seeks = cap, 0
        def set(self, prop, val):
            self.seeks += 1
            return self._cap.set(prop, val)
        def __getattr__(self, name):
            return getattr(self._cap, name)
    vf.cap = SpyCap(vf.cap)
    for i in range(6):
        vf(i, frame)
    assert vf.cap.seeks == 0    # repeated indices reuse the frame — no seeks

    assert rvm.make_fill("blur", 200, 100) is None            # blur mode -> no fill provider


def test_cut_detector_locate():
    from face_reframe.detect import CutDetector
    a = np.full((1080, 1920, 3), (60, 90, 120), np.uint8)
    b = np.full((1080, 1920, 3), (160, 40, 40), np.uint8)
    det = CutDetector()
    det(a)                                    # state at the last sampled frame
    seed = det.state()
    assert det(b)                             # the stride-spaced fire
    assert det.locate([b], *seed) == 0        # boundary is the first gap frame
    det2 = CutDetector()
    det2(a)
    seed2 = det2.state()
    det2(b)                                   # fire again at the next sample
    assert det2.locate([a, a, b], *seed2) == 2     # boundary pinned mid-gap
    det3 = CutDetector()
    det3(a)
    s3 = det3.state()
    assert not det3(a) and det3.locate([a, a], *s3) is None   # no cut -> None


def test_look_tables():
    rng = np.random.default_rng(0)
    img = rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)
    assert look.apply(img, look.Look("none")) is img
    bw = look.apply(img, look.Look("bw"))
    assert np.abs(bw[..., 0].astype(int) - bw[..., 2]).max() <= 2  # grey
    warm = look.apply(img, look.Look("warm")).astype(int)
    assert warm[..., 2].mean() > img[..., 2].mean() and warm[..., 0].mean() < img[..., 0].mean()
    half = look.apply(img, look.Look("pop", 0.5)).astype(int)
    full = look.apply(img, look.Look("pop", 1.0)).astype(int)
    assert np.abs(half - img).mean() < np.abs(full - img).mean()
    wb = look.apply(img, look.Look("none", 1.0, (1.2, 1.0, 0.8))).astype(int)
    assert wb[..., 2].mean() > img[..., 2].mean()


def test_auto_emoji_and_beats():
    from face_reframe.app.emoji import for_words
    from face_reframe.app import motion
    assert for_words(["I", "love", "this", "secret"]) == ["❤️", "🤫"]
    assert for_words(["soldi", "gratis"]) == ["💰", "🆓"]
    assert for_words(["hello", "there"]) == []
    assert len(for_words(["money", "fire", "love", "top"])) == 2  # cap at 2
    sr = 16000
    pcm = np.zeros(8 * sr, np.float32)
    for b in np.arange(0.5, 8, 0.5):       # clicks on a strict 0.5 s grid
        i = int(b * sr)
        pcm[i:i + 200] += np.sin(2 * np.pi * 120 * np.arange(200) / sr) * 0.8
    bs = motion.beats(pcm, sr)
    assert len(bs) >= 10
    assert all(min(abs(b - t) for t in np.arange(0.5, 8, 0.5)) < 0.12 for b in bs)
    # beat-sync: triggers snap to the nearest detected beat, unmatched drop
    snapped = motion._snap_to_beat([0.45, 1.1, 1.52], bs, tol=0.15)
    assert snapped == [bs[0], bs[2]], snapped   # 1.1 s has no beat within tol
    flat = np.zeros(4 * sr, np.float32)
    assert motion.beats(flat, sr) == []


def test_cube_lut(tmp_path):
    p = tmp_path / "id.cube"
    rows = ["LUT_3D_SIZE 2"] + ["%d %d %d" % (r, g, b)
                                for b in range(2) for g in range(2) for r in range(2)]
    p.write_text("\n".join(rows))
    lut = look.parse_cube(str(p))
    assert lut is not None and lut[0].shape == (2, 2, 2, 3)
    x = np.array([[0.2, 0.5, 0.8]], np.float32)
    assert np.allclose(look._cube_apply(x, lut), x, atol=0.02)
    p.write_text("\n".join(rows[:1] + ["%d %d %d" % (1 - r, 1 - g, 1 - b)
                                       for b in range(2) for g in range(2) for r in range(2)]))
    lut = look.load_cube(str(p))
    assert np.allclose(look._cube_apply(x, lut), 1 - x, atol=0.02)
    lk = look.Look("cube", 1.0, None, None, lut)
    assert lk.active and np.allclose(look.transform(x, "cube", None, 1.0, None, lut), 1 - x, atol=0.02)
    assert not look.Look("cube", 1.0).active
    assert look.parse_cube(str(tmp_path / "missing.cube")) is None
    (tmp_path / "bad.cube").write_text("TITLE hi\nLUT_1D_SIZE 8\n")
    assert look.parse_cube(str(tmp_path / "bad.cube")) is None


def test_grade_controls():
    rng = np.random.default_rng(0)
    img = rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)
    # neutral grade is a no-op: the frame is returned untouched
    assert look.apply(img, look.Look("none", 1.0, None, look.Grade())) is img
    # a grade alone (no preset) activates the Look and shifts pixels
    g = look.Grade(temp=0.6, exposure=0.4, contrast=1.2, saturation=1.3)
    out = look.apply(img, look.Look("none", 1.0, None, g)).astype(int)
    assert not np.array_equal(out, img.astype(int))
    assert out[..., 2].mean() > img[..., 2].mean()   # warm temp lifts red
    # vignette darkens edges more than the centre
    v = look.apply(img, look.Look("none", 1.0, None, look.Grade(vignette=0.8))).astype(int)
    assert v[0, 0].mean() < v[32, 32].mean()
    # vignette without any LUT ops still runs (spatial-only grade)
    assert not np.array_equal(v, img.astype(int))


def test_timemap_inverse_and_source_frames():
    tm = TimeMap([(0.0, 1.0), (2.0, 3.0)])
    assert tm.inverse(0.5) == pytest.approx(0.5) and tm.inverse(1.5) == pytest.approx(2.5)
    f = source_frames(50, 25.0, [(0.0, 1.0), (2.0, 3.0)], 10)
    assert f[0] == 10 and f[30] == 10 + 2.2 * 25


def test_stabilization_cancels_jitter(tmp_path):
    import cv2
    from face_reframe.app.stabilize import estimate
    rng = np.random.default_rng(1)
    base = (rng.random((400, 700, 3)) * 255).astype(np.uint8)
    base = cv2.GaussianBlur(base, (0, 0), 2)
    path = str(tmp_path / "shaky.mp4")
    out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 25, (640, 360))
    jit = []
    for i in range(80):
        dx, dy = int(rng.integers(-8, 9)), int(rng.integers(-6, 7))
        jit.append((dx, dy))
        out.write(base[20 + dy:380 + dy, 30 + dx:670 + dx])
    out.release()
    st = estimate(path, smooth_s=1.0)
    assert st is not None and len(st.dx) == 80
    # each frame is a window shifted by +dx into a still scene: to keep looking at
    # the same scene, the crop must move by -dx (minus the smooth part)
    jx = np.array([j[0] for j in jit], float)
    jx -= np.convolve(np.pad(jx, 12, mode="edge"), np.ones(25) / 25, mode="valid")
    corr = np.corrcoef(st.dx, jx)[0, 1]
    assert corr < -0.8, corr


def test_stabilization_rotation_sign(tmp_path):
    import cv2
    from face_reframe.app.stabilize import estimate
    rng = np.random.default_rng(2)
    base = cv2.GaussianBlur((rng.random((360, 640, 3)) * 255).astype(np.uint8), (0, 0), 2)
    path = str(tmp_path / "rot.mp4")
    out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 25, (640, 360))
    ang = rng.uniform(-2, 2, 60)
    for a in ang:
        out.write(cv2.warpAffine(base, cv2.getRotationMatrix2D((320, 180), float(a), 1.0), (640, 360),
                                 borderMode=cv2.BORDER_REFLECT))
    out.release()
    st = estimate(path, smooth_s=1.0)
    hp = ang - np.convolve(np.pad(ang, 12, mode="edge"), np.ones(25) / 25, mode="valid")
    # output rotation (getRotationMatrix2D convention) must undo the frame's
    assert np.corrcoef(st.rot, hp)[0, 1] < -0.8


def test_video_args_software_and_hardware(monkeypatch):
    from face_reframe.app import jobs
    s = JobSettings(encoder="software", crf=20)
    assert video_args(s, 1080, 1920)[:2] == ["-c:v", "libx264"]
    monkeypatch.setattr(jobs, "hw_encoder_available", lambda: True)
    a = video_args(JobSettings(encoder="auto", crf=18), 1080, 1920)
    assert a[1] == "h264_videotoolbox" and a[3] == "12.0M"
