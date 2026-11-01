"""Two similar people in a wide shot: the camera must lock onto one."""

import numpy as np

from face_reframe.detect import Analysis, Face, FrameSample
from face_reframe.track import TrackSettings, build_camera_path


def test_wide_two_people_no_flipping():
    rng = np.random.default_rng(0)
    samples = []
    for f in range(0, 200, 2):
        # left person slightly bigger on average, but sizes jitter so the
        # per-frame "biggest" alternates between them
        a = Face(600 + rng.normal(0, 4), 400, 60 * (1 + rng.normal(0, 0.08)), 70, 0.9)
        b = Face(1350 + rng.normal(0, 4), 410, 57 * (1 + rng.normal(0, 0.08)), 66, 0.9)
        faces = [a, b] if f % 6 else [b]  # left one sometimes missed
        samples.append(FrameSample(f, faces))
    an = Analysis(1920, 1080, 25.0, 0, 200, samples)
    path = build_camera_path(an, TrackSettings(smoothing=0.3))
    assert np.abs(path.x - 600).max() < 40   # stays on the left person


def _two_people(talk_left):
    """talk_left(frame) -> True when the left person is the one talking."""
    samples = []
    for f in range(0, 300, 2):
        left = talk_left(f)
        a = Face(600, 400, 60, 70, 0.9, mouth=4.0 if left else 0.3)
        b = Face(1350, 410, 57, 66, 0.9, mouth=0.3 if left else 4.0)
        samples.append(FrameSample(f, [a, b]))
    return Analysis(1920, 1080, 25.0, 0, 300, samples)


def test_speaker_mode_cuts_to_whoever_talks():
    an = _two_people(lambda f: f < 150)
    path = build_camera_path(an, TrackSettings(subject="speaker", smoothing=0.3))
    assert abs(path.x[60] - 600) < 40
    assert abs(path.x[260] - 1350) < 40
    assert len(path.segments) == 2             # one hard cut at the switch
    assert 130 <= path.segments[1][0] <= 175


def test_speaker_mode_ignores_short_interjections():
    # right person says "sì" for ~0.3 s: no cut
    an = _two_people(lambda f: not (140 <= f < 148))
    path = build_camera_path(an, TrackSettings(subject="speaker"))
    assert len(path.segments) == 1


def test_main_mode_unchanged_by_mouths():
    an = _two_people(lambda f: f < 150)
    path = build_camera_path(an, TrackSettings(subject="main"))
    assert len(path.segments) == 1


def _sig(main_bin):
    v = np.full(96, 0.002, np.float32)
    v[main_bin] = 1.0
    v[48 + main_bin] = 1.0
    return v / v.sum() * 2


def test_find_people_across_shots_and_follow_choice():
    from face_reframe.track import find_people
    red, blue = _sig(1), _sig(30)
    samples = []
    for f in range(0, 300, 2):
        if f < 100:      # close-up on blue
            faces = [Face(960, 400, 300, 340, 0.9, sig=blue)]
        else:            # wide shot: red (a bit bigger) on the left, blue on the right
            faces = [Face(500, 420, 70, 80, 0.9, sig=red), Face(1400, 420, 60, 70, 0.9, sig=blue)]
        samples.append(FrameSample(f, faces, cut=(f == 100)))
    an = Analysis(1920, 1080, 25.0, 0, 300, samples)
    people = find_people(an)
    assert len(people) == 2
    blue_p = next(p for p in people if np.argmax(p.sig[:48]) == 30)
    assert len(blue_p.tracks) == 2           # recognised in both shots
    auto = build_camera_path(an, TrackSettings())
    assert abs(auto.x[200] - 500) < 40       # default: the bigger face (red)
    picked = build_camera_path(an, TrackSettings(person=blue))
    assert abs(picked.x[200] - 1400) < 40    # chosen person in the wide shot


def test_follow_accepts_several_thumbnails():
    """Same person split in two thumbnails (close-up vs wide not matched):
    picking both follows them in both shots."""
    red, blue, blue_wide = _sig(1), _sig(30), _sig(40)
    samples = []
    for f in range(0, 300, 2):
        if f < 100:
            faces = [Face(700, 400, 300, 340, 0.9, sig=blue)]
        else:
            faces = [Face(500, 420, 70, 80, 0.9, sig=red), Face(1400, 420, 60, 70, 0.9, sig=blue_wide)]
        samples.append(FrameSample(f, faces, cut=(f == 100)))
    an = Analysis(1920, 1080, 25.0, 0, 300, samples)
    path = build_camera_path(an, TrackSettings(person=[blue, blue_wide]))
    assert abs(path.x[50] - 700) < 40
    assert abs(path.x[200] - 1400) < 40


def test_embeddings_merge_close_up_and_wide_despite_colours():
    """Different light in the wide shot breaks the colour match; the face
    embedding still says it's the same person."""
    from face_reframe.track import find_people
    e_blue = np.zeros(128, np.float32); e_blue[0] = 1
    e_red = np.zeros(128, np.float32); e_red[1] = 1
    samples = []
    for f in range(0, 300, 2):
        if f < 100:
            faces = [Face(700, 400, 300, 340, 0.9, sig=_sig(30), emb=e_blue)]
        else:
            faces = [Face(500, 420, 70, 80, 0.9, sig=_sig(1), emb=e_red),
                     Face(1400, 420, 60, 70, 0.9, sig=_sig(5), emb=e_blue)]
        samples.append(FrameSample(f, faces, cut=(f == 100)))
    people = find_people(Analysis(1920, 1080, 25.0, 0, 300, samples))
    assert len(people) == 2
    assert sorted(len(p.tracks) for p in people) == [1, 2]


def test_real_sface_same_person_two_scales():
    import cv2, pytest
    from face_reframe.detect import FaceDetector, FaceEmbedder, identity, identity_distance
    emb = FaceEmbedder()
    if not emb.available:
        pytest.skip("SFace model missing")
    skimage = pytest.importorskip("skimage.data")
    img = cv2.cvtColor(skimage.astronaut(), cv2.COLOR_RGB2BGR)
    det = FaceDetector()
    big = det.detect(img)[0]
    small_img = cv2.resize(img, None, fx=0.35, fy=0.35)
    small = det.detect(small_img)[0]
    a, b = emb(img, big), emb(small_img, small)
    assert a is not None and b is not None
    assert identity_distance(identity(None, a), identity(None, b)) < 0.38
