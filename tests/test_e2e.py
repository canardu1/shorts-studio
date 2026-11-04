import json
import os
import shutil

import cv2
import pytest
from make_sample import make, make_wide

from face_reframe.cli import main
from face_reframe.detect import FaceDetector

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")


def test_cli_keeps_face_centred(tmp_path):
    src, out, track = str(tmp_path / "in.mp4"), str(tmp_path / "out.mp4"), str(tmp_path / "t.json")
    make(src)
    assert main([src, "-o", out, "--json", track]) == 0
    with open(track) as fh:
        data = json.load(fh)
    # cut detection runs on sampled frames, so the boundary may land on the
    # next sample (75 -> 76 at stride 2); the camera still cuts hard there
    assert len(data["cuts"]) == 1 and abs(data["cuts"][0] - 75) <= 1
    assert len(data["frames"]) == 150

    cap = cv2.VideoCapture(out)
    assert (cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) == (1080, 1920)
    det = FaceDetector()
    offsets = []
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % 10 == 0:
            faces = det.detect(frame)
            assert faces, "no face in output frame %d" % i
            offsets.append(abs(max(faces, key=lambda f: f.area).cx - 540) / 1080)
        i += 1
    assert i == 150
    assert max(offsets) < 0.15, offsets
    assert os.path.getsize(out) > 0


def test_cli_zooms_wide_shot(tmp_path):
    src, out, track = str(tmp_path / "in.mp4"), str(tmp_path / "out.mp4"), str(tmp_path / "t.json")
    make_wide(src)
    assert main([src, "-o", out, "--json", track]) == 0
    with open(track) as fh:
        data = json.load(fh)
    # cut detection runs on sampled frames, so the boundary may land on the
    # next sample (75 -> 76 at stride 2); the camera still cuts hard there
    assert len(data["cuts"]) == 1 and abs(data["cuts"][0] - 75) <= 1
    kinds = [s["kind"] for s in data["shots"]]
    assert kinds == ["close", "wide"], data["shots"]
    assert data["shots"][0]["zoom"] == 1.0 and data["shots"][1]["zoom"] > 1.5

    cap = cv2.VideoCapture(out)
    det = FaceDetector()
    for i in range(150):
        ok, frame = cap.read()
        assert ok
        if i in (40, 120):
            face = max(det.detect(frame), key=lambda f: f.area)
            assert abs(face.cx - 540) / 1080 < 0.1
            if i == 120:
                # source face is ~4% of the height; the 2x punch-in should roughly double it
                assert face.h / 1920 > 0.075
