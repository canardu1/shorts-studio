"""Face restoration: alignment, feathered paste-back, job integration.

Needs a downloaded model (set SHORTS_STUDIO_MODELS to a folder holding
gpen_bfr_256.onnx); skipped otherwise."""

import numpy as np
import pytest

from face_reframe import restore


def _have():
    try:
        import onnxruntime  # noqa: F401
    except ImportError:
        return False
    return restore.is_available("gpen256")


pytestmark = pytest.mark.skipif(not _have(), reason="GPEN-256 model / onnxruntime not available")


def _degraded():
    import cv2
    from skimage import data
    img = cv2.cvtColor(data.astronaut(), cv2.COLOR_RGB2BGR)
    small = cv2.resize(img, None, fx=1 / 3, fy=1 / 3, interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (1024, 1024), interpolation=cv2.INTER_CUBIC)


def test_restores_only_the_face_without_seams():
    deg = _degraded()
    r = restore.FaceRestorer("gpen256", blend=1.0)
    for _ in range(8):              # the restoration fades in over ~7 frames
        out = r(deg)
    d = np.abs(out.astype(int) - deg.astype(int)).sum(axis=2)
    ys, xs = np.nonzero(d > 6)
    assert len(xs), "the face should change"
    # the change stays around the face (astronaut's face is ~x 330-560, y 90-350)
    assert 250 < xs.min() and xs.max() < 700 and ys.max() < 450
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    edge = np.concatenate([d[y0:y1, x0:x0 + 3].ravel(), d[y0:y1, x1 - 3:x1].ravel()])
    centre = d[(y0 + y1) // 2 - 20:(y0 + y1) // 2 + 20, (x0 + x1) // 2 - 20:(x0 + x1) // 2 + 20]
    assert edge.max() < 30 and centre.mean() > edge.mean()


def test_blend_zero_like_and_no_face_untouched():
    deg = _degraded()
    blank = np.full((640, 360, 3), 90, np.uint8)
    r = restore.FaceRestorer("gpen256", blend=0.8)
    assert np.array_equal(r(blank), blank)
    rw = restore.FaceRestorer("gpen256", blend=0.1)
    rs = restore.FaceRestorer("gpen256", blend=1.0)
    weak = [rw(deg) for _ in range(8)][-1]      # fade-in ramp, like the render loop
    strong = [rs(deg) for _ in range(8)][-1]
    assert np.abs(weak.astype(int) - deg).sum() < np.abs(strong.astype(int) - deg).sum()


def test_estimate():
    assert restore.estimate_seconds("codeformer", 1500) > restore.estimate_seconds("gpen256", 1500) * 20
