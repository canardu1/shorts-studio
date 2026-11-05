"""Face beauty: skin smoothing, blemish removal, wrinkle and under-eye softening.

Works on the finished output frame, one face at a time, inside a soft skin
mask built from the YuNet landmarks (face ellipse minus eyes, brows and
mouth) intersected with the face's own skin colour (so beard, hair, glasses
and background are left alone):

* smoothing  — edge-preserving (bilateral) low-pass blended in, with a little
  fine grain added back so skin keeps a natural texture (no "plastic");
* blemishes  — small dark/red spots that stand out from a median-filtered
  version of the skin are replaced by it, with feathered edges;
* wrinkles   — dark mid-frequency detail (fine lines, creases) is lifted
  rather than blurred, plus a gentle brighten under each eye.

Cost: ~10-25 ms per face on the output frame (only the face box is touched).
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from .i18n import tr


@dataclass
class BeautySettings:
    smooth: float = 0.5
    blemish: float = 0.5
    wrinkles: float = 0.4

    @property
    def active(self) -> bool:
        return self.smooth > 0 or self.blemish > 0 or self.wrinkles > 0


# ------------------------------------------------------------------ parsing
PARSE_FILE = "bisenet_resnet_34.onnx"


def parse_model_path() -> str:
    from .restore import BASE_URL, models_dir
    return os.path.join(models_dir(), PARSE_FILE)


def parse_available() -> bool:
    p = parse_model_path()
    return os.path.exists(p) and os.path.getsize(p) > 50_000_000


def download_parse(progress=None) -> str:
    path = parse_model_path()
    if parse_available():
        return path
    from .restore import BASE_URL, fetch
    return fetch(BASE_URL + PARSE_FILE, path, progress)


_PSESS = None


class FaceParser:
    """Precise skin mask from a BiSeNet face-parsing net (CelebAMask-HQ).

    Tightens the landmark ellipse: jaw, ears, hairline, glasses and beard are
    classified properly instead of approximated. Weights are the shared
    bisenet_resnet_34 export (89 MB), downloaded once.
    """

    SKIN = (1, 2, 17)          # skin, nose, neck — not eyes/brows/lips/hair

    def __call__(self, roi: np.ndarray) -> np.ndarray:
        """(h, w) float32 mask 0..1 of skin-only regions in ``roi``."""
        import cv2
        global _PSESS
        if _PSESS is None:
            import onnxruntime as ort
            so = ort.SessionOptions()
            so.log_severity_level = 3
            from . import models
            _PSESS = ort.InferenceSession(parse_model_path(), so,
                                          providers=models.providers())
        h, w = roi.shape[:2]
        x = cv2.resize(roi, (512, 512), interpolation=cv2.INTER_AREA)
        x = x[..., ::-1].astype(np.float32) / 255.0
        x = ((x - (0.485, 0.456, 0.406)) / (0.229, 0.224, 0.225)).astype(np.float32)
        out = _PSESS.run(["output"], {"input": x.transpose(2, 0, 1)[None]})[0]
        lab = out.argmax(1)[0]
        m = np.isin(lab, self.SKIN).astype(np.float32)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        return cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR)


def build(s: BeautySettings, log=print, progress=None) -> "Beautifier":
    """Beautifier, with the parsing mask when the model is (or gets) there."""
    parser = None
    if s.active:
        try:
            if not parse_available():
                log(tr("Beauty: downloading the face-parsing model (89 MB, once)…"))
                download_parse(progress)
            parser = FaceParser()
        except Exception as exc:
            log(tr("  face parsing unavailable (%s): landmark mask") % exc)
    return Beautifier(s, parser=parser)


def _odd(n: float) -> int:
    n = max(3, int(round(n)))
    return n if n % 2 else n + 1


class Beautifier:
    """Callable ``frame -> frame``; keeps light landmark smoothing across frames."""

    def __init__(self, s: BeautySettings, max_faces: int = 2, min_face: int = 60, parser=None):
        from .detect import FaceDetector
        self.s = s
        self.max_faces, self.min_face = max_faces, min_face
        self.detector = FaceDetector(min_score=0.6)
        self.parser = parser
        self._prev: List[np.ndarray] = []

    def _smooth_lm(self, lm: np.ndarray) -> np.ndarray:
        c = lm.mean(axis=0)
        for p in self._prev:
            if np.linalg.norm(p.mean(axis=0) - c) < 0.25 * np.linalg.norm(lm[0] - lm[1]) + 4:
                return 0.5 * lm + 0.5 * p
        return lm

    def __call__(self, frame: np.ndarray) -> np.ndarray:
        if not self.s.active:
            return frame
        faces = [f for f in self.detector.detect(frame) if f.landmarks is not None and f.h >= self.min_face]
        faces.sort(key=lambda f: -f.area)
        out, cur = frame, []
        for f in faces[:self.max_faces]:
            lm = self._smooth_lm(np.array(f.landmarks, np.float32).reshape(5, 2))
            cur.append(lm)
            if out is frame:
                out = frame.copy()
            self._face(out, f, lm)
        self._prev = cur
        return out

    # ------------------------------------------------------------ one face
    def _face(self, img: np.ndarray, f, lm: np.ndarray) -> None:
        import cv2
        H, W = img.shape[:2]
        (rex, rey), (lex, ley), (nx, ny), (rmx, rmy), (lmx, lmy) = lm
        ed = math.hypot(lex - rex, ley - rey)
        if ed < 12:
            return
        x0, x1 = int(max(0, f.cx - 0.75 * f.w)), int(min(W, f.cx + 0.75 * f.w))
        y0, y1 = int(max(0, f.cy - 0.85 * f.h)), int(min(H, f.cy + 0.75 * f.h))
        if x1 - x0 < 16 or y1 - y0 < 16:
            return
        roi = img[y0:y1, x0:x1]
        h, w = roi.shape[:2]
        ox, oy = x0, y0
        P = lambda x, y: (int(round(x - ox)), int(round(y - oy)))  # noqa: E731

        # --- region masks (float 0..1)
        face = np.zeros((h, w), np.float32)
        cv2.ellipse(face, P(f.cx, f.cy + 0.05 * f.h), (int(0.52 * f.w), int(0.66 * f.h)), 0, 0, 360, 1.0, -1)
        excl = np.zeros((h, w), np.float32)
        for ex, ey in ((rex, rey), (lex, ley)):
            cv2.ellipse(excl, P(ex, ey), (int(0.30 * ed), int(0.17 * ed)), 0, 0, 360, 1.0, -1)   # eye
            cv2.ellipse(excl, P(ex, ey - 0.33 * ed), (int(0.38 * ed), int(0.12 * ed)), 0, 0, 360, 1.0, -1)  # brow
        mw = math.hypot(lmx - rmx, lmy - rmy)
        cv2.ellipse(excl, P((rmx + lmx) / 2, (rmy + lmy) / 2 + 0.02 * ed),
                    (int(0.62 * mw), int(0.30 * mw + 0.08 * ed)), 0, 0, 360, 1.0, -1)              # lips
        cv2.circle(excl, P(nx, ny + 0.05 * ed), int(0.16 * ed), 0.6, -1)                            # nostrils
        feather = max(1.0, 0.06 * ed)
        face = cv2.GaussianBlur(face, (0, 0), feather * 1.5)
        excl = cv2.GaussianBlur(excl, (0, 0), feather)
        # skin colour likelihood around this face's own skin (Lab a/b)
        lab = cv2.cvtColor(roi.astype(np.float32) / 255.0, cv2.COLOR_BGR2LAB)
        from .skin import _patches
        samp = []
        for cx, cy, r in _patches(f):
            a, b = P(cx, cy)
            r = max(2, int(r))
            patch = lab[max(0, b - r):b + r, max(0, a - r):a + r].reshape(-1, 3)
            if len(patch):
                samp.append(patch)
        if not samp:
            return
        samp = np.concatenate(samp)
        ma, mb = float(np.median(samp[:, 1])), float(np.median(samp[:, 2]))
        sd = max(6.0, 2.5 * float(np.median(np.hypot(samp[:, 1] - ma, samp[:, 2] - mb))))
        d2 = ((lab[..., 1] - ma) ** 2 + (lab[..., 2] - mb) ** 2) / (sd * sd)
        skinlike = (np.exp(-0.5 * d2) * np.clip((lab[..., 0] - 12) / 10, 0, 1)).astype(np.float32)
        # close small holes: a blemish is *not* skin-coloured, yet it is on the skin
        kc = _odd(0.22 * ed)
        skinlike = cv2.morphologyEx(skinlike, cv2.MORPH_CLOSE,
                                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kc, kc)))
        skinlike = cv2.GaussianBlur(skinlike, (0, 0), feather)
        mask = face * (1.0 - excl) * skinlike
        if self.parser is not None:
            try:
                mask = mask * self.parser(roi)
            except Exception:
                self.parser = None          # broken model: landmark mask only
        mask = np.clip(mask, 0, 1)[..., None]

        src = roi.astype(np.float32)
        res = src.copy()
        s = self.s

        # --- blemishes: small spots that differ from a median-filtered skin
        if s.blemish > 0:
            # median wider than a spot (spots are ~5-10% of the eye distance)
            k = min(_odd(0.22 * ed), 45)
            med = cv2.medianBlur(roi, k).astype(np.float32)
            Lr = lab[..., 0]
            Lm = cv2.cvtColor(med / 255.0, cv2.COLOR_BGR2LAB)
            dark = Lm[..., 0] - Lr                       # darker than surroundings
            red = lab[..., 1] - Lm[..., 1]               # redder than surroundings
            # only outliers: thresholds adapt to this face's normal skin texture
            # (pores, shading), so the skin isn't flattened and lightened overall
            inside = mask[..., 0] > 0.5
            if inside.sum() < 50:
                return
            def thr(v):
                m = np.median(v[inside])
                return m + max(5.0, 3.0 * 1.4826 * np.median(np.abs(v[inside] - m)))
            td, tr = thr(dark), thr(red)
            spot = np.clip(np.clip((dark - td) / 5.0, 0, 1) + np.clip((red - tr) / 5.0, 0, 1), 0, 1)
            # grow a little before feathering, or a faint ring of the spot's edge stays
            kd = _odd(0.05 * ed)
            spot = cv2.dilate(spot.astype(np.float32), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kd, kd)))
            spot = cv2.GaussianBlur(spot, (0, 0), max(1.0, 0.02 * ed))
            sm = (spot * s.blemish)[..., None] * mask
            res = res * (1 - sm) + med * sm

        # --- wrinkles: lift dark mid-frequency detail; brighten under the eyes
        if s.wrinkles > 0:
            g1 = cv2.GaussianBlur(res, (0, 0), max(0.8, 0.012 * ed))
            g2 = cv2.GaussianBlur(res, (0, 0), max(2.0, 0.07 * ed))
            band = g1 - g2
            # only clearly dark detail (lines, creases), not the overall pore
            # texture: lifting that would just wash the skin out
            lift = np.clip(-band - 2.0, 0, None) * (0.9 * s.wrinkles)
            res = res + lift * mask
            under = np.zeros((h, w), np.float32)
            for ex, ey in ((rex, rey), (lex, ley)):
                cv2.ellipse(under, P(ex, ey + 0.30 * ed), (int(0.34 * ed), int(0.15 * ed)), 0, 0, 360, 1.0, -1)
            under = cv2.GaussianBlur(under, (0, 0), 0.08 * ed)[..., None] * mask
            res = res + under * (10.0 * s.wrinkles)

        # --- smoothing: bilateral low-pass + a bit of fine grain back
        if s.smooth > 0:
            sigma_s = max(2.0, 0.035 * ed)
            low = cv2.bilateralFilter(np.clip(res, 0, 255).astype(np.uint8), 0, 18 + 22 * s.smooth, sigma_s)
            low = low.astype(np.float32)
            grain = res - cv2.GaussianBlur(res, (0, 0), 0.8)
            sm = mask * (0.85 * s.smooth)
            res = res * (1 - sm) + (low + 0.35 * grain) * sm

        roi[:] = np.clip(res, 0, 255).astype(np.uint8)
