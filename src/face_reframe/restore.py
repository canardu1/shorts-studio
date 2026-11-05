"""Face restoration (GPEN / GFPGAN / CodeFormer) on the finished output frame.

Each face is found with YuNet, aligned to the FFHQ 5-point template, run
through the ONNX model, and pasted back with a feathered mask — only the
face changes, so the cost is one model run per face per frame.

Measured on an Apple M1 (CPU; CoreML is slower for these graphs):
GPEN-256 ~0.15 s, GPEN-512 ~1.8 s, GFPGAN 1.4 ~2.1 s, CodeFormer ~5 s per face.
Models are downloaded on first use (75-380 MB) into :func:`models_dir`.
Licences: GFPGAN is Apache-2.0; CodeFormer (S-Lab) and GPEN are for
non-commercial use only.
"""

from __future__ import annotations

import os
import sys
import time
import urllib.error
import urllib.request

from .i18n import tr
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import numpy as np

BASE_URL = "https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/"


@dataclass(frozen=True)
class RestoreModel:
    key: str
    file: str
    size: int
    seconds: float
    """Typical seconds per face on an M1 CPU (for time estimates)."""
    mb: int
    weight: bool = False
    """Takes CodeFormer's fidelity input."""


MODELS: Dict[str, RestoreModel] = {
    "gpen256": RestoreModel("gpen256", "gpen_bfr_256.onnx", 256, 0.15, 76),
    "gpen512": RestoreModel("gpen512", "gpen_bfr_512.onnx", 512, 1.8, 284),
    "gfpgan": RestoreModel("gfpgan", "gfpgan_1.4.onnx", 512, 2.1, 340),
    "codeformer": RestoreModel("codeformer", "codeformer.onnx", 512, 5.0, 377, weight=True),
}

# FFHQ alignment template (eyes, nose, mouth corners), normalised to [0, 1];
# same point order as YuNet: subject's right eye (image left) first
TEMPLATE = np.array([[0.37691676, 0.46864664], [0.62285697, 0.46912813], [0.50123859, 0.61331904],
                     [0.39308822, 0.72541100], [0.61150205, 0.72490465]], np.float32)


def models_dir() -> str:
    d = os.environ.get("SHORTS_STUDIO_MODELS")
    if not d:
        if sys.platform == "darwin":
            d = os.path.expanduser("~/Library/Application Support/ShortsStudio/models")
        elif os.name == "nt":
            d = os.path.join(os.environ.get("APPDATA", os.path.expanduser("~")), "ShortsStudio", "models")
        else:
            d = os.path.join(os.environ.get("XDG_DATA_HOME", os.path.expanduser("~/.local/share")),
                             "shorts_studio", "models")
    os.makedirs(d, exist_ok=True)
    return d


def model_path(key: str) -> str:
    return os.path.join(models_dir(), MODELS[key].file)


def is_available(key: str) -> bool:
    p = model_path(key)
    return os.path.exists(p) and os.path.getsize(p) > 1_000_000


def fetch(url: str, path: str, progress: Optional[Callable[[float], None]] = None) -> str:
    """Download ``url`` to ``path`` (via a .part file). Python builds without
    CA certificates (python.org on macOS before "Install Certificates") fail
    TLS: then certifi, then the system curl (Keychain certificates) are used."""
    import ssl
    tmp = path + ".part"

    def via_urllib(ctx=None):
        with urllib.request.urlopen(url, timeout=60, context=ctx) as r, open(tmp, "wb") as fh:
            total = int(r.headers.get("Content-Length") or 0)
            done = 0
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                fh.write(chunk)
                done += len(chunk)
                if progress and total:
                    progress(done / total)

    try:
        via_urllib()
    except (urllib.error.URLError, ssl.SSLError) as exc:
        if "CERTIFICATE" not in str(exc).upper():
            raise
        try:
            import certifi
            via_urllib(ssl.create_default_context(cafile=certifi.where()))
        except ImportError:
            import shutil
            import subprocess
            curl = shutil.which("curl")
            if not curl:
                raise
            subprocess.run([curl, "-fsSL", "--retry", "2", "-o", tmp, url], check=True, timeout=600)
            if progress:
                progress(1.0)
    os.replace(tmp, path)
    return path


def download(key: str, progress: Optional[Callable[[float], None]] = None) -> str:
    """Fetch the model once."""
    path = model_path(key)
    if is_available(key):
        return path
    return fetch(BASE_URL + MODELS[key].file, path, progress)


def estimate_seconds(key: str, frames: int, faces: float = 1.0) -> float:
    return MODELS[key].seconds * frames * faces


class FaceRestorer:
    """Callable ``frame -> frame`` restoring up to ``max_faces`` faces.

    ``blend`` mixes the restored face over the original (1 = fully restored);
    ``fidelity`` is CodeFormer's w (higher = closer to the input identity).
    Landmarks are smoothed frame to frame to keep the restored texture steady.
    """

    def __init__(self, key: str = "gpen256", fidelity: float = 0.7, blend: float = 0.8,
                 max_faces: int = 2, min_face: int = 48, tone: float = 0.6, log=print):
        import onnxruntime as ort
        from .detect import FaceDetector
        self.model = MODELS[key]
        if not is_available(key):
            raise FileNotFoundError(tr("model %s not downloaded") % self.model.file)
        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        from . import models
        self.sess = ort.InferenceSession(model_path(key), opts,
                                         providers=models.providers())
        self.inp = self.sess.get_inputs()[0].name
        self.fidelity, self.blend = float(fidelity), float(blend)
        self.max_faces, self.min_face = max_faces, min_face
        self.tone = float(tone)
        self.detector = FaceDetector(min_score=0.6)
        self._prev: List[dict] = []   # {lm, w}: per-face state for the fade
        s = self.model.size
        mask = np.zeros((s, s), np.float32)
        # feather fully inside the square: the edge must reach exactly 0 or a
        # faint box outline shows where the face was pasted
        pad = int(0.15 * s)
        mask[pad:s - pad, pad:s - pad] = 1.0
        import cv2
        self.mask = cv2.GaussianBlur(mask, (0, 0), 0.05 * s)
        self.template = TEMPLATE * s

    def _run(self, crop_bgr: np.ndarray) -> np.ndarray:
        x = crop_bgr[:, :, ::-1].astype(np.float32) / 127.5 - 1.0
        feed = {self.inp: x.transpose(2, 0, 1)[None]}
        if self.model.weight:
            feed["weight"] = np.array([self.fidelity], np.float64)
        y = self.sess.run(None, feed)[0][0]
        y = (np.clip(y, -1.0, 1.0) + 1.0) * 127.5
        return y.transpose(1, 2, 0)[:, :, ::-1].astype(np.uint8)

    def _gate(self, lm: np.ndarray, W: int) -> bool:
        """Don't restore a face the camera never really captured — tiny or
        profile ones get detail *invented* (the classic "glasses nobody wore").

        Landmark order is [eye, eye, nose, mouth, mouth]: eyes closer than
        ~12 px on a 640-wide frame is a smear, and the nose far off-centre
        between the eyes means a profile view."""
        eye = float(np.linalg.norm(lm[0] - lm[1]))
        if eye < 0.019 * W:
            return False
        n = lm[2]
        d0, d1 = abs(lm[0][0] - n[0]), abs(lm[1][0] - n[0])
        return min(d0, d1) >= 0.28 * max(d0, d1)

    def __call__(self, frame: np.ndarray) -> np.ndarray:
        import cv2
        faces = [f for f in self.detector.detect(frame)
                 if f.landmarks is not None and f.h >= self.min_face]
        faces.sort(key=lambda f: -f.area)
        out = frame
        cur = []
        H, W = frame.shape[:2]
        s = self.model.size
        for f in faces[:self.max_faces]:
            raw = np.array(f.landmarks, np.float32).reshape(5, 2)
            c = raw.mean(axis=0)
            prev = None
            for p in self._prev:
                if np.linalg.norm(p["lm"].mean(axis=0) - c) < 0.25 * np.linalg.norm(raw[0] - raw[1]) + 4:
                    prev = p
                    break
            lm = raw if prev is None else 0.5 * raw + 0.5 * prev["lm"]
            M, _ = cv2.estimateAffinePartial2D(lm, self.template, method=cv2.LMEDS)
            if M is None:
                cur.append({"lm": lm, "w": prev["w"] if prev is not None else 0.0})
                continue
            crop = cv2.warpAffine(frame, M, (s, s), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            # NB: no blur gate — restoring soft faces is the model's job here;
            # size + yaw are what stop it from inventing features
            ok = self._gate(lm, W)
            # fade the restoration in/out instead of switching — a face that
            # appears for a moment (or turns away) shouldn't pop
            w = (prev["w"] if prev is not None else 0.0) + (0.14 if ok else -0.18)
            w = min(self.blend, max(0.0, w))
            cur.append({"lm": lm, "w": w})
            if w < 0.04:
                continue                    # faded out: skip the network run
            fixed = self._run(crop)
            if self.tone > 0:
                # keep the source's broad tones: the model only adds detail
                sig = max(1.0, s * 0.03)
                diff = (cv2.GaussianBlur(crop, (0, 0), sig).astype(np.float32)
                        - cv2.GaussianBlur(fixed, (0, 0), sig).astype(np.float32))
                fixed = np.clip(fixed.astype(np.float32) + diff * self.tone, 0, 255).astype(np.uint8)
            IM = cv2.invertAffineTransform(M)
            # paste back only inside the face's bounding box (much cheaper)
            corners = np.array([[0, 0, 1], [s, 0, 1], [0, s, 1], [s, s, 1]], np.float32) @ IM.T
            x0, y0 = np.floor(corners.min(axis=0)).astype(int)
            x1, y1 = np.ceil(corners.max(axis=0)).astype(int)
            x0, y0, x1, y1 = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
            if x1 - x0 < 4 or y1 - y0 < 4:
                continue
            IMr = IM.copy()
            IMr[:, 2] -= (x0, y0)
            size = (x1 - x0, y1 - y0)
            paste = cv2.warpAffine(fixed, IMr, size, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            m = cv2.warpAffine(self.mask, IMr, size, flags=cv2.INTER_LINEAR)[..., None] * w
            if out is frame:
                out = frame.copy()
            roi = out[y0:y1, x0:x1].astype(np.float32)
            out[y0:y1, x0:x1] = (roi * (1.0 - m) + paste.astype(np.float32) * m).astype(np.uint8)
        self._prev = cur
        return out

    def timing(self) -> float:
        """Seconds per face measured on this machine (one warm run)."""
        s = self.model.size
        dummy = np.full((s, s, 3), 128, np.uint8)
        self._run(dummy)
        t = time.time()
        self._run(dummy)
        return time.time() - t
