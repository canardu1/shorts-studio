"""Face detection over a video's frames.

Uses OpenCV's YuNet DNN detector (bundled ONNX model) and falls back to a
Haar cascade if the DNN module is unavailable.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable, Iterator, List, Optional, Sequence, Tuple

import cv2
import numpy as np

MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "face_detection_yunet_2023mar.onnx")
SFACE_PATH = os.path.join(os.path.dirname(__file__), "models", "face_recognition_sface_2021dec_int8.onnx")
SIG_LEN = 96
"""Length of the colour part of an identity vector (see :func:`identity`)."""
DETECT_WIDTH = 640
RETRY_WIDTH = 1280


@dataclass
class Face:
    cx: float
    cy: float
    w: float
    h: float
    score: float
    landmarks: Optional[Tuple[float, ...]] = None
    """YuNet's 5 points as (x, y) pairs: right eye, left eye, nose, right and
    left mouth corner — in source pixels. ``None`` with the Haar fallback."""
    mouth: Optional[float] = None
    """Mouth-region motion vs. the previous sample (speaker detection);
    ``None`` when it couldn't be measured."""
    sig: Optional[np.ndarray] = None
    """Appearance signature (colour histograms of face/hair and clothes),
    used to recognise the same person across shots."""
    emb: Optional[np.ndarray] = None
    """SFace identity embedding (128-d, L2-normalised) when the face
    recognition model is available; computed on a subset of samples."""

    @property
    def area(self) -> float:
        return self.w * self.h


@dataclass
class FrameSample:
    frame: int
    faces: List[Face]
    cut: bool = False
    salient: Optional[Tuple[float, float, float, float]] = None
    """(cx, cy, w, h) of the most salient object when no face was found —
    filled by ``analyze_video(saliency=...)`` (U2-Net-p) or by the object
    tracker; used by the "follow the object" framing mode."""


@dataclass
class Analysis:
    width: int
    height: int
    fps: float
    start_frame: int
    frame_count: int
    samples: List[FrameSample]
    salient_pass: bool = False
    """True when saliency was already offered on the faceless samples (by
    ``analyze_video(saliency=...)`` or :func:`saliency.backfill`) — keeps a
    cached analysis from being re-scanned for nothing."""


class FaceDetector:
    def __init__(self, min_score: float = 0.6):
        self.min_score = min_score
        self._yunet = None
        self._haar = None
        self._yunet_size = None
        if hasattr(cv2, "FaceDetectorYN") and os.path.exists(MODEL_PATH):
            self._yunet = cv2.FaceDetectorYN.create(MODEL_PATH, "", (320, 320), min_score, 0.3, 5000)
        else:
            self._haar = cv2.CascadeClassifier(
                os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
            )

    def detect(self, image: np.ndarray) -> List[Face]:
        faces = self._detect(image, DETECT_WIDTH)
        if not faces and image.shape[1] > DETECT_WIDTH:
            faces = self._detect(image, RETRY_WIDTH)
        return faces

    def _detect(self, image: np.ndarray, detect_width: int) -> List[Face]:
        h, w = image.shape[:2]
        scale = min(1.0, detect_width / float(w))
        small = cv2.resize(image, (int(round(w * scale)), int(round(h * scale)))) if scale < 1.0 else image
        sh, sw = small.shape[:2]
        faces: List[Face] = []
        if self._yunet is not None:
            if self._yunet_size != (sw, sh):
                self._yunet.setInputSize((sw, sh))
                self._yunet_size = (sw, sh)
            _, dets = self._yunet.detect(small)
            if dets is not None:
                for d in dets:
                    x, y, fw, fh, score = float(d[0]), float(d[1]), float(d[2]), float(d[3]), float(d[-1])
                    if score >= self.min_score:
                        lm = tuple(float(v) / scale for v in d[4:14]) if len(d) >= 15 else None
                        faces.append(Face((x + fw / 2) / scale, (y + fh / 2) / scale, fw / scale, fh / scale,
                                          score, lm))
        else:
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            for (x, y, fw, fh) in self._haar.detectMultiScale(gray, 1.1, 5, minSize=(24, 24)):
                faces.append(Face((x + fw / 2) / scale, (y + fh / 2) / scale, fw / scale, fh / scale, 1.0))
        return faces


class CutDetector:
    """Flags hard cuts from colour-histogram changes or sudden structural changes.

    The structural cue catches cuts inside the same set (close-up -> wide shot),
    where the colours barely change but the picture layout does.
    """

    def __init__(self, hist_threshold: float = 0.45, change_threshold: float = 0.06, change_ratio: float = 4.0):
        self.hist_threshold = hist_threshold
        self.change_threshold = change_threshold
        self.change_ratio = change_ratio
        self._hist = None
        self._gray = None
        self._diffs: List[float] = []

    def _feats(self, image: np.ndarray):
        small = cv2.resize(image, (160, 90), interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [16, 8], [0, 180, 0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        gray = cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), (5, 5), 0).astype(np.float32)
        return hist, gray

    def __call__(self, image: np.ndarray) -> bool:
        hist, gray = self._feats(image)
        cut = False
        if self._hist is not None:
            hist_diff = cv2.compareHist(self._hist, hist, cv2.HISTCMP_BHATTACHARYYA)
            changed = float(np.mean(np.abs(gray - self._gray) > 20))
            typical = float(np.median(self._diffs)) if self._diffs else 0.0
            cut = hist_diff > self.hist_threshold or (
                changed > self.change_threshold and changed > self.change_ratio * max(typical, 0.01)
            )
            self._diffs = [] if cut else (self._diffs + [changed])[-25:]
        self._hist, self._gray = hist, gray
        return cut

    def state(self):
        """Feature state of the last frame fed — the seed for ``locate``."""
        return self._hist, self._gray

    def locate(self, frames, hist, gray) -> Optional[int]:
        """Replay consecutive ``frames`` seeded by ``state()`` and return the
        index of the first real cut inside, or ``None``.

        Used when detection runs on sampled frames only: a fire means a cut
        happened somewhere in the skipped gap, and rescanning the buffered
        frames pins the exact boundary. A fire that matches no consecutive
        pair was a stride-spaced false positive (fast pan, fade) — a hard cut
        always trips on adjacent frames."""
        diffs = list(self._diffs)
        for k, img in enumerate(frames):
            h, g = self._feats(img)
            if hist is not None:
                d = cv2.compareHist(hist, h, cv2.HISTCMP_BHATTACHARYYA)
                ch = float(np.mean(np.abs(g - gray) > 20))
                typ = float(np.median(diffs)) if diffs else 0.0
                if d > self.hist_threshold or (
                        ch > self.change_threshold and ch > self.change_ratio * max(typ, 0.01)):
                    return k
                diffs = (diffs + [ch])[-25:]
            hist, gray = h, g
        return None


def _roi_motion(gray: np.ndarray, prev: np.ndarray, x0: float, y0: float, x1: float, y1: float,
                dx: int = 0, dy: int = 0) -> Optional[float]:
    """Mean abs difference of a box in ``gray`` vs. the same box shifted by
    (-dx, -dy) in ``prev`` — i.e. with the head's own movement cancelled."""
    h, w = gray.shape[:2]
    a, b = int(max(0, x0, dx)), int(min(w, x1, w + dx))
    c, d = int(max(0, y0, dy)), int(min(h, y1, h + dy))
    if b - a < 4 or d - c < 4:
        return None
    return float(np.mean(cv2.absdiff(gray[c:d, a:b], prev[c - dy:d - dy, a - dx:b - dx])))


def mouth_activity(face: Face, gray: np.ndarray, prev: np.ndarray,
                   prev_face: Optional[Face] = None) -> Optional[float]:
    """How much the mouth moved since ``prev``, minus how much the upper face
    changed (so nodding, blinking and camera shake don't count as talking).
    ``prev_face`` (the same person in ``prev``) cancels head translation."""
    if face.landmarks is None:
        return None
    dx = dy = 0
    if prev_face is not None:
        dx = int(round(face.cx - prev_face.cx))
        dy = int(round(face.cy - prev_face.cy))
    rex, rey, lex, ley, nx, ny, rmx, rmy, lmx, lmy = face.landmarks
    mw = max(4.0, abs(lmx - rmx))
    mx, my = (rmx + lmx) / 2.0, (rmy + lmy) / 2.0
    eye_y = (rey + ley) / 2.0
    ref_box = (min(rex, lex) - 0.2 * mw, eye_y - 0.3 * mw, max(rex, lex) + 0.2 * mw, ny - 0.1 * mw)
    mouth_box = (mx - 0.75 * mw, my - 0.45 * mw, mx + 0.75 * mw, my + 0.75 * mw)
    # refine the head shift on the upper face (detector boxes jitter by a few
    # px, which on textured skin reads as "motion")
    r = max(2, int(round(0.08 * mw)))
    best = None
    for ddy in range(-r, r + 1, max(1, r // 3)):
        for ddx in range(-r, r + 1, max(1, r // 3)):
            v = _roi_motion(gray, prev, *ref_box, dx + ddx, dy + ddy)
            if v is not None and (best is None or v < best[0]):
                best = (v, dx + ddx, dy + ddy)
    if best is None:
        return None
    ref, dx, dy = best
    mouth = _roi_motion(gray, prev, *mouth_box, dx, dy)
    if mouth is None:
        return None
    return max(0.0, mouth - ref)


def appearance(frame: np.ndarray, face: Face) -> np.ndarray:
    """Colour signature of a person: H-S histograms of the head (skin, hair)
    and of the torso just below it (clothes) — cheap, and in an interview
    the clothes alone usually tell people apart."""
    h, w = frame.shape[:2]

    def hist(x0, y0, x1, y1):
        a, b = int(max(0, x0)), int(min(w, x1))
        c, d = int(max(0, y0)), int(min(h, y1))
        if b - a < 4 or d - c < 4:
            return np.full(48, 1.0 / 48, np.float32)
        roi = frame[c:d, a:b]
        step = max(1, int(max(roi.shape[:2]) / 64))  # ~64 px is plenty
        hsv = cv2.cvtColor(roi[::step, ::step], cv2.COLOR_BGR2HSV)
        hh = cv2.calcHist([hsv], [0, 1], None, [12, 4], [0, 180, 0, 256]).flatten()
        return (hh / max(1e-6, hh.sum())).astype(np.float32)

    head = hist(face.cx - 0.6 * face.w, face.cy - 0.75 * face.h, face.cx + 0.6 * face.w, face.cy + 0.5 * face.h)
    body = hist(face.cx - 1.0 * face.w, face.cy + 0.9 * face.h, face.cx + 1.0 * face.w, face.cy + 2.4 * face.h)
    return np.concatenate([head, body])


def signature_distance(a: np.ndarray, b: np.ndarray) -> float:
    """0 = same look, 1 = completely different (mean Bhattacharyya of the parts)."""
    n = len(a) // 2
    d1 = cv2.compareHist(a[:n], b[:n], cv2.HISTCMP_BHATTACHARYYA)
    d2 = cv2.compareHist(a[n:], b[n:], cv2.HISTCMP_BHATTACHARYYA)
    return float(0.4 * d1 + 0.6 * d2)


class FaceEmbedder:
    """OpenCV SFace: aligned 112x112 face -> 128-d identity embedding.

    Recognises the same person across a close-up and a wide shot, where the
    colour signature alone often fails (different light, crop, scale)."""

    def __init__(self, path: str = SFACE_PATH):
        self._rec = None
        if hasattr(cv2, "FaceRecognizerSF") and os.path.exists(path):
            try:
                self._rec = cv2.FaceRecognizerSF.create(path, "")
            except cv2.error:
                self._rec = None

    @property
    def available(self) -> bool:
        return self._rec is not None

    def __call__(self, frame: np.ndarray, face: Face) -> Optional[np.ndarray]:
        if self._rec is None or face.landmarks is None or min(face.w, face.h) < 12:
            return None
        row = np.array([face.cx - face.w / 2, face.cy - face.h / 2, face.w, face.h,
                        *face.landmarks, face.score], np.float32)
        try:
            feat = self._rec.feature(self._rec.alignCrop(frame, row)).flatten().astype(np.float32)
        except cv2.error:
            return None
        n = float(np.linalg.norm(feat))
        return feat / n if n > 0 else None


def identity(sig: Optional[np.ndarray], emb: Optional[np.ndarray]) -> np.ndarray:
    """One vector per person/track: colour signature + embedding (zeros when
    there is none). Compared with :func:`identity_distance`."""
    sig = sig if sig is not None else np.full(SIG_LEN, 1.0 / SIG_LEN, np.float32)
    emb = emb if emb is not None else np.zeros(128, np.float32)
    return np.concatenate([sig.astype(np.float32), emb.astype(np.float32)])


SFACE_SAME = 0.363
"""SFace cosine similarity above which two faces are the same person."""


def identity_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Distance on the colour-signature scale (same person below ~0.38).

    Uses the face embedding when both sides have one, else the colours."""
    if len(a) > SIG_LEN and len(b) > SIG_LEN:
        ea, eb = a[SIG_LEN:], b[SIG_LEN:]
        na, nb = float(np.linalg.norm(ea)), float(np.linalg.norm(eb))
        if na > 0 and nb > 0:
            cos = float(np.dot(ea, eb) / (na * nb))
            return 0.38 * (1.0 - cos) / (1.0 - SFACE_SAME)
    return signature_distance(a[:SIG_LEN], b[:SIG_LEN])


def merge_identity(a: np.ndarray, wa: float, b: np.ndarray, wb: float) -> np.ndarray:
    """Weighted mean of two identity vectors (colour and embedding averaged
    separately; an embedding missing on one side doesn't dilute the other)."""
    sig = ((a[:SIG_LEN] * wa + b[:SIG_LEN] * wb) / max(1e-6, wa + wb)).astype(np.float32)
    ea = a[SIG_LEN:] if len(a) > SIG_LEN and np.any(a[SIG_LEN:]) else None
    eb = b[SIG_LEN:] if len(b) > SIG_LEN and np.any(b[SIG_LEN:]) else None
    if ea is None and eb is None:
        return sig
    emb = (ea * wa if ea is not None else 0) + (eb * wb if eb is not None else 0)
    emb = np.asarray(emb, np.float32)
    return identity(sig, emb / max(1e-6, float(np.linalg.norm(emb))))


def _same_face(face: Face, candidates: Sequence[Face]) -> Optional[Face]:
    best, best_d = None, 0.5 * face.w
    for c in candidates:
        d = abs(c.cx - face.cx) + abs(c.cy - face.cy)
        if d < best_d:
            best, best_d = c, d
    return best

def iter_frames(cap: "cv2.VideoCapture", start: int, count: int) -> Iterator[np.ndarray]:
    if start > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    for _ in range(count):
        ok, frame = cap.read()
        if not ok:
            return
        yield frame


def analyze_video(
    path: str,
    start_frame: int = 0,
    frame_count: Optional[int] = None,
    stride: int = 2,
    cut_threshold: float = 0.45,
    min_score: float = 0.6,
    progress: Optional[Callable[[float], None]] = None,
    saliency: Optional[Callable[[np.ndarray], Optional[Tuple[float, float, float, float]]]] = None,
) -> Analysis:
    """Detect faces on every ``stride``-th frame and flag hard cuts.

    Frame numbers in the result are relative to ``start_frame``. When
    ``saliency`` (e.g. :class:`saliency.Saliency`) is given, it is run on the
    sampled frames that have no face and the blob is stored on the sample —
    so the camera can follow products/pets/B-roll instead of the centre.
    """
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise IOError("Cannot open video: %s" % path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count is None or frame_count <= 0:
        frame_count = max(0, total - start_frame) if total > 0 else 10**9

    detector = FaceDetector(min_score=min_score)
    embedder = FaceEmbedder()
    # identity embeddings are the slow part: ~2 per second per face is plenty
    embed_every = max(1, int(round(0.5 * fps / max(1, stride))))
    is_cut = CutDetector(hist_threshold=cut_threshold)
    samples: List[FrameSample] = []
    buf: List[np.ndarray] = []       # frames skipped since the last sample
    prev_gray = None
    prev_faces: List[Face] = []
    n = 0

    def _take_sample(idx: int, frame: np.ndarray, cut: bool):
        nonlocal prev_gray, prev_faces
        faces = detector.detect(frame)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        for f in faces:
            f.sig = appearance(frame, f)
        if embedder.available and (cut or len(samples) % embed_every == 0):
            for f in faces:
                f.emb = embedder(frame, f)
        if prev_gray is not None and not cut:
            for f in faces:
                f.mouth = mouth_activity(f, gray, prev_gray, _same_face(f, prev_faces))
        prev_faces = faces
        smp = FrameSample(idx, faces, cut=cut)
        if not faces and saliency is not None:
            try:
                smp.salient = saliency(frame)
            except Exception:
                smp.salient = None
        samples.append(smp)
        prev_gray = gray

    try:
        for i, frame in enumerate(iter_frames(cap, start_frame, frame_count)):
            n = i + 1
            if i % stride == 0:
                # cut detection on sampled frames only (its downscale+hist was
                # ~40% of the pump). A fire means the boundary sits somewhere in
                # the gap since the last sample: rescan the buffered frames to
                # pin it exactly, and sample that frame so the first frame of a
                # new shot always carries a real detection. A fire matching no
                # consecutive pair is a stride-spaced false positive — dropped.
                seed_h, seed_g = is_cut.state()
                cut = is_cut(frame)
                if cut and buf:
                    j = is_cut.locate(buf + [frame], seed_h, seed_g)
                    if j is None:
                        cut = False
                    elif j < len(buf):
                        _take_sample(i - len(buf) + j, buf[j], True)
                        cut = False      # the boundary sample carries the flag
                    # j == len(buf): the boundary is this frame itself
                _take_sample(i, frame, cut)
                buf = []
            else:
                buf.append(frame)
            if progress and frame_count < 10**9 and i % 25 == 0:
                progress(min(1.0, i / float(frame_count)))
    finally:
        cap.release()
    if progress:
        progress(1.0)
    res = Analysis(width, height, fps, start_frame, n, samples)
    res.salient_pass = saliency is not None and getattr(saliency, "ok", True)
    return res


def pick_subject(faces: Sequence[Face], previous: Optional[Face], frame_w: int) -> Optional[Face]:
    """Choose the main face: big, confident and close to the one tracked before."""
    if not faces:
        return None
    best, best_score = None, -1.0
    for f in faces:
        score = f.area * f.score
        if previous is not None:
            dist = abs(f.cx - previous.cx) / float(frame_w)
            score *= 1.0 / (1.0 + 8.0 * dist)
        if score > best_score:
            best, best_score = f, score
    return best
