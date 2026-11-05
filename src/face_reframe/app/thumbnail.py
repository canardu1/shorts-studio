"""Click-worthy thumbnails (YouTube 16:9, Shorts 9:16) built from the video.

No generative model: the most expressive sharp frame, the person cut out
with a tiny segmentation net (U²-Net-p, 4.6 MB, ~0.2 s on an M1 CPU), a
white outline and shadow, a punchy background and 2-4 big words.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..i18n import tr

SEG_URL = "https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2netp.onnx"
SEG_FILE = "u2netp.onnx"

STYLES = [("pop", "Pop (blurred background)"), ("colore", "Bright colour"), ("zoom", "Close-up"),
          ("neon", "Dark with glow")]

SIZES = {"16:9": (1280, 720), "9:16": (1080, 1920)}

# text drawn as (lines, highlighted word index over all words, width px, height px) -> BGRA
TextRenderer = Callable[[List[str], Optional[int], int, int, str], np.ndarray]   # words, hl, w, h, style


# ------------------------------------------------------------------ segmentation
def seg_model_path() -> str:
    from ..restore import models_dir
    return os.path.join(models_dir(), SEG_FILE)


def seg_available() -> bool:
    p = seg_model_path()
    return os.path.exists(p) and os.path.getsize(p) > 1_000_000


def download_seg(progress=None) -> str:
    path = seg_model_path()
    if seg_available():
        return path
    from ..restore import fetch
    return fetch(SEG_URL, path, progress)


class Segmenter:
    """Person/subject alpha matte at the frame's size (float32 0..1)."""

    def __init__(self, path: Optional[str] = None):
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.log_severity_level = 3
        from .. import models
        self.sess = ort.InferenceSession(path or seg_model_path(), so,
                                         providers=models.providers())
        self.inp = self.sess.get_inputs()[0].name

    def __call__(self, img: np.ndarray, face=None) -> np.ndarray:
        import cv2
        x = cv2.resize(img, (320, 320), interpolation=cv2.INTER_AREA)[..., ::-1].astype(np.float32) / 255.0
        x = (x - (0.485, 0.456, 0.406)) / (0.229, 0.224, 0.225)
        out = self.sess.run(None, {self.inp: x.transpose(2, 0, 1)[None].astype(np.float32)})[0][0, 0]
        out = (out - out.min()) / max(1e-6, float(out.max() - out.min()))
        a = cv2.resize(out, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)
        a = np.clip((a - 0.25) / 0.5, 0, 1).astype(np.float32)      # firmer edges
        if face is not None:
            a = a * person_prior(a.shape, face)   # salient props beside the person go
        return _keep_subject(a, face)


class RvmSegmenter:
    """Same job as :class:`Segmenter` through the RVM matte (better people
    edges); used when the background-blur model is already downloaded."""

    def __init__(self):
        from .rvm import Matter
        self.matter = Matter()

    def __call__(self, img: np.ndarray, face=None) -> np.ndarray:
        self.matter.reset()                     # stills: no temporal memory
        a = self.matter(img)
        a = np.clip((a - 0.1) / 0.8, 0, 1).astype(np.float32)
        if face is not None:
            a = a * person_prior(a.shape, face)
        return _keep_subject(a, face)


def _segmenter(log=print):
    """The best available cutout backend: RVM when downloaded (it is once the
    background blur was used), else U2NetP downloaded now, else None."""
    from .rvm import rvm_available
    if rvm_available():
        try:
            return RvmSegmenter()
        except Exception as exc:
            log(tr("Thumbnails: RVM cutout failed (%s), falling back") % exc)
    try:
        if not seg_available():
            log(tr("Thumbnails: downloading the cutout model (4.6 MB, once)…"))
            download_seg()
        return Segmenter()
    except Exception as exc:   # offline / no onnxruntime: soft head-and-shoulders shape
        log(tr("Thumbnails: cutout unavailable (%s), using a soft shape") % exc)
    return None


def person_prior(shape, face) -> np.ndarray:
    """Soft region where a person around ``face`` can be: head plus a body
    widening below it (keeps flags, lamps, posters out of the cut-out)."""
    import cv2
    h, w = shape[:2]
    m = np.zeros((h, w), np.float32)
    cx, cy, fw, fh = face.cx, face.cy, face.w, face.h
    cv2.ellipse(m, (int(cx), int(cy - 0.1 * fh)), (int(fw * 1.0), int(fh * 1.05)), 0, 0, 360, 1, -1)
    top = cy + 0.35 * fh
    pts = np.array([[cx - 0.9 * fw, top], [cx + 0.9 * fw, top], [cx + 3.4 * fw, top + 2.2 * fh],
                    [cx + 3.6 * fw, h + fh], [cx - 3.6 * fw, h + fh], [cx - 3.4 * fw, top + 2.2 * fh]],
                   np.int32)
    cv2.fillPoly(m, [pts], 1)
    return cv2.GaussianBlur(m, (0, 0), max(3.0, fw * 0.25))


def _keep_subject(a: np.ndarray, face) -> np.ndarray:
    """Drop blobs not connected to the person (props, background objects)."""
    import cv2
    m = (a > 0.5).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    if n <= 2:
        return a
    keep = None
    if face is not None:
        x, y = int(np.clip(face.cx, 0, a.shape[1] - 1)), int(np.clip(face.cy, 0, a.shape[0] - 1))
        keep = lab[y, x] if lab[y, x] > 0 else None
    if keep is None:
        keep = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    mask = (lab == keep).astype(np.uint8)
    mask = cv2.dilate(mask, np.ones((9, 9), np.uint8))
    return a * cv2.GaussianBlur(mask.astype(np.float32), (0, 0), 3)


def face_matte(img: np.ndarray, face) -> np.ndarray:
    """Fallback without the model: a soft head-and-shoulders shape."""
    import cv2
    a = np.zeros(img.shape[:2], np.float32)
    cv2.ellipse(a, (int(face.cx), int(face.cy)), (int(face.w * 0.75), int(face.h * 0.95)), 0, 0, 360, 1, -1)
    cv2.ellipse(a, (int(face.cx), int(face.cy + face.h * 1.9)), (int(face.w * 1.9), int(face.h * 1.2)),
                0, 0, 360, 1, -1)
    return cv2.GaussianBlur(a, (0, 0), max(2.0, face.w * 0.05))


# ------------------------------------------------------------------ frames
@dataclass
class Shot:
    frame: int
    face: object
    image: np.ndarray
    score: float


def _smile(f) -> float:
    if f.landmarks is None:
        return 0.0
    rex, rey, lex, ley, _nx, _ny, rmx, rmy, lmx, lmy = f.landmarks[:10]
    eye = max(1.0, float(np.hypot(lex - rex, ley - rey)))
    return float(np.hypot(lmx - rmx, lmy - rmy)) / eye   # ~0.75 neutral, >0.9 big smile


def pick_frames(path: str, analysis, count: int = 4, min_gap_s: float = 2.0,
                max_read: int = 40) -> List[Shot]:
    """Sharp, large, frontal, expressive faces, spread over the video."""
    import cv2
    fps = getattr(analysis, "fps", 25.0) or 25.0
    cands = []
    for smp in analysis.samples:
        faces = [f for f in smp.faces if f.landmarks is not None]
        if not faces:
            continue
        f = max(faces, key=lambda x: x.area)
        rex, rey, lex, ley, nx, ny = f.landmarks[:6]
        frontal = 1.0 - min(1.0, abs(nx - 0.5 * (rex + lex)) / max(1.0, abs(lex - rex)))
        expr = 1.0 + 2.0 * max(0.0, _smile(f) - 0.72) + 0.6 * min(1.0, (f.mouth or 0.0) / 10.0)
        cands.append((f.area * f.score * (0.4 + frontal) * expr, smp.frame, f))
    if not cands:
        # No faces at all: aesthetic fallback — salient subject, sharp frame.
        # Scores the U2-Net-p boxes the reframe can already produce; rule-of-
        # thirds placement beats a perfectly centred blob.
        for smp in analysis.samples:
            if smp.salient is None:
                continue
            cx, cy, bw, bh = smp.salient
            W_, H_ = analysis.width, analysis.height
            area = bw * bh / float(W_ * H_)
            if not 0.005 < area < 0.6:
                continue
            tx, ty = W_ / 3.0, H_ / 3.0
            third = max(abs(cx - tx), abs(cx - 2 * tx)) + max(abs(cy - ty), abs(cy - 2 * ty))
            third_score = 1.0 - min(1.0, third / (0.7 * (W_ + H_)))
            cands.append((area * (0.5 + third_score) * W_ * H_, smp.frame, smp.salient))
        if not cands:
            return []
    cands.sort(key=lambda c: -c[0])
    # spread the reads over the video: best few per time bucket
    picked, seen = [], []
    for c in cands:
        if all(abs(c[1] - s) >= 0.5 * min_gap_s * fps for s in seen):
            picked.append(c)
            seen.append(c[1])
        if len(picked) >= max_read:
            break
    cap = cv2.VideoCapture(path)
    shots = []
    for score, fr, f in picked:
        cap.set(cv2.CAP_PROP_POS_FRAMES, getattr(analysis, "start_frame", 0) + fr)
        ok, img = cap.read()
        if not ok:
            continue
        if isinstance(f, tuple):
            # salient box (cx, cy, w, h) → Face-shaped shim for the pipeline
            cx, cy, bw, bh = f
            f = type("Obj", (), {"cx": cx, "cy": cy, "w": bw, "h": bh,
                                 "area": bw * bh, "score": 1.0,
                                 "landmarks": None, "mouth": 0.0})()
        if f is None:
            crop = img
        else:
            x0, y0 = int(max(0, f.cx - f.w / 2)), int(max(0, f.cy - f.h / 2))
            crop = img[y0:int(y0 + f.h), x0:int(x0 + f.w)]
        if crop.size == 0:
            continue
        g = cv2.cvtColor(cv2.resize(crop, (128, 128)), cv2.COLOR_BGR2GRAY)
        sharp = float(cv2.Laplacian(g, cv2.CV_64F).var())
        shots.append(Shot(fr, f, img, score * (sharp ** 0.5)))
    cap.release()
    shots.sort(key=lambda s: -s.score)
    out: List[Shot] = []
    for s in shots:
        if all(abs(s.frame - o.frame) >= min_gap_s * fps for o in out):
            out.append(s)
        if len(out) >= count:
            break
    return out


# ------------------------------------------------------------------ text
def thumb_text(words: Sequence, hook: str = "", max_words: int = 4) -> Tuple[str, Optional[int]]:
    """2-4 punchy words around the strongest one, and which to highlight."""
    from .motion import STRONG
    from .social import STOP
    src = (hook or "").split() or [w.text.strip() for w in (words or [])[:30]]
    toks = [t for t in src if t.strip()]
    if not toks:
        return "", None

    def strength(t):
        n = re.sub(r"[^\wàèéìòù]+", "", t.lower())
        if not n or n in STOP:
            return 0.0
        s = len(n) / 10.0
        if n in STRONG:
            s += 3
        if re.search(r"\d", n):
            s += 2.5
        if t.endswith(("!", "?")):
            s += 1
        return s

    sc = [strength(t) for t in toks]
    k = int(np.argmax(sc))
    a = max(0, min(k - 1, len(toks) - max_words))
    win = toks[a:a + max_words]
    # don't start or end on a filler word
    while len(win) > 2 and strength(win[0]) == 0 and a < k:
        win, a = win[1:], a + 1
    while len(win) > 2 and strength(win[-1]) == 0 and a + len(win) - 1 > k:
        win = win[:-1]
    text = " ".join(re.sub(r"[.,;:]+$", "", t) for t in win).upper()
    return text, k - a


def partitions(words: Sequence[str], max_lines: int = 4):
    """Every way of breaking ``words`` into 1..max_lines consecutive lines."""
    words = list(words)
    n = len(words)
    if n == 0:
        return
    from itertools import combinations
    for k in range(0, min(max_lines, n) ):
        for cuts in combinations(range(1, n), k):
            b = (0,) + cuts + (n,)
            yield [" ".join(words[b[i]:b[i + 1]]) for i in range(len(b) - 1)]


def wrap(text: str, per_line: int = 2) -> List[str]:
    w = text.split()
    if len(w) <= 2:
        return [" ".join(w)] if w else []
    if len(w) == 3:
        return [w[0] + " " + w[1], w[2]] if len(w[0] + w[1]) < len(w[1] + w[2]) else [w[0], w[1] + " " + w[2]]
    return [" ".join(w[i:i + per_line]) for i in range(0, len(w), per_line)]


def cv_text(words, hl: Optional[int], max_w: int, max_h: int, style: str) -> np.ndarray:
    """OpenCV fallback for the big text (no Qt)."""
    import cv2
    lines = wrap(" ".join(words) if not isinstance(words, str) else words)
    font, th = cv2.FONT_HERSHEY_DUPLEX, 1
    scale = 4.0
    while scale > 0.6:
        sizes = [cv2.getTextSize(l, font, scale, int(scale * 3))[0] for l in lines]
        if max(s[0] for s in sizes) <= max_w and sum(s[1] * 1.35 for s in sizes) <= max_h:
            break
        scale *= 0.9
    th = max(2, int(scale * 3))
    lh = int(max(s[1] for s in sizes) * 1.35)
    W, H = max(s[0] for s in sizes) + 4 * th, lh * len(lines) + 4 * th
    img = np.zeros((H, W, 4), np.uint8)
    idx = 0
    for k, l in enumerate(lines):
        x, y = 2 * th, 2 * th + (k + 1) * lh - lh // 4
        for wd in l.split():
            col = (0, 212, 255, 255) if idx == hl else (255, 255, 255, 255)
            cv2.putText(img, wd, (x, y), font, scale, (0, 0, 0, 255), th * 3, cv2.LINE_AA)
            cv2.putText(img, wd, (x, y), font, scale, col, th, cv2.LINE_AA)
            x += cv2.getTextSize(wd + " ", font, scale, th)[0][0]
            idx += 1
    return img


# ------------------------------------------------------------------ compose
def _paste(dst: np.ndarray, img: np.ndarray, x: int, y: int) -> np.ndarray:
    h, w = img.shape[:2]
    H, W = dst.shape[:2]
    x0, y0, x1, y1 = max(0, x), max(0, y), min(W, x + w), min(H, y + h)
    if x1 <= x0 or y1 <= y0:
        return dst
    sub = img[y0 - y:y1 - y, x0 - x:x1 - x]
    a = sub[..., 3:4].astype(np.float32) / 255.0
    roi = dst[y0:y1, x0:x1].astype(np.float32)
    dst[y0:y1, x0:x1] = (roi * (1 - a) + sub[..., :3].astype(np.float32) * a).astype(np.uint8)
    return dst


def _pop(img: np.ndarray, sat: float = 1.25, contrast: float = 1.1) -> np.ndarray:
    import cv2
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 1] = np.clip(hsv[..., 1] * sat, 0, 255)
    out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)
    out = np.clip((out - 118) * contrast + 124, 0, 255)
    blur = cv2.GaussianBlur(out, (0, 0), 1.6)
    return np.clip(out * 1.5 - blur * 0.5, 0, 255).astype(np.uint8)


def _gradient(W: int, H: int, c0, c1, radial: bool = True) -> np.ndarray:
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    if radial:
        t = np.clip(np.hypot((xx - W * 0.68) / W, (yy - H * 0.45) / H) * 1.6, 0, 1)
    else:
        t = yy / H
    c0, c1 = np.array(c0, np.float32), np.array(c1, np.float32)
    return (c0 * (1 - t[..., None]) + c1 * t[..., None]).astype(np.uint8)


def _placement(shot: Shot, W: int, H: int, style: str) -> Tuple[float, float, float]:
    """Scale and offset mapping the source frame onto the canvas."""
    f = shot.face
    sh, sw = shot.image.shape[:2]
    wide = W > H
    face_h = (0.62 if style == "zoom" else 0.40) * H if wide else (0.30 if style == "zoom" else 0.21) * H
    s = face_h / max(1.0, f.h)
    fx = (0.68 if style != "zoom" else 0.66) * W if wide else 0.5 * W
    fy = (0.42 if style != "zoom" else 0.45) * H if wide else (0.62 if style == "zoom" else 0.56) * H
    tx, ty = fx - f.cx * s, fy - f.cy * s
    # the body runs out of the frame's bottom edge: never leave a gap under it
    bottom = ty + sh * s
    if bottom < H:
        ty += H - bottom
    if style == "zoom":     # full-bleed photo: cover the canvas
        s2 = max(s, W / sw, H / sh)
        if s2 > s:
            tx, ty = fx - f.cx * s2, fy - f.cy * s2
            s = s2
        tx = min(0.0, max(W - sw * s, tx))
        ty = min(0.0, max(H - sh * s, ty))
    return s, tx, ty


def compose(shot: Shot, matte: Optional[np.ndarray], text: str, hl: Optional[int], style: str,
            size: Tuple[int, int], text_renderer: Optional[TextRenderer] = None) -> np.ndarray:
    import cv2
    W, H = size
    wide = W > H
    s, tx, ty = _placement(shot, W, H, style)
    M = np.float32([[s, 0, tx], [0, s, ty]])
    src = _pop(shot.image)
    fg = cv2.warpAffine(src, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    a = matte if matte is not None else face_matte(shot.image, shot.face)
    al = cv2.warpAffine(a, M, (W, H), flags=cv2.INTER_LINEAR, borderValue=0)[..., None]
    if style == "zoom":
        bg = fg.copy()
    elif style == "colore":
        bg = _gradient(W, H, (0, 215, 255), (0, 90, 255))           # yellow -> orange (BGR)
    elif style == "neon":
        bg = _gradient(W, H, (70, 25, 20), (12, 6, 6))
    else:   # pop: the frame itself, blurred and darkened
        small = cv2.resize(fg, (W // 4, H // 4), interpolation=cv2.INTER_AREA)
        small = cv2.GaussianBlur(small, (0, 0), 6)
        bg = (cv2.resize(small, (W, H), interpolation=cv2.INTER_LINEAR).astype(np.float32) * 0.55).astype(np.uint8)
    out = bg.astype(np.float32)
    if style == "zoom":
        # vignette + darker side for the text
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        v = 1 - 0.55 * np.clip(np.hypot((xx - W * 0.6) / W, (yy - H * 0.45) / H) * 1.4, 0, 1)
        if wide:
            v *= np.clip(0.45 + xx / (W * 0.55), 0, 1) ** 0.7
        else:
            v *= np.clip(0.45 + yy / (H * 0.35), 0, 1) ** 0.7
        out = out * v[..., None]
    else:
        r = max(4, int(H * (0.012 if wide else 0.008)))
        hard = (al[..., 0] > 0.5).astype(np.uint8)
        # drop shadow
        sh = cv2.GaussianBlur(np.roll(np.roll(hard.astype(np.float32), r * 2, 0), r, 1), (0, 0), r * 2)
        out = out * (1 - 0.55 * sh[..., None])
        if style == "neon":
            glow = cv2.GaussianBlur(cv2.dilate(hard, np.ones((r * 3, r * 3), np.uint8)).astype(np.float32),
                                    (0, 0), r * 4)[..., None]
            out = out + glow * np.array((255, 220, 0), np.float32) * 0.9         # cyan halo
        ring = cv2.GaussianBlur(cv2.dilate(hard, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1,) * 2))
                                .astype(np.float32), (0, 0), 1.2)[..., None]
        out = out * (1 - ring) + 255.0 * ring                                    # white outline
        out = out * (1 - al) + fg.astype(np.float32) * al
    out = np.clip(out, 0, 255).astype(np.uint8)
    if text.strip():
        if wide:
            box_w, box_h = int(W * (0.50 if style != "zoom" else 0.52)), int(H * 0.80)
        else:
            box_w, box_h = int(W * 0.92), int(H * 0.26)
        img = (text_renderer or cv_text)(text.split(), hl, box_w, box_h, style)
        if wide:
            x, y = int(W * 0.04), (H - img.shape[0]) // 2
        else:
            x, y = (W - img.shape[1]) // 2, int(H * 0.07)
        out = _paste(out, img, x, y)
    return out


# ------------------------------------------------------------------ high level
@dataclass
class Variant:
    style: str
    shot: Shot
    matte: Optional[np.ndarray]


def prepare(path: str, analysis, count: int = 4, progress=None, log=print) -> List[Variant]:
    """Frames + mattes for ``count`` variants (one style each, cycling)."""
    shots = pick_frames(path, analysis, count=count)
    if not shots:
        return []
    seg = _segmenter(log)
    out = []
    styles = [k for k, _ in STYLES]
    for i in range(count):
        sh = shots[i % len(shots)]
        matte = seg(sh.image, sh.face) if seg is not None else None
        out.append(Variant(styles[i % len(styles)], sh, matte))
        if progress:
            progress((i + 1) / count)
    return out


def render(v: Variant, text: str, hl: Optional[int], aspect: str = "16:9",
           text_renderer: Optional[TextRenderer] = None) -> np.ndarray:
    return compose(v.shot, v.matte, text, hl, v.style, SIZES[aspect], text_renderer)


def save(img: np.ndarray, path: str) -> str:
    import cv2
    cv2.imwrite(path, img, [cv2.IMWRITE_JPEG_QUALITY, 93])
    return path
