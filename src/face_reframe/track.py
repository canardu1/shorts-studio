"""Turn noisy per-frame face detections into a smooth virtual-camera path."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from .detect import Analysis, Face, identity, identity_distance, merge_identity


@dataclass
class TrackSettings:
    smoothing: float = 0.5
    """Gaussian smoothing strength in seconds (0 disables)."""
    dead_zone: float = 0.05
    """Fraction of the crop size the face may drift from centre before the camera follows."""
    max_speed: float = 0.6
    """Maximum camera pan speed in source widths per second (0 = unlimited)."""
    outlier_window: float = 0.3
    """Median filter window in seconds used to reject detection glitches."""
    subject: str = "main"
    """Who to follow when several people are in a shot: ``main`` (the most
    prominent face, locked for the whole shot), ``speaker`` (whoever is
    talking, judged from mouth movement; switches are hard cuts) or
    ``object`` (the salient/picked object wins over faces wherever the
    samples carry one — the "track this thing" pick)."""
    noface: str = "center"
    """What the camera aims at in shots with no face: ``center`` holds the
    frame centre, ``object`` follows the most salient blob (U2-Net-p boxes
    stored on the analysis samples)."""
    speaker_hold: float = 1.2
    """Minimum seconds on one speaker before the camera may switch."""
    manual: Optional[List[Tuple[int, Optional[float], Optional[float]]]] = None
    """Hand corrections ``(frame, x, y)`` in source pixels: from ``frame`` on
    (until the next correction or the end of the shot) the camera aims at
    (x, y) instead of the tracked face. ``x=None`` hands control back to the
    tracker from that frame; ``y=None`` leaves the vertical axis alone."""
    person: Optional[object] = None
    """Appearance signature(s) of the person to follow wherever they're on
    screen (see :func:`find_people`); overrides ``subject`` in those shots.
    One array, or a list of them — several thumbnails of the same person
    (e.g. close-up and wide shot not matched automatically)."""
    look_room: float = 0.0
    """Leave room toward the gaze: the crop target shifts up to this fraction
    of the crop width in the direction the face looks (from the nose-vs-eyes
    landmarks; 0 disables, ~0.15 gives a visible off-centre composition)."""
    voice: Optional[List[Tuple[float, float, int]]] = None
    """Voice-tagged speech spans ``(start_s, end_s, speaker_id)`` from
    ECAPA-TDNN (:mod:`speakerid`): with ``subject="speaker"`` the camera
    switches on actual voice changes; the mouth heuristic only maps each
    voice to a face."""


@dataclass
class ShotSettings:
    wide_mode: str = "zoom"
    """What to do with wide shots: ``zoom`` in on the speaker or ``keep`` the full height."""
    wide_threshold: float = 0.12
    """A shot is wide when the face is shorter than this fraction of the frame height."""
    target_face: float = 0.18
    """Face height (fraction of the crop height) to aim for when zooming a wide shot."""
    max_zoom: float = 2.0
    """Upper limit for the wide-shot punch-in, to protect image quality."""


@dataclass
class Shot:
    start: int
    end: int
    kind: str
    """``close``, ``wide`` or ``noface``."""
    face_frac: float
    zoom: float


@dataclass
class CameraPath:
    x: np.ndarray
    y: np.ndarray
    face_h: np.ndarray
    has_face: np.ndarray
    segments: List[Tuple[int, int]]
    zoom: np.ndarray
    shots: List[Shot]


def _size_jump_cuts(analysis: Analysis, ratio: float = 2.0, window: int = 3) -> List[int]:
    """Cuts inferred from the face suddenly changing size (e.g. close-up -> wide shot)."""
    pts = [(s.frame, max(f.h for f in s.faces)) for s in analysis.samples if s.faces]
    cuts, k = [], window
    while k <= len(pts) - window:
        before = float(np.median([h for _, h in pts[k - window:k]]))
        after = float(np.median([h for _, h in pts[k:k + window]]))
        if max(before, after) > ratio * min(before, after):
            cuts.append(pts[k][0])
            k += window
        else:
            k += 1
    return cuts


def _segments(analysis: Analysis) -> List[Tuple[int, int]]:
    cuts = sorted({s.frame for s in analysis.samples if s.cut and s.frame > 0})
    for c in _size_jump_cuts(analysis):
        if all(abs(c - x) > 3 for x in cuts):
            cuts.append(c)
    cuts.sort()
    bounds = [0] + cuts + [analysis.frame_count]
    return [(a, b) for a, b in zip(bounds[:-1], bounds[1:]) if b > a]


def _median_filter(values: np.ndarray, window: int) -> np.ndarray:
    if window < 3 or len(values) < 3:
        return values.copy()
    half = window // 2
    padded = np.pad(values, half, mode="edge")
    view = np.lib.stride_tricks.sliding_window_view(padded, 2 * half + 1)
    return np.median(view, axis=1)


def _gaussian(values: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0.5 or len(values) < 2:
        return values.copy()
    radius = int(3 * sigma)
    k = np.arange(-radius, radius + 1, dtype=float)
    kernel = np.exp(-0.5 * (k / sigma) ** 2)
    kernel /= kernel.sum()
    if radius >= len(values):
        radius = len(values) - 1
        k = np.arange(-radius, radius + 1, dtype=float)
        kernel = np.exp(-0.5 * (k / sigma) ** 2)
        kernel /= kernel.sum()
    padded = np.pad(values, radius, mode="reflect", reflect_type="odd")
    return np.convolve(padded, kernel, mode="valid")


def _dead_zone(values: np.ndarray, zone: float) -> np.ndarray:
    if zone <= 0 or len(values) == 0:
        return values.copy()
    out = np.empty_like(values)
    cam = values[0]
    for i, v in enumerate(values):
        if v > cam + zone:
            cam = v - zone
        elif v < cam - zone:
            cam = v + zone
        out[i] = cam
    return out


def _limit_speed(values: np.ndarray, max_step: float) -> np.ndarray:
    if max_step <= 0 or len(values) == 0:
        return values.copy()
    out = np.empty_like(values)
    out[0] = values[0]
    for i in range(1, len(values)):
        out[i] = out[i - 1] + np.clip(values[i] - out[i - 1], -max_step, max_step)
    return out


def classify_shot(face_frac: Optional[float], settings: ShotSettings) -> Tuple[str, float]:
    if face_frac is None:
        return "noface", 1.0
    if face_frac >= settings.wide_threshold:
        return "close", 1.0
    if settings.wide_mode != "zoom":
        return "wide", 1.0
    zoom = settings.target_face / max(face_frac, 1e-6)
    return "wide", float(min(max(zoom, 1.0), max(1.0, settings.max_zoom)))


def _fill(frames: List[int], vals: List[float], n: int, default: float) -> np.ndarray:
    if not frames:
        return np.full(n, default, dtype=float)
    return np.interp(np.arange(n), np.asarray(frames, float), np.asarray(vals, float))



def _face_tracks(samples, frame_w: float, max_jump: float = 0.08, max_ratio: float = 1.6):
    """Group a shot's detections into per-person tracks.

    Each face joins the track whose last face is closest horizontally (within
    ``max_jump`` frame widths and a similar size); otherwise it starts a new
    track. Returns lists of ``(frame, Face)``.
    """
    tracks: List[List[Tuple[int, Face]]] = []
    for smp in samples:
        taken = set()
        for face in sorted(smp.faces, key=lambda f: -f.area * f.score):
            best, best_d = None, max_jump * frame_w
            for k, tr in enumerate(tracks):
                if k in taken:
                    continue
                last = tr[-1][1]
                ratio = max(face.h, last.h) / max(1e-6, min(face.h, last.h))
                d = abs(face.cx - last.cx)
                if ratio <= max_ratio and d <= best_d:
                    best, best_d = k, d
            if best is None:
                tracks.append([(smp.frame, face)])
                taken.add(len(tracks) - 1)
            else:
                tracks[best].append((smp.frame, face))
                taken.add(best)
    return tracks


def _salient_track(samples) -> List[Tuple[int, Face]]:
    """Pseudo face track out of the samples' salient-object boxes
    (``FrameSample.salient``), so a faceless shot can follow the product /
    pet / B-roll subject through the same smoothing pipeline."""
    out = []
    for smp in samples:
        if smp.salient is None:
            continue
        cx, cy, w, h = smp.salient
        out.append((smp.frame, Face(cx, cy, w, h, 1.0)))
    return out


def _main_track(samples, frame_w: float) -> List[Tuple[int, Face]]:
    """The one person to follow for a whole shot.

    Locking the subject per shot stops the camera from hovering between two
    similar-sized people in a wide shot (the per-frame "biggest face" pick
    flips between them). The winner is the track with the most screen
    presence: mean face size x confidence, discounted for tracks seen in
    less than half the shot (passers-by, false detections).
    """
    tracks = _face_tracks(samples, frame_w)
    if not tracks:
        return []
    n = max(1, sum(1 for smp in samples if smp.faces))

    def weight(tr):
        # bigger face wins; a track seen in under half the shot is discounted
        size = float(np.mean([f.area * f.score for _, f in tr]))
        return size * min(1.0, len(tr) / (0.5 * n))

    tracks.sort(key=weight, reverse=True)
    # tracks that never share the screen with the chosen one are the same
    # person after a jump (a missed cut, a fast move): chain them in
    chain = list(tracks[0])
    spans = [(tracks[0][0][0], tracks[0][-1][0])]
    for tr in tracks[1:]:
        a, b = tr[0][0], tr[-1][0]
        if all(b < x or a > y for x, y in spans):
            chain.extend(tr)
            spans.append((a, b))
    chain.sort(key=lambda item: item[0])
    return chain



def _speaker_pieces(samples, frame_w: float, fps: float, hold_s: float,
                    voice: Optional[List[Tuple[float, float, int]]] = None):
    """Split a shot by who is talking.

    Returns ``[(start_frame, end_frame, track)]`` covering the shot's
    samples, or ``None`` when there aren't two people with mouth data (then
    the caller falls back to the main-subject lock). With ``voice``
    (ECAPA speaker tags, seconds) the cuts happen where the voice changes
    and mouth activity only decides which face owns that voice.
    """
    if len(samples) < 4:
        return None
    tracks = [tr for tr in _face_tracks(samples, frame_w)
              if len(tr) >= 0.25 * len(samples)
              and sum(1 for _, f in tr if f.mouth is not None) >= 0.5 * len(tr)]
    if len(tracks) < 2:
        return None
    if voice:
        # ECAPA: cut where the voice changes; the face is the track whose
        # mouth moved most inside that voice's span
        t0, t1 = samples[0].frame / fps, samples[-1].frame / fps
        segs = [(max(a, t0), min(b, t1), lab) for a, b, lab in voice
                if lab >= 0 and b > t0 and a < t1]
        segs.sort()
        if len({lab for _, _, lab in segs}) < 2:
            return None
        pieces, cur = [], None
        for va, vb, lab in segs:
            best, best_act = cur, 0.0
            for tr in tracks:
                act = sum(f.mouth or 0.0 for f, face in tr
                          if va * fps <= f <= vb * fps)
                if act > best_act:
                    best, best_act = tr, act
            if best is None:
                continue
            cur = best
            fa, fb = int(round(va * fps)), int(round(vb * fps))
            if pieces and pieces[-1][2] is cur:
                pieces[-1] = (pieces[-1][0], fb, cur)
            else:
                pieces.append((fa, fb, cur))
        if pieces:
            # cover the pauses between speech: each piece lasts until the next
            for k in range(len(pieces) - 1):
                pieces[k] = (pieces[k][0], pieces[k + 1][0], pieces[k][2])
        return pieces or None
    frames = np.array([smp.frame for smp in samples], float)
    act = np.zeros((len(tracks), len(samples)))
    for k, tr in enumerate(tracks):
        fr = np.array([f for f, face in tr if face.mouth is not None], float)
        val = np.array([face.mouth for _, face in tr if face.mouth is not None], float)
        if len(fr):
            v = np.interp(frames, fr, val, left=0.0, right=0.0)
            # outside the track's lifetime that person isn't on screen
            v[(frames < tr[0][0]) | (frames > tr[-1][0])] = 0.0
            act[k] = v
    step = max(1.0, float(np.median(np.diff(frames)))) if len(frames) > 1 else 1.0
    win = max(1, int(round(0.8 * fps / step)))
    kernel = np.ones(win) / win
    act = np.array([np.convolve(row, kernel, mode="same") for row in act])

    hold = max(1, int(round(hold_s * fps / step)))
    head = act[:, :min(len(samples), 2 * hold)].mean(axis=1)
    cur = int(np.argmax(head))
    since = 0
    labels = np.empty(len(samples), int)
    for i in range(len(samples)):
        best = int(np.argmax(act[:, i]))
        since += 1
        if best != cur and since >= hold and act[best, i] > 1.3 * act[cur, i] + 0.3:
            # challenger must keep the lead for a bit before we cut to them
            j = min(len(samples), i + max(1, hold // 2))
            if np.all(act[best, i:j] >= act[cur, i:j]):
                cur, since = best, 0
        labels[i] = cur
    pieces, a = [], 0
    for i in range(1, len(samples) + 1):
        if i == len(samples) or labels[i] != labels[a]:
            pieces.append((samples[a].frame, samples[i].frame if i < len(samples) else None,
                           tracks[labels[a]]))
            a = i
    return pieces



@dataclass
class Person:
    id: int
    sig: np.ndarray
    tracks: Dict[int, List[Tuple[int, Face]]]
    """Segment index -> that person's track in the segment."""
    samples: int
    best: Tuple[int, Face]
    """(frame, face) of the clearest view, for a thumbnail."""


def _track_sig(tr) -> Optional[np.ndarray]:
    """Identity vector of a track: mean colour signature + mean embedding."""
    sigs = [f.sig for _, f in tr if f.sig is not None]
    embs = [f.emb for _, f in tr if f.emb is not None]
    if not sigs and not embs:
        return None
    sig = np.mean(sigs, axis=0).astype(np.float32) if sigs else None
    emb = None
    if embs:
        emb = np.mean(embs, axis=0)
        emb = (emb / max(1e-6, float(np.linalg.norm(emb)))).astype(np.float32)
    if emb is None and sig is not None:
        return sig
    return identity(sig, emb)


def find_people(analysis: Analysis, merge: float = 0.38) -> List[Person]:
    """Everyone seen in the video, matched across shots by appearance.

    Per-shot face tracks are clustered greedily (most-seen first); two tracks
    in the same shot are never the same person. Deterministic, and cached on
    the analysis, so the UI and the render agree on who is who.
    """
    cached = getattr(analysis, "_people", None)
    if cached is not None:
        return cached
    by_frame = {smp.frame: smp for smp in analysis.samples}
    items = []
    for si, (a, b) in enumerate(_segments(analysis)):
        samples = [by_frame[f] for f in sorted(k for k in by_frame if a <= k < b)]
        for tr in _face_tracks(samples, analysis.width):
            sig = _track_sig(tr)
            if sig is not None and len(tr) >= 3:
                items.append((si, tr, sig))
    items.sort(key=lambda it: -len(it[1]))
    people: List[Person] = []
    for si, tr, sig in items:
        best, best_d = None, merge
        for p in people:
            if si in p.tracks:
                continue
            d = identity_distance(sig, p.sig)
            if d < best_d:
                best, best_d = p, d
        top = max(tr, key=lambda it: it[1].area * it[1].score)
        if best is None:
            people.append(Person(len(people), sig, {si: tr}, len(tr), top))
        else:
            k = best.samples
            best.sig = merge_identity(best.sig, k, sig, len(tr))
            best.tracks[si] = tr
            best.samples += len(tr)
            if top[1].area * top[1].score > best.best[1].area * best.best[1].score:
                best.best = top
    people.sort(key=lambda p: -p.samples)
    for i, p in enumerate(people):
        p.id = i
    analysis._people = people
    return people


def match_person(people: List[Person], sig: np.ndarray, max_dist: float = 0.45) -> Optional[Person]:
    best = min(people, key=lambda p: identity_distance(sig, p.sig), default=None)
    if best is None or identity_distance(sig, best.sig) > max_dist:
        return None
    return best



def _apply_manual(seg_x, seg_y, has, a: int, b: int, manual):
    """Overwrite the raw target inside [a, b) with the hand corrections.

    A correction made earlier in the same shot still holds at ``a`` (the
    shot may have been split by a speaker change). Returns the masks and
    values so they can be re-imposed after the dead zone."""
    n = b - a
    mask_x, mask_y = np.zeros(n, bool), np.zeros(n, bool)
    val_x, val_y = np.zeros(n), np.zeros(n)
    keys = sorted((int(f), x, y) for f, x, y in manual if f < b)
    active = None
    for f, x, y in keys:
        if f <= a:
            active = (x, y)
    marks = [(a, active)] if active is not None else []
    marks += [(f, (x, y)) for f, x, y in keys if a < f < b]
    for k, (f, (x, y)) in enumerate(marks):
        end = marks[k + 1][0] if k + 1 < len(marks) else b
        if x is None:
            continue
        sl = slice(f - a, end - a)
        seg_x[sl] = x
        mask_x[sl], val_x[sl] = True, x
        if y is not None:
            seg_y[sl] = y
            mask_y[sl], val_y[sl] = True, y
        has[f:end] = True
    return (mask_x, val_x), (mask_y, val_y)


def build_camera_path(
    analysis: Analysis,
    settings: Optional[TrackSettings] = None,
    crop_size: Optional[Tuple[float, float]] = None,
    shot_settings: Optional[ShotSettings] = None,
) -> CameraPath:
    s = settings or TrackSettings()
    shots_cfg = shot_settings or ShotSettings(wide_mode="keep")
    n, w, h, fps = analysis.frame_count, analysis.width, analysis.height, analysis.fps
    xs = np.full(n, w / 2.0)
    ys = np.full(n, h / 2.0)
    fh = np.full(n, h / 4.0)
    has = np.zeros(n, dtype=bool)
    zoom = np.ones(n)
    shots: List[Shot] = []
    segments = _segments(analysis)
    zone_w, zone_h = crop_size or (w, h)

    by_frame = {smp.frame: smp for smp in analysis.samples}
    # (start, end, [(frame, Face)]) pieces: one per shot, or several when the
    # camera cuts between speakers inside a shot
    pieces: List[Tuple[int, int, List[Tuple[int, Face]]]] = []
    chosen = []
    if s.person is not None:
        sigs = s.person if isinstance(s.person, (list, tuple)) else [s.person]
        everyone = find_people(analysis)
        for sig in sigs:
            p = match_person(everyone, np.asarray(sig, np.float32))
            if p is not None and p not in chosen:
                chosen.append(p)
    for si, (a, b) in enumerate(segments):
        seg_samples = [by_frame[f] for f in sorted(k for k in by_frame if a <= k < b)]
        here = [p.tracks[si] for p in chosen if si in p.tracks]
        if here:
            # several picks on screen together: the most prominent of them
            pieces.append((a, b, max(here, key=lambda tr: float(np.mean(
                [f.area * f.score for _, f in tr])) * len(tr))))
            continue
        if s.subject == "object":
            obj = _salient_track(seg_samples)
            if obj:
                pieces.append((a, b, obj))
                continue
        split = None
        if s.subject == "speaker":
            split = _speaker_pieces(seg_samples, w, fps, s.speaker_hold, s.voice)
        if not split:
            pieces.append((a, b, _main_track(seg_samples, w)))
            continue
        for k, (pa, pb, tr) in enumerate(split):
            pa = a if k == 0 else pa
            pb = b if pb is None else pb
            pieces.append((pa, pb, [(f, face) for f, face in tr if pa <= f < pb]))
    segments = [(a, b) for a, b, _ in pieces]

    last_face_center: Optional[Tuple[float, float, float]] = None
    for a, b, track in pieces:
        if not track and s.noface == "object":
            track = _salient_track([by_frame[f] for f in sorted(k for k in by_frame if a <= k < b)])
        frames, sx, sy, sh = [], [], [], []
        for f, face in track:
            frames.append(f - a)
            cx = face.cx
            if s.look_room > 0 and face.landmarks is not None:
                # yaw from the 5 YuNet landmarks: nose tip drifting right of
                # the eye midpoint means the face looks toward image-right,
                # so the crop slides right and leaves room in the gaze path
                rex, rey, lex, ley, nx, ny = face.landmarks[:6]
                ed = abs(lex - rex)
                if ed > 4:
                    yaw = (nx - (rex + lex) / 2.0) / ed
                    cx += s.look_room * zone_w * np.clip(yaw / 0.4, -1.0, 1.0)
            sx.append(cx)
            sy.append(face.cy)
            sh.append(face.h)
        length = b - a
        if frames:
            default = (sx[0], sy[0], sh[0])
            last_face_center = (sx[-1], sy[-1], sh[-1])
        else:
            default = last_face_center or (w / 2.0, h / 2.0, h / 4.0)
        seg_x = _fill(frames, sx, length, default[0])
        seg_y = _fill(frames, sy, length, default[1])
        seg_h = _fill(frames, sh, length, default[2])
        if frames:
            has[a + frames[0]: a + frames[-1] + 1] = True
        man = _apply_manual(seg_x, seg_y, has, a, b, s.manual) if s.manual else None
        face_frac = float(np.median(sh)) / h if frames else None
        kind, seg_zoom = classify_shot(face_frac, shots_cfg)
        shots.append(Shot(a, b, kind, face_frac or 0.0, seg_zoom))
        zoom[a:b] = seg_zoom

        win = int(round(s.outlier_window * fps)) | 1
        seg_x = _median_filter(seg_x, win)
        seg_y = _median_filter(seg_y, win)
        seg_x = _dead_zone(seg_x, s.dead_zone * zone_w / seg_zoom)
        seg_y = _dead_zone(seg_y, s.dead_zone * zone_h / seg_zoom)
        if man is not None:
            # a hand correction is exact: the dead zone must not stop short of it
            (mx, vx), (my, vy) = man
            seg_x[mx], seg_y[my] = vx[mx], vy[my]
        sigma = s.smoothing * fps
        seg_x = _gaussian(seg_x, sigma)
        seg_y = _gaussian(seg_y, sigma)
        seg_h = _gaussian(_median_filter(seg_h, win), sigma)
        step = s.max_speed * w / fps if s.max_speed > 0 else 0
        seg_x = _limit_speed(seg_x, step)
        seg_y = _limit_speed(seg_y, step)
        xs[a:b], ys[a:b], fh[a:b] = seg_x, seg_y, seg_h
    return CameraPath(xs, ys, fh, has, segments, zoom, shots)
