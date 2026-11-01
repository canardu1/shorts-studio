"""Crop-window geometry for a target aspect ratio."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Union

import numpy as np

ArrayLike = Union[float, bool, np.ndarray]


def parse_aspect(text: str) -> float:
    """``"9:16"`` -> 0.5625 (width / height)."""
    if ":" in text:
        a, b = text.split(":", 1)
        return float(a) / float(b)
    if "/" in text:
        a, b = text.split("/", 1)
        return float(a) / float(b)
    return float(text)


def crop_size(src_w: int, src_h: int, aspect: float, zoom: ArrayLike = 1.0):
    """Largest ``aspect`` window that fits the source, shrunk by ``zoom``."""
    zoom = np.maximum(1.0, np.asarray(zoom, float))
    crop_h = src_h / zoom
    crop_w = crop_h * aspect
    too_wide = crop_w > src_w / zoom
    crop_w = np.where(too_wide, src_w / zoom, crop_w)
    crop_h = np.where(too_wide, crop_w / aspect, crop_h)
    if crop_w.ndim == 0:
        return float(crop_w), float(crop_h)
    return crop_w, crop_h


@dataclass
class CropPlan:
    src_w: int
    src_h: int
    crop_w: np.ndarray
    crop_h: np.ndarray
    cx: np.ndarray
    cy: np.ndarray

    def box(self, i: int):
        """Integer (x, y, w, h) crop box for frame ``i``."""
        w, h = int(round(self.crop_w[i])), int(round(self.crop_h[i]))
        x = int(round(self.cx[i] - self.crop_w[i] / 2.0))
        y = int(round(self.cy[i] - self.crop_h[i] / 2.0))
        x = max(0, min(self.src_w - w, x))
        y = max(0, min(self.src_h - h, y))
        return x, y, w, h


def plan_crops(
    src_w: int,
    src_h: int,
    path_x: np.ndarray,
    path_y: np.ndarray,
    aspect: float,
    zoom: ArrayLike = 1.0,
    follow_vertical: ArrayLike = False,
    face_height_pos: float = 0.4,
) -> CropPlan:
    """Size the crop for ``aspect`` and centre it on the camera path.

    ``zoom`` and ``follow_vertical`` may be per-frame arrays. ``face_height_pos``
    is where the face sits vertically in the crop (0 = top, 0.5 = middle) when
    following vertically.
    """
    n = len(path_x)
    zoom = np.broadcast_to(np.asarray(zoom, float), (n,))
    follow = np.broadcast_to(np.asarray(follow_vertical, bool), (n,))
    crop_w, crop_h = crop_size(src_w, src_h, aspect, zoom)
    cx = np.clip(np.asarray(path_x, float), crop_w / 2.0, src_w - crop_w / 2.0)
    cy = np.where(follow, np.asarray(path_y, float) + (0.5 - face_height_pos) * crop_h, src_h / 2.0)
    cy = np.clip(cy, crop_h / 2.0, src_h - crop_h / 2.0)
    return CropPlan(src_w, src_h, crop_w, crop_h, cx, cy)
