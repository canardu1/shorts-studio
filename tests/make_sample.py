"""Generate a synthetic 1920x1080 clip with a moving face and one hard cut."""

import sys

import cv2
import numpy as np
from skimage import data


def face_x(i, n, w=1920):
    if i < n // 2:
        return int(400 + (w - 800) * i / float(n // 2))
    return 500


def make(path, n=150, fps=25):
    face = cv2.cvtColor(data.astronaut(), cv2.COLOR_RGB2BGR)[0:300, 100:380]
    fh, fw = face.shape[:2]
    out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (1920, 1080))
    for i in range(n):
        bg = (60, 90, 120) if i < n // 2 else (160, 40, 40)
        frame = np.full((1080, 1920, 3), bg, np.uint8)
        x = face_x(i, n) - fw // 2
        frame[300:300 + fh, x:x + fw] = face
        out.write(frame)
    out.release()


def make_wide(path, n=150, fps=25):
    """Close-up then a wide shot of the same set: a small seated speaker on a stool."""
    face = cv2.cvtColor(data.astronaut(), cv2.COLOR_RGB2BGR)[0:300, 100:380]
    big = cv2.resize(face, None, fx=1.6, fy=1.6)
    small = cv2.resize(face, None, fx=0.45, fy=0.45)
    out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (1920, 1080))
    for i in range(n):
        frame = np.full((1080, 1920, 3), (60, 90, 120), np.uint8)
        if i < n // 2:
            ph, pw = big.shape[:2]
            frame[150:150 + ph, 700:700 + pw] = big
        else:
            ph, pw = small.shape[:2]
            x0, y0 = 1250, 380
            frame[y0:y0 + ph, x0:x0 + pw] = small
            cv2.rectangle(frame, (x0 + 10, y0 + ph), (x0 + pw - 10, y0 + ph + 260), (40, 40, 140), -1)
            cv2.rectangle(frame, (x0 + 20, y0 + ph + 260), (x0 + pw - 20, y0 + ph + 420), (20, 20, 20), -1)
        out.write(frame)
    out.release()


if __name__ == "__main__":
    (make_wide if "--wide" in sys.argv else make)(sys.argv[1])
