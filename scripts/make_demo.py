#!/usr/bin/env python3
"""Make README demo media from your own footage.

Side-by-side "before → after" GIF (original 16:9 next to the vertical short):

    python scripts/make_demo.py compare original.mp4 short.mp4 -o docs/media/reframe.gif --start 12 --dur 6

Just the short as a GIF (captions, punch-in, moves):

    python scripts/make_demo.py clip short.mp4 -o docs/media/captions.gif --start 3 --dur 5

Thumbnail grid from the images saved by "Copertine…":

    python scripts/make_demo.py grid a.jpg b.jpg c.jpg d.jpg -o docs/media/thumbnails.jpg

GIFs are kept small for GitHub (default 12 fps, 360 px tall, 128-colour palette).
"""

import argparse
import os
import subprocess
import sys


def ffmpeg() -> str:
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        return "ffmpeg"


def run(args):
    subprocess.run([ffmpeg(), "-y", "-loglevel", "error"] + args, check=True)


def gif(filter_graph: str, inputs, out: str, fps: int):
    pal = (f"{filter_graph},fps={fps},split[a][b];[a]palettegen=max_colors=128:stats_mode=diff[p];"
           f"[b][p]paletteuse=dither=bayer:bayer_scale=4")
    run(inputs + ["-filter_complex", pal, "-loop", "0", out])


def compare(a: argparse.Namespace):
    h = a.height
    label = ("drawtext=text='{t}':x=12:y=12:fontsize={fs}:fontcolor=white:box=1:boxcolor=black@0.55:boxborderw=6"
             if a.labels else None)
    fs = max(14, h // 18)
    left = f"[0:v]trim=start={a.start}:duration={a.dur},setpts=PTS-STARTPTS,scale=-2:{h}"
    right = f"[1:v]trim=start={a.start_short}:duration={a.dur},setpts=PTS-STARTPTS,scale=-2:{h}"
    if label:
        left += "," + label.format(t="Before", fs=fs)
        right += "," + label.format(t="After", fs=fs)
    graph = f"{left}[l];{right}[r];[l][r]hstack=inputs=2"
    inputs = ["-i", a.original, "-i", a.short]
    if a.out.endswith(".gif"):
        gif(graph, inputs, a.out, a.fps)
    else:
        run(inputs + ["-filter_complex", graph, "-an", "-c:v", "libx264", "-crf", "23", "-pix_fmt", "yuv420p", a.out])


def clip(a: argparse.Namespace):
    graph = f"[0:v]trim=start={a.start}:duration={a.dur},setpts=PTS-STARTPTS,scale=-2:{a.height}"
    gif(graph, ["-i", a.video], a.out, a.fps)


def grid(a: argparse.Namespace):
    import cv2
    import numpy as np
    imgs = [cv2.imread(p) for p in a.images]
    if any(i is None for i in imgs):
        sys.exit("cannot read one of the images")
    h = a.height
    imgs = [cv2.resize(i, (int(i.shape[1] * h / i.shape[0]), h), interpolation=cv2.INTER_AREA) for i in imgs]
    cols = 2 if len(imgs) == 4 else len(imgs)
    rows = [np.hstack(imgs[r:r + cols]) for r in range(0, len(imgs), cols)]
    w = max(r.shape[1] for r in rows)
    rows = [np.pad(r, ((0, 0), (0, w - r.shape[1]), (0, 0))) for r in rows]
    cv2.imwrite(a.out, np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 88])


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("compare", help="original and short side by side")
    c.add_argument("original")
    c.add_argument("short")
    c.add_argument("--start", type=float, default=0.0, help="start in the original (s)")
    c.add_argument("--start-short", type=float, default=None,
                   help="start in the short (s); defaults to --start (differs if pauses were cut)")
    c.add_argument("--no-labels", dest="labels", action="store_false")
    k = sub.add_parser("clip", help="one video as a GIF")
    k.add_argument("video")
    k.add_argument("--start", type=float, default=0.0)
    g = sub.add_parser("grid", help="thumbnail grid")
    g.add_argument("images", nargs="+")
    for s in (c, k):
        s.add_argument("--dur", type=float, default=6.0)
        s.add_argument("--fps", type=int, default=12)
        s.add_argument("--height", type=int, default=360)
    g.add_argument("--height", type=int, default=270)
    for s in (c, k, g):
        s.add_argument("-o", "--out", required=True)
    a = p.parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    if a.cmd == "compare":
        if a.start_short is None:
            a.start_short = a.start
        try:
            compare(a)
        except subprocess.CalledProcessError:
            if not a.labels:
                raise
            print("this ffmpeg has no drawtext filter: making the GIF without labels")
            a.labels = False
            compare(a)
    elif a.cmd == "clip":
        clip(a)
    else:
        grid(a)
    print(a.out, "%.0f KB" % (os.path.getsize(a.out) / 1024))


if __name__ == "__main__":
    main()
