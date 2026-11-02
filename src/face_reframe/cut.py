"""Command line: jump-cut a file to keep only speech.

    python -m face_reframe.cut in.mp4 -o out.mp4
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from typing import List, Tuple

from .silence import SpeechParams, extract_audio, find_ffmpeg, speech_intervals


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="face_reframe.cut",
                                description="Cut silent pauses out of a video, keeping only speech.")
    p.add_argument("input")
    p.add_argument("-o", "--output")
    p.add_argument("--min-pause", type=float, default=0.35, help="keep pauses shorter than this (s)")
    p.add_argument("--padding", type=float, default=0.16, help="seconds kept around each speech span")
    p.add_argument("--margin-db", type=float, default=8.0, help="dB above the noise floor that counts as speech")
    p.add_argument("--release-db", type=float, default=4.0, help="dB above the noise floor where speech stops (hysteresis)")
    p.add_argument("--min-speech", type=float, default=0.15, help="drop speech blips shorter than this (s)")
    p.add_argument("--engine", choices=("vad", "db"), default="vad",
                   help="speech detection: neural Silero VAD, or dB levels above the noise floor")
    p.add_argument("--json", help="write kept intervals to this JSON file")
    p.add_argument("--no-render", action="store_true", help="only analyse")
    return p


def concat_filter(spans: List[Tuple[float, float]]) -> str:
    parts = []
    for i, (a, b) in enumerate(spans):
        parts.append("[0:v]trim=start=%.3f:end=%.3f,setpts=PTS-STARTPTS[v%d]" % (a, b, i))
        parts.append("[0:a]atrim=start=%.3f:end=%.3f,asetpts=PTS-STARTPTS[a%d]" % (a, b, i))
    join = "".join("[v%d][a%d]" % (i, i) for i in range(len(spans)))
    parts.append("%sconcat=n=%d:v=1:a=1[v][a]" % (join, len(spans)))
    return ";".join(parts)


def render_cut(video: str, spans: List[Tuple[float, float]], out: str) -> str:
    cmd = [find_ffmpeg(), "-y", "-loglevel", "error", "-i", video,
           "-filter_complex", concat_filter(spans), "-map", "[v]", "-map", "[a]",
           "-c:v", "libx264", "-crf", "18", "-preset", "medium", "-pix_fmt", "yuv420p",
           "-c:a", "aac", "-b:a", "192k", out]
    subprocess.run(cmd, check=True)
    return out


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    params = SpeechParams(vad=args.engine == "vad", min_pause=args.min_pause, padding=args.padding,
                          margin_db=args.margin_db, release_db=args.release_db, min_speech=args.min_speech)
    pcm = extract_audio(args.input)
    spans = speech_intervals(pcm, params)
    kept = sum(b - a for a, b in spans)
    sys.stderr.write("speech: %d segment(s), %.1fs kept of %.1fs\n"
                     % (len(spans), kept, len(pcm) / 16000.0))
    if args.json:
        with open(args.json, "w") as fh:
            json.dump({"spans": spans, "kept_seconds": kept}, fh)
    if args.no_render or not spans:
        return 0
    import os
    out = args.output or os.path.splitext(args.input)[0] + "_nopause.mp4"
    render_cut(args.input, spans, out)
    sys.stderr.write("wrote %s\n" % out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
