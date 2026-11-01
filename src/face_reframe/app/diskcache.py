"""On-disk cache for the slow analyses (faces, transcripts), so reopening the
app — or rendering after the preview — doesn't redo them.

Entries are keyed by the file's path, size and mtime plus the parameters
that change the result, and by :data:`VERSION` (bump it when the analysis
code changes shape)."""

from __future__ import annotations

import hashlib
import os
import pickle
import sys
from typing import Any, Optional

VERSION = 3


def cache_dir() -> str:
    d = os.environ.get("SHORTS_STUDIO_CACHE")
    if not d:
        if sys.platform == "darwin":
            d = os.path.expanduser("~/Library/Caches/ShortsStudio")
        elif os.name == "nt":
            d = os.path.join(os.environ.get("LOCALAPPDATA", os.path.expanduser("~")), "ShortsStudio", "cache")
        else:
            d = os.path.join(os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache")), "shorts_studio")
    os.makedirs(d, exist_ok=True)
    return d


def _key(path: str, kind: str, extra: str) -> Optional[str]:
    try:
        st = os.stat(path)
    except OSError:
        return None
    raw = "%d|%s|%d|%d|%s|%s" % (VERSION, os.path.abspath(path), st.st_size, int(st.st_mtime), kind, extra)
    return os.path.join(cache_dir(), "%s_%s.pkl" % (kind, hashlib.sha1(raw.encode("utf-8")).hexdigest()[:20]))


def load(path: str, kind: str, extra: str = "") -> Any:
    k = _key(path, kind, extra)
    if not k or not os.path.exists(k):
        return None
    try:
        with open(k, "rb") as fh:
            return pickle.load(fh)
    except Exception:
        return None


def save(path: str, kind: str, obj: Any, extra: str = "") -> None:
    k = _key(path, kind, extra)
    if not k:
        return
    tmp = k + ".tmp"
    try:
        with open(tmp, "wb") as fh:
            pickle.dump(obj, fh, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, k)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass


def size_mb() -> float:
    d = cache_dir()
    return sum(os.path.getsize(os.path.join(d, n)) for n in os.listdir(d)) / 1e6


def clear() -> None:
    d = cache_dir()
    for n in os.listdir(d):
        if n.endswith(".pkl") or (n.startswith("proxy_") and n.endswith(".mp4")):
            try:
                os.unlink(os.path.join(d, n))
            except OSError:
                pass
