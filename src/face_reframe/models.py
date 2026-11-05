"""Shared registry for the optional neural models (downloaded on first use).

Everything here lives in :func:`restore.models_dir` next to the face-restoration
models, is fetched lazily through :func:`restore.fetch`, and runs on the CPU
through cached ONNX sessions — no remote calls after the download.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable, Dict, Optional

from .i18n import tr
from .restore import fetch, models_dir

_GH_RAW = "https://raw.githubusercontent.com"
_HF = "https://huggingface.co"


@dataclass(frozen=True)
class Model:
    key: str
    file: str
    url: str
    min_size: int
    """Sanity floor for :func:`available` (bytes) — catches truncated/HTML pages."""
    mb: int
    label: str


MODELS: Dict[str, Model] = {
    # salient-object segmentation; shared with the thumbnail cut-out
    "u2netp": Model("u2netp", "u2netp.onnx",
                    "https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2netp.onnx",
                    4_000_000, 5, "U2-Net-p (saliency)"),
    # low-light enhancement
    "uretinex": Model("uretinex", "uretinex_net_180x320.onnx",
                      _GH_RAW + "/royshil/obs-backgroundremoval/main/data/models/uretinex_net_180x320.onnx",
                      1_000_000, 2, "URetinex-Net (low light)"),
    "sg_llie": Model("sg_llie", "semantic_guided_llie_180x324.onnx",
                     _GH_RAW + "/royshil/obs-backgroundremoval/main/data/models/semantic_guided_llie_180x324.onnx",
                     40_000, 1, "SG-LLIE (low light, fast)"),
    # audio events
    "yamnet": Model("yamnet", "yamnet.onnx",
                    _HF + "/zeropointnine/yamnet-onnx/resolve/main/yamnet.onnx",
                    15_000_000, 16, "YAMNet (audio events)"),
    # speaker voice-prints (raw wav in, 192-d embedding out)
    "ecapa": Model("ecapa", "ecapa-any-length.onnx",
                   _HF + "/kenrunio/transcriber-speaker-onnx/resolve/main/ecapa-any-length.onnx",
                   80_000_000, 84, "ECAPA-TDNN (voice id)"),
    # frame interpolation (slow motion)
    "rife": Model("rife", "rife48_ensemble_True_scale_1_sim.onnx",
                  _HF + "/yuvraj108c/rife-onnx/resolve/main/rife48_ensemble_True_scale_1_sim.onnx",
                  15_000_000, 21, "RIFE 4.8 (slow motion)"),
    # photographic portrait matting — much finer edges than RVM-MobileNet
    "modnet": Model("modnet", "modnet.onnx",
                    "https://github.com/facefusion/facefusion-assets/releases/download/models-3.5.0/modnet.onnx",
                    20_000_000, 24, "MODNet (portrait matting)"),
    # multilingual sentence embeddings — the highlight "is it a complete
    # thought?" scorer (Xenova mirror: quantised for every CPU, ~118 MB)
    "minilm": Model("minilm", "minilm-q8.onnx",
                    _HF + "/Xenova/paraphrase-multilingual-MiniLM-L12-v2/resolve/main/onnx/model_quantized.onnx",
                    100_000_000, 118, "MiniLM sentence embeddings"),
    "minilm_tok": Model("minilm_tok", "minilm-tokenizer.json",
                        _HF + "/Xenova/paraphrase-multilingual-MiniLM-L12-v2/resolve/main/tokenizer.json",
                        10_000_000, 17, "MiniLM tokenizer"),
    # generic object tracking (OpenCV TrackerNano wants two files)
    "nanotrack_backbone": Model("nanotrack_backbone", "nanotrack_backbone_sim.onnx",
                                _GH_RAW + "/HonglinChu/SiamTrackers/master/NanoTrack/models/nanotrackv2/nanotrack_backbone_sim.onnx",
                                900_000, 2, "NanoTrack backbone"),
    "nanotrack_head": Model("nanotrack_head", "nanotrack_head_sim.onnx",
                            _GH_RAW + "/HonglinChu/SiamTrackers/master/NanoTrack/models/nanotrackv2/nanotrack_head_sim.onnx",
                            600_000, 1, "NanoTrack head"),
}

_sessions: Dict[str, object] = {}


def path(key: str) -> str:
    return os.path.join(models_dir(), MODELS[key].file)


def available(key: str) -> bool:
    p = path(key)
    return os.path.exists(p) and os.path.getsize(p) >= MODELS[key].min_size


def download(key: str, progress: Optional[Callable[[float], None]] = None) -> str:
    """Fetch the model once; returns the local path."""
    dst = path(key)
    if available(key):
        return dst
    return fetch(MODELS[key].url, dst, progress)


def session(key: str):
    """Cached CPU-only :class:`onnxruntime.InferenceSession` for the model."""
    if key in _sessions:
        return _sessions[key]
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.log_severity_level = 3
    sess = ort.InferenceSession(path(key), so, providers=["CPUExecutionProvider"])
    _sessions[key] = sess
    return sess


def reset_sessions() -> None:
    """Tests: drop cached sessions when ``SHORTS_STUDIO_MODELS`` is repointed."""
    _sessions.clear()


def ensure(key: str, progress: Optional[Callable[[float], None]] = None,
           log: Callable[[str], None] = print) -> Optional[str]:
    """Download when needed and return the path, or None with a log line."""
    try:
        if not available(key):
            m = MODELS[key]
            log(tr("Downloading %s (%d MB, once)…") % (m.label, m.mb))
            download(key, progress)
        return path(key)
    except Exception as exc:
        log(tr("%s unavailable (%s)") % (MODELS[key].label, exc))
        return None
