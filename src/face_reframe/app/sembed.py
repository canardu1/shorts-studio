"""Sentence embeddings for the highlight "complete thought" check.

Uses ``paraphrase-multilingual-MiniLM-L12-v2`` (XLM-RoBERTa, 384-d) as a
quantised ONNX from the Xenova mirror — ~135 MB with the tokenizer, downloaded
once through :mod:`face_reframe.models` and ~10 ms per sentence on CPU.  Works
for both English and Italian (and ~50 other languages).
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np

from .. import models
from ..i18n import tr

MAX_LEN = 128


class SentenceEmbedder:
    """Map sentence strings to L2-normalised embeddings (call it like a fn)."""

    def __init__(self):
        from tokenizers import Tokenizer
        self.tok = Tokenizer.from_file(models.path("minilm_tok"))
        self.tok.enable_truncation(MAX_LEN)
        self.tok.enable_padding()
        self.sess = models.session("minilm")
        self._names = {i.name for i in self.sess.get_inputs()}

    def __call__(self, texts) -> Optional[np.ndarray]:
        if not texts:
            return None
        # sentence embedders expect whole sentences, not word fragments —
        # truncate from the right when a sentence is longer than the window
        enc = self.tok.encode_batch(list(texts))
        ids = np.array([e.ids for e in enc], dtype=np.int64)
        mask = np.array([e.attention_mask for e in enc], dtype=np.int64)
        feed = {"input_ids": ids, "attention_mask": mask}
        if "token_type_ids" in self._names:
            feed["token_type_ids"] = np.zeros_like(ids)
        hidden = self.sess.run(None, feed)[0]          # (n, seq, 384)
        emb = (hidden * mask[..., None]).sum(1) / np.clip(mask.sum(1, keepdims=True), 1, None)
        emb /= np.clip(np.linalg.norm(emb, axis=1, keepdims=True), 1e-9, None)
        return emb.astype(np.float32)


def get(progress: Optional[Callable[[float], None]] = None,
        log: Callable[[str], None] = print) -> Optional[SentenceEmbedder]:
    """Load the embedder, downloading ~135 MB once; None when unavailable —
    the caller's heuristics keep working without it."""
    try:
        for k in ("minilm", "minilm_tok"):
            if models.ensure(k, progress, log) is None:
                return None
        return SentenceEmbedder()
    except Exception as exc:
        log(tr("  sentence embeddings unavailable (%s)") % exc)
        return None
