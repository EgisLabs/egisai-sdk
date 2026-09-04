"""MiniLM sentence embedder (allow-only semantic_guard first stage).

Loads a local ONNX export of ``all-MiniLM-L6-v2`` (INT8 / quantized).
Mean-pools the last hidden state with the attention mask, then L2-
normalizes. Constructor-injected session + tokenizer so tests can
swap a fake without the ~22 MB weights.

Never downloads on ``egisai.init()``. Missing files → ``None``
embedder; callers fail open to the LLM judge.
"""

from __future__ import annotations

import logging
import os
import threading
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol

LOGGER = logging.getLogger("egisai.semantic_local")

__all__ = [
    "Embedder",
    "OnnxMiniLMEmbedder",
    "resolve_model_dir",
]

_DEFAULT_DIR = Path("/opt/egisai-models/minilm")
_MODEL_FILES = ("model.onnx", "tokenizer.json")


class Embedder(Protocol):
    def embed(self, texts: Sequence[str]) -> list[list[float]] | None:
        """Return one L2-normalized vector per text, or ``None`` on failure."""


def resolve_model_dir() -> Path | None:
    raw = (os.getenv("EGISAI_SEMANTIC_LOCAL_MODEL_PATH") or "").strip()
    if raw:
        path = Path(raw)
        return path if path.is_dir() else None
    if _DEFAULT_DIR.is_dir():
        return _DEFAULT_DIR
    return None


def _ort_threads() -> int:
    raw = (os.getenv("EGISAI_SEMANTIC_ORT_THREADS") or "1").strip()
    try:
        n = int(raw)
        return n if n > 0 else 1
    except ValueError:
        return 1


class OnnxMiniLMEmbedder:
    """Mean-pool + L2 MiniLM over an ONNX session."""

    def __init__(self, session: Any, tokenizer: Any) -> None:
        self._session = session
        self._tokenizer = tokenizer
        self._lock = threading.Lock()
        inputs = {i.name for i in session.get_inputs()}
        self._input_names = inputs
        self._output_name = session.get_outputs()[0].name

    def embed(self, texts: Sequence[str]) -> list[list[float]] | None:
        if not texts:
            return []
        try:
            import numpy as np
        except Exception:  # noqa: BLE001
            return None
        try:
            encoded = self._tokenizer.encode_batch(list(texts))
            max_len = max(len(e.ids) for e in encoded)
            batch_ids: list[list[int]] = []
            batch_mask: list[list[int]] = []
            batch_types: list[list[int]] = []
            for e in encoded:
                pad = max_len - len(e.ids)
                ids = list(e.ids) + [0] * pad
                mask = list(e.attention_mask) + [0] * pad
                types = list(getattr(e, "type_ids", [0] * len(e.ids))) + [0] * pad
                batch_ids.append(ids)
                batch_mask.append(mask)
                batch_types.append(types)
            feeds: dict[str, Any] = {}
            if "input_ids" in self._input_names:
                feeds["input_ids"] = np.asarray(batch_ids, dtype=np.int64)
            if "attention_mask" in self._input_names:
                feeds["attention_mask"] = np.asarray(batch_mask, dtype=np.int64)
            if "token_type_ids" in self._input_names:
                feeds["token_type_ids"] = np.asarray(batch_types, dtype=np.int64)
            with self._lock:
                outputs = self._session.run([self._output_name], feeds)
            hidden = np.asarray(outputs[0], dtype=np.float32)
            mask_arr = np.asarray(batch_mask, dtype=np.float32)
            if hidden.ndim != 3:
                return None
            mask_exp = mask_arr[:, :, None]
            summed = (hidden * mask_exp).sum(axis=1)
            counts = np.clip(mask_arr.sum(axis=1, keepdims=True), 1e-6, None)
            pooled = summed / counts
            norms = np.linalg.norm(pooled, axis=1, keepdims=True)
            norms = np.clip(norms, 1e-12, None)
            pooled = pooled / norms
            return [row.tolist() for row in pooled]
        except Exception:  # noqa: BLE001 — fail open
            LOGGER.debug("MiniLM embed failed", exc_info=True)
            return None


def try_load() -> OnnxMiniLMEmbedder | None:
    """Load from disk. Returns ``None`` when files or deps are missing."""
    model_dir = resolve_model_dir()
    if model_dir is None:
        return None
    model_path = model_dir / "model.onnx"
    tok_path = model_dir / "tokenizer.json"
    if not model_path.is_file() or not tok_path.is_file():
        return None
    try:
        import onnxruntime as ort  # type: ignore[import-untyped]
        from tokenizers import Tokenizer  # type: ignore[import-untyped]
    except Exception:  # noqa: BLE001
        return None
    try:
        so = ort.SessionOptions()
        threads = _ort_threads()
        so.intra_op_num_threads = threads
        so.inter_op_num_threads = 1
        session = ort.InferenceSession(
            str(model_path),
            sess_options=so,
            providers=["CPUExecutionProvider"],
        )
        tokenizer = Tokenizer.from_file(str(tok_path))
        return OnnxMiniLMEmbedder(session, tokenizer)
    except Exception:  # noqa: BLE001
        LOGGER.debug("MiniLM ONNX load failed", exc_info=True)
        return None
