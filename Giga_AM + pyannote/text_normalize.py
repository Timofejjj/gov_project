"""Нормализация текста ASR для артефактов и читаемости."""
from __future__ import annotations

import re
import unicodedata


_WS_RE = re.compile(r"\s+")
_REPEAT_PUNCT_RE = re.compile(r"([.,!?;:…\-—])\1{2,}")


def normalize_transcription_text(s: str) -> str:
    if not s:
        return ""
    t = unicodedata.normalize("NFC", s)
    t = t.replace("\u00a0", " ").strip()
    t = _REPEAT_PUNCT_RE.sub(r"\1\1", t)
    t = _WS_RE.sub(" ", t)
    return t.strip()
