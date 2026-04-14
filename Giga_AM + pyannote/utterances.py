"""Сборка utterances для JSON: секунды внутри логики, миллисекунды только в выходе."""
from __future__ import annotations

from typing import Any, List, Optional

from text_normalize import normalize_transcription_text


def seconds_to_json_ms(t_sec: float) -> int:
    """Абсолютное время в секундах → целые миллисекунды для JSON."""
    return int(round(float(t_sec) * 1000.0))


def json_ms_to_seconds(ms: int) -> float:
    return float(ms) / 1000.0


def build_utterances_payload(
    full_text: str,
    words: list | None,
    speaker: str = "—",
    *,
    sec_to_ms=seconds_to_json_ms,
) -> list[dict]:
    """Один спикер и список слов — формат как в transcribed_text из AssemblyAI."""
    full_text = normalize_transcription_text(full_text)
    if not words:
        return [
            {
                "speaker": speaker,
                "text": full_text,
                "confidence": None,
                "start": 0,
                "end": 0,
                "words": [],
            }
        ]

    word_objs: list[dict] = []
    for w in words:
        if isinstance(w, dict):
            text = w.get("text", "")
            start, end = w.get("start"), w.get("end")
            conf = w.get("confidence")
        else:
            text = getattr(w, "text", "") or ""
            start = getattr(w, "start", None)
            end = getattr(w, "end", None)
            conf = getattr(w, "confidence", None)
        word_objs.append(
            {
                "text": text,
                "start": sec_to_ms(float(start)) if start is not None else 0,
                "end": sec_to_ms(float(end)) if end is not None else 0,
                "confidence": conf,
                "speaker": speaker,
            }
        )

    starts = [x["start"] for x in word_objs]
    ends = [x["end"] for x in word_objs]
    t0 = min(starts) if starts else 0
    t1 = max(ends) if ends else 0

    return [
        {
            "speaker": speaker,
            "text": full_text,
            "confidence": None,
            "start": t0,
            "end": t1,
            "words": word_objs,
        }
    ]


def merge_chunk_texts(parts: list[str]) -> str:
    return normalize_transcription_text(" ".join(p for p in parts if p))
