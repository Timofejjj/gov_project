from __future__ import annotations

from typing import Any, TypedDict


class DiarizationState(TypedDict, total=False):
    """Состояние графа. Поля добавляйте сюда при расширении пайплайна."""

    # вход
    local_wav_path: str

    # AssemblyAI
    upload_url: str
    transcript_id: str
    job_status: str

    # результат
    transcript_text: str
    utterances: list[dict[str, Any]]
    raw_transcript: dict[str, Any]

    # ошибки
    error: str
