"""Загрузка .env и проброс HF-токена в окружение для huggingface_hub / SpeechBrain."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

_REPO_ROOT = Path(__file__).resolve().parent.parent


def load_dotenv_repo() -> None:
    """Читает переменные из .env в корне репозитория (идемпотентно)."""
    load_dotenv(_REPO_ROOT / ".env")


def ensure_hf_hub_token(*, log: bool = True, log_fn=None) -> bool:
    """
    Копирует HF_TOKEN (или аналоги из Pipline_1_New) в HUGGING_FACE_HUB_TOKEN,
    чтобы SpeechBrain и huggingface_hub ходили на Hub с аутентификацией.
    """
    load_dotenv_repo()
    raw = (
        os.getenv("HF_TOKEN")
        or os.getenv("HUGGING_FACE_HUB_TOKEN")
        or os.getenv("HF_ACCESS_TOKEN")
        or ""
    ).strip()
    if not raw:
        if log and log_fn:
            log_fn(
                "Hugging Face: в .env не найден HF_TOKEN / HUGGING_FACE_HUB_TOKEN — "
                "загрузки с Hub без токена (ниже лимиты, возможны предупреждения)"
            )
        return False
    os.environ["HUGGING_FACE_HUB_TOKEN"] = raw
    if not (os.getenv("HF_TOKEN") or "").strip():
        os.environ["HF_TOKEN"] = raw
    if log and log_fn:
        log_fn("Hugging Face: токен из .env применён для запросов к Hub")
    return True
