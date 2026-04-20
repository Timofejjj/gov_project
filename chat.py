import os
import sys
from pathlib import Path

import requests

try:
    from dotenv import load_dotenv  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover
    load_dotenv = None  # type: ignore[assignment]

REPO_ROOT = Path(__file__).resolve().parent
if load_dotenv is not None:
    load_dotenv(REPO_ROOT / ".env")

API_KEY = (os.getenv("OPENAI_API_KEY") or os.getenv("AI_GOV_API_KEY") or "").strip()
if not API_KEY:
    print(
        "Нет ключа: задайте OPENAI_API_KEY (или AI_GOV_API_KEY) в .env рядом с проектом.",
        file=sys.stderr,
    )
    raise SystemExit(2)

MODEL_NAME = (os.getenv("CHAT_MODEL") or os.getenv("LLM_MODEL") or "google/gemma-4-26b-a4b-it").strip()

url = (os.getenv("OPENAI_BASE_URL") or "https://ai.gov.by/api/openai/v1").rstrip("/") + "/chat/completions"

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
}

data = {
    "model": MODEL_NAME,
    "messages": [
        {"role": "user", "content": "Привет! Напиши короткое стихотворение про программиста."}
    ],
}

print("Отправляем запрос, подождите...")
response = requests.post(url, headers=headers, json=data, timeout=120)

try:
    answer = response.json()["choices"][0]["message"]["content"]
    print("\nОтвет нейросети:\n", answer)
except KeyError:
    print("\nЧто-то пошло не так. Ответ сервера:", response.json())
