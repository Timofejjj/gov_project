from __future__ import annotations

import os
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import requests

if TYPE_CHECKING:
    from audio_diarization.state import DiarizationState

_PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_project_dotenv() -> None:
    """Подхватывает Compare_Mod/.env даже без установленного python-dotenv."""
    path = _PROJECT_ROOT / ".env"
    if not path.is_file():
        return
    try:
        from dotenv import load_dotenv

        load_dotenv(path, override=False)
    except ImportError:
        pass
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError:
        return
    for line in raw.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, _, val = line.partition("=")
        key, val = key.strip(), val.strip().strip('"').strip("'")
        if key:
            os.environ.setdefault(key, val)


_load_project_dotenv()

BASE_URL = "https://api.assemblyai.com"
POLL_INTERVAL_SEC = 3


def _headers() -> dict[str, str]:
    key = os.environ.get("ASSEMBLYAI_API_KEY", "").strip()
    if not key:
        raise RuntimeError(
            "Задайте ASSEMBLYAI_API_KEY (переменная окружения или строка в "
            f"{_PROJECT_ROOT / '.env'})"
        )
    return {"authorization": key}


def upload_wav(state: DiarizationState) -> dict[str, Any]:
    path = state["local_wav_path"]
    with open(path, "rb") as f:
        r = requests.post(
            f"{BASE_URL}/v2/upload",
            headers=_headers(),
            data=f,
            timeout=120,
        )
    r.raise_for_status()
    return {"upload_url": r.json()["upload_url"]}


def start_transcription(state: DiarizationState) -> dict[str, Any]:
    payload = {
        "audio_url": state["upload_url"],
        "speaker_labels": True,
        "language_detection": True,
        "speech_models": ["universal-3-pro", "universal-2"],
    }
    r = requests.post(
        f"{BASE_URL}/v2/transcript",
        json=payload,
        headers=_headers(),
        timeout=60,
    )
    r.raise_for_status()
    tid = r.json()["id"]
    return {"transcript_id": tid, "job_status": "queued"}


def poll_once(state: DiarizationState) -> dict[str, Any]:
    tid = state["transcript_id"]
    r = requests.get(
        f"{BASE_URL}/v2/transcript/{tid}",
        headers=_headers(),
        timeout=60,
    )
    r.raise_for_status()
    body = r.json()
    status = body["status"]
    out: dict[str, Any] = {"job_status": status, "raw_transcript": body}
    if status == "completed":
        out["transcript_text"] = body.get("text") or ""
        out["utterances"] = body.get("utterances") or []
    elif status == "error":
        err = body.get("error") or body
        out["error"] = str(err)
    return out


def wait_between_polls(_state: DiarizationState) -> dict[str, Any]:
    time.sleep(POLL_INTERVAL_SEC)
    return {}


def fail_fast(state: DiarizationState) -> dict[str, Any]:
    return {"error": state.get("error") or "unknown AssemblyAI error"}
