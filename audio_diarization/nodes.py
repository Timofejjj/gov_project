from __future__ import annotations

import audioop
import os
import threading
import wave
from collections import OrderedDict
from pathlib import Path
from typing import TYPE_CHECKING, Any

import assemblyai as aai
from assemblyai.streaming.v3 import StreamingClient, StreamingClientOptions
from assemblyai.streaming.v3.models import (
    Encoding,
    SpeechModel,
    StreamingEvents,
    StreamingParameters,
    TurnEvent,
)

if TYPE_CHECKING:
    from audio_diarization.state import DiarizationState

_PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Размер чанка PCM при отправке (~100 ms при 16 kHz mono s16le).
_STREAM_CHUNK_BYTES = 3200


def _load_project_dotenv() -> None:
    path = _PROJECT_ROOT / ".env"
    if not path.is_file():
        return
    try:
        from dotenv import load_dotenv

        load_dotenv(path, override=False)
    except ImportError:
        pass
    try:
        # utf-8-sig: если .env сохранён с BOM, первая строка всё равно распарсится как ключ.
        raw = path.read_text(encoding="utf-8-sig")
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
        if not key:
            continue
        # Пустая переменная в окружении (часто из профиля/IDE) не даёт подставить .env при
        # load_dotenv(override=False) и setdefault — подставляем значение из файла.
        existing = os.environ.get(key)
        if existing is None or not str(existing).strip():
            os.environ[key] = val


_load_project_dotenv()


def _api_key() -> str:
    key = os.environ.get("ASSEMBLYAI_API_KEY", "").strip()
    if not key:
        raise RuntimeError(
            "Задайте ASSEMBLYAI_API_KEY (переменная окружения или строка в "
            f"{_PROJECT_ROOT / '.env'})"
        )
    return key


def _wav_to_pcm_s16le_16k_mono(path: str) -> bytes:
    """WAV без сжатия → моно int16 little-endian, 16 kHz (как ожидает streaming PCM)."""
    with wave.open(path, "rb") as wf:
        nch = wf.getnchannels()
        sw = wf.getsampwidth()
        fr = wf.getframerate()
        if wf.getcomptype() != "NONE":
            raise ValueError(
                f"Нужен несжатый PCM WAV (получено compression={wf.getcomptype()!r}): {path}"
            )
        frames = wf.readframes(wf.getnframes())
    if nch == 2:
        frames = audioop.tomono(frames, sw, 0.5, 0.5)
    elif nch != 1:
        raise ValueError(f"Поддерживаются 1 или 2 канала, получено {nch}: {path}")
    if sw != 2:
        frames = audioop.lin2lin(frames, sw, 2)
        sw = 2
    if fr != 16000:
        frames, _ = audioop.ratecv(frames, 2, 1, fr, 16000, None)
    return frames


def _streaming_parameters() -> StreamingParameters:
    return StreamingParameters(
        sample_rate=16_000,
        encoding=Encoding.pcm_s16le,
        speech_model=SpeechModel.whisper_rt,
        language_detection=False,
        speaker_labels=True,
        format_turns=True,
        end_of_turn_confidence_threshold=0.25,
        vad_threshold=0.4,

        min_turn_silence=250,
        max_turn_silence=1280,
    )


def _turn_to_utterance_dict(turn: TurnEvent) -> dict[str, Any]:
    words_out: list[dict[str, Any]] = []
    for w in turn.words:
        wd = w.model_dump() if hasattr(w, "model_dump") else w.dict()
        if turn.speaker_label is not None:
            wd["speaker"] = turn.speaker_label
        words_out.append(wd)
    confs = [float(w.confidence) for w in turn.words] if turn.words else [0.0]
    conf = sum(confs) / len(confs) if confs else 0.0
    start_ms = int(turn.words[0].start) if turn.words else 0
    end_ms = int(turn.words[-1].end) if turn.words else 0
    return {
        "speaker": turn.speaker_label or "",
        "text": turn.transcript,
        "confidence": conf,
        "start": start_ms,
        "end": end_ms,
        "words": words_out,
    }


def _dedupe_utterances_same_turn(utterances: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Стриминг может прислать два end_of_turn на одну реплику: сначала без speaker_label,
    затем с меткой. Ключ (start, end, text) совпадает — оставляем одну запись.
    """
    by_key: OrderedDict[tuple[int, int, str], dict[str, Any]] = OrderedDict()
    for u in utterances:
        key = (int(u["start"]), int(u["end"]), str(u.get("text", "")).strip())
        if key not in by_key:
            by_key[key] = u
            continue
        prev = by_key[key]
        prev_sp = str(prev.get("speaker") or "")
        cur_sp = str(u.get("speaker") or "")
        if cur_sp and not prev_sp:
            by_key[key] = u
        elif prev_sp and not cur_sp:
            continue
        else:
            by_key[key] = u
    return list(by_key.values())


def transcribe_streaming(state: DiarizationState) -> dict[str, Any]:
    """
    Диаризация через AssemblyAI Streaming v3 (Whisper RT) с параметрами как в UI.
    """
    path = state["local_wav_path"]
    try:
        pcm = _wav_to_pcm_s16le_16k_mono(path)
    except Exception as exc:
        return {"error": f"Не удалось прочитать WAV: {exc}", "job_status": "error"}

    lock = threading.Lock()
    turn_events: list[dict[str, Any]] = []
    stream_error: list[str] = []

    def on_turn(_client: StreamingClient, msg: TurnEvent) -> None:
        payload = msg.model_dump() if hasattr(msg, "model_dump") else msg.dict()
        with lock:
            turn_events.append(payload)

    def on_error(_client: StreamingClient, err: Any) -> None:
        with lock:
            stream_error.append(getattr(err, "message", str(err)))

    aai.settings.api_key = _api_key()
    streaming = StreamingClient(
        StreamingClientOptions(api_key=aai.settings.api_key),
    )
    streaming.on(StreamingEvents.Turn, on_turn)
    streaming.on(StreamingEvents.Error, on_error)

    try:
        streaming.connect(_streaming_parameters())
        for i in range(0, len(pcm), _STREAM_CHUNK_BYTES):
            streaming.stream(pcm[i : i + _STREAM_CHUNK_BYTES])
        streaming.disconnect(terminate=True)
    except Exception as exc:
        try:
            streaming.disconnect(terminate=False)
        except Exception:
            pass
        return {"error": str(exc), "job_status": "error"}

    with lock:
        if stream_error:
            return {
                "error": stream_error[0],
                "job_status": "error",
                "raw_transcript": {"streaming_turns": list(turn_events)},
            }

    params = _streaming_parameters()
    if hasattr(params, "model_dump"):
        params_dump = params.model_dump(exclude_none=True, mode="json")
    else:
        params_dump = params.dict(exclude_none=True)

    utterances: list[dict[str, Any]] = []
    for raw in turn_events:
        turn = TurnEvent.model_validate(raw)
        if turn.end_of_turn and (turn.transcript or "").strip():
            utterances.append(_turn_to_utterance_dict(turn))
    utterances = _dedupe_utterances_same_turn(utterances)

    lines = [u["text"] for u in utterances if u.get("text")]
    transcript_text = "\n".join(lines) if lines else ""

    return {
        "job_status": "completed",
        "transcript_text": transcript_text,
        "utterances": utterances,
        "raw_transcript": {
            "streaming_turns": turn_events,
            "streaming_params": params_dump,
        },
    }


def fail_fast(state: DiarizationState) -> dict[str, Any]:
    return {"error": state.get("error") or "unknown AssemblyAI error"}
