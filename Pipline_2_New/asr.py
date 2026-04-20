"""Ветка контента: ASR по сегменту (GigaAM или Whisper).

Функции записи WAV и нарезки по времени согласованы с Pipline_1_New/run_pipeline.py
(transcribe_segments_gigaam, _write_wav_pcm16, _time_chunks).
"""

from __future__ import annotations

import os
import tempfile
import wave
from typing import Literal, Optional

import torch

from Pipline_2_New.constants import MAX_ASR_CHUNK_SEC, MIN_UTTERANCE_SEC, SAMPLE_RATE

AsrBackend = Literal["gigaam", "whisper"]


def _time_chunks(t0: float, t1: float, max_sec: float) -> list[tuple[float, float]]:
    if t1 <= t0:
        return []
    chunks: list[tuple[float, float]] = []
    cur = t0
    while cur < t1 - 1e-6:
        nxt = min(cur + max_sec, t1)
        chunks.append((cur, nxt))
        cur = nxt
    return chunks


def _write_wav_pcm16(path: str, mono_float: torch.Tensor, sr: int) -> None:
    x = mono_float.squeeze(0).detach().cpu().float().clamp(-1.0, 1.0)
    x_i16 = (x * 32767.0).round().to(torch.int16).numpy()
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(x_i16.tobytes())


def transcribe_segment_gigaam(
    model,
    wav_mono: torch.Tensor,
    t0: float,
    t1: float,
    *,
    word_timestamps: bool = False,
) -> str:
    total = wav_mono.shape[-1]
    if t1 - t0 < MIN_UTTERANCE_SEC:
        return ""
    text_parts: list[str] = []
    for cs, ce in _time_chunks(t0, t1, MAX_ASR_CHUNK_SEC):
        i0 = max(0, int(cs * SAMPLE_RATE))
        i1 = min(total, int(ce * SAMPLE_RATE))
        if i1 <= i0:
            continue
        seg = wav_mono[:, i0:i1].cpu().float()
        if seg.shape[-1] < int(0.05 * SAMPLE_RATE):
            continue
        fd, tmp_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        try:
            _write_wav_pcm16(tmp_path, seg, SAMPLE_RATE)
            result = model.transcribe(tmp_path, word_timestamps=word_timestamps)
        finally:
            try:
                os.remove(tmp_path)
            except OSError:
                pass
        chunk_text = (result.text if hasattr(result, "text") else str(result)).strip()
        if chunk_text:
            text_parts.append(chunk_text)
    return " ".join(text_parts).strip()


def transcribe_segment_whisper(
    model,
    wav_mono: torch.Tensor,
    t0: float,
    t1: float,
) -> str:
    total = wav_mono.shape[-1]
    if t1 - t0 < MIN_UTTERANCE_SEC:
        return ""
    i0 = max(0, int(t0 * SAMPLE_RATE))
    i1 = min(total, int(t1 * SAMPLE_RATE))
    if i1 <= i0:
        return ""
    seg = wav_mono[:, i0:i1].cpu().float()
    if seg.shape[-1] < int(0.05 * SAMPLE_RATE):
        return ""
    fd, tmp_path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    try:
        _write_wav_pcm16(tmp_path, seg, SAMPLE_RATE)
        r = model.transcribe(tmp_path)
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
    if isinstance(r, dict):
        return str(r.get("text", "")).strip()
    return str(getattr(r, "text", r)).strip()


def load_gigaam(model_name: str, device: str):
    import gigaam

    return gigaam.load_model(model_name, device=device)


def load_whisper(model_size: str = "base", *, device: str = "cpu"):
    try:
        import whisper
    except ImportError as e:
        raise RuntimeError("Для --asr whisper установите пакет openai-whisper") from e
    # whisper ожидает "cuda" | "cpu" (и опционально индекс через переменные окружения)
    dev = device if device.startswith("cuda") or device == "cpu" else "cpu"
    if dev == "mps":
        dev = "cpu"
    return whisper.load_model(model_size, device=dev)


def transcribe_segment(
    backend: AsrBackend,
    model,
    wav_mono: torch.Tensor,
    t0: float,
    t1: float,
) -> str:
    if backend == "gigaam":
        return transcribe_segment_gigaam(model, wav_mono, t0, t1)
    return transcribe_segment_whisper(model, wav_mono, t0, t1)
