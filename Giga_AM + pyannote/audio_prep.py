"""Загрузка аудио и проверка ffmpeg/ffprobe (16 kHz mono для диаризации и ASR)."""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import torch

SAMPLE_RATE = 16000


def ensure_ffmpeg_available() -> None:
    """Явная проверка до инференса; понятная ошибка вместо сбоя внутри ffmpeg."""
    for name in ("ffmpeg", "ffprobe"):
        if shutil.which(name) is None:
            raise RuntimeError(
                f"В PATH не найден «{name}». Установите ffmpeg (включая ffprobe) "
                "и добавьте каталог с бинарниками в PATH, затем повторите запуск."
            )
    try:
        subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        )
        subprocess.run(
            ["ffprobe", "-version"],
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        )
    except (subprocess.CalledProcessError, FileNotFoundError, OSError) as e:
        raise RuntimeError(
            "ffmpeg/ffprobe установлены, но не удалось выполнить проверочный запуск. "
            f"Детали: {e}"
        ) from e


def ffprobe_audio_channels(audio_path: str) -> int:
    """Число каналов первого аудиопотока (минимум 1)."""
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "a:0",
        "-show_entries",
        "stream=channels",
        "-of",
        "default=nokey=1:noprint_wrappers=1",
        audio_path,
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=120)
        line = (r.stdout or "").strip().splitlines()
        if not line:
            return 1
        return max(1, min(int(line[0].strip()), 16))
    except (subprocess.CalledProcessError, ValueError, OSError, IndexError):
        return 1


def load_waveform_16k_mono(
    audio_path: str,
    *,
    sample_rate: int = SAMPLE_RATE,
) -> tuple[torch.Tensor, int]:
    """
    Моно 16 kHz float32, форма (1, T). Для pyannote и GigaAM — один и тот же вход
    (стерео при необходимости сводится в ffmpeg, без двухканального тензора).
    """
    _ = ffprobe_audio_channels(audio_path)  # диагностика; ffmpeg всё равно даёт mono
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-threads",
        "0",
        "-i",
        audio_path,
        "-f",
        "s16le",
        "-acodec",
        "pcm_s16le",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-",
    ]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    import numpy as np

    arr = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    wav = torch.from_numpy(arr).unsqueeze(0).float().contiguous()
    return wav, sample_rate
