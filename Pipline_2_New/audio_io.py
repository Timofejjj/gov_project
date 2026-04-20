"""Загрузка аудио 16 kHz mono, Silero VAD, эвристики merge/split сегментов.

Логика взята из Pipline_1_New/run_pipeline.py (load_waveform_16k_mono, silero_speech_intervals,
_merge_intervals, _pad_and_split_windows).
"""

from __future__ import annotations

import subprocess
from typing import List, Tuple

import numpy as np
import torch

from Pipline_2_New.constants import (
    MAX_SEGMENT_LEN_SEC,
    MERGE_MAX_GAP_SEC,
    MERGE_MIN_LEN_SEC,
    PAD_SEC,
    SAMPLE_RATE,
    VAD_MIN_SILENCE_MS,
    VAD_MIN_SPEECH_MS,
    VAD_SPEECH_PAD_MS,
)

Interval = Tuple[float, float]


def _ffprobe_audio_channels(audio_path: str) -> int:
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


def load_waveform_16k_mono(audio_path: str) -> torch.Tensor:
    """(1, num_samples) float32 [-1..1], 16 kHz mono."""
    extra_ac = ["-ac", "1"]
    _ffprobe_audio_channels(audio_path)
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
        "-ar",
        str(SAMPLE_RATE),
        *extra_ac,
        "-",
    ]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    arr = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    wav = torch.from_numpy(arr).unsqueeze(0)
    return wav.float().contiguous()


def silero_speech_intervals(
    wav_mono: torch.Tensor,
    sample_rate: int,
    *,
    torch_device: str | torch.device | None = None,
    threshold: float = 0.5,
    min_speech_ms: int = VAD_MIN_SPEECH_MS,
    min_silence_ms: int = VAD_MIN_SILENCE_MS,
    speech_pad_ms: int = VAD_SPEECH_PAD_MS,
) -> List[Interval]:
    """torch_device: cpu | cuda | cuda:0 | mps — весь VAD на указанном устройстве (тишина остаётся на CPU)."""
    model, utils = torch.hub.load(
        repo_or_dir="snakers4/silero-vad",
        model="silero_vad",
        force_reload=False,
        onnx=False,
        trust_repo=True,
    )
    dev = torch.device("cpu")
    if torch_device is not None:
        dev = torch.device(torch_device)
        if dev.type == "cuda" and not torch.cuda.is_available():
            dev = torch.device("cpu")
        if dev.type == "mps":
            if not getattr(torch.backends, "mps", None) or not torch.backends.mps.is_available():
                dev = torch.device("cpu")
    w = wav_mono.squeeze(0).float()
    get_speech_timestamps = utils[0]

    def _run_on(d: torch.device) -> list:
        m = model.to(d)
        ww = w.to(d)
        return get_speech_timestamps(
            ww,
            m,
            sampling_rate=sample_rate,
            threshold=threshold,
            min_speech_duration_ms=min_speech_ms,
            min_silence_duration_ms=min_silence_ms,
            speech_pad_ms=speech_pad_ms,
            return_seconds=True,
        )

    try:
        ts = _run_on(dev)
    except Exception:
        if dev.type != "cpu":
            ts = _run_on(torch.device("cpu"))
        else:
            raise
    return [(float(x["start"]), float(x["end"])) for x in ts]


def merge_intervals(
    intervals: List[Interval],
    *,
    max_gap: float,
    min_len: float,
) -> List[Interval]:
    if not intervals:
        return []
    ints = [(float(a), float(b)) for a, b in intervals if b > a]
    if not ints:
        return []
    ints.sort(key=lambda x: (x[0], x[1]))
    out: List[Interval] = []
    a0, b0 = ints[0]
    for a, b in ints[1:]:
        if a - b0 <= max_gap:
            b0 = max(b0, b)
        else:
            if b0 - a0 >= min_len:
                out.append((a0, b0))
            a0, b0 = a, b
    if b0 - a0 >= min_len:
        out.append((a0, b0))
    return out


def pad_and_split_windows(
    windows: List[Interval],
    *,
    pad: float,
    max_len: float,
    total_dur: float,
) -> List[Interval]:
    out: List[Interval] = []
    for a, b in windows:
        a2 = max(0.0, a - pad)
        b2 = min(total_dur, b + pad)
        if b2 <= a2:
            continue
        cur = a2
        while cur < b2 - 1e-6:
            nxt = min(cur + max_len, b2)
            out.append((cur, nxt))
            cur = nxt
    if not out:
        return out
    out.sort(key=lambda x: (x[0], x[1]))
    non_overlapping: List[Interval] = []
    prev_end = -1.0
    for a, b in out:
        a3 = max(float(a), float(prev_end))
        b3 = float(b)
        if b3 <= a3 + 1e-6:
            continue
        non_overlapping.append((a3, b3))
        prev_end = b3
    return non_overlapping


def vad_to_speech_segments(
    vad_intervals: List[Interval],
    total_dur_sec: float,
) -> List[Interval]:
    merged = merge_intervals(
        vad_intervals,
        max_gap=MERGE_MAX_GAP_SEC,
        min_len=MERGE_MIN_LEN_SEC,
    )
    return pad_and_split_windows(
        merged,
        pad=PAD_SEC,
        max_len=MAX_SEGMENT_LEN_SEC,
        total_dur=total_dur_sec,
    )


def crop_segment(wav: torch.Tensor, t0: float, t1: float, sample_rate: int = SAMPLE_RATE) -> torch.Tensor:
    """Возвращает (1, samples) float32."""
    i0 = max(0, int(t0 * sample_rate))
    i1 = min(int(wav.shape[-1]), int(t1 * sample_rate))
    if i1 <= i0:
        return wav[:, :0]
    return wav[:, i0:i1].contiguous()
