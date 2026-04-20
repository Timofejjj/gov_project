"""Ветка спикера: ECAPA-TDNN (SpeechBrain), один эмбеддинг на сегмент.

Реализация опирается на Pipline_1_New/run_pipeline.py (EncoderClassifier, _best_subsegment_for_embedding).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from Pipline_2_New.constants import EMBED_FRAME_MS, EMBED_MIN_SEGMENT_SEC, SAMPLE_RATE

REPO_ROOT = Path(__file__).resolve().parent.parent


def best_subsegment_for_embedding(
    x: torch.Tensor,
    sr: int,
    *,
    min_len_sec: float,
    frame_ms: int,
) -> Optional[tuple[int, int]]:
    if x.numel() == 0:
        return None
    x1 = x.squeeze(0).detach().cpu().float()
    n = int(x1.shape[0])
    frame = max(1, int(sr * (frame_ms / 1000.0)))
    hop = frame
    if n < frame:
        return None
    xs = x1[: (n // hop) * hop].view(-1, hop)
    if int(xs.shape[0]) == 0:
        return None
    rms = torch.sqrt(torch.mean(xs * xs, dim=1) + 1e-12)
    rms_med = float(torch.median(rms))
    rms_max = float(torch.max(rms))
    thr = max(2e-4, 0.25 * rms_med, 0.08 * rms_max)
    voiced = (rms >= thr).numpy().astype(np.bool_)
    idx = np.flatnonzero(voiced)
    if idx.size == 0:
        return None
    breaks = np.flatnonzero(np.diff(idx) > 1)
    starts = np.concatenate(([0], breaks + 1))
    ends = np.concatenate((breaks + 1, [idx.size]))
    min_frames = max(1, int(np.ceil((min_len_sec * sr) / hop)))
    energy_segments: list[tuple[int, int]] = []
    for s, e in zip(starts, ends):
        a = int(idx[s])
        b = int(idx[e - 1]) + 1
        if (b - a) >= min_frames:
            energy_segments.append((a * hop, min(n, b * hop)))
    if not energy_segments:
        return None

    def _pitch_hz_acf(seg: torch.Tensor) -> float:
        y = (seg - seg.mean()).float()
        m = int(y.numel())
        if m < int(0.06 * sr):
            return 0.0
        nfft = 1 << (int(m - 1).bit_length())
        Y = torch.fft.rfft(y, n=nfft)
        acf = torch.fft.irfft(Y * torch.conj(Y), n=nfft)[:m]
        acf = acf / (acf[0] + 1e-9)
        min_lag = max(1, int(sr / 350.0))
        max_lag = max(min_lag + 1, int(sr / 70.0))
        if max_lag >= m:
            max_lag = m - 1
        if max_lag - min_lag <= 2:
            return 0.0
        lag = int(torch.argmax(acf[min_lag:max_lag]).item()) + min_lag
        if lag <= 0:
            return 0.0
        return float(sr / lag)

    best: Optional[tuple[int, int]] = None
    best_score = -1.0
    sub_len = max(1, int(round(0.12 * sr)))
    for si0, si1 in energy_segments:
        if si1 - si0 < int(min_len_sec * sr):
            continue
        seg = x1[si0:si1]
        pitches: list[float] = []
        for t in range(0, max(1, int(seg.numel()) - sub_len + 1), sub_len):
            pitches.append(_pitch_hz_acf(seg[t : t + sub_len]))
        if not pitches or all(p <= 0.0 for p in pitches):
            score = float(si1 - si0)
            if score > best_score:
                best_score = score
                best = (si0, si1)
            continue
        p = np.asarray(pitches, dtype=np.float32)
        dp = np.abs(np.diff(p))
        cut = np.flatnonzero(dp > 45.0)
        if cut.size == 0:
            score = float(si1 - si0) / (1.0 + float(np.nanstd(p)))
            if score > best_score:
                best_score = score
                best = (si0, si1)
            continue
        parts = np.split(np.arange(len(p), dtype=np.int32), cut + 1)
        for part in parts:
            if part.size == 0:
                continue
            a = int(part[0] * sub_len)
            b = int((part[-1] + 1) * sub_len)
            ii0 = si0 + a
            ii1 = min(si1, si0 + b)
            if ii1 - ii0 < int(min_len_sec * sr):
                continue
            pv = p[part]
            score = float(ii1 - ii0) / (1.0 + float(np.nanstd(pv)))
            if score > best_score:
                best_score = score
                best = (ii0, ii1)
    return best


_encoder = None
_encoder_device: str | None = None


def get_ecapa_encoder(device: str):
    global _encoder, _encoder_device
    d = str(device).strip()
    if _encoder is not None and _encoder_device == d:
        return _encoder
    from speechbrain.inference.classifiers import EncoderClassifier
    from speechbrain.utils.fetching import FetchConfig, LocalStrategy

    sb_dir = Path(__file__).resolve().parent / "_ecapa_pretrained"
    sb_dir.mkdir(parents=True, exist_ok=True)
    use_hf_token = bool(
        (os.getenv("HUGGING_FACE_HUB_TOKEN") or os.getenv("HF_TOKEN") or "").strip()
    )
    fetch_config = FetchConfig(token=use_hf_token)
    _encoder = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir=str(sb_dir),
        run_opts={"device": d},
        local_strategy=LocalStrategy.COPY,
        fetch_config=fetch_config,
    )
    _encoder_device = d
    return _encoder


def ecapa_embedding_for_segment(
    wav_segment: torch.Tensor,
    *,
    device: str,
    min_rms: float = 2e-4,
) -> Optional[np.ndarray]:
    """wav_segment: (1, N) float32 на CPU или GPU."""
    if wav_segment is None or wav_segment.numel() == 0:
        return None
    dev = torch.device(device)
    x = wav_segment.float().to(dev)
    rms = float(torch.sqrt(torch.mean(x * x) + 1e-12).item())
    if rms < min_rms:
        return None
    sub = best_subsegment_for_embedding(
        x.cpu(),
        SAMPLE_RATE,
        min_len_sec=EMBED_MIN_SEGMENT_SEC,
        frame_ms=EMBED_FRAME_MS,
    )
    if sub is not None:
        si0, si1 = sub
        if si1 - si0 >= int(EMBED_MIN_SEGMENT_SEC * SAMPLE_RATE):
            x = x[:, si0:si1]
    if x.shape[-1] < int(EMBED_MIN_SEGMENT_SEC * SAMPLE_RATE):
        return None
    enc = get_ecapa_encoder(device)
    with torch.inference_mode():
        emb = enc.encode_batch(x)
    vec = emb.squeeze(0).detach().cpu().numpy().reshape(-1)
    return vec.astype(np.float32)
