"""Разбиение длинных участков на чанки для ASR с учётом пауз (RMS), не только по времени."""
from __future__ import annotations

import torch

Frame = tuple[int, int]  # sample indices


def _frame_rms_mono(waveform_1xt: torch.Tensor, frame_samples: int, hop_samples: int) -> torch.Tensor:
    """RMS по коротким кадрам, waveform (1, T) float."""
    x = waveform_1xt.squeeze(0).contiguous()
    if x.numel() < frame_samples:
        return torch.tensor([], dtype=x.dtype, device=x.device)
    cols = x.unfold(0, frame_samples, hop_samples)
    if cols.numel() == 0:
        return torch.tensor([], dtype=x.dtype, device=x.device)
    return cols.pow(2).mean(dim=1).sqrt()


def silence_split_candidates_sec(
    waveform_mono: torch.Tensor,
    sample_rate: int,
    t0_sec: float,
    t1_sec: float,
    *,
    frame_ms: float = 20.0,
    hop_ms: float = 10.0,
    silence_quantile: float = 0.25,
    min_silence_sec: float = 0.28,
) -> list[float]:
    """
    Возвращает времена (сек) внутри [t0_sec, t1_sec], где локально тихо —
    предпочтительные границы резки. Без torchaudio/webrtcvad.
    """
    total = waveform_mono.shape[-1]
    i0 = max(0, int(t0_sec * sample_rate))
    i1 = min(total, int(t1_sec * sample_rate))
    if i1 <= i0 + int(0.05 * sample_rate):
        return []
    sl = waveform_mono[:, i0:i1].float()
    fs = max(int(frame_ms * sample_rate / 1000.0), 160)
    hs = max(int(hop_ms * sample_rate / 1000.0), 80)
    rms = _frame_rms_mono(sl, fs, hs)
    if rms.numel() < 3:
        return []
    thr = float(torch.quantile(rms, silence_quantile))
    thr = max(thr, float(rms.mean()) * 0.35)
    silent = rms < thr
    # центры тихих окон → секунды относительно t0
    hop_sec = hs / float(sample_rate)
    frame_sec = fs / float(sample_rate)
    candidates: list[float] = []
    run_start: int | None = None
    for k in range(silent.numel()):
        if bool(silent[k].item()):
            if run_start is None:
                run_start = k
        else:
            if run_start is not None:
                run_len = k - run_start
                if run_len * hop_sec >= min_silence_sec:
                    mid = run_start + run_len // 2
                    t_mid = t0_sec + mid * hop_sec + 0.5 * frame_sec
                    candidates.append(float(t_mid))
                run_start = None
    if run_start is not None:
        run_len = silent.numel() - run_start
        if run_len * hop_sec >= min_silence_sec:
            mid = run_start + run_len // 2
            t_mid = t0_sec + mid * hop_sec + 0.5 * frame_sec
            candidates.append(float(t_mid))
    return [c for c in candidates if t0_sec + 0.15 < c < t1_sec - 0.15]


def speech_aware_time_chunks(
    t0_sec: float,
    t1_sec: float,
    waveform_mono: torch.Tensor,
    sample_rate: int,
    max_chunk_sec: float,
    *,
    min_chunk_sec: float = 1.5,
) -> list[tuple[float, float]]:
    """
    Чанки не длиннее max_chunk_sec; границы по возможности ставятся на тишину внутри интервала.
    """
    if t1_sec <= t0_sec:
        return []
    total_dur = t1_sec - t0_sec
    if total_dur <= max_chunk_sec + 1e-6:
        return [(t0_sec, t1_sec)]
    cuts = silence_split_candidates_sec(waveform_mono, sample_rate, t0_sec, t1_sec)
    cuts = sorted({c for c in cuts if t0_sec + min_chunk_sec < c < t1_sec - min_chunk_sec})
    chunks: list[tuple[float, float]] = []
    cur = t0_sec
    while cur < t1_sec - 1e-6:
        hard_end = min(cur + max_chunk_sec, t1_sec)
        if hard_end >= t1_sec - 1e-6:
            chunks.append((cur, t1_sec))
            break
        # ищем последний cut в (cur + min_chunk, hard_end]
        best_cut: float | None = None
        low = cur + min_chunk_sec
        for c in cuts:
            if c <= low:
                continue
            if c <= hard_end - min_chunk_sec:
                best_cut = c
            else:
                break
        if best_cut is not None:
            chunks.append((cur, best_cut))
            cur = best_cut
        else:
            chunks.append((cur, hard_end))
            cur = hard_end
    return chunks
