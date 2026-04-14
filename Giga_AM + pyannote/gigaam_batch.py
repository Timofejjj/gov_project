"""ASR GigaAM: инференс с тензора (без WAV на чанк) и батчинг."""
from __future__ import annotations

from typing import Any, List, Optional, Tuple

import torch

from gigaam.model import LONGFORM_THRESHOLD


@torch.inference_mode()
def transcribe_mono_segments_batch(
    model: Any,
    segments_1d: list[torch.Tensor],
    *,
    word_timestamps: bool,
    batch_size: int = 8,
) -> list[Tuple[str, Optional[List[Any]]]]:
    """
    segments_1d: список 1D float32 тензоров (CPU или CUDA), длина каждого ≤ LONGFORM_THRESHOLD.
    Возвращает список (text, words) в том же порядке.
    """
    if not segments_1d:
        return []
    device = model._device
    dtype = model._dtype
    out: list[Tuple[str, Optional[List[Any]]]] = [("", None)] * len(segments_1d)

    for start in range(0, len(segments_1d), batch_size):
        batch = segments_1d[start : start + batch_size]
        lengths_list = [int(s.shape[-1]) for s in batch]
        for ln in lengths_list:
            if ln > LONGFORM_THRESHOLD:
                raise ValueError(
                    f"Сегмент {ln} сэмплов длиннее порога GigaAM ({LONGFORM_THRESHOLD}); "
                    "уменьшите MAX_ASR_CHUNK_SEC."
                )
        tmax = max(lengths_list)
        wav = torch.zeros(len(batch), tmax, device=device, dtype=dtype)
        length = torch.tensor(lengths_list, device=device, dtype=torch.long)
        for i, s in enumerate(batch):
            sl = s.to(device=device, dtype=dtype)
            wav[i, : sl.shape[-1]] = sl

        encoded, encoded_len = model.forward(wav, length)
        dec_list = model.decoding.decode(model.head, encoded, encoded_len)
        if word_timestamps:
            from gigaam.timestamps_utils import compute_frame_shift, frames_to_words
        for j, (token_ids, token_frames) in enumerate(dec_list):
            idx = start + j
            audio_len = lengths_list[j]
            text = model.decoding.tokenizer.decode(token_ids)
            words_out: Optional[List[Any]] = None
            if word_timestamps:
                el = int(encoded_len[j].item())
                frame_shift = compute_frame_shift(audio_len, el)
                words_out = frames_to_words(
                    model.decoding.tokenizer,
                    token_ids,
                    token_frames,
                    frame_shift,
                )
            out[idx] = (text, words_out)
    return out


def words_to_absolute_dicts(
    words: Optional[List[Any]],
    chunk_t0_sec: float,
    speaker_letter: str,
    sec_to_ms,
) -> list[dict]:
    if not words:
        return []
    rows: list[dict] = []
    for w in words:
        if hasattr(w, "text"):
            wt, ws, we = w.text, w.start, w.end
            conf = getattr(w, "confidence", None)
        else:
            wt = w.get("text", "")
            ws = w.get("start")
            we = w.get("end")
            conf = w.get("confidence")
        if ws is None:
            continue
        abs_s = chunk_t0_sec + float(ws)
        abs_e = chunk_t0_sec + float(we) if we is not None else abs_s
        rows.append(
            {
                "text": wt,
                "start": sec_to_ms(abs_s),
                "end": sec_to_ms(abs_e),
                "confidence": conf,
                "speaker": speaker_letter,
            }
        )
    return rows
