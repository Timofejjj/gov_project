"""Загрузка pyannote и получение сырых сегментов (start, end, cluster_id)."""
from __future__ import annotations

import torch

_DEFAULT_PYANNOTE_MODELS = (
    "pyannote/speaker-diarization-3.1",
    "pyannote/speaker-diarization-community-1",
)


def pyannote_diarization_turns(
    waveform_mono: torch.Tensor,
    sample_rate: int,
    hf_token: str,
    device: str,
    model_id: str | None,
) -> tuple[list[tuple[float, float, str]], str]:
    """
    waveform_mono: (1, T) float32, 16 kHz mono (рекомендуемый вход для pyannote).
    """
    from pyannote.audio import Pipeline

    dev = torch.device(device)
    wf = waveform_mono.to(dev) if waveform_mono.device != dev else waveform_mono

    candidates = (model_id,) if model_id else _DEFAULT_PYANNOTE_MODELS
    last_err: Exception | None = None
    for mid in candidates:
        try:
            pipeline = Pipeline.from_pretrained(mid, token=hf_token)
            pipeline.to(dev)
            output = pipeline({"uri": "recording", "waveform": wf, "sample_rate": sample_rate})
            diarization = (
                getattr(output, "exclusive_speaker_diarization", None)
                or getattr(output, "speaker_diarization", None)
                or getattr(output, "diarization", None)
                or getattr(output, "annotation", None)
            )
            if diarization is None:
                try:
                    import collections.abc as cabc

                    if isinstance(output, cabc.Mapping):
                        diarization = (
                            output.get("exclusive_speaker_diarization")
                            or output.get("speaker_diarization")
                            or output.get("diarization")
                            or output.get("annotation")
                        )
                except Exception:
                    pass
            diarization = diarization or output
            turns: list[tuple[float, float, str]] = []
            if not hasattr(diarization, "itertracks"):
                raise RuntimeError(
                    f"Unexpected diarization output type: {type(diarization)}"
                )
            for segment, _, label in diarization.itertracks(yield_label=True):
                turns.append((float(segment.start), float(segment.end), str(label)))
            turns.sort(key=lambda x: (x[0], x[1]))
            return turns, mid
        except Exception as e:  # noqa: BLE001
            last_err = e
            continue
    raise RuntimeError(
        f"Не удалось запустить pyannote ({candidates}). Последняя ошибка: {last_err}"
    )
