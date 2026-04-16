"""
Пайплайн (LangGraph): VAD (Silero) → pyannote segmentation-3.0 → ECAPA (SpeechBrain)
→ кластеризация → опциональное сглаживание → GigaAM ASR.

Выход: JSON-массив объектов { "speaker": "SPEAKER_1", "start", "end", "text" } (время в секундах).

Нужны: ffmpeg в PATH, HF_TOKEN (или HUGGING_FACE_HUB_TOKEN), принятие условий на
https://huggingface.co/pyannote/segmentation-3.0

Установка доп. пакетов:
  .venv\\Scripts\\python.exe -m pip install -r Pipline_1_New\\requirements.txt

Запуск:
  .venv\\Scripts\\python.exe Pipline_1_New\\run_pipeline.py путь\\к\\audio.wav --out result.json
  .venv\\Scripts\\python.exe Pipline_1_New\\run_pipeline.py
    (без аргумента — диалог выбора аудио: tkinter / WinForms на Windows)
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import re
import subprocess
import sys
import tempfile
import warnings
import wave
from pathlib import Path
from typing import Any, Literal, TypedDict

# До импорта torch/torchaudio/pyannote: на Windows torchcodec часто без рабочих DLL —
# pyannote всё равно выдаёт огромный UserWarning при import; мы аудио подаём как waveform.
# SpeechBrain тянет torchaudio и дублирует предупреждение про backends.
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message=r"[\s\S]*torchcodec is not installed correctly[\s\S]*",
)
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message=r"[\s\S]*torchaudio\._backend\.list_audio_backends[\s\S]*",
)
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message=r"[\s\S]*transition TorchAudio into a maintenance phase[\s\S]*",
)

# Снижает шанс принудительного выбора torchcodec-диспетчера в новых torchaudio (если переменная поддерживается).
os.environ.setdefault("TORCHAUDIO_USE_BACKEND_DISPATCHER", "0")

import numpy as np
import torch
from dotenv import load_dotenv
from langgraph.graph import END, StateGraph
from pyannote.audio import Audio, Inference
from pyannote.audio.pipelines.utils import get_model
from pyannote.audio.pipelines.utils.diarization import SpeakerDiarizationMixin
from pyannote.audio.utils.signal import binarize
from pyannote.core import Annotation, Segment, SlidingWindowFeature

REPO_ROOT = Path(__file__).resolve().parent.parent


def _reconstruct_discrete(
    segmentations: SlidingWindowFeature,
    hard_clusters: np.ndarray,
    count: SlidingWindowFeature,
) -> SlidingWindowFeature:
    """Аналог pyannote.audio.pipelines.speaker_diarization.SpeakerDiarization.reconstruct."""
    num_chunks, num_frames, local_num_speakers = segmentations.data.shape
    num_clusters = int(np.max(hard_clusters)) + 1
    clustered_segmentations = np.nan * np.zeros(
        (num_chunks, num_frames, num_clusters),
        dtype=np.float32,
    )
    for c, (cluster, (_, segmentation)) in enumerate(
        zip(hard_clusters, segmentations)
    ):
        for k in np.unique(cluster):
            if k == -2:
                continue
            clustered_segmentations[c, :, k] = np.max(
                segmentation[:, cluster == k], axis=1
            )
    clustered_swf = SlidingWindowFeature(
        clustered_segmentations, segmentations.sliding_window
    )
    return SpeakerDiarizationMixin.to_diarization(clustered_swf, count)


SAMPLE_RATE = 16_000
MAX_ASR_CHUNK_SEC = 20.0
MIN_UTTERANCE_SEC = 0.12
SEGMENTATION_WINDOW_SEC = 5.0  # нужен контекст для segmentation-3.0
SEGMENTATION_STEP_SEC = 0.5    # ~10% от окна
EMBED_MIN_SEGMENT_SEC = 0.6    # 0.4 часто мало для стабильного ECAPA
EMBED_FRAME_MS = 20


def _pipeline_log(node: str, message: str, *, indent: int = 0) -> None:
    """Подробный вывод этапов LangGraph-пайплайна в консоль (UTF-8)."""
    pad = "  " * max(0, indent)
    print(f"[PIPELINE]{pad}[{node}] {message}", flush=True)


def _llm_log(message: str, *, indent: int = 0) -> None:
    _pipeline_log("llm_post", message, indent=indent)


class PipelineGraphState(TypedDict, total=False):
    """Состояние графа LangGraph: входные параметры и промежуточные артефакты."""

    audio_path: str
    device: str
    giga_model_name: str
    clusterer: str
    num_speakers: int | None
    agglomerative_threshold: float
    hdbscan_min_cluster_size: int
    refine_merge_gap: float
    resegment: bool
    skip_vad_trim: bool
    segmentation_step_ratio: float

    hf_token: str
    wav: Any
    vad_intervals: list[tuple[float, float]]
    speech_windows: list[tuple[float, float]]
    packs: list[dict[str, Any]]
    hard_clusters_packs: list[Any]

    turns_named: list[tuple[float, float, str]]
    turns_reseg: list[tuple[float, float, str]]
    asr_rows: list[dict[str, Any]]


def node_load_hf_waveform(state: PipelineGraphState) -> dict[str, Any]:
    """(0) Загружает HF_TOKEN и waveform (16k mono) в состояние графа."""
    audio_path = str(state.get("audio_path") or "").strip()
    device = str(state.get("device") or "cpu").strip() or "cpu"
    if not audio_path:
        raise RuntimeError("audio_path пуст — нечего обрабатывать")

    hf = _hf_token()
    if not hf:
        raise RuntimeError("Нужен HF_TOKEN в окружении или в .env для pyannote/segmentation-3.0")
    _pipeline_log("load_hf_waveform", "HF_TOKEN найден (длина скрыта)")

    _pipeline_log("load_hf_waveform", f"ffmpeg → waveform 16k mono: {audio_path!r}", indent=1)
    wav = load_waveform_16k_mono(audio_path)
    dur_sec = float(wav.shape[-1] / SAMPLE_RATE) if wav.numel() else 0.0
    _pipeline_log(
        "load_hf_waveform",
        f"waveform: shape={tuple(wav.shape)}, ~{dur_sec:.2f} с @ {SAMPLE_RATE} Hz",
        indent=1,
    )
    wav = wav.to(torch.device(device))
    _pipeline_log("load_hf_waveform", "узел завершён → переход к vad", indent=1)
    return {"hf_token": hf, "wav": wav}


def _ensure_utf8_stdio() -> None:
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
            sys.stderr.reconfigure(encoding="utf-8")
        except (AttributeError, OSError):
            pass


def _load_dotenv() -> None:
    load_dotenv(REPO_ROOT / ".env")


def _prompt_num_speakers() -> int | None:
    """Интерактивный ввод количества спикеров (Enter = авто)."""
    try:
        if not sys.stdin or not sys.stdin.isatty():
            return None
    except Exception:
        return None
    try:
        s = input("Введите точное число спикеров (Enter = авто): ").strip()
    except (EOFError, KeyboardInterrupt):
        return None
    if not s:
        return None
    try:
        n = int(s)
    except ValueError:
        return None
    return n if n > 0 else None


def _pick_audio_path_tkinter() -> str | None:
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    try:
        root.attributes("-topmost", True)
    except tk.TclError:
        pass
    path = filedialog.askopenfilename(
        title="Выберите аудиофайл (диаризация)",
        filetypes=[
            ("Аудио", "*.wav *.mp3 *.m4a *.flac *.ogg *.webm"),
            ("WAV", "*.wav"),
            ("Все файлы", "*.*"),
        ],
    )
    root.destroy()
    return path if path else None


def _pick_audio_path_powershell() -> str | None:
    ps = r"""
Add-Type -AssemblyName System.Windows.Forms
$d = New-Object System.Windows.Forms.OpenFileDialog
$d.Title = 'Выберите аудиофайл (диаризация)'
$d.Filter = 'Аудио|*.wav;*.mp3;*.m4a;*.flac;*.ogg;*.webm|WAV|*.wav|Все файлы|*.*'
if ($d.ShowDialog() -eq [System.Windows.Forms.DialogResult]::OK) {
    [Console]::Out.Write($d.FileName)
}
""".strip()
    enc = base64.b64encode(ps.encode("utf-16-le")).decode("ascii")
    try:
        proc = subprocess.run(
            [
                "powershell.exe",
                "-NoProfile",
                "-Sta",
                "-EncodedCommand",
                enc,
            ],
            capture_output=True,
            text=True,
            timeout=3600,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    path = (proc.stdout or "").strip()
    if proc.returncode != 0 or not path:
        return None
    return path


def _pick_audio_path_applescript() -> str | None:
    script = r"""
try
    tell application "System Events" to activate
    set f to choose file with prompt "Выберите аудиофайл (диаризация)"
    return POSIX path of f
on error number -128
    return ""
end try
"""
    try:
        proc = subprocess.run(
            ["osascript", "-e", script],
            capture_output=True,
            text=True,
            timeout=3600,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    path = (proc.stdout or "").strip()
    if proc.returncode != 0 or not path:
        return None
    return path


def _pick_audio_path_stdin() -> str | None:
    print(
        "Диалог недоступен. Введите полный путь к аудиофайлу и нажмите Enter:",
        file=sys.stderr,
    )
    line = sys.stdin.readline()
    if not line:
        return None
    p = Path(line.strip().strip('"').strip("'"))
    return str(p.expanduser().resolve()) if p.expanduser().exists() else None


def pick_audio_path() -> str | None:
    try:
        return _pick_audio_path_tkinter()
    except (ImportError, ModuleNotFoundError, RuntimeError):
        pass

    if sys.platform == "darwin":
        path = _pick_audio_path_applescript()
        if path:
            return path

    if sys.platform == "win32":
        path = _pick_audio_path_powershell()
        if path:
            return path

    return _pick_audio_path_stdin()


def _hf_token() -> str | None:
    return (
        os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        or os.environ.get("HF_ACCESS_TOKEN")
    )


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
    ch_in = _ffprobe_audio_channels(audio_path)
    extra_ac = ["-ac", "1"]  # mono для ECAPA / ASR
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
    threshold: float = 0.5,
    min_speech_ms: int = 250,
    min_silence_ms: int = 150,
    speech_pad_ms: int = 50,
) -> list[tuple[float, float]]:
    """(1) VAD: Silero VAD → список интервалов речи в секундах."""
    model, utils = torch.hub.load(
        repo_or_dir="snakers4/silero-vad",
        model="silero_vad",
        force_reload=False,
        onnx=False,
        trust_repo=True,
    )
    get_speech_timestamps = utils[0]
    ts = get_speech_timestamps(
        wav_mono.squeeze(0),
        model,
        sampling_rate=sample_rate,
        threshold=threshold,
        min_speech_duration_ms=min_speech_ms,
        min_silence_duration_ms=min_silence_ms,
        speech_pad_ms=speech_pad_ms,
        return_seconds=True,
    )
    return [(float(x["start"]), float(x["end"])) for x in ts]


def _merge_intervals(
    intervals: list[tuple[float, float]],
    *,
    max_gap: float,
    min_len: float,
) -> list[tuple[float, float]]:
    if not intervals:
        return []
    ints = [(float(a), float(b)) for a, b in intervals if b > a]
    if not ints:
        return []
    ints.sort(key=lambda x: (x[0], x[1]))
    out: list[tuple[float, float]] = []
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


def _pad_and_split_windows(
    windows: list[tuple[float, float]],
    *,
    pad: float,
    max_len: float,
    total_dur: float,
) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
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
    # ВАЖНО: после паддинга соседние окна могут начать перекрываться (особенно между разными VAD-интервалами).
    # Перекрытие приводит к повторной диаризации/ASR одного и того же времени и “взрыву” числа вызовов ASR.
    out.sort(key=lambda x: (x[0], x[1]))
    non_overlapping: list[tuple[float, float]] = []
    prev_end = -1.0
    for a, b in out:
        a3 = max(float(a), float(prev_end))
        b3 = float(b)
        if b3 <= a3 + 1e-6:
            continue
        non_overlapping.append((a3, b3))
        prev_end = b3
    return non_overlapping


def _annotation_to_turns(ann: Annotation) -> list[tuple[float, float, str]]:
    turns: list[tuple[float, float, str]] = []
    for seg, _, label in ann.itertracks(yield_label=True):
        turns.append((float(seg.start), float(seg.end), str(label)))
    turns.sort(key=lambda x: (x[0], x[1]))
    return turns


def _clip_segment_to_vad(
    t0: float, t1: float, vad: list[tuple[float, float]]
) -> list[tuple[float, float]]:
    if not vad:
        return [(t0, t1)]
    out: list[tuple[float, float]] = []
    for a, b in vad:
        s, e = max(t0, a), min(t1, b)
        if e - s >= MIN_UTTERANCE_SEC:
            out.append((s, e))
    return out


def _merge_short_gaps(turns: list[tuple[float, float, str]], max_gap: float) -> list[tuple[float, float, str]]:
    if not turns:
        return []
    merged: list[tuple[float, float, str]] = [turns[0]]
    for t0, t1, spk in turns[1:]:
        p0, p1, ps = merged[-1]
        if spk == ps and t0 - p1 <= max_gap:
            merged[-1] = (p0, max(p1, t1), ps)
        else:
            merged.append((t0, t1, spk))
    return merged


def _flatten_turns_no_overlap(turns: list[tuple[float, float, str]]) -> list[tuple[float, float, str]]:
    """Убирает временные перекрытия между соседними сегментами (в т.ч. разных спикеров).

    GigaAM вызывается по независимым WAV-чанкам; если два сегмента пересекаются по времени,
    один и тот же кусок аудио может попасть в распознавание дважды.
    """
    if not turns:
        return []
    turns_sorted = sorted(turns, key=lambda x: (x[0], x[1], x[2]))
    cur_t = float("-inf")
    out: list[tuple[float, float, str]] = []
    for t0, t1, spk in turns_sorted:
        s = max(float(t0), cur_t)
        e = float(t1)
        if e - s >= MIN_UTTERANCE_SEC:
            out.append((s, e, spk))
            cur_t = max(cur_t, e)
    return out


_LLM_SPEAKER_SYSTEM = """Ты — эксперт-редактор стенограмм. Твоя цель: превратить сырой ASR-вывод в структурированный диалог.

ОСНОВНЫЕ ПРАВИЛА:
1. РАЗДЕЛЕНИЕ (SPLIT): Если внутри одного сегмента текст начинается с тире '—' или содержит его внутри (например, "— Да. — А вы?"), ты ОБЯЗАН разбить этот сегмент на разные объекты.
2. ИДЕНТИФИКАЦИЯ: Вместо SPEAKER_1, SPEAKER_2 используй ИМЕНА (Анна, Марио) или РОЛИ (Скаут, Менеджер, Фотограф), если они упоминаются в тексте или понятны из контекста.
3. СОХРАННОСТЬ: Не меняй слова, не исправляй грамматику, если это не явная ошибка распознавания (например, "куками" -> "куклами").
4. СМЫСЛ: Следи за логикой. Если один спрашивает, а другой отвечает — это разные люди.

ФОРМАТ ОТВЕТА:
Строгий JSON-массив объектов:
{
  "id": <int>,           // ID исходного сегмента
  "speaker": "<Имя>",    // Конкретное имя или роль
  "text": "<текст>",     // Часть текста, принадлежащая этому спикеру
  "source_ids": [<int>]  // Всегда [id] исходного сегмента
}
"""


_LLM_REFINE_SYSTEM = """Ты — строгий редактор JSON-стенограммы.

У тебя есть:
1) исходный ASR (список реплик с id/start/end/duration_sec/text/speaker)
2) черновик правок после первого прохода (JSON-массив)

Подсказка про число спикеров:
- Во входных метаданных может быть num_speakers_hint. Если это целое число > 0, проверь, что итоговые speaker не противоречат этому числу.
  Если видишь, что всё схлопнулось в одного — постарайся разделить роли/людей; если speaker слишком много — постарайся объединить очевидные дубликаты.

Задача второго прохода:
- Проверь логику диалога и согласованность спикеров.
- Исправь оставшиеся ошибки merge/split (source_ids), если они противоречат времени/смыслу.
- Сохрани текст максимально близко к ASR/черновику.

Верни СТРОГО один JSON-массив в том же формате, что и в первом проходе (см. system первого прохода).
"""


def _truthy_env(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None or str(v).strip() == "":
        return default
    s = str(v).strip().lower()
    if s in {"1", "true", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "no", "n", "off"}:
        return False
    return default


def _extract_json_array(text: str) -> str:
    s = text.strip()
    if "```" in s:
        parts = s.split("```")
        for p in parts:
            pp = p.strip()
            if pp.startswith("{") or pp.startswith("["):
                s = pp
                break
    m = re.search(r"\[[\s\S]*\]\s*$", s)
    if m:
        return m.group(0).strip()
    i = s.find("[")
    j = s.rfind("]")
    if i >= 0 and j > i:
        return s[i : j + 1].strip()
    return s


def _llm_chat_json_array(
    *,
    system: str,
    user: str,
    model: str,
    temperature: float,
    timeout_sec: float,
) -> list[dict]:
    groq_key = (os.getenv("GROQ_API_KEY") or "").strip()
    openai_key = (os.getenv("OPENAI_API_KEY") or "").strip()
    base_url = (os.getenv("OPENAI_BASE_URL") or os.getenv("LLM_BASE_URL") or "").strip() or None

    if groq_key and not openai_key:
        try:
            from groq import Groq  # type: ignore[import-not-found]
        except ImportError as e:
            raise RuntimeError("Для GROQ_API_KEY установите пакет groq (см. Pipline_1_New/requirements.txt)") from e
        client = Groq(api_key=groq_key, timeout=timeout_sec)
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=temperature,
        )
        content = (resp.choices[0].message.content or "").strip()
    else:
        if not openai_key:
            raise RuntimeError("Нужен OPENAI_API_KEY (или только GROQ_API_KEY для Groq SDK)")
        try:
            from openai import OpenAI  # type: ignore[import-not-found]
        except ImportError as e:
            raise RuntimeError("Для LLM установите пакет openai (см. Pipline_1_New/requirements.txt)") from e
        kwargs: dict = {"api_key": openai_key, "timeout": timeout_sec}
        if base_url:
            kwargs["base_url"] = base_url
        client = OpenAI(**kwargs)
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=temperature,
        )
        content = (resp.choices[0].message.content or "").strip()

    raw = _extract_json_array(content)
    data = json.loads(raw)
    if not isinstance(data, list):
        raise RuntimeError("LLM вернул не JSON-массив")
    out: list[dict] = []
    for it in data:
        if isinstance(it, dict):
            out.append(it)
    return out


def llm_speaker_correction(
    rows: list[dict],
    *,
    model: str,
    num_speakers: int | None,
    temperature: float,
    timeout_sec: float,
    second_pass: bool,
) -> list[dict]:
    if not rows:
        return rows

    _llm_log(
        f"старт: входных реплик={len(rows)}, model={model!r}, second_pass={bool(second_pass)}",
        indent=0,
    )

    indexed: list[dict] = []
    for i, r in enumerate(rows, start=1):
        try:
            s = float(r.get("start"))
            e = float(r.get("end"))
        except Exception:
            continue
        txt = str(r.get("text", "")).strip()
        if not txt:
            continue
        indexed.append(
            {
                "id": i,
                "start": round(s, 3),
                "end": round(e, 3),
                "duration_sec": round(max(0.0, e - s), 3),
                "speaker": str(r.get("speaker", "")).strip(),
                "text": txt,
            }
        )
    if not indexed:
        _llm_log("нет валидных строк для отправки (пустые/битые) → пропуск", indent=0)
        return rows

    speakers = sorted({str(x.get("speaker", "")).strip() for x in indexed if str(x.get("speaker", "")).strip()})
    meta_lines = [
        f"model={model}",
        f"num_speakers_hint={num_speakers if num_speakers is not None else 'unknown'}",
        f"speakers_observed={len(speakers)} ({', '.join(speakers)})",
    ]
    user_pass1 = "\n".join(
        [
            "Ниже JSON массива реплик (ASR). Исправь speaker/text.",
            "Метаданные:",
            *meta_lines,
            "",
            json.dumps(indexed, ensure_ascii=False),
        ]
    )

    _llm_log(f"pass1: отправка {len(indexed)} реплик…", indent=0)
    patch1 = _llm_chat_json_array(
        system=_LLM_SPEAKER_SYSTEM,
        user=user_pass1,
        model=model,
        temperature=temperature,
        timeout_sec=timeout_sec,
    )
    _llm_log(f"pass1: получено объектов={len(patch1)}", indent=0)

    patch_final = patch1
    if second_pass:
        draft = json.dumps(patch1, ensure_ascii=False)
        user_pass2 = "\n".join(
            [
                "Сделай второй проход (refine) по черновику.",
                "Метаданные:",
                *meta_lines,
                "",
                "ASR:",
                json.dumps(indexed, ensure_ascii=False),
                "",
                "Черновик (pass1):",
                draft,
            ]
        )
        _llm_log("pass2: refine…", indent=0)
        patch2 = _llm_chat_json_array(
            system=_LLM_REFINE_SYSTEM,
            user=user_pass2,
            model=model,
            temperature=temperature,
            timeout_sec=timeout_sec,
        )
        _llm_log(f"pass2: получено объектов={len(patch2)}", indent=0)
        patch_final = patch2 if patch2 else patch1

    by_id: dict[int, dict] = {}
    for x in indexed:
        try:
            by_id[int(x["id"])] = x
        except Exception:
            continue

    def _parse_src_ids(p: dict) -> list[int] | None:
        src = p.get("source_ids", None)
        if isinstance(src, list) and src:
            ids: list[int] = []
            for it in src:
                if str(it).strip().lstrip("-").isdigit():
                    ids.append(int(it))
            ids = sorted(set(ids))
            return ids if ids and all(i in by_id for i in ids) else None
        pid = p.get("id", None)
        if str(pid).strip().lstrip("-").isdigit():
            i = int(pid)
            return [i] if i in by_id else None
        return None

    class _DSU:
        def __init__(self) -> None:
            self.p: dict[int, int] = {}

        def find(self, x: int) -> int:
            self.p.setdefault(x, x)
            if self.p[x] != x:
                self.p[x] = self.find(self.p[x])
            return self.p[x]

        def union(self, a: int, b: int) -> None:
            ra, rb = self.find(a), self.find(b)
            if ra != rb:
                self.p[rb] = ra

    dsu = _DSU()
    for p in patch_final:
        src = _parse_src_ids(p)
        if not src or len(src) < 2:
            continue
        a0 = src[0]
        for b in src[1:]:
            dsu.union(a0, b)

    merge_root: dict[int, int] = {i: dsu.find(i) for i in by_id.keys()}

    singles: dict[int, list[dict]] = {}
    multis: list[tuple[tuple[int, ...], dict]] = []
    for p in patch_final:
        src = _parse_src_ids(p)
        if not src:
            continue
        if len(src) == 1:
            singles.setdefault(src[0], []).append(p)
        else:
            multis.append((tuple(src), p))

    consumed: set[int] = set()
    rebuilt: list[dict] = []

    def _emit_split(i: int, parts: list[dict]) -> None:
        base = by_id[i]
        t0 = float(base["start"])
        t1 = float(base["end"])
        duration = t1 - t0

        valid_parts = [p for p in parts if str(p.get("text", "")).strip()]
        if not valid_parts:
            rebuilt.append(
                {
                    "speaker": str(base.get("speaker", "")),
                    "start": round(t0, 3),
                    "end": round(t1, 3),
                    "text": str(base.get("text", "")),
                }
            )
            return

        total_chars = sum(len(str(p.get("text", ""))) for p in valid_parts)

        cur_t = t0
        for j, p in enumerate(valid_parts):
            p_text = str(p.get("text", "")).strip()
            p_spk = str(p.get("speaker", "")).strip() or str(base.get("speaker", ""))

            clean_text = re.sub(r"^[—\-\s]+", "", p_text).capitalize()

            share = len(p_text) / total_chars
            seg_dur = share * duration

            seg_end = cur_t + seg_dur
            if j < len(valid_parts) - 1:
                actual_end = max(cur_t + 0.1, seg_end - 0.05)
            else:
                actual_end = t1

            rebuilt.append(
                {
                    "speaker": p_spk,
                    "start": round(cur_t, 3),
                    "end": round(actual_end, 3),
                    "text": clean_text,
                }
            )
            cur_t = actual_end + 0.05

    def _emit_merge(ids: list[int], parts: list[dict]) -> None:
        ids = sorted(set(ids))
        t0 = min(float(by_id[i]["start"]) for i in ids)
        t1 = max(float(by_id[i]["end"]) for i in ids)
        texts: list[str] = []
        spk = ""
        for p in parts:
            tx = str(p.get("text", "")).strip()
            if tx:
                texts.append(tx)
            sp = str(p.get("speaker", "")).strip()
            if sp and not spk:
                spk = sp
        merged_text = " ".join(texts).strip()
        if not merged_text:
            merged_text = " ".join(str(by_id[i].get("text", "")) for i in ids).strip()
        if not spk:
            spk = str(by_id[ids[0]].get("speaker", ""))
        rebuilt.append({"speaker": spk, "start": round(t0, 3), "end": round(t1, 3), "text": merged_text})

    # 1) Обработка merge-компонент (по source_ids длиной > 1 и DSU)
    roots_done: set[int] = set()
    for i in sorted(by_id.keys()):
        r = merge_root[i]
        if r in roots_done:
            continue
        members = sorted({k for k, rr in merge_root.items() if rr == r})
        if len(members) <= 1:
            continue
        roots_done.add(r)

        mparts: list[dict] = []
        for tup, p in multis:
            st = set(tup)
            if st.issubset(set(members)) and len(st) > 1:
                mparts.append(p)
        if not mparts:
            # Нет явного merge-объекта — оставим как отдельные реплики
            continue

        _emit_merge(members, mparts)
        consumed.update(members)

    # 2) Одиночные id: либо update, либо split (несколько объектов на один id)
    for i in sorted(by_id.keys()):
        if i in consumed:
            continue
        parts = singles.get(i, [])
        if len(parts) >= 2:
            _emit_split(i, parts)
            consumed.add(i)
            continue
        if len(parts) == 1:
            p = parts[0]
            base = by_id[i]
            sp = str(p.get("speaker", "")).strip() or str(base.get("speaker", ""))
            tx = str(p.get("text", "")).strip() or str(base.get("text", ""))
            rebuilt.append(
                {
                    "speaker": sp,
                    "start": round(float(base["start"]), 3),
                    "end": round(float(base["end"]), 3),
                    "text": tx,
                }
            )
            consumed.add(i)
            continue

        # Нет явного патча для id — сохраняем как в ASR
        base = by_id[i]
        rebuilt.append(
            {
                "speaker": str(base.get("speaker", "")),
                "start": round(float(base["start"]), 3),
                "end": round(float(base["end"]), 3),
                "text": str(base.get("text", "")),
            }
        )
        consumed.add(i)

    if not rebuilt:
        _llm_log("выход пустой после сборки → оставляем исходные строки", indent=0)
        return rows

    rebuilt.sort(key=lambda r: (float(r["start"]), float(r["end"])))
    _llm_log(f"готово: выходных реплик={len(rebuilt)}", indent=0)
    return rebuilt


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


def transcribe_segments_gigaam(
    model,
    wav_mono: torch.Tensor,
    sample_rate: int,
    turns: list[tuple[float, float, str]],
    *,
    word_timestamps: bool = False,
    pipeline_trace: bool = False,
) -> list[dict]:
    """(7) ASR по сегментам; (3) длинные режем по MAX_ASR_CHUNK_SEC."""
    total = wav_mono.shape[-1]
    rows: list[dict] = []
    n_turns = len(turns)
    for idx, (t0, t1, spk) in enumerate(turns, start=1):
        if pipeline_trace:
            _pipeline_log(
                "gigaam_asr",
                f"ASR сегмент {idx}/{n_turns}: {t0:.3f}–{t1:.3f} с, спикер={spk}, длительность={t1 - t0:.3f} с",
                indent=1,
            )
        if t1 - t0 < MIN_UTTERANCE_SEC:
            if pipeline_trace:
                _pipeline_log(
                    "gigaam_asr",
                    f"  пропуск: короче MIN_UTTERANCE_SEC ({MIN_UTTERANCE_SEC} с)",
                    indent=1,
                )
            continue
        text_parts: list[str] = []
        for cs, ce in _time_chunks(t0, t1, MAX_ASR_CHUNK_SEC):
            i0 = max(0, int(cs * sample_rate))
            i1 = min(total, int(ce * sample_rate))
            if i1 <= i0:
                continue
            seg = wav_mono[:, i0:i1].cpu().float()
            if seg.shape[-1] < int(0.05 * sample_rate):
                continue
            if pipeline_trace:
                _pipeline_log(
                    "gigaam_asr",
                    f"  подчанк ASR {cs:.3f}–{ce:.3f} с → временный WAV → model.transcribe()",
                    indent=1,
                )
            fd, tmp_path = tempfile.mkstemp(suffix=".wav")
            os.close(fd)
            try:
                _write_wav_pcm16(tmp_path, seg, sample_rate)
                result = model.transcribe(tmp_path, word_timestamps=word_timestamps)
            finally:
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass
            chunk_text = (result.text if hasattr(result, "text") else str(result)).strip()
            if chunk_text:
                text_parts.append(chunk_text)
        full_text = " ".join(text_parts).strip()
        if not full_text:
            continue
        rows.append({"speaker": spk, "start": round(t0, 3), "end": round(t1, 3), "text": full_text})
    return rows


def _speaker_label_chronological(turns: list[tuple[float, float, str]]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    n = 0
    for t0, _, lab in sorted(turns, key=lambda x: (x[0], x[1])):
        if lab not in mapping:
            mapping[lab] = f"SPEAKER_{n + 1}"
            n += 1
    return mapping


def run_pipeline_legacy_broken(
    audio_path: str,
    *,
    device: str,
    giga_model_name: str,
    clusterer: str,
    agglomerative_threshold: float,
    hdbscan_min_cluster_size: int,
    refine_merge_gap: float,
    skip_vad_trim: bool,
    segmentation_step_ratio: float,
    use_llm_speaker_correction: bool,
    llm_model: str | None,
    num_speakers: int | None,
) -> list[dict]:
    """
    Устаревший фрагмент, оставлен для истории.

    ВНИМАНИЕ: эта функция не используется; её логика перенесена в node_load_hf_waveform + run_pipeline.
    """
    raise RuntimeError(
        "run_pipeline_legacy_broken не должен вызываться. Используйте run_pipeline() / LangGraph."
    )


def node_vad(state: PipelineGraphState) -> dict[str, Any]:
    _pipeline_log(
        "vad",
        "узел: Silero VAD — поиск интервалов речи (torch.hub silero-vad)",
    )
    wav = state["wav"]
    vad_intervals = silero_speech_intervals(wav.cpu(), SAMPLE_RATE)
    if vad_intervals:
        t0, t1 = vad_intervals[0][0], vad_intervals[-1][1]
        _pipeline_log(
            "vad",
            f"  интервалов речи: {len(vad_intervals)} (первая реплика с {t0:.2f} с, последняя до {t1:.2f} с)",
            indent=1,
        )
    else:
        _pipeline_log("vad", "  интервалов речи: 0 (тишина по VAD)", indent=1)
    _pipeline_log("vad", "узел завершён → переход к speech_windows", indent=1)
    return {"vad_intervals": vad_intervals}


def node_speech_windows(state: PipelineGraphState) -> dict[str, Any]:
    _pipeline_log(
        "speech_windows",
        "узел: VAD → speech-окна (merge/паддинг/сплит). Только по ним считаем segmentation/embeddings",
    )
    vad_intervals = list(state.get("vad_intervals") or [])
    wav = state["wav"]
    total_dur = float(wav.shape[-1] / SAMPLE_RATE)

    # Важно для границ реплик: слишком агрессивный merge/pad склеивает вопрос+ответ в одно окно.
    # Но слишком маленький max_gap дробит речь на слова/микрофрагменты.
    merged = _merge_intervals(vad_intervals, max_gap=0.80, min_len=0.25)
    windows = _pad_and_split_windows(merged, pad=0.05, max_len=45.0, total_dur=total_dur)
    _pipeline_log(
        "speech_windows",
        f"  VAD интервалов: {len(vad_intervals)} → merged: {len(merged)} → окон: {len(windows)}",
        indent=1,
    )
    if not windows:
        _pipeline_log("speech_windows", "  нет speech-окон → ранний выход", indent=1)
    else:
        a0, b0 = windows[0]
        a1, b1 = windows[-1]
        _pipeline_log(
            "speech_windows",
            f"  пример: первое окно {a0:.2f}–{b0:.2f} с; последнее {a1:.2f}–{b1:.2f} с",
            indent=1,
        )
    _pipeline_log("speech_windows", "узел завершён → переход к pyannote_segment", indent=1)
    return {"speech_windows": windows}


def node_pyannote_segment(state: PipelineGraphState) -> dict[str, Any]:
    _pipeline_log(
        "pyannote_segment",
        "узел: pyannote segmentation-3.0 — сегментация/оверлап спикеров",
    )
    hf = state["hf_token"]
    wav = state["wav"]
    device = state["device"]
    audio_path = state["audio_path"]
    _segmentation_step_ratio = state["segmentation_step_ratio"]
    speech_windows: list[tuple[float, float]] = list(state.get("speech_windows") or [])
    _pipeline_log(
        "pyannote_segment",
        f"  загрузка модели pyannote/segmentation-3.0 на {device}…",
        indent=1,
    )

    seg_model = get_model("pyannote/segmentation-3.0", token=hf)
    seg_model.eval()
    seg_model.to(torch.device(device))
    specs = seg_model.specifications
    spec0 = specs[0] if isinstance(specs, tuple) else specs
    # Критично: segmentation работает локально. Длинные окна (≈10s) размывают границы смены спикера.
    duration = float(min(max(0.5, SEGMENTATION_WINDOW_SEC), spec0.duration))
    step = float(SEGMENTATION_STEP_SEC)
    _pipeline_log(
        "pyannote_segment",
        f"  Inference: window={duration:.3f} с, step={step:.4f} с, batch_size=8",
        indent=1,
    )
    seg_inf = Inference(
        seg_model,
        duration=duration,
        step=step,
        batch_size=8,
        device=torch.device(device),
    )
    # Overlapped Speech Detection (OSD): помогает при перебиваниях, чтобы они не “прилипали” к одному спикеру.
    osd = None
    try:
        from pyannote.audio import Pipeline  # type: ignore[import-not-found]

        _pipeline_log(
            "pyannote_segment",
            "  загрузка модели pyannote/overlapped-speech-detection…",
            indent=1,
        )
        osd = Pipeline.from_pretrained(
            "pyannote/overlapped-speech-detection",
            use_auth_token=hf,
        )
        try:
            osd.to(torch.device(device))
        except Exception:
            pass
    except Exception as e:
        _pipeline_log(
            "pyannote_segment",
            f"  OSD недоступен ({e}); overlap detection пропущен",
            indent=1,
        )
    if not speech_windows:
        _pipeline_log(
            "pyannote_segment",
            "  speech_windows пуст → нет чего сегментировать",
            indent=1,
        )
        return {"packs": []}

    _audio = Audio(sample_rate=SAMPLE_RATE, mono="downmix")
    file_full: dict[str, Any] = {"uri": Path(audio_path).stem, "waveform": wav, "sample_rate": SAMPLE_RATE}

    packs: list[dict[str, Any]] = []
    receptive_field = seg_model.receptive_field

    for wi, (w0, w1) in enumerate(speech_windows, start=1):
        _pipeline_log("pyannote_segment", f"  окно {wi}/{len(speech_windows)}: {w0:.2f}–{w1:.2f} с", indent=1)
        seg = Segment(w0, w1)
        w_chunk, _ = _audio.crop(file_full, seg, mode="pad")
        w_chunk = w_chunk.to(torch.device(device))
        fd: dict[str, Any] = {
            "uri": f"{Path(audio_path).stem}__vadwin_{wi}",
            "waveform": w_chunk,
            "sample_rate": SAMPLE_RATE,
        }
        overlap_detected = False
        overlap_intervals: list[tuple[float, float]] = []
        if osd is not None:
            try:
                ov = osd(fd)
                for s in ov.itersegments():
                    overlap_detected = True
                    overlap_intervals.append((float(s.start), float(s.end)))
            except Exception as e:
                _pipeline_log(
                    "pyannote_segment",
                    f"    OSD ошибка ({e}); overlap detection для окна пропущен",
                    indent=1,
                )
        _pipeline_log("pyannote_segment", "    seg_inf(fd) — segmentations…", indent=1)
        segmentations = seg_inf(fd)
        if spec0.powerset:
            binarized = segmentations
        else:
            binarized = binarize(segmentations, onset=0.5, initial_state=False)
        count = SpeakerDiarizationMixin.speaker_count(
            binarized,
            receptive_field,
            warm_up=(0.0, 0.0),
        )
        mx = float(np.nanmax(count.data)) if count is not None else 0.0
        nc, nf, ls = binarized.data.shape
        _pipeline_log(
            "pyannote_segment",
            f"    binarized shape=({nc}, {nf}, {ls}), max(count)={mx:.4f}",
            indent=1,
        )
        packs.append(
            {
                "window": (w0, w1),
                "file_dict": fd,
                "segmentations": segmentations,
                "binarized": binarized,
                "count": count,
                "overlap_detected": overlap_detected,
                "overlap_intervals": overlap_intervals,
            }
        )

    total_chunks = sum(int(p["binarized"].data.shape[0]) for p in packs)
    _pipeline_log(
        "pyannote_segment",
        f"  всего окон: {len(packs)}, всего чанков (по всем окнам): {total_chunks}",
        indent=1,
    )
    _pipeline_log(
        "pyannote_segment",
        "узел завершён → условное ребро _route_after_pyannote (embed | no_speech)",
        indent=1,
    )

    return {"packs": packs}


def _route_after_pyannote(state: PipelineGraphState) -> Literal["no_speech", "embed"]:
    packs = list(state.get("packs") or [])
    if not packs:
        _pipeline_log(
            "route",
            "_route_after_pyannote: packs пуст → ветка no_speech → узел finish_empty",
        )
        return "no_speech"
    mxs: list[float] = []
    for p in packs:
        c = p.get("count")
        if c is None:
            mxs.append(0.0)
        else:
            try:
                mxs.append(float(np.nanmax(c.data)))
            except Exception:
                mxs.append(0.0)
    mx = float(max(mxs) if mxs else 0.0)
    if mx <= 0.0:
        _pipeline_log(
            "route",
            "_route_after_pyannote: max(count)=0 → ветка no_speech → узел finish_empty",
        )
        return "no_speech"
    _pipeline_log(
        "route",
        f"_route_after_pyannote: max(count)={mx:.6f} → ветка embed → узел ecapa_embeddings",
    )
    return "embed"


def node_finish_empty(_state: PipelineGraphState) -> dict[str, Any]:
    _pipeline_log(
        "finish_empty",
        "узел: ранний выход — нет речи для ASR или нет кластеров; asr_rows=[] → END",
    )
    return {"asr_rows": []}


def node_ecapa_embeddings(state: PipelineGraphState) -> dict[str, Any]:
    _pipeline_log(
        "ecapa_embeddings",
        "узел: SpeechBrain ECAPA — эмбеддинги по маскам локальных спикеров",
    )
    from speechbrain.inference.classifiers import EncoderClassifier
    from speechbrain.utils.fetching import LocalStrategy

    packs: list[dict[str, Any]] = list(state.get("packs") or [])
    device = state["device"]

    sb_dir = REPO_ROOT / "Pipline_1_New" / "_ecapa_pretrained"
    sb_dir.mkdir(parents=True, exist_ok=True)
    _pipeline_log(
        "ecapa_embeddings",
        f"  загрузка EncoderClassifier speechbrain/spkrec-ecapa-voxceleb → {sb_dir}",
        indent=1,
    )
    encoder = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir=str(sb_dir),
        run_opts={"device": device},
        local_strategy=LocalStrategy.COPY,
    )
    dev = torch.device(device)
    with torch.inference_mode():
        _probe = torch.zeros(1, int(0.25 * SAMPLE_RATE), device=dev, dtype=torch.float32)
        emb_dim = int(encoder.encode_batch(_probe).shape[-1])

    if not packs:
        _pipeline_log("ecapa_embeddings", "  packs пуст → эмбеддингов нет", indent=1)
        return {"packs": []}

    _audio = Audio(sample_rate=SAMPLE_RATE, mono="downmix")

    def _longest_true_run(mask: np.ndarray) -> tuple[int, int] | None:
        """Вернёт [start,end) для самой длинной непрерывной серии True."""
        if mask.size == 0:
            return None
        m = mask.astype(np.bool_)
        # найти границы серий True
        idx = np.flatnonzero(m)
        if idx.size == 0:
            return None
        # split по разрывам
        breaks = np.flatnonzero(np.diff(idx) > 1)
        starts = np.concatenate(([0], breaks + 1))
        ends = np.concatenate((breaks + 1, [idx.size]))
        best_len = -1
        best: tuple[int, int] | None = None
        for s, e in zip(starts, ends):
            a = int(idx[s])
            b = int(idx[e - 1]) + 1
            ln = b - a
            if ln > best_len:
                best_len = ln
                best = (a, b)
        return best

    def _pick_exclusive_region(
        masks: np.ndarray,  # (num_frames, local_spk)
        s_idx: int,
        *,
        thr_on: float = 0.35,
        thr_margin: float = 0.10,
        min_frames: int = 5,
    ) -> np.ndarray:
        m = np.nan_to_num(masks, nan=0.0).astype(np.float32)
        own = m[:, s_idx]
        if m.shape[1] > 1:
            other_max = np.max(np.delete(m, s_idx, axis=1), axis=1)
        else:
            other_max = np.zeros_like(own)
        score = own - other_max
        keep = (own >= thr_on) & (score >= thr_margin)
        if int(np.sum(keep)) < min_frames:
            keep = own >= (thr_on + 0.10)
        return keep.astype(np.bool_)

    def _best_subsegment_for_embedding(
        x: torch.Tensor,  # (1, samples)
        sr: int,
        *,
        min_len_sec: float,
        frame_ms: int,
    ) -> tuple[int, int] | None:
        """
        Агрессивный split ДО embeddings:
        - split по паузам / energy drop (RMS по фреймам)
        - split по резкой смене pitch (грубая ACF через FFT)
        Возвращает [i0, i1) в сэмплах относительно x.
        """
        if x.numel() == 0:
            return None
        x1 = x.squeeze(0).detach().cpu().float()
        n = int(x1.shape[0])
        frame = max(1, int(sr * (frame_ms / 1000.0)))
        hop = frame
        if n < frame:
            return None
        # Energy (RMS)
        xs = x1[: (n // hop) * hop].view(-1, hop)
        if int(xs.shape[0]) == 0:
            return None
        rms = torch.sqrt(torch.mean(xs * xs, dim=1) + 1e-12)  # (T,)
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
            b = int(idx[e - 1]) + 1  # [a,b) в фреймах
            if (b - a) >= min_frames:
                energy_segments.append((a * hop, min(n, b * hop)))
        if not energy_segments:
            return None

        def _pitch_hz_acf(seg: torch.Tensor) -> float:
            # Возвращает pitch в Hz или 0.0 если не удалось.
            y = (seg - seg.mean()).float()
            m = int(y.numel())
            if m < int(0.06 * sr):
                return 0.0
            # FFT autocorrelation: irfft(|rfft(y)|^2)
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

        best: tuple[int, int] | None = None
        best_score = -1.0
        # Оцениваем pitch по под-окнам ~120ms и режем, если скачки > 45 Hz
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

    for pi, p in enumerate(packs, start=1):
        binarized = p["binarized"]
        file_dict = p["file_dict"]
        num_chunks, num_frames, local_spk = binarized.data.shape
        embeddings = np.full((num_chunks, local_spk, emb_dim), np.nan, dtype=np.float32)
        overlap_flags = np.zeros((num_chunks, local_spk), dtype=np.uint8)
        _pipeline_log(
            "ecapa_embeddings",
            f"  окно {pi}/{len(packs)}: чанков={num_chunks}, local_spk={local_spk}, emb_dim={emb_dim}",
            indent=1,
        )

        for c, (chunk, masks) in enumerate(binarized):
            waveform, _ = _audio.crop(file_dict, chunk, mode="pad")
            w = waveform.float().to(device)
            chunk_len_sec = float(w.shape[-1] / SAMPLE_RATE)
            rel_centers = (np.arange(num_frames, dtype=np.float64) + 0.5) * (
                chunk_len_sec / max(num_frames, 1)
            )

            for s_idx in range(local_spk):
                keep = _pick_exclusive_region(masks, s_idx)
                run = _longest_true_run(keep)
                if run is None:
                    continue
                f0, f1 = run  # [f0,f1)
                if (f1 - f0) < max(3, int(0.03 * num_frames)):
                    continue

                m = np.nan_to_num(masks, nan=0.0).astype(np.float32)
                own = m[:, s_idx]
                if m.shape[1] > 1:
                    other_max = np.max(np.delete(m, s_idx, axis=1), axis=1)
                else:
                    other_max = np.zeros_like(own)
                # overlap proxy на выбранном run: если "другие" тоже сильные — не используем как основной embedding
                ov_frac = float(np.mean(other_max[f0:f1] >= 0.35))
                if ov_frac >= 0.20:
                    overlap_flags[c, s_idx] = 1
                    continue

                # Берём «чистый» (не маскированный) участок по эксклюзивным кадрам,
                # чтобы не делать ECAPA на подавленном сигнале как единственном источнике.
                t0 = float(rel_centers[f0])
                t1 = float(rel_centers[f1 - 1]) if (f1 - 1) > f0 else float(
                    rel_centers[f0] + (chunk_len_sec / max(num_frames, 1))
                )
                t0 = max(0.0, t0 - 0.05)
                t1 = min(chunk_len_sec, t1 + 0.05)
                i0 = max(0, int(t0 * SAMPLE_RATE))
                i1 = min(int(w.shape[-1]), int(t1 * SAMPLE_RATE))
                if i1 - i0 < int(EMBED_MIN_SEGMENT_SEC * SAMPLE_RATE):
                    continue
                x = w[:, i0:i1]
                # очень тихие/шумные куски пропускаем
                rms = float(torch.sqrt(torch.mean(x**2)))
                if rms < 2e-4:
                    continue
                sub = _best_subsegment_for_embedding(
                    x,
                    SAMPLE_RATE,
                    min_len_sec=EMBED_MIN_SEGMENT_SEC,
                    frame_ms=EMBED_FRAME_MS,
                )
                if sub is None:
                    continue
                si0, si1 = sub
                if si1 - si0 < int(EMBED_MIN_SEGMENT_SEC * SAMPLE_RATE):
                    continue
                x = x[:, si0:si1]
                with torch.inference_mode():
                    emb = encoder.encode_batch(x)
                vec = emb.squeeze(0).detach().cpu().numpy().reshape(-1)
                embeddings[c, s_idx] = vec.astype(np.float32)

        p["embeddings"] = embeddings
        p["overlap_flags"] = overlap_flags
        p["emb_dim"] = emb_dim
        p["local_spk"] = int(local_spk)
        p["num_chunks"] = int(num_chunks)

    _pipeline_log("ecapa_embeddings", "узел завершён → переход к cluster", indent=1)
    return {"packs": packs}


def node_cluster(state: PipelineGraphState) -> dict[str, Any]:
    _pipeline_log(
        "cluster",
        "узел: кластеризация эмбеддингов (hdbscan | agglomerative + spectral)",
    )
    packs: list[dict[str, Any]] = list(state.get("packs") or [])
    clusterer = state["clusterer"]
    num_speakers = state["num_speakers"]
    agglomerative_threshold = state["agglomerative_threshold"]
    hdbscan_min_cluster_size = state["hdbscan_min_cluster_size"]

    active: list[tuple[int, int, int]] = []  # (pack_idx, chunk_idx, local_spk_idx)
    flat: list[np.ndarray] = []
    for p_idx, p in enumerate(packs):
        emb = p.get("embeddings")
        if emb is None:
            continue
        emb = np.asarray(emb)
        if emb.ndim != 3:
            continue
        num_chunks, local_spk, emb_dim = int(emb.shape[0]), int(emb.shape[1]), int(emb.shape[2])
        for c in range(num_chunks):
            for s in range(local_spk):
                v = emb[c, s]
                if np.any(np.isnan(v)):
                    continue
                flat.append(v)
                active.append((p_idx, c, s))
    flat_arr = np.stack(flat, axis=0) if flat else np.zeros((0, 1), dtype=np.float32)
    _pipeline_log(
        "cluster",
        f"  активных векторов для кластеризации: {flat_arr.shape[0]} (по всем speech-окнам)",
        indent=1,
    )

    hard_clusters = np.full((num_chunks, local_spk), -2, dtype=np.int64)
    if flat_arr.shape[0] == 0:
        _pipeline_log("cluster", "  нет валидных эмбеддингов → no_clusters", indent=1)
        return {"packs": packs, "hard_clusters_packs": []}

    X = flat_arr - flat_arr.mean(axis=0, keepdims=True)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    X = flat_arr / norms

    if clusterer == "hdbscan":
        _pipeline_log("cluster", f"  режим: hdbscan (min_cluster_size≥{max(2, hdbscan_min_cluster_size)})", indent=1)
        try:
            import hdbscan  # type: ignore[import-untyped]
        except ImportError as e:
            raise RuntimeError("Для --clusterer hdbscan установите пакет hdbscan") from e
        cl = hdbscan.HDBSCAN(
            min_cluster_size=max(2, hdbscan_min_cluster_size),
            metric="euclidean",
            cluster_selection_epsilon=0.0,
        )
        labels = cl.fit_predict(X)
        noise = labels == -1
        if np.all(noise):
            labels = np.zeros(len(labels), dtype=np.int64)
        else:
            next_id = int(labels.max()) + 1
            for i in range(len(labels)):
                if labels[i] == -1:
                    labels[i] = next_id
                    next_id += 1
        labels = np.asarray(labels, dtype=np.int64)
        _pipeline_log(
            "cluster",
            f"  hdbscan: уникальных меток после обработки шума: {len(set(labels.tolist()))}",
            indent=1,
        )
    else:
        from sklearn.cluster import AgglomerativeClustering
        from sklearn.metrics.pairwise import cosine_similarity

        _pipeline_log(
            "cluster",
            f"  режим: agglomerative (однозначно, без Spectral); num_speakers={num_speakers!r}, agg_threshold={agglomerative_threshold}",
            indent=1,
        )
        n = X.shape[0]
        if n == 1:
            _pipeline_log("cluster", "  один вектор → один кластер", indent=1)
            labels = np.zeros((1,), dtype=np.int64)
        else:
            S = cosine_similarity(X)
            D = 1.0 - np.clip(S, -1.0, 1.0)
            np.fill_diagonal(D, 0.0)
            if num_speakers is not None and int(num_speakers) > 0:
                _pipeline_log(
                    "cluster",
                    f"  AgglomerativeClustering(n_clusters={int(num_speakers)}, metric=precomputed, linkage=average)",
                    indent=1,
                )
                agg = AgglomerativeClustering(
                    n_clusters=int(num_speakers),
                    linkage="average",
                    metric="precomputed",
                )
                labels = agg.fit_predict(D).astype(np.int64)
            else:
                _pipeline_log(
                    "cluster",
                    "  AgglomerativeClustering(distance_threshold, metric=precomputed, linkage=average)",
                    indent=1,
                )
                agg = AgglomerativeClustering(
                    n_clusters=None,
                    distance_threshold=agglomerative_threshold,
                    linkage="average",
                    metric="precomputed",
                )
                labels = agg.fit_predict(D).astype(np.int64)

    # Разворачиваем метки обратно в packs → hard_clusters по каждому speech-окну
    hard_clusters_packs: list[np.ndarray] = []
    for p in packs:
        b = p["binarized"]
        nc, _nf, ls = b.data.shape
        hard_clusters_packs.append(np.full((int(nc), int(ls)), -2, dtype=np.int64))

    for i, (p_idx, c, s) in enumerate(active):
        hard_clusters_packs[p_idx][c, s] = int(labels[i])

    hc_max = int(max((int(np.max(hc)) for hc in hard_clusters_packs), default=-1))
    _pipeline_log(
        "cluster",
        f"  итог: max id кластера (по всем окнам)={hc_max}",
        indent=1,
    )
    _pipeline_log(
        "cluster",
        "узел завершён → условное ребро _route_after_cluster (diarize | no_clusters)",
        indent=1,
    )

    return {"packs": packs, "hard_clusters_packs": hard_clusters_packs}


def node_resegment(state: PipelineGraphState) -> dict[str, Any]:
    """
    Пост-кластерная resegmentation (VB-HMM-like по эффекту, но лёгкая):
    - строим постериоры по глобальным кластерам на сетке кадров из binarized masks
    - делаем Viterbi со штрафом за переключение (в overlap-окнах штраф ниже)
    - конвертируем обратно в turns по времени
    """
    if not bool(state.get("resegment", True)):
        _pipeline_log("resegment", "узел отключён (resegment=False) → пропуск", indent=0)
        return {"turns_reseg": []}

    packs: list[dict[str, Any]] = list(state.get("packs") or [])
    hard_clusters_packs: list[np.ndarray] = list(state.get("hard_clusters_packs") or [])
    if not packs or not hard_clusters_packs or len(packs) != len(hard_clusters_packs):
        _pipeline_log("resegment", "нет packs/hard_clusters_packs → пропуск", indent=0)
        return {"turns_reseg": []}

    k_max = int(max((int(np.max(hc)) for hc in hard_clusters_packs), default=-1))
    if k_max < 0:
        _pipeline_log("resegment", "max(cluster_id)<0 → пропуск", indent=0)
        return {"turns_reseg": []}
    K = k_max + 1
    _pipeline_log("resegment", f"узел: resegmentation по {K} кластерам", indent=0)

    turns_all: list[tuple[float, float, str]] = []
    eps = 1e-6
    min_on = 0.10  # сек, отсекаем микросегменты после Viterbi

    for pi, (p, hc) in enumerate(zip(packs, hard_clusters_packs), start=1):
        w0, w1 = p["window"]
        binarized = p["binarized"]  # SlidingWindowFeature iterable по чанкам
        overlap = bool(p.get("overlap_detected"))

        # Параметры Viterbi: уменьшаем "залипание", чтобы легче переключаться на короткие реплики/перебивания
        stay_bias = 1.4 if not overlap else 1.2
        switch_bias = 0.2
        silence_bias = 0.6  # склонность к "тишине" если постериоры слабые

        for c, (chunk, masks) in enumerate(binarized):
            masks = np.nan_to_num(np.asarray(masks, dtype=np.float32), nan=0.0)  # (F, local_spk)
            F = int(masks.shape[0])
            LS = int(masks.shape[1]) if masks.ndim == 2 else 0
            if F <= 1 or LS <= 0:
                continue

            # постериоры по глобальным кластерам
            post = np.zeros((F, K), dtype=np.float32)
            for s in range(LS):
                k = int(hc[c, s])
                if k < 0 or k >= K:
                    continue
                post[:, k] = np.maximum(post[:, k], masks[:, s])
            mx = np.max(post, axis=1)  # (F,)
            # "тишина" как остаточная вероятность
            sil = np.clip(1.0 - mx, 0.0, 1.0).astype(np.float32)

            # Эмиссии в log-space для Viterbi: states = K + silence
            loge = np.log(np.clip(post, eps, 1.0))
            log_sil = np.log(np.clip(sil, eps, 1.0))
            loge = np.concatenate([loge, log_sil[:, None]], axis=1)  # (F, K+1)
            S = K + 1

            # Матрица переходов: предпочитаем оставаться, разрешаем смены; silence отдельно
            trans = np.full((S, S), -stay_bias, dtype=np.float32)
            np.fill_diagonal(trans, 0.0)
            trans[:, :] += switch_bias
            # переходы в/из silence
            trans[:, K] -= silence_bias
            trans[K, :] -= silence_bias
            trans[K, K] = 0.0

            # Viterbi
            dp = np.full((F, S), -1e9, dtype=np.float32)
            bp = np.zeros((F, S), dtype=np.int32)
            dp[0] = loge[0]
            for t in range(1, F):
                scores = dp[t - 1][:, None] + trans  # (S,S)
                bp[t] = np.argmax(scores, axis=0).astype(np.int32)
                dp[t] = scores[bp[t], np.arange(S, dtype=np.int32)] + loge[t]
            st = np.zeros((F,), dtype=np.int32)
            st[-1] = int(np.argmax(dp[-1]))
            for t in range(F - 1, 0, -1):
                st[t - 1] = bp[t, st[t]]

            # Время кадра
            chunk_len_sec = float((chunk.end - chunk.start))
            dt = chunk_len_sec / max(F, 1)
            # Схлопываем в сегменты (игнорируя silence=K)
            cur = st[0]
            seg_start = 0
            for t in range(1, F + 1):
                nxt = st[t] if t < F else -999
                if nxt != cur:
                    if cur != K:
                        a = float(chunk.start + seg_start * dt)
                        b = float(chunk.start + t * dt)
                        if b - a >= min_on:
                            turns_all.append((a + float(w0), b + float(w0), str(cur)))
                    cur = nxt
                    seg_start = t

        _pipeline_log(
            "resegment",
            f"  окно {pi}/{len(packs)}: overlap={overlap} → turns_accum={len(turns_all)}",
            indent=1,
        )

    if not turns_all:
        _pipeline_log("resegment", "результат пуст → пропуск", indent=0)
        return {"turns_reseg": []}

    turns_all.sort(key=lambda x: (x[0], x[1]))
    lab_map = _speaker_label_chronological(turns_all)
    turns_named = [(a, b, lab_map.get(lab, lab)) for a, b, lab in turns_all]
    _pipeline_log("resegment", f"узел завершён: turns_reseg={len(turns_named)}", indent=0)
    return {"turns_reseg": turns_named}


def _route_after_cluster(state: PipelineGraphState) -> Literal["no_clusters", "diarize"]:
    hcs = list(state.get("hard_clusters_packs") or [])
    if not hcs:
        _pipeline_log(
            "route",
            "_route_after_cluster: max(hard_clusters)<0 → ветка no_clusters → finish_empty",
        )
        return "no_clusters"
    m = int(max((int(np.max(x)) for x in hcs), default=-1))
    if m < 0:
        _pipeline_log(
            "route",
            "_route_after_cluster: max(hard_clusters)<0 → ветка no_clusters → finish_empty",
        )
        return "no_clusters"
    _pipeline_log(
        "route",
        f"_route_after_cluster: max(hard_clusters)={m} → ветка diarize → узел build_turns",
    )
    return "diarize"


def node_build_turns(state: PipelineGraphState) -> dict[str, Any]:
    _pipeline_log(
        "build_turns",
        "узел: сбор реплик — _reconstruct_discrete → to_annotation → SPEAKER_* → опционально merge/VAD-обрезка",
    )
    turns_reseg = list(state.get("turns_reseg") or [])
    if turns_reseg:
        _pipeline_log(
            "build_turns",
            f"  resegmentation включён: используем turns_reseg={len(turns_reseg)} (пропуск reconstruct/to_annotation)",
            indent=1,
        )
        turns_named = turns_reseg
    else:
        turns_named = None
    packs: list[dict[str, Any]] = list(state.get("packs") or [])
    hard_clusters_packs: list[np.ndarray] = list(state.get("hard_clusters_packs") or [])
    refine_merge_gap = state["refine_merge_gap"]
    skip_vad_trim = state["skip_vad_trim"]
    vad_intervals = state["vad_intervals"]
    if turns_named is None and (not packs or not hard_clusters_packs or len(packs) != len(hard_clusters_packs)):
        _pipeline_log("build_turns", "  нет packs/hard_clusters_packs → turns пустые", indent=1)
        return {"turns_named": []}

    if turns_named is None:
        all_turns: list[tuple[float, float, str]] = []
        for pi, (p, hc) in enumerate(zip(packs, hard_clusters_packs), start=1):
            w0, w1 = p["window"]
            segmentations = p["segmentations"]
            count = p["count"]
            if bool(p.get("overlap_detected")):
                # При перебиваниях не “обрезаем” speaker count до 1 — иначе overlap никогда не проявится (max(count)=1.0),
                # и смена спикера внутри окна подавляется.
                count_exc = count
            else:
                count_exc = SlidingWindowFeature(
                    np.minimum(count.data, 1).astype(np.int8),
                    count.sliding_window,
                )
            _pipeline_log(
                "build_turns",
                f"  окно {pi}/{len(packs)}: reconstruct + to_annotation (сдвиг на {w0:.2f} с)",
                indent=1,
            )
            discrete_exc = _reconstruct_discrete(segmentations, hc, count_exc)
            ann = SpeakerDiarizationMixin.to_annotation(
                discrete_exc,
                min_duration_on=0.0,
                min_duration_off=0.1,
            )
            turns = _annotation_to_turns(ann)
            for a, b, x in turns:
                all_turns.append((a + float(w0), b + float(w0), str(x)))

        _pipeline_log("build_turns", f"  turns всего (после сдвига): {len(all_turns)}", indent=1)

        lab_map = _speaker_label_chronological(
            [(a, b, str(int(x)) if str(x).isdigit() else str(x)) for a, b, x in all_turns]
        )
        turns_named = []
        for a, b, x in all_turns:
            k = str(int(x)) if str(x).isdigit() else str(x)
            turns_named.append((a, b, lab_map.get(k, k)))

    if refine_merge_gap > 0:
        _pipeline_log(
            "build_turns",
            f"  _merge_short_gaps(gap={refine_merge_gap} с)",
            indent=1,
        )
        turns_named = _merge_short_gaps(turns_named, refine_merge_gap)

    if not skip_vad_trim and vad_intervals:
        clipped: list[tuple[float, float, str]] = []
        _pipeline_log(
            "build_turns",
            f"  обрезка реплик по VAD: {len(vad_intervals)} интервалов речи",
            indent=1,
        )
        for a, b, spk in turns_named:
            for ca, cb in _clip_segment_to_vad(a, b, vad_intervals):
                clipped.append((ca, cb, spk))
        turns_named = clipped
    elif skip_vad_trim:
        _pipeline_log("build_turns", "  обрезка по VAD отключена (skip_vad_trim)", indent=1)

    # ВАЖНО: убираем временные нахлёсты между сегментами, чтобы ASR не распознавал одно и то же дважды.
    turns_named = _flatten_turns_no_overlap(turns_named)

    _pipeline_log(
        "build_turns",
        f"  итог turns_named: {len(turns_named)} реплик",
        indent=1,
    )
    _pipeline_log("build_turns", "узел завершён → переход к gigaam_asr", indent=1)

    return {"turns_named": turns_named}


def node_gigaam_asr(state: PipelineGraphState) -> dict[str, Any]:
    _pipeline_log(
        "gigaam_asr",
        "узел: GigaAM — загрузка модели и построчная транскрибация turns_named",
    )
    try:
        import gigaam
    except ImportError as e:
        raise RuntimeError(
            "Установите gigaam (см. requirements_gigaam.txt в корне проекта)"
        ) from e

    device = state["device"]
    giga_model_name = state["giga_model_name"]
    wav = state["wav"]
    turns_named = state["turns_named"]
    _pipeline_log(
        "gigaam_asr",
        f"  gigaam.load_model({giga_model_name!r}, device={device!r})…",
        indent=1,
    )

    gmodel = gigaam.load_model(
        giga_model_name,
        device=device,
        fp16_encoder=False if device == "cpu" else True,
        use_flash=False,
    )
    _pipeline_log(
        "gigaam_asr",
        f"  transcribe_segments_gigaam: реплик {len(turns_named)}, длинные режутся по {MAX_ASR_CHUNK_SEC} с",
        indent=1,
    )
    rows = transcribe_segments_gigaam(
        gmodel,
        wav.cpu(),
        SAMPLE_RATE,
        turns_named,
        word_timestamps=False,
        pipeline_trace=True,
    )
    _pipeline_log(
        "gigaam_asr",
        f"узел завершён: строк ASR в JSON-массиве: {len(rows)} → END",
        indent=1,
    )
    return {"asr_rows": rows}


def build_pipeline_graph() -> Any:
    g = StateGraph(PipelineGraphState)
    g.add_node("load_hf_waveform", node_load_hf_waveform)
    g.add_node("vad", node_vad)
    g.add_node("speech_windows", node_speech_windows)
    g.add_node("pyannote_segment", node_pyannote_segment)
    g.add_node("finish_empty", node_finish_empty)
    g.add_node("ecapa_embeddings", node_ecapa_embeddings)
    g.add_node("cluster", node_cluster)
    g.add_node("resegment", node_resegment)
    g.add_node("build_turns", node_build_turns)
    g.add_node("gigaam_asr", node_gigaam_asr)

    g.set_entry_point("load_hf_waveform")
    g.add_edge("load_hf_waveform", "vad")
    g.add_edge("vad", "speech_windows")
    g.add_edge("speech_windows", "pyannote_segment")
    g.add_conditional_edges(
        "pyannote_segment",
        _route_after_pyannote,
        {
            "no_speech": "finish_empty",
            "embed": "ecapa_embeddings",
        },
    )
    g.add_edge("ecapa_embeddings", "cluster")
    g.add_conditional_edges(
        "cluster",
        _route_after_cluster,
        {
            "no_clusters": "finish_empty",
            "diarize": "resegment",
        },
    )
    g.add_edge("resegment", "build_turns")
    g.add_edge("build_turns", "gigaam_asr")
    g.add_edge("gigaam_asr", END)
    g.add_edge("finish_empty", END)
    return g.compile()


def _llm_temperature() -> float:
    try:
        return float(os.getenv("LLM_TEMPERATURE", "0.0").strip() or "0.0")
    except Exception:
        return 0.0


def _llm_timeout_sec() -> float:
    try:
        return float(os.getenv("LLM_TIMEOUT_SEC", "120").strip() or "120")
    except Exception:
        return 120.0


def _llm_second_pass_default() -> bool:
    # Всегда делаем второй проход (refine) для корректировки merge/split/speaker.
    return True


def run_pipeline(
    audio_path: str,
    *,
    device: str,
    giga_model_name: str,
    clusterer: str,
    num_speakers: int | None,
    agglomerative_threshold: float,
    hdbscan_min_cluster_size: int,
    refine_merge_gap: float,
    skip_vad_trim: bool,
    segmentation_step_ratio: float,
    resegment: bool = True,
) -> list[dict]:
    _pipeline_log("run_pipeline", "=" * 60)
    _pipeline_log(
        "run_pipeline",
        f"старт графа LangGraph: аудио={audio_path!r}, device={device!r}, clusterer={clusterer!r}",
    )
    app = build_pipeline_graph()
    _pipeline_log(
        "run_pipeline",
        "граф скомпилирован; порядок: load_hf_waveform → vad → speech_windows → pyannote_segment → "
        "(ecapa_embeddings|finish_empty) → cluster → (build_turns|finish_empty) → gigaam_asr|END",
    )
    initial: PipelineGraphState = {
        "audio_path": audio_path,
        "device": device,
        "giga_model_name": giga_model_name,
        "clusterer": clusterer,
        "num_speakers": num_speakers,
        "agglomerative_threshold": agglomerative_threshold,
        "hdbscan_min_cluster_size": hdbscan_min_cluster_size,
        "refine_merge_gap": refine_merge_gap,
        "resegment": bool(resegment),
        "skip_vad_trim": skip_vad_trim,
        "segmentation_step_ratio": segmentation_step_ratio,
    }
    final: PipelineGraphState = app.invoke(initial)
    _pipeline_log(
        "run_pipeline",
        f"invoke завершён; asr_rows: {len(list(final.get('asr_rows') or []))} записей",
    )
    _pipeline_log("run_pipeline", "=" * 60)
    return list[dict[str, Any]](final.get("asr_rows") or [])


_LLM_SPEAKER_SYSTEM_LEGACY = (
    "Ты — ИИ-редактор. Тебе на вход подается JSON-массив сегментов аудио. "
    "В каждом объекте есть 'id', 'speaker' и 'text'. "
    "Часто акустическая система сливает быстрый диалог двух людей в один сегмент "
    "(например, когда текст идет через тире). "
    "Твоя задача: если внутри одного сегмента говорят разные люди, РАЗБЕЙ его на несколько объектов. "
    "Ты не должен ничего придумывать или менять слова, только распределить существующий текст. "
    "В ответе верни JSON-объект {\"segments\": [...]}. "
    "В каждом выходном объекте обязательно укажи оригинальный 'id' сегмента, который ты сейчас обрабатываешь, "
    "нужный 'speaker' и соответствующий 'text'. "
    "Тайминги (start, end) возвращать НЕ НУЖНО, скрипт посчитает их сам. "
    "Используй метки SPEAKER_1, SPEAKER_2 и т.д. Никакого markdown, только сырой JSON."
)


def _llm_norm_segment_id(val: Any, n_rows: int) -> int | None:
    """Индекс исходного сегмента или None, если значение некорректно или вне диапазона."""
    if val is None or isinstance(val, bool):
        return None
    if isinstance(val, int):
        i = val
    elif isinstance(val, float) and val == int(val) and not (val != val):  # not NaN
        i = int(val)
    elif isinstance(val, str):
        s = val.strip()
        if not s or not s.lstrip("-").isdigit():
            return None
        i = int(s)
    else:
        return None
    if i < 0 or i >= n_rows:
        return None
    return i


def _llm_speaker_str(val: Any, fallback: str) -> str:
    if isinstance(val, str) and val.strip():
        return val.strip()
    if val is not None:
        s = str(val).strip()
        if s:
            return s
    return fallback


def llm_speaker_correction_legacy_openai_segments(
    rows: list[dict],
    api_key: str,
    base_url: str | None = None,
    model_name: str = "gpt-4o-mini",
) -> list[dict]:
    """Постобработка: LLM может дробить сегменты и менять speaker; тайминги восстанавливаются по доле текста."""
    if not rows:
        return rows
    try:
        from openai import OpenAI  # type: ignore[import-not-found]
    except ImportError:
        print(
            "LLM-коррекция: пакет openai не установлен (pip install openai). Пропуск.",
            file=sys.stderr,
        )
        return rows

    client_kwargs: dict = {"api_key": api_key}
    if base_url:
        client_kwargs["base_url"] = base_url.rstrip("/")

    client = OpenAI(**client_kwargs)
    payload = [
        {
            "id": i,
            "speaker": str(rows[i].get("speaker", "SPEAKER_1")),
            "text": str(rows[i].get("text", "")),
        }
        for i in range(len(rows))
    ]
    user_content = (
        "Ниже JSON-массив сегментов (поля только id, speaker, text). "
        "Верни один JSON-объект с ключом \"segments\" — массив объектов с полями id, speaker, text. "
        "Тайминги start/end не указывай. При необходимости разбей один id на несколько объектов с одним и тем же id "
        "или с повторением id по частям; скрипт сопоставит части по id и распределит время.\n"
        + json.dumps(payload, ensure_ascii=False)
    )

    try:
        completion = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": _LLM_SPEAKER_SYSTEM_LEGACY},
                {"role": "user", "content": user_content},
            ],
            temperature=0.0,
        )
        raw = (completion.choices[0].message.content or "").strip()
        data = json.loads(raw)
    except Exception as e:
        print(f"LLM-коррекция(legacy): ошибка API или разбора ответа: {e}", file=sys.stderr)
        return rows

    if not isinstance(data, dict):
        print("LLM-коррекция: ответ не JSON-объект, пропуск.", file=sys.stderr)
        return rows

    segs = data.get("segments")
    if not isinstance(segs, list):
        print("LLM-коррекция: нет ключа segments или неверный тип, пропуск.", file=sys.stderr)
        return rows

    n = len(rows)
    by_id: dict[int, list[dict[str, Any]]] = {}
    for item in segs:
        if not isinstance(item, dict):
            continue
        sid = _llm_norm_segment_id(item.get("id"), n)
        if sid is None:
            continue
        by_id.setdefault(sid, []).append(item)

    out: list[dict[str, Any]] = []
    for i, orig in enumerate(rows):
        pieces = by_id.get(i, [])
        if not pieces:
            out.append(
                {
                    "start": orig["start"],
                    "end": orig["end"],
                    "text": orig.get("text", ""),
                    "speaker": str(orig.get("speaker", "SPEAKER_1")),
                }
            )
            continue

        t0 = float(orig["start"])
        t1 = float(orig["end"])
        span = max(0.0, t1 - t0)
        fb_sp = str(orig.get("speaker", "SPEAKER_1"))

        if len(pieces) == 1:
            p = pieces[0]
            out.append(
                {
                    "start": round(t0, 3),
                    "end": round(t1, 3),
                    "text": str(p.get("text", orig.get("text", ""))),
                    "speaker": _llm_speaker_str(p.get("speaker"), fb_sp),
                }
            )
            continue

        texts = [str(p.get("text", "")) for p in pieces]
        lens = [max(0, len(tx)) for tx in texts]
        total_chars = sum(lens)
        n_p = len(pieces)
        if total_chars <= 0:
            weights = [1.0 / n_p] * n_p
        else:
            weights = [lc / total_chars for lc in lens]

        acc = 0.0
        boundaries = [t0]
        for j in range(n_p - 1):
            acc += weights[j]
            boundaries.append(t0 + span * acc)
        boundaries.append(t1)

        for j, p in enumerate(pieces):
            out.append(
                {
                    "start": round(boundaries[j], 3),
                    "end": round(boundaries[j + 1], 3),
                    "text": texts[j],
                    "speaker": _llm_speaker_str(p.get("speaker"), fb_sp),
                }
            )

    return out


def main() -> int:
    _ensure_utf8_stdio()
    _load_dotenv()
    parser = argparse.ArgumentParser(description="Диаризация (Silero + segmentation-3.0 + ECAPA + ASR GigaAM), LangGraph")
    parser.add_argument(
        "audio",
        nargs="?",
        default=None,
        help="Путь к аудио (wav/mp3/…). Если не указан — откроется диалог выбора файла.",
    )
    parser.add_argument("--out", type=Path, default=None, help="JSON выход (по умолчанию рядом с аудио)")
    parser.add_argument("--device", default="cpu", help="cpu или cuda")
    parser.add_argument(
        "--giga-model",
        default="e2e_rnnt",
        help="Имя модели для gigaam.load_model (например e2e_rnnt)",
    )
    parser.add_argument(
        "--clusterer",
        choices=("agglomerative", "hdbscan"),
        default="agglomerative",
        help="Кластеризация эмбеддингов",
    )
    parser.add_argument(
        "--num-speakers",
        type=int,
        default=None,
        help="Точное количество спикеров (если известно). Улучшает разделение похожих голосов.",
    )
    parser.add_argument(
        "--agg-threshold",
        type=float,
        default=0.58,
        help="Порог distance_threshold для Agglomerative (cosine); по умолчанию 0.58 (диапазон 0.55–0.60 чаще отделяет похожие голоса). Меньше — больше спикеров",
    )
    parser.add_argument(
        "--cosine-sim-threshold",
        type=float,
        default=None,
        help="Альтернатива для Agglomerative: порог cosine similarity (например 0.6–0.7). Будет конвертирован в distance_threshold=1-sim.",
    )
    parser.add_argument(
        "--hdbscan-min-size",
        type=int,
        default=2,
        help="min_cluster_size для HDBSCAN",
    )
    parser.add_argument(
        "--refine-gap",
        type=float,
        default=0.25,
        help="Слияние соседних фраз одного спикера, если пауза ≤ N сек (0 — выкл)",
    )
    parser.add_argument(
        "--merge-off",
        action="store_true",
        help="Полностью отключить merge реплик (эквивалент --refine-gap 0)",
    )
    parser.add_argument(
        "--skip-vad-trim",
        action="store_true",
        help="Не обрезать хвосты сегментов по Silero VAD",
    )
    parser.add_argument(
        "--seg-step-ratio",
        type=float,
        default=0.1,
        help="Шаг окна сегментации как доля длины окна модели (как в pyannote, по умолчанию 0.1)",
    )
    parser.add_argument(
        "--no-resegment",
        action="store_true",
        help="Отключить resegmentation после clustering (по умолчанию включено: лучше ловит быстрые смены спикера)",
    )
    parser.add_argument(
        "--use-llm",
        action="store_true",
        help="Принудительно включить LLM-постобработку спикеров (если настроены ключи/зависимости).",
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Принудительно выключить LLM-постобработку, даже если ключи/USE_LLM заданы.",
    )
    parser.add_argument(
        "--llm-model",
        default="gpt-4o-mini",
        help="Модель для LLM-коррекции (OpenAI-совместимый API)",
    )
    parser.add_argument(
        "--llm-base-url",
        default=None,
        help="Базовый URL API (proxy, vLLM, Groq: https://api.groq.com/openai/v1 и т.п.)",
    )
    args = parser.parse_args()
    if args.merge_off:
        args.refine_gap = 0.0
    if args.cosine_sim_threshold is not None:
        sim = float(args.cosine_sim_threshold)
        if sim <= -1.0:
            sim = -1.0
        if sim >= 1.0:
            sim = 0.999
        args.agg_threshold = float(1.0 - sim)
    if not args.audio or not str(args.audio).strip():
        picked = pick_audio_path()
        if not picked:
            print("Файл не выбран — выход.", file=sys.stderr)
            return 2
        args.audio = picked
    if args.num_speakers is None:
        args.num_speakers = _prompt_num_speakers()
    audio_path = str(Path(args.audio).expanduser().resolve())
    print(f"Аудио: {audio_path}", flush=True)
    ap = Path(audio_path)
    out_path = args.out or ap.with_name(ap.stem + "_diarization_asr.json")
    if out_path.suffix.lower() != ".json":
        out_path = out_path.with_suffix(".json")

    groq_key = (os.getenv("GROQ_API_KEY") or "").strip()
    openai_key = (os.getenv("OPENAI_API_KEY") or "").strip()
    auto_llm = bool(groq_key or openai_key) and _truthy_env("USE_LLM", True)
    use_llm = (auto_llm or bool(args.use_llm)) and (not bool(args.no_llm))

    num_speakers = args.num_speakers
    if num_speakers is None:
        env_ns = (os.getenv("NUM_SPEAKERS") or "").strip()
        if env_ns.isdigit():
            num_speakers = int(env_ns)
    if use_llm and num_speakers is None and sys.stdin.isatty():
        try:
            s = input("Сколько говорящих ожидается (целое число)? [Enter = неизвестно]: ").strip()
        except EOFError:
            s = ""
        if s.isdigit():
            num_speakers = int(s)

    llm_model = (os.getenv("LLM_MODEL") or "").strip() or None
    if groq_key and not openai_key:
        if (not llm_model) or (("/" not in llm_model) and llm_model.startswith("gpt-")):
            llm_model = "openai/gpt-oss-120b"
            print(f"LLM: auto model для Groq: {llm_model}", flush=True)

    if use_llm and not groq_key and not openai_key:
        print("LLM: ключи не найдены (нужен GROQ_API_KEY или OPENAI_API_KEY) — LLM выключено", file=sys.stderr, flush=True)
        use_llm = False

    if use_llm:
        print(
            "LLM: включено ("
            f"USE_LLM={os.getenv('USE_LLM')!r}, "
            f"LLM_SECOND_PASS={os.getenv('LLM_SECOND_PASS')!r}, "
            f"model={llm_model!r}"
            ")",
            flush=True,
        )
    else:
        print("LLM: выключено", flush=True)

    try:
        rows = run_pipeline(
            audio_path,
            device=args.device,
            giga_model_name=args.giga_model,
            clusterer=args.clusterer,
            num_speakers=num_speakers,
            agglomerative_threshold=args.agg_threshold,
            hdbscan_min_cluster_size=args.hdbscan_min_size,
            refine_merge_gap=args.refine_gap,
            skip_vad_trim=args.skip_vad_trim,
            segmentation_step_ratio=args.seg_step_ratio,
            # resegment=True (по умолчанию): пост-обработка границ по кластерам, устойчивее к быстрым переключениям.
            resegment=not bool(args.no_resegment),
        )
    except Exception as e:
        print(f"Ошибка: {e}", file=sys.stderr)
        return 1

    if use_llm:
        llm_model_final = (llm_model or str(args.llm_model).strip() or "gpt-4o-mini").strip()
        try:
            _llm_log(
                f"запуск после GigaAM: строк ASR={len(rows)}, model={llm_model_final!r}",
                indent=0,
            )
            rows = llm_speaker_correction(
                rows,
                model=llm_model_final,
                num_speakers=num_speakers,
                temperature=_llm_temperature(),
                timeout_sec=_llm_timeout_sec(),
                second_pass=_llm_second_pass_default(),
            )
            _llm_log(f"завершено: строк после LLM={len(rows)}", indent=0)
        except Exception as e:
            print(f"LLM: ошибка постобработки ({e}) — пропуск.", file=sys.stderr, flush=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Записано: {out_path} ({len(rows)} сегментов)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
