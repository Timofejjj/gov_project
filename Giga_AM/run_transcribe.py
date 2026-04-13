"""
Транскрипция через GigaAM с записью файлов как в test_2 (Whisper-пайплайн):

  model_parameters.txt
  info.txt
  transcribed_text.txt

Спикеры A, B, …: через pyannote (нужен HF_TOKEN и пакет pyannote.audio).
Для pyannote звук грузится в 16 kHz без принудительного моно (стерео, если есть в файле);
GigaAM по-прежнему получает моно-сегменты при transcribe.
Без токена или с --no-diarize — один проход ASR без разделения спикеров.

Запуск из корня репозитория:
  .venv\\Scripts\\python.exe Giga_AM\\run_transcribe.py

Токен Hugging Face для pyannote: переменная HF_TOKEN или файл .env в корне репозитория
(шаблон: .env.example). При старте вызывается load_dotenv из python-dotenv.

Зависимости: install_requirements.ps1 / .bat
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

SAMPLE_RATE = 16000
# GigaAM transcribe ~25 с; берём запас
MAX_ASR_CHUNK_SEC = 20.0
MIN_SEGMENT_SEC = 0.12

_DEFAULT_PYANNOTE_MODELS = (
    "pyannote/speaker-diarization-3.1",
    "pyannote/speaker-diarization-community-1",
)


def _ensure_cpu_env() -> None:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")


def _load_project_dotenv() -> None:
    """Читает .env из корня репозитория (Gov_pl_2), чтобы подтянуть HF_TOKEN и др."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    repo_root = Path(__file__).resolve().parent.parent
    load_dotenv(repo_root / ".env")


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
        title="Выберите аудиофайл (GigaAM)",
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
$d.Title = 'Выберите аудиофайл (GigaAM)'
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
    set f to choose file with prompt "Выберите аудиофайл (GigaAM)"
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


def _speaker_label_from_index(idx: int) -> str:
    if idx < 26:
        return chr(ord("A") + idx)
    return f"S{idx - 25}"


def _speaker_map_chronological(
    turns: list[tuple[float, float, str]],
) -> dict[str, str]:
    mapping: dict[str, str] = {}
    n = 0
    for t0, _, lab in sorted(turns, key=lambda x: (x[0], x[1])):
        if lab not in mapping:
            mapping[lab] = _speaker_label_from_index(n)
            n += 1
    return mapping


def _ffprobe_audio_channels(audio_path: str) -> int:
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


def _load_waveform_16k_for_diarization(audio_path: str):
    """16 kHz для pyannote: моно как моно, стерео как стерео; >2 каналов — сводим в stereo (не в моно)."""
    from subprocess import run

    import numpy as np
    import torch

    ch_in = _ffprobe_audio_channels(audio_path)
    # >2 каналов — stereo; иначе сохраняем исходную размерность (1 или 2)
    extra_ac = ["-ac", "2"] if ch_in >= 2 else []
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
    raw = run(cmd, capture_output=True, check=True).stdout
    ch = 2 if ch_in >= 2 else 1
    arr = np.frombuffer(raw, dtype=np.int16)
    if ch == 1:
        x = arr.astype(np.float32) / 32768.0
        wav = torch.from_numpy(x).unsqueeze(0)
    else:
        n = arr.size // 2
        arr = arr[: n * 2].reshape(n, 2).T
        wav = torch.from_numpy(arr.astype(np.float32) / 32768.0)
    return wav.float().contiguous(), SAMPLE_RATE


def _pyannote_diarization_turns(
    waveform,
    sample_rate: int,
    hf_token: str,
    device: str,
    model_id: str | None,
) -> tuple[list[tuple[float, float, str]], str]:
    from pyannote.audio import Pipeline
    import torch

    dev = torch.device(device)
    wf = waveform.to(dev) if waveform.device != dev else waveform

    candidates = (model_id,) if model_id else _DEFAULT_PYANNOTE_MODELS
    last_err: Exception | None = None
    for mid in candidates:
        try:
            pipeline = Pipeline.from_pretrained(mid, token=hf_token)
            pipeline.to(dev)
            output = pipeline({"uri": "recording", "waveform": wf, "sample_rate": sample_rate})
            # pyannote может вернуть либо Annotation (есть itertracks),
            # либо DiarizeOutput со вложенной Annotation:
            # - speaker_diarization / exclusive_speaker_diarization (pyannote.audio 4)
            # - diarization / annotation (на всякий случай)
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
        except Exception as e:  # noqa: BLE001 — пробуем следующую модель
            last_err = e
            continue
    raise RuntimeError(
        f"Не удалось запустить pyannote ({candidates}). Последняя ошибка: {last_err}"
    )


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


def _transcribe_diarized(
    model,
    waveform,
    sample_rate: int,
    turns: list[tuple[float, float, str]],
    speaker_map: dict[str, str],
    word_timestamps: bool,
) -> list[dict]:
    import wave
    import torch

    utterances: list[dict] = []
    total_samples = waveform.shape[-1]

    def write_wav_pcm16(path: str, mono_float: "torch.Tensor", sr: int) -> None:
        """Сохраняет (1, time) float32 [-1..1] как PCM16 WAV без torchaudio."""
        x = mono_float.squeeze(0)
        if x.is_cuda:
            x = x.cpu()
        x = x.detach().to(torch.float32).clamp(-1.0, 1.0)
        x_i16 = (x * 32767.0).round().to(torch.int16).numpy()
        with wave.open(path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # int16
            wf.setframerate(sr)
            wf.writeframes(x_i16.tobytes())

    for t0, t1, raw_spk in turns:
        if t1 - t0 < MIN_SEGMENT_SEC:
            continue
        letter = speaker_map.get(raw_spk, "?")
        text_parts: list[str] = []
        all_words: list[dict] = []

        for cs, ce in _time_chunks(t0, t1, MAX_ASR_CHUNK_SEC):
            i0 = max(0, int(cs * sample_rate))
            i1 = min(total_samples, int(ce * sample_rate))
            if i1 <= i0:
                continue
            seg = waveform[:, i0:i1].cpu().float()
            if seg.shape[-1] < int(0.05 * sample_rate):
                continue
            # GigaAM ожидает моно; pyannote мог работать по стерео-тензору
            if seg.shape[0] > 1:
                seg = seg.mean(dim=0, keepdim=True)

            fd, tmp_path = tempfile.mkstemp(suffix=".wav")
            os.close(fd)
            try:
                write_wav_pcm16(tmp_path, seg, sample_rate)
                result = model.transcribe(tmp_path, word_timestamps=word_timestamps)
            finally:
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass

            chunk_text = (result.text if hasattr(result, "text") else str(result)).strip()
            if chunk_text:
                text_parts.append(chunk_text)

            if word_timestamps and hasattr(result, "words") and result.words:
                for w in result.words:
                    if isinstance(w, dict):
                        wt, ws, we = w.get("text", ""), w.get("start"), w.get("end")
                        conf = w.get("confidence")
                    else:
                        wt = getattr(w, "text", "") or ""
                        ws = getattr(w, "start", None)
                        we = getattr(w, "end", None)
                        conf = getattr(w, "confidence", None)
                    if ws is None:
                        continue
                    abs_s = cs + float(ws)
                    abs_e = cs + float(we) if we is not None else abs_s
                    all_words.append(
                        {
                            "text": wt,
                            "start": _sec_to_ms(abs_s),
                            "end": _sec_to_ms(abs_e),
                            "confidence": conf,
                            "speaker": letter,
                        }
                    )

        full_text = " ".join(text_parts).strip()
        if not full_text:
            continue

        if all_words:
            starts = [w["start"] for w in all_words]
            ends = [w["end"] for w in all_words]
            u0, u1 = min(starts), max(ends)
        else:
            u0, u1 = _sec_to_ms(t0), _sec_to_ms(t1)

        utterances.append(
            {
                "speaker": letter,
                "text": full_text,
                "confidence": None,
                "start": u0,
                "end": u1,
                "words": all_words if word_timestamps else [],
            }
        )

    return utterances


def _sec_to_ms(t: float) -> int:
    return int(round(float(t) * 1000.0))


def _build_utterances_payload(
    full_text: str, words: list | None, speaker: str = "—"
) -> list[dict]:
    """Один спикер и список слов — формат как в transcribed_text из AssemblyAI."""
    if not words:
        return [
            {
                "speaker": speaker,
                "text": full_text,
                "confidence": None,
                "start": 0,
                "end": 0,
                "words": [],
            }
        ]

    word_objs = []
    for w in words:
        if isinstance(w, dict):
            text = w.get("text", "")
            start, end = w.get("start"), w.get("end")
            conf = w.get("confidence")
        else:
            text = getattr(w, "text", "") or ""
            start = getattr(w, "start", None)
            end = getattr(w, "end", None)
            conf = getattr(w, "confidence", None)
        word_objs.append(
            {
                "text": text,
                "start": _sec_to_ms(start) if start is not None else 0,
                "end": _sec_to_ms(end) if end is not None else 0,
                "confidence": conf,
                "speaker": speaker,
            }
        )

    starts = [x["start"] for x in word_objs if x["start"]]
    ends = [x["end"] for x in word_objs if x["end"]]
    t0 = min(starts) if starts else 0
    t1 = max(ends) if ends else 0

    return [
        {
            "speaker": speaker,
            "text": full_text,
            "confidence": None,
            "start": t0,
            "end": t1,
            "words": word_objs,
        }
    ]


def _write_model_parameters(
    out_dir: Path,
    model_name: str,
    device: str,
    *,
    diarization: bool,
    pyannote_model: str | None,
) -> None:
    resolved = model_name
    if model_name in ("ctc", "rnnt", "e2e_ctc", "e2e_rnnt", "ssl"):
        resolved = f"v3_{model_name}"
    diar_line = (
        f"  pyannote: {pyannote_model} (спикеры A, B, … по хронологии первого появления)"
        if diarization and pyannote_model
        else "  pyannote: не использовалась"
    )
    diar_flag = (
        "включена (локально, через pyannote; метки A/B — относительные, не имена людей)"
        if diarization
        else "выключена (один спикер «—» или только ASR)"
    )
    text = f"""Параметры модели (локальный запуск GigaAM)

Модель
  GigaAM — {resolved} (локально, без AssemblyAI)

Инференс
  Устройство: {device}
  Режим: transcribe по сегментам диаризации (каждый кусок до ~{int(MAX_ASR_CHUNK_SEC)} с)

Язык
  Russian (модель ориентирована на русский речевой корпус)

Функции
  Speaker diarization (диаризация спикеров): {diar_flag}
{diar_line}
  Вход pyannote: 16 kHz, стерео если в файле ≥2 канала (без принудительного моно); >2 каналов — сводка в stereo.
  GigaAM по сегментам: моно (среднее по каналам перед transcribe).
  Language detection: не применимо (фиксированный домен русского ASR)

Примечания
  Для HF-моделей pyannote: токен в HF_TOKEN / HUGGING_FACE_HUB_TOKEN и согласие с условиями на странице модели на huggingface.co.
  Длинные сегменты режутся по времени; при необходимости см. GigaAM transcribe_longform в документации пакета gigaam.
"""
    (out_dir / "model_parameters.txt").write_text(text, encoding="utf-8")


def _write_info(
    out_dir: Path,
    audio_path: str,
    generated_at: str,
    *,
    diarization: bool,
    extra: str = "",
) -> None:
    diar_note = (
        "Спикеры A, B, …: результат pyannote + отдельная транскрипция GigaAM на каждый сегмент.\n"
        if diarization
        else "Диаризация не выполнялась — в JSON один условный спикер или без разбиения по людям.\n"
    )
    text = f"""Сравнение с пайплайном Whisper в этом репозитории

В папках test_* результат AssemblyAI (Whisper-Streaming) часто включает диаризацию и
обогащённые метаданные. Здесь используется локальная связка pyannote (кто когда говорил)
и GigaAM (текст по сегментам).

{diar_note}{extra}
Запуск без аргумента к пути файлу открывает диалог выбора аудио (tkinter / WinForms на Windows).

Аудио: {audio_path}
Время записи артефактов (UTC): {generated_at}
"""
    (out_dir / "info.txt").write_text(text, encoding="utf-8")


def main() -> int:
    _ensure_cpu_env()
    _load_project_dotenv()
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
            sys.stderr.reconfigure(encoding="utf-8")
        except (AttributeError, OSError):
            pass

    pkg_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="GigaAM + опционально pyannote → как test_*")
    parser.add_argument(
        "audio",
        nargs="?",
        default=None,
        help="Путь к WAV/другому файлу. Если не указан — диалог выбора файла.",
    )
    parser.add_argument(
        "--model",
        default="e2e_rnnt",
        help="Имя для gigaam.load_model: ctc, rnnt, e2e_ctc, e2e_rnnt, ssl или полное v3_*.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Куда писать txt (по умолчанию — эта папка Giga_AM).",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Устройство torch, по умолчанию cpu.",
    )
    parser.add_argument(
        "--no-word-timestamps",
        action="store_true",
        help="Не запрашивать пословную разметку (меньше работы декодера).",
    )
    parser.add_argument(
        "--no-diarize",
        action="store_true",
        help="Не вызывать pyannote (один проход ASR на весь файл, без спикеров A/B).",
    )
    parser.add_argument(
        "--pyannote-model",
        default=None,
        help="ID модели на HuggingFace (иначе по очереди пробуются встроенные варианты).",
    )
    args = parser.parse_args()

    out_dir = (args.out_dir or pkg_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        import gigaam
    except ImportError:
        print(
            "Не установлен пакет gigaam/torch в этом интерпретаторе.\n"
            "Выполните из корня проекта: .\\install_requirements.bat\n"
            "или: powershell -ExecutionPolicy Bypass -File .\\install_requirements.ps1\n"
            "Либо вручную:\n"
            '  .venv\\Scripts\\python.exe -m pip install -r requirements_gigaam.txt',
            file=sys.stderr,
        )
        return 1

    device = args.device
    if device == "cpu":
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

    if args.audio:
        audio_path = str(Path(args.audio).expanduser().resolve())
    else:
        picked = pick_audio_path()
        if not picked:
            print("Файл не выбран — выход.", file=sys.stderr)
            return 1
        audio_path = str(Path(picked).expanduser().resolve())

    print(f"Аудио: {audio_path}", flush=True)
    print(f"Вывод: {out_dir}", flush=True)
    print(f"Модель: {args.model}, устройство: {device}", flush=True)

    model = gigaam.load_model(
        args.model,
        device=device,
        fp16_encoder=False if device == "cpu" else True,
        use_flash=False,
    )

    word_ts = not args.no_word_timestamps
    hf = _hf_token()
    use_diarize = not args.no_diarize and bool(hf)
    info_extra = ""
    pyannote_model_used: str | None = None
    utterances: list[dict]

    if use_diarize:
        try:
            import pyannote.audio  # noqa: F401
        except ImportError:
            print(
                "Диаризация отключена: не установлен pyannote.audio. "
                "Установите: pip install -r requirements_pyannote.txt",
                file=sys.stderr,
            )
            use_diarize = False
            info_extra = "pyannote не установлен — см. requirements_pyannote.txt\n"

    if use_diarize:
        try:
            wf, sr = _load_waveform_16k_for_diarization(audio_path)
            turns, pyannote_model_used = _pyannote_diarization_turns(
                wf, sr, hf, device, args.pyannote_model
            )
            if not turns:
                raise RuntimeError("pyannote вернул пустой результат")
            spk_map = _speaker_map_chronological(turns)
            utterances = _transcribe_diarized(
                model, wf, sr, turns, spk_map, word_ts
            )
            if not utterances:
                raise RuntimeError("После диаризации не получилось ни одной непустой транскрипции")
            header = "\n--- utterances (GigaAM + pyannote: спикеры A, B, …) ---\n"
        except Exception as e:  # noqa: BLE001
            err = str(e)
            print(f"Диаризация не удалась ({e}); выполняется один проход ASR.", file=sys.stderr)
            if "403" in err or "gated" in err.lower() or "restricted" in err.lower():
                print(
                    "Нужен доступ к gated-модели pyannote: войдите на huggingface.co под тем же\n"
                    "аккаунтом, что и токен в .env, и нажмите «Agree» на странице хотя бы одной модели:\n"
                    "  https://huggingface.co/pyannote/speaker-diarization-3.1\n"
                    "  https://huggingface.co/pyannote/speaker-diarization-community-1\n"
                    "Также требуется согласие для базовой модели сегментации:\n"
                    "  https://huggingface.co/pyannote/segmentation-3.0\n"
                    "Затем повторите запуск.",
                    file=sys.stderr,
                )
            use_diarize = False
            info_extra = f"Ошибка диаризации (один проход ASR): {e}\n"
            result = model.transcribe(audio_path, word_timestamps=word_ts)
            full_text = result.text if hasattr(result, "text") else str(result)
            words = result.words if hasattr(result, "words") else None
            utterances = _build_utterances_payload(full_text, words, speaker="—")
            header = "\n--- utterances (локальная транскрипция GigaAM, без диаризации) ---\n"
    else:
        if not args.no_diarize and not hf:
            print(
                "Подсказка: для спикеров A/B задайте HF_TOKEN (или HUGGING_FACE_HUB_TOKEN) "
                "и установите pyannote.audio; иначе используйте явно --no-diarize.",
                file=sys.stderr,
            )
        result = model.transcribe(audio_path, word_timestamps=word_ts)
        full_text = result.text if hasattr(result, "text") else str(result)
        words = result.words if hasattr(result, "words") else None
        utterances = _build_utterances_payload(full_text, words, speaker="—")
        header = "\n--- utterances (локальная транскрипция GigaAM, без диаризации) ---\n"

    transcribed_body = header + json.dumps(utterances, ensure_ascii=False, indent=2) + "\n"
    (out_dir / "transcribed_text.txt").write_text(transcribed_body, encoding="utf-8")

    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    _write_model_parameters(
        out_dir,
        args.model,
        device,
        diarization=use_diarize,
        pyannote_model=pyannote_model_used,
    )
    _write_info(out_dir, audio_path, now, diarization=use_diarize, extra=info_extra)

    print("Готово: transcribed_text.txt, model_parameters.txt, info.txt")
    print("---")
    for u in utterances:
        print(f"{u.get('speaker', '?')}: {u.get('text', '')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
