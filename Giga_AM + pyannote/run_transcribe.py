"""
Транскрипция через GigaAM с артефактами как в test_* (model_parameters.txt, info.txt, transcribed_text.json в txt).

Пайплайн: ffmpeg/ffprobe → моно 16 kHz → pyannote (опционально) → постобработка сегментов
(overlap, склейка спикера, сглаживание) → ASR по чанкам с паузо-зависимыми границами → JSON (время в мс).

Запуск:
  .venv\\Scripts\\python.exe \"Giga_AM + pyannote\\run_transcribe.py\" [аудио]

Токен HF для pyannote: HF_TOKEN / HUGGING_FACE_HUB_TOKEN или .env в корне Gov_pl_2.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from audio_prep import ensure_ffmpeg_available, load_waveform_16k_mono
from chunk_audio import speech_aware_time_chunks
from diarize_post import postprocess_diarization_turns
from diarize_pyannote import pyannote_diarization_turns
from gigaam_batch import transcribe_mono_segments_batch, words_to_absolute_dicts
from text_normalize import normalize_transcription_text
from utterances import (
    build_utterances_payload,
    merge_chunk_texts,
    seconds_to_json_ms,
)

# --- константы (секунды float внутри пайплайна; в JSON только миллисекунды) ---
MAX_ASR_CHUNK_SEC = 20.0
MIN_SEGMENT_SEC = 0.12
MERGE_SAME_SPEAKER_GAP_SEC = 0.4
MAX_INTRUSION_SEC = 0.18
BOUNDARY_PAD_SEC = 0.04
ASR_BATCH_SIZE = 8


def _ensure_cpu_env() -> None:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")


def _load_project_dotenv() -> None:
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
    import subprocess

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
    import subprocess

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


def _transcribe_diarized(
    model,
    waveform_mono: "torch.Tensor",
    sample_rate: int,
    turns: list[tuple[float, float, str]],
    speaker_map: dict[str, str],
    word_timestamps: bool,
    asr_batch_size: int,
) -> list[dict]:
    """ASR по обработанным сегментам диаризации; чанки с учётом тишины; инференс батчами без WAV на чанк."""
    import torch

    utterances: list[dict] = []
    total_samples = waveform_mono.shape[-1]

    for _turn_idx, (t0, t1, raw_spk) in enumerate(turns):
        if t1 - t0 < MIN_SEGMENT_SEC:
            continue
        letter = speaker_map.get(raw_spk, "?")
        chunk_starts: list[float] = []
        segment_tensors: list[torch.Tensor] = []

        for cs, ce in speech_aware_time_chunks(
            t0, t1, waveform_mono, sample_rate, MAX_ASR_CHUNK_SEC
        ):
            i0 = max(0, int(cs * sample_rate))
            i1 = min(total_samples, int(ce * sample_rate))
            if i1 <= i0:
                continue
            seg = waveform_mono[:, i0:i1].float().squeeze(0).cpu().contiguous()
            if seg.numel() < int(0.05 * sample_rate):
                continue
            chunk_starts.append(float(cs))
            segment_tensors.append(seg)

        if not segment_tensors:
            continue

        batch_out = transcribe_mono_segments_batch(
            model,
            segment_tensors,
            word_timestamps=word_timestamps,
            batch_size=asr_batch_size,
        )

        text_parts: list[str] = []
        all_words: list[dict] = []
        for cs, (result_text, result_words) in zip(chunk_starts, batch_out):
            chunk_text = normalize_transcription_text(
                result_text if isinstance(result_text, str) else str(result_text)
            )
            if chunk_text:
                text_parts.append(chunk_text)
            all_words.extend(
                words_to_absolute_dicts(
                    result_words,
                    chunk_t0_sec=cs,
                    speaker_letter=letter,
                    sec_to_ms=seconds_to_json_ms,
                )
            )

        full_text = merge_chunk_texts(text_parts)
        if not full_text:
            continue

        if all_words:
            starts_ms = [w["start"] for w in all_words]
            ends_ms = [w["end"] for w in all_words]
            u0, u1 = min(starts_ms), max(ends_ms)
        else:
            u0 = seconds_to_json_ms(t0)
            u1 = seconds_to_json_ms(t1)

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


def _write_model_parameters(
    out_dir: Path,
    model_name: str,
    device: str,
    *,
    diarization: bool,
    pyannote_model: str | None,
    asr_batch_size: int,
) -> None:
    resolved = model_name
    if model_name in ("ctc", "rnnt", "e2e_ctc", "e2e_rnnt", "ssl"):
        resolved = f"v3_{model_name}"
    diar_line = (
        f"  pyannote: {pyannote_model} (метки A, B, … — по хронологии первого появления кластера)"
        if diarization and pyannote_model
        else "  pyannote: не использовалась"
    )
    diar_flag = (
        "включена (локально; A/B не идентичны между разными файлами — относительные кластеры)"
        if diarization
        else "выключена (один условный спикер «—»)"
    )
    text = f"""Параметры модели (локальный запуск GigaAM)

Модель
  GigaAM — {resolved} (локально, без AssemblyAI)

Инференс
  Устройство: {device}
  Диаризованный режим: ASR батчами (до {asr_batch_size} чанков за проход), без записи WAV на каждый чанк (тензоры в памяти).
  Макс. длина одного ASR-чанка: ~{int(MAX_ASR_CHUNK_SEC)} с; границы чанков — по паузам (RMS), не только по таймеру.

Язык
  Russian (модель ориентирована на русский речевой корпус)

Функции
  Speaker diarization: {diar_flag}
{diar_line}
  Вход pyannote: 16 kHz mono (стерео сводится в ffmpeg; в модель не подаётся 2-канальный тензор).
  Постобработка диаризации: разрешение пересечений сегментов, склейка соседних кусков одного спикера,
  слияние очень коротких врезок, лёгкое смещение границ.
  GigaAM: моно-сегменты из того же моно-волнового фронта.
  Нормализация текста: пробелы/повторы пунктуации/Unicode NFC для transcribed_text.
  Language detection: не применимо (фиксированный домен русского ASR)

Примечания
  Для HF-моделей pyannote: токен в HF_TOKEN и согласие с условиями на странице модели.
  ffmpeg/ffprobe должны быть в PATH (проверка при старте скрипта).
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
        "Спикеры A, B, …: pyannote + постобработка сегментов + GigaAM по чанкам (границы по тишине где возможно).\n"
        "Метки A/B относительны: спикер A в одном файле не соответствует A в другом.\n"
        if diarization
        else "Диаризация не выполнялась — в JSON один условный спикер.\n"
    )
    text = f"""Сравнение с пайплайном Whisper в этом репозитории

В папках test_* результат AssemblyAI (Whisper-Streaming) часто включает диаризацию и
обогащённые метаданные. Здесь используется локальная связка pyannote и GigaAM.

{diar_note}{extra}
Запуск без аргумента пути открывает диалог выбора аудио (tkinter / WinForms на Windows).

Аудио: {audio_path}
Время записи артефактов (UTC): {generated_at}
"""
    (out_dir / "info.txt").write_text(text, encoding="utf-8")


def load_gigaam_model(model_name: str, device: str):
    import gigaam

    return gigaam.load_model(
        model_name,
        device=device,
        fp16_encoder=False if device == "cpu" else True,
        use_flash=False,
    )


def run_diarization_branch(
    audio_path: str,
    hf_token: str,
    device: str,
    pyannote_model: str | None,
) -> tuple[list[tuple[float, float, str]], str, "torch.Tensor", int]:
    """Загрузка волны, pyannote, постобработка сегментов."""
    import torch

    wf, sr = load_waveform_16k_mono(audio_path)
    wf = wf.to(torch.device(device)) if device != "cpu" else wf
    raw_turns, used_model = pyannote_diarization_turns(
        wf, sr, hf_token, device, pyannote_model
    )
    if not raw_turns:
        raise RuntimeError("pyannote вернул пустой результат")
    turns = postprocess_diarization_turns(
        raw_turns,
        min_segment_sec=MIN_SEGMENT_SEC,
        merge_same_speaker_gap_sec=MERGE_SAME_SPEAKER_GAP_SEC,
        max_intrusion_sec=MAX_INTRUSION_SEC,
        boundary_pad_sec=BOUNDARY_PAD_SEC,
    )
    if not turns:
        raise RuntimeError("После постобработки не осталось сегментов диаризации")
    return turns, used_model, wf, sr


def serialize_transcribed_text(out_dir: Path, utterances: list[dict], header: str) -> None:
    transcribed_body = header + json.dumps(utterances, ensure_ascii=False, indent=2) + "\n"
    (out_dir / "transcribed_text.txt").write_text(transcribed_body, encoding="utf-8")


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
        help="Куда писать txt (по умолчанию — эта папка).",
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
        help="Не вызывать pyannote (один проход ASR на весь файл).",
    )
    parser.add_argument(
        "--pyannote-model",
        default=None,
        help="ID модели на HuggingFace (иначе встроенный список кандидатов).",
    )
    parser.add_argument(
        "--asr-batch-size",
        type=int,
        default=ASR_BATCH_SIZE,
        help=f"Размер батча для GigaAM по чанкам (по умолчанию {ASR_BATCH_SIZE}).",
    )
    args = parser.parse_args()

    out_dir = (args.out_dir or pkg_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        ensure_ffmpeg_available()
    except RuntimeError as e:
        print(str(e), file=sys.stderr)
        return 1

    try:
        import gigaam  # noqa: F401
    except ImportError:
        print(
            "Не установлен пакет gigaam/torch в этом интерпретаторе.\n"
            "Выполните из корня проекта: .\\install_requirements.bat\n"
            "или: powershell -ExecutionPolicy Bypass -File .\\install_requirements.ps1\n",
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

    model = load_gigaam_model(args.model, device)

    word_ts = not args.no_word_timestamps
    hf = _hf_token()
    use_diarize = not args.no_diarize and bool(hf)
    info_extra = ""
    pyannote_model_used: str | None = None
    utterances: list[dict]
    asr_batch = max(1, int(args.asr_batch_size))

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
            turns, pyannote_model_used, wf, sr = run_diarization_branch(
                audio_path, hf, device, args.pyannote_model
            )
            spk_map = _speaker_map_chronological(turns)
            utterances = _transcribe_diarized(
                model, wf, sr, turns, spk_map, word_ts, asr_batch
            )
            if not utterances:
                raise RuntimeError("После диаризации не получилось ни одной непустой транскрипции")
            header = "\n--- utterances (GigaAM + pyannote: спикеры A, B, …) ---\n"
        except Exception as e:  # noqa: BLE001
            err = str(e)
            print(f"Диаризация не удалась ({e}); выполняется один проход ASR.", file=sys.stderr)
            if "403" in err or "gated" in err.lower() or "restricted" in err.lower():
                print(
                    "Нужен доступ к gated-модели pyannote на huggingface.co (Agree) для модели и сегментации.\n",
                    file=sys.stderr,
                )
            use_diarize = False
            info_extra = f"Ошибка диаризации (один проход ASR): {e}\n"
            result = model.transcribe(audio_path, word_timestamps=word_ts)
            full_text = result.text if hasattr(result, "text") else str(result)
            words = result.words if hasattr(result, "words") else None
            utterances = build_utterances_payload(
                normalize_transcription_text(full_text), words, speaker="—"
            )
            header = "\n--- utterances (локальная транскрипция GigaAM, без диаризации) ---\n"
    else:
        if not args.no_diarize and not hf:
            print(
                "Подсказка: для спикеров A/B задайте HF_TOKEN и установите pyannote.audio; "
                "или используйте --no-diarize.",
                file=sys.stderr,
            )
        result = model.transcribe(audio_path, word_timestamps=word_ts)
        full_text = result.text if hasattr(result, "text") else str(result)
        words = result.words if hasattr(result, "words") else None
        utterances = build_utterances_payload(
            normalize_transcription_text(full_text), words, speaker="—"
        )
        header = "\n--- utterances (локальная транскрипция GigaAM, без диаризации) ---\n"

    serialize_transcribed_text(out_dir, utterances, header)

    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    _write_model_parameters(
        out_dir,
        args.model,
        device,
        diarization=use_diarize,
        pyannote_model=pyannote_model_used,
        asr_batch_size=asr_batch,
    )
    _write_info(out_dir, audio_path, now, diarization=use_diarize, extra=info_extra)

    print("Готово: transcribed_text.txt, model_parameters.txt, info.txt")
    print("---")
    for u in utterances:
        print(f"{u.get('speaker', '?')}: {u.get('text', '')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
