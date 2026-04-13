"""
Транскрипция через GigaAM с теми же артефактами, что и Giga_AM (model_parameters, info, transcribed_text).

Отличие от Giga_AM: вместо pyannote (gated HF-пайплайны) — диаризация по эмбеддингам WavLM
(transformers): скользящие окна → вектор признаков → кластеризация → сегменты спикеров.
Веса WavLM по умолчанию подгружаются с Hugging Face Hub как публичная модель (токен не нужен).

Запуск из корня репозитория:
  .venv\\Scripts\\python.exe Giga_AM2\\run_transcribe.py
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
MAX_ASR_CHUNK_SEC = 20.0
MIN_SEGMENT_SEC = 0.12

# Окна для эмбеддингов WavLM (секунды)
WAVLM_WIN_SEC = 0.65
WAVLM_HOP_SEC = 0.20

_DEFAULT_WAVLM_MODEL = "microsoft/wavlm-base-plus"


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
        title="Выберите аудиофайл (GigaAM2 / WavLM)",
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
$d.Title = 'Выберите аудиофайл (GigaAM2)'
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
    set f to choose file with prompt "Выберите аудиофайл (GigaAM2)"
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
    """16 kHz: как в Giga_AM — моно как моно, ≥2 канала → стерео (для согласованности входа)."""
    from subprocess import run

    import numpy as np
    import torch

    ch_in = _ffprobe_audio_channels(audio_path)
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


def _estimate_num_speakers(embs: "object", max_spk: int = 8) -> int:
    import numpy as np
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import normalize

    X = np.asarray(embs, dtype=np.float64)
    X = normalize(X, norm="l2")
    n = X.shape[0]
    if n < 4:
        return min(2, max(1, n))
    best_k, best_score = 2, -1.0
    upper = min(max_spk + 1, n)
    for k in range(2, upper):
        try:
            lab = AgglomerativeClustering(
                n_clusters=k, linkage="average", metric="cosine"
            ).fit_predict(X)
            if len(set(lab)) < 2:
                continue
            sc = float(silhouette_score(X, lab, metric="cosine"))
            if sc > best_score:
                best_score, best_k = sc, k
        except Exception:
            continue
    return best_k if best_score >= 0 else 2


def _wavlm_window_embeddings(
    mono_wav: "object",
    sample_rate: int,
    device: str,
    model_id: str,
    win_sec: float,
    hop_sec: float,
) -> tuple[list[tuple[int, int]], "object"]:
    """mono_wav: torch (1, T) или (T,). Возвращает список (i0,i1) в сэмплах и матрицу эмбеддингов (N, H)."""
    import numpy as np
    import torch
    from transformers import AutoFeatureExtractor, WavLMModel

    if mono_wav.dim() == 1:
        w = mono_wav.float()
    else:
        w = mono_wav.squeeze(0).float()
    w = w.detach().cpu()
    n_tot = int(w.shape[0])
    if n_tot < int(0.05 * sample_rate):
        return [], torch.zeros(0, 1)

    win = max(int(win_sec * sample_rate), int(0.25 * sample_rate))
    hop = max(int(hop_sec * sample_rate), int(0.08 * sample_rate))
    if n_tot < win:
        spans = [(0, n_tot)]
    else:
        spans = []
        s = 0
        while s < n_tot:
            e = min(s + win, n_tot)
            spans.append((s, e))
            if e >= n_tot:
                break
            s += hop

    dev = torch.device(device)
    extractor = AutoFeatureExtractor.from_pretrained(model_id)
    mdl = WavLMModel.from_pretrained(model_id).to(dev)
    mdl.eval()

    # По одному окну: батч переменной длины с padding=True даёт рассинхрон маски/внимания в WavLM на части сборок torch/transformers.
    embs_list: list[torch.Tensor] = []
    for i0, i1 in spans:
        x = w[i0:i1].numpy().astype(np.float32)
        inputs = extractor(
            x,
            sampling_rate=sample_rate,
            return_tensors="pt",
            padding=True,
        )
        inputs = {k: v.to(dev) for k, v in inputs.items()}
        with torch.no_grad():
            out = mdl(**inputs)
        h = out.last_hidden_state
        # attention_mask в длине входных сэмплов, last_hidden_state — после CNN (другая T); для одного окна усредняем по кадрам.
        pooled = h.mean(dim=1)
        embs_list.append(pooled.cpu())

    embs = torch.cat(embs_list, dim=0) if embs_list else torch.zeros(0, 1)
    return spans, embs


def _wavlm_diarization_turns(
    waveform,
    sample_rate: int,
    device: str,
    wavlm_model_id: str,
    num_speakers: int,
    win_sec: float,
    hop_sec: float,
) -> tuple[list[tuple[float, float, str]], str]:
    import numpy as np
    import torch
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.preprocessing import normalize

    wf = waveform.float().cpu()
    if wf.shape[0] > 1:
        mono = wf.mean(dim=0, keepdim=True)
    else:
        mono = wf

    spans, embs = _wavlm_window_embeddings(
        mono, sample_rate, device, wavlm_model_id, win_sec, hop_sec
    )
    if not spans or embs.shape[0] == 0:
        dur = float(wf.shape[-1]) / float(sample_rate)
        return [(0.0, dur, "SPK_0")], wavlm_model_id

    X = normalize(embs.numpy().astype(np.float64), norm="l2")
    n = X.shape[0]
    if num_speakers <= 0:
        k = _estimate_num_speakers(X, max_spk=8)
    else:
        k = min(num_speakers, n)
    if k < 2:
        k = min(2, n)
    if k < 2:
        dur = float(wf.shape[-1]) / float(sample_rate)
        return [(0.0, dur, "SPK_0")], wavlm_model_id

    labels = AgglomerativeClustering(
        n_clusters=k, linkage="average", metric="cosine"
    ).fit_predict(X)

    raw_turns: list[tuple[float, float, int]] = []
    for (i0, i1), lab in zip(spans, labels):
        t0 = i0 / float(sample_rate)
        t1 = i1 / float(sample_rate)
        if t1 - t0 < 1e-4:
            continue
        raw_turns.append((t0, t1, int(lab)))

    raw_turns.sort(key=lambda x: (x[0], x[1]))
    merged: list[tuple[float, float, int]] = []
    for t0, t1, lab in raw_turns:
        if not merged:
            merged.append((t0, t1, lab))
            continue
        p0, p1, pl = merged[-1]
        if lab == pl and t0 <= p1 + (hop_sec * 1.5):
            merged[-1] = (p0, max(p1, t1), pl)
        else:
            merged.append((t0, t1, lab))

    turns: list[tuple[float, float, str]] = [
        (a, b, f"SPK_{c}") for a, b, c in merged if b - a >= MIN_SEGMENT_SEC * 0.5
    ]
    if not turns:
        dur = float(wf.shape[-1]) / float(sample_rate)
        return [(0.0, dur, "SPK_0")], wavlm_model_id
    return turns, wavlm_model_id


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
        x = mono_float.squeeze(0)
        if x.is_cuda:
            x = x.cpu()
        x = x.detach().to(torch.float32).clamp(-1.0, 1.0)
        x_i16 = (x * 32767.0).round().to(torch.int16).numpy()
        with wave.open(path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
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
    wavlm_model: str | None,
    num_speakers: int,
    win_sec: float,
    hop_sec: float,
) -> None:
    resolved = model_name
    if model_name in ("ctc", "rnnt", "e2e_ctc", "e2e_rnnt", "ssl"):
        resolved = f"v3_{model_name}"
    diar_line = (
        f"  WavLM: {wavlm_model} (окна {win_sec} с, шаг {hop_sec} с; "
        f"число кластеров: {'авто (silhouette)' if num_speakers <= 0 else num_speakers})"
        if diarization and wavlm_model
        else "  WavLM: не использовалась"
    )
    diar_flag = (
        "включена (эмбеддинги WavLM + AgglomerativeClustering; метки A/B — относительные)"
        if diarization
        else "выключена (один проход ASR)"
    )
    text = f"""Параметры модели (локальный запуск GigaAM2)

Модель ASR
  GigaAM — {resolved} (локально)

Диаризация (вместо pyannote из Giga_AM)
  {diar_flag}
{diar_line}
  Вход для WavLM: моно (среднее по каналам после загрузки 16 kHz; исходный файл как в Giga_AM: стерео при ≥2 каналах).
  Кластеризация: sklearn AgglomerativeClustering (cosine, average linkage).

Инференс
  Устройство: {device}
  Режим: transcribe по сегментам (каждый кусок до ~{int(MAX_ASR_CHUNK_SEC)} с)

Язык
  Russian (GigaAM)

Примечания
  Веса WavLM загружаются через transformers (публичные чекпоинты Hub; токен не обязателен).
  Качество диаризации проще, чем у end-to-end pyannote; для точных границ лучше pyannote или специализированные системы.
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
        "Спикеры A, B, …: WavLM-эмбеддинги по окнам + кластеризация + GigaAM на каждый сегмент.\n"
        if diarization
        else "Диаризация не выполнялась — один проход ASR.\n"
    )
    text = f"""Giga_AM2 — тот же идеологический пайплайн, что Giga_AM, но диаризация на WavLM, а не pyannote.

{diar_note}{extra}
Запуск без аргумента открывает диалог выбора аудио.

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
    parser = argparse.ArgumentParser(description="GigaAM + WavLM-диаризация → артефакты как в test_*")
    parser.add_argument(
        "audio",
        nargs="?",
        default=None,
        help="Путь к аудио. Если не указан — диалог выбора.",
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
        help="Куда писать txt (по умолчанию — папка Giga_AM2).",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Устройство torch (cpu/cuda).",
    )
    parser.add_argument(
        "--no-word-timestamps",
        action="store_true",
        help="Не запрашивать пословную разметку.",
    )
    parser.add_argument(
        "--no-diarize",
        action="store_true",
        help="Один проход ASR без WavLM-диаризации.",
    )
    parser.add_argument(
        "--wavlm-model",
        default=_DEFAULT_WAVLM_MODEL,
        help="ID модели WavLM на Hugging Face Hub (например microsoft/wavlm-base-plus).",
    )
    parser.add_argument(
        "--num-speakers",
        type=int,
        default=0,
        help="Число кластеров (спикеров). 0 — оценка по silhouette (2..8).",
    )
    parser.add_argument(
        "--wavlm-win-sec",
        type=float,
        default=WAVLM_WIN_SEC,
        help=f"Длина окна для эмбеддинга (по умолчанию {WAVLM_WIN_SEC}).",
    )
    parser.add_argument(
        "--wavlm-hop-sec",
        type=float,
        default=WAVLM_HOP_SEC,
        help=f"Шаг окна в секундах (по умолчанию {WAVLM_HOP_SEC}).",
    )
    args = parser.parse_args()

    out_dir = (args.out_dir or pkg_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        import gigaam
    except ImportError:
        print(
            "Не установлен пакет gigaam.\n"
            "Выполните: .\\install_requirements.bat\n"
            "или pip install -r requirements_gigaam.txt",
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
    print(f"GigaAM: {args.model}, устройство: {device}", flush=True)

    model = gigaam.load_model(
        args.model,
        device=device,
        fp16_encoder=False if device == "cpu" else True,
        use_flash=False,
    )

    word_ts = not args.no_word_timestamps
    use_diarize = not args.no_diarize
    info_extra = ""
    wavlm_used: str | None = None
    utterances: list[dict]

    if use_diarize:
        try:
            import transformers  # noqa: F401
            import sklearn  # noqa: F401
        except ImportError:
            print(
                "Диаризация отключена: нужны transformers и scikit-learn.\n"
                "Установите: pip install -r requirements_giga_am2.txt",
                file=sys.stderr,
            )
            use_diarize = False
            info_extra = "transformers/sklearn не установлены — см. requirements_giga_am2.txt\n"

    if use_diarize:
        try:
            wf, sr = _load_waveform_16k_for_diarization(audio_path)
            turns, wavlm_used = _wavlm_diarization_turns(
                wf,
                sr,
                device,
                args.wavlm_model,
                args.num_speakers,
                args.wavlm_win_sec,
                args.wavlm_hop_sec,
            )
            if not turns:
                raise RuntimeError("WavLM-диаризация вернула пустой результат")
            spk_map = _speaker_map_chronological(turns)
            utterances = _transcribe_diarized(
                model, wf, sr, turns, spk_map, word_ts
            )
            if not utterances:
                raise RuntimeError("После диаризации нет непустых транскрипций")
            header = "\n--- utterances (GigaAM + WavLM: спикеры A, B, …) ---\n"
        except Exception as e:  # noqa: BLE001
            print(f"Диаризация не удалась ({e}); один проход ASR.", file=sys.stderr)
            use_diarize = False
            info_extra = f"Ошибка WavLM-диаризации: {e}\n"
            result = model.transcribe(audio_path, word_timestamps=word_ts)
            full_text = result.text if hasattr(result, "text") else str(result)
            words = result.words if hasattr(result, "words") else None
            utterances = _build_utterances_payload(full_text, words, speaker="—")
            header = "\n--- utterances (GigaAM без диаризации) ---\n"
    else:
        result = model.transcribe(audio_path, word_timestamps=word_ts)
        full_text = result.text if hasattr(result, "text") else str(result)
        words = result.words if hasattr(result, "words") else None
        utterances = _build_utterances_payload(full_text, words, speaker="—")
        header = "\n--- utterances (GigaAM без диаризации) ---\n"

    transcribed_body = header + json.dumps(utterances, ensure_ascii=False, indent=2) + "\n"
    (out_dir / "transcribed_text.txt").write_text(transcribed_body, encoding="utf-8")

    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    _write_model_parameters(
        out_dir,
        args.model,
        device,
        diarization=use_diarize,
        wavlm_model=wavlm_used,
        num_speakers=args.num_speakers,
        win_sec=args.wavlm_win_sec,
        hop_sec=args.wavlm_hop_sec,
    )
    _write_info(out_dir, audio_path, now, diarization=use_diarize, extra=info_extra)

    print("Готово: transcribed_text.txt, model_parameters.txt, info.txt")
    print("---")
    for u in utterances:
        print(f"{u.get('speaker', '?')}: {u.get('text', '')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
