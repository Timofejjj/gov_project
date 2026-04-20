"""
Пайплайн v2: Silero VAD → сегменты → параллельные *логические* ветки ECAPA + ASR
→ первичная кластеризация → joint graph refinement → сглаживание → LLM → (опц.) centroid merge → turns.

Выход: JSON [{ "speaker", "start", "end", "text" }] (время в секундах).

Запуск:
  python Pipline_2_New/run_pipeline.py path/to/audio.wav --out out.json
  (по умолчанию --device auto: тяжёлые шаги на GPU при наличии CUDA/MPS)

См. Pipline_2_New/pipline_structure.txt и Pipline_1_New/run_pipeline.py (LLM, ffmpeg, GigaAM).
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import subprocess
import sys
import warnings
from pathlib import Path
from typing import Any, List, Optional, Sequence

# Запуск как `python Pipline_2_New/run_pipeline.py` — нужен корень репо на sys.path.
_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

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

import numpy as np
import torch

REPO_ROOT = _REPO

# Дефолт и fallback имени модели LLM — как в Pipline_1_New/run_pipeline.py.
_DEFAULT_LLM_MODEL = "google/gemma-4-26b-a4b-it"

from Pipline_2_New.asr import AsrBackend, load_gigaam, load_whisper, transcribe_segment
from Pipline_2_New.audio_io import crop_segment, load_waveform_16k_mono, silero_speech_intervals, vad_to_speech_segments
from Pipline_2_New.clustering import cluster_embeddings, merge_clusters_by_centroid, merge_speaker_labels_by_embedding
from Pipline_2_New.constants import MIN_UTTERANCE_SEC, REFINE_MERGE_GAP_SEC, SAMPLE_RATE
from Pipline_2_New.device_utils import describe_device, log_why_cpu_if_needed, resolve_compute_device
from Pipline_2_New.hf_env import ensure_hf_hub_token, load_dotenv_repo
from Pipline_2_New.embedding import ecapa_embedding_for_segment
from Pipline_2_New.joint_refinement import graph_partition_labels, joint_similarity_matrix
from Pipline_2_New.smoothing import hmm_smooth_labels, median_smooth_labels, remove_micro_switches


def _log(msg: str) -> None:
    print(f"[P2] {msg}", flush=True)


def _ensure_utf8_stdio() -> None:
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
            sys.stderr.reconfigure(encoding="utf-8")
        except (AttributeError, OSError):
            pass


def _load_dotenv() -> None:
    load_dotenv_repo()


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


def _prompt_num_speakers() -> int | None:
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
        title="Выберите аудиофайл (Pipline_2_New)",
        filetypes=[
            ("Аудио", "*.wav *.mp3 *.m4a *.flac *.ogg *.webm"),
            ("Все файлы", "*.*"),
        ],
    )
    root.destroy()
    return path if path else None


def _pick_audio_path_powershell() -> str | None:
    ps = r"""
Add-Type -AssemblyName System.Windows.Forms
$d = New-Object System.Windows.Forms.OpenFileDialog
$d.Title = 'Выберите аудиофайл (Pipline_2_New)'
$d.Filter = 'Аудио|*.wav;*.mp3;*.m4a;*.flac;*.ogg;*.webm|Все файлы|*.*'
if ($d.ShowDialog() -eq [System.Windows.Forms.DialogResult]::OK) {
    [Console]::Out.Write($d.FileName)
}
""".strip()
    enc = base64.b64encode(ps.encode("utf-16-le")).decode("ascii")
    try:
        proc = subprocess.run(
            ["powershell.exe", "-NoProfile", "-Sta", "-EncodedCommand", enc],
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
    print("Введите полный путь к аудиофайлу:", file=sys.stderr)
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
    if sys.platform == "win32":
        path = _pick_audio_path_powershell()
        if path:
            return path
    return _pick_audio_path_stdin()


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
    return False


def _fill_embeddings(embs: Sequence[Optional[np.ndarray]]) -> np.ndarray:
    dim = 192
    for e in embs:
        if e is not None:
            dim = int(e.shape[0])
            break
    rows: List[np.ndarray] = []
    valid = [e for e in embs if e is not None]
    mean = np.zeros((dim,), dtype=np.float32)
    if valid:
        mean = np.stack([v.astype(np.float32) for v in valid], axis=0).mean(axis=0)
    for e in embs:
        rows.append((e.astype(np.float32) if e is not None else mean.copy()))
    return np.stack(rows, axis=0)


def _speaker_str_from_int_labels(labels: np.ndarray) -> List[str]:
    return [f"SPEAKER_{int(x) + 1}" for x in labels.tolist()]


def _renumber_labels(labels: Sequence[int]) -> np.ndarray:
    labels = [int(x) for x in labels]
    mapping: dict[int, int] = {}
    out: List[int] = []
    nxt = 0
    for x in labels:
        if x not in mapping:
            mapping[x] = nxt
            nxt += 1
        out.append(mapping[x])
    return np.asarray(out, dtype=np.int64)


def build_turns(
    rows: list[dict[str, Any]],
    *,
    refine_merge_gap: float = REFINE_MERGE_GAP_SEC,
) -> list[dict[str, Any]]:
    """Склейка соседних сегментов одного спикера только если пауза ≤ refine_merge_gap (как _merge_short_gaps в Pipline_1_New)."""
    if not rows:
        return []
    rows = sorted(rows, key=lambda r: (float(r["start"]), float(r["end"])))
    acc: list[dict[str, Any]] = [
        {
            "speaker": str(rows[0]["speaker"]),
            "start": float(rows[0]["start"]),
            "end": float(rows[0]["end"]),
            "text": str(rows[0].get("text", "")),
        }
    ]
    gap = float(refine_merge_gap)
    for r in rows[1:]:
        sp = str(r["speaker"])
        t0, t1 = float(r["start"]), float(r["end"])
        tx = str(r.get("text", ""))
        prev_end = float(acc[-1]["end"])
        if sp == acc[-1]["speaker"] and (t0 - prev_end) <= gap + 1e-6:
            acc[-1]["end"] = max(acc[-1]["end"], t1)
            a = str(acc[-1].get("text", "")).strip()
            b = tx.strip()
            acc[-1]["text"] = (a + " " + b).strip() if a and b else (a or b)
        else:
            acc.append({"speaker": sp, "start": t0, "end": t1, "text": tx})
    for x in acc:
        x["start"] = round(float(x["start"]), 3)
        x["end"] = round(float(x["end"]), 3)
    return acc


def run_multimodal_pipeline(
    audio_path: str,
    *,
    device: str = "auto",
    asr_backend: AsrBackend = "gigaam",
    giga_model_name: str = "e2e_rnnt",
    whisper_model_size: str = "base",
    clusterer: str = "agglomerative",
    num_speakers: int | None = None,
    agglomerative_threshold: float = 0.53,
    hdbscan_min_cluster_size: int = 2,
    joint_alpha: float = 0.55,
    joint_beta: float = 0.25,
    joint_gamma: float = 0.20,
    joint_temporal_sigma_sec: float = 3.0,
    smooth_median_window: int = 1,
    micro_switch_sec: float = 0.15,
    use_hmm_smooth: bool = False,
) -> tuple[list[dict[str, Any]], np.ndarray]:
    """
    Полный проход до финальной склейки реплик.

    Возвращает (строки по сегментам, матрица эмбеддингов (n, dim)) для optional post-LLM centroid merge.
    """
    ap = str(Path(audio_path).expanduser().resolve())
    ensure_hf_hub_token(log=False, log_fn=None)
    _log(f"Аудио: {ap}")
    resolved = resolve_compute_device(device)
    if str(device).strip().lower().startswith("cuda") and resolved == "cpu" and not torch.cuda.is_available():
        _log("запрошена CUDA, но torch.cuda.is_available()=False — используется CPU")
    _log(f"устройство вычислений: {describe_device(resolved)}")
    log_why_cpu_if_needed(device, resolved, _log)
    wav = load_waveform_16k_mono(ap)
    total_dur = float(wav.shape[-1] / SAMPLE_RATE)
    wav_dev = wav.to(torch.device(resolved))

    _log("Silero VAD…")
    vad_iv = silero_speech_intervals(wav_dev, SAMPLE_RATE, torch_device=resolved)
    segments = vad_to_speech_segments(vad_iv, total_dur)
    segments = [(float(a), float(b)) for a, b in segments if b - a >= MIN_UTTERANCE_SEC]
    _log(f"VAD интервалов: {len(vad_iv)} → сегментов для ECAPA/ASR: {len(segments)}")
    if not segments:
        return [], np.zeros((0, 192), dtype=np.float32)

    # --- 4A / 4B: одни и те же интервалы; ветки не используют результаты друг друга ---
    _log("ветка 4A: ECAPA-TDNN по сегментам…")
    emb_list: List[Optional[np.ndarray]] = []
    for t0, t1 in segments:
        seg = crop_segment(wav_dev, t0, t1)
        emb_list.append(ecapa_embedding_for_segment(seg, device=resolved))

    _log(f"ветка 4B: ASR ({asr_backend}) по сегментам…")
    if asr_backend == "gigaam":
        asr_model = load_gigaam(giga_model_name, device=resolved)
    else:
        asr_model = load_whisper(whisper_model_size, device=resolved)

    texts: List[str] = []
    for t0, t1 in segments:
        texts.append(
            transcribe_segment(asr_backend, asr_model, wav_dev, t0, t1).strip()
        )

    X = _fill_embeddings(emb_list)
    _log("первичная кластеризация по эмбеддингам…")
    init_lab = cluster_embeddings(
        X,
        method=clusterer,
        num_speakers=num_speakers,
        agglomerative_threshold=agglomerative_threshold,
        hdbscan_min_cluster_size=hdbscan_min_cluster_size,
    )
    init_lab = _renumber_labels(init_lab.tolist())

    if num_speakers is not None and int(num_speakers) > 0:
        n_graph = int(num_speakers)
    else:
        n_graph = int(init_lab.max()) + 1 if init_lab.size else 1
    n_graph = max(1, min(n_graph, len(segments)))

    _log(
        f"joint refinement (graph): α={joint_alpha} β={joint_beta} γ={joint_gamma}, K={n_graph}…"
    )
    S = joint_similarity_matrix(
        X,
        texts,
        segments,
        alpha=joint_alpha,
        beta=joint_beta,
        gamma=joint_gamma,
        temporal_sigma_sec=joint_temporal_sigma_sec,
    )
    refined = graph_partition_labels(S, n_graph)
    refined = _renumber_labels(refined.tolist())

    _log("temporal smoothing…")
    lab_seq = median_smooth_labels(refined.tolist(), window=smooth_median_window)
    lab_seq = remove_micro_switches(segments, lab_seq, min_run_sec=micro_switch_sec)
    if use_hmm_smooth:
        lab_seq = hmm_smooth_labels(lab_seq, segments)
    refined_arr = np.asarray(lab_seq, dtype=np.int64)

    speakers = _speaker_str_from_int_labels(refined_arr)
    rows: list[dict[str, Any]] = []
    for (t0, t1), sp, tx in zip(segments, speakers, texts):
        rows.append(
            {
                "speaker": sp,
                "start": round(float(t0), 3),
                "end": round(float(t1), 3),
                "text": tx,
            }
        )

    _log(f"сегментов с разметкой: {len(rows)} (до build_turns / LLM)")
    return rows, X


def _apply_pre_llm_centroid_merge(
    rows: list[dict[str, Any]],
    emb_X: np.ndarray,
    cos_thresh: float,
) -> None:
    """In-place: схлопывание числовых кластеров SPEAKER_* по близости центроидов."""
    if not rows or len(rows) != int(emb_X.shape[0]):
        return
    lab: list[int] = []
    for r in rows:
        s = str(r.get("speaker", "")).strip().upper()
        if s.startswith("SPEAKER_") and s[8:].isdigit():
            lab.append(int(s[8:]) - 1)
        else:
            lab.append(0)
    arr = np.asarray(lab, dtype=np.int64)
    merged = merge_clusters_by_centroid(emb_X, arr, merge_cosine_min=float(cos_thresh))
    for r, k in zip(rows, merged.tolist()):
        r["speaker"] = f"SPEAKER_{int(k) + 1}"


def main() -> int:
    _ensure_utf8_stdio()
    _load_dotenv()
    ensure_hf_hub_token(log=True, log_fn=_log)
    parser = argparse.ArgumentParser(
        description="Диаризация v2: VAD → ECAPA + ASR → graph joint refinement (без pyannote)"
    )
    parser.add_argument("audio", nargs="?", default=None, help="Путь к аудио")
    parser.add_argument("--out", type=Path, default=None, help="JSON выход")
    parser.add_argument(
        "--device",
        default="auto",
        help="auto (CUDA/MPS при наличии) | cuda | cuda:N | mps | cpu",
    )
    parser.add_argument("--asr", choices=("gigaam", "whisper"), default="gigaam")
    parser.add_argument("--giga-model", default="e2e_rnnt", help="gigaam.load_model")
    parser.add_argument("--whisper-model", default="base", help="whisper.load_model size")
    parser.add_argument("--clusterer", choices=("agglomerative", "hdbscan"), default="agglomerative")
    parser.add_argument("--num-speakers", type=int, default=None)
    parser.add_argument(
        "--agg-threshold",
        type=float,
        default=0.53,
        help="Порог distance_threshold для Agglomerative (precomputed 1−cos). Меньше — меньше риск склеить похожие голоса.",
    )
    parser.add_argument(
        "--cosine-sim-threshold",
        type=float,
        default=None,
        help="Альтернатива: порог cosine similarity; будет agg_threshold = 1 − sim (как в Pipline_1_New)",
    )
    parser.add_argument("--hdbscan-min-size", type=int, default=2)
    parser.add_argument("--joint-alpha", type=float, default=0.55)
    parser.add_argument("--joint-beta", type=float, default=0.25)
    parser.add_argument("--joint-gamma", type=float, default=0.20)
    parser.add_argument("--joint-temporal-sigma", type=float, default=3.0)
    parser.add_argument("--smooth-window", type=int, default=1)
    parser.add_argument("--micro-switch-sec", type=float, default=0.15)
    parser.add_argument("--hmm-smooth", action="store_true", help="Опционально hmmlearn CategoricalHMM")
    parser.add_argument(
        "--pre-llm-centroid-merge",
        action="store_true",
        help="До LLM: схлопнуть кластеры по близости центроидов эмбеддингов (на уровне сегментов)",
    )
    parser.add_argument("--pre-llm-centroid-cos", type=float, default=0.92)
    parser.add_argument(
        "--post-llm-centroid-merge",
        action="store_true",
        help="После LLM: если число строк не изменилось — объединить близкие по эмбеддингу метки спикеров",
    )
    parser.add_argument("--post-llm-centroid-cos", type=float, default=0.92)
    parser.add_argument(
        "--refine-gap",
        type=float,
        default=REFINE_MERGE_GAP_SEC,
        help="Склейка соседних фраз одного спикера в turns, если пауза ≤ N с (0 — только стык без зазора)",
    )
    parser.add_argument(
        "--merge-off",
        action="store_true",
        help="Отключить склейку по паузе (эквивалент --refine-gap 0)",
    )
    parser.add_argument("--use-llm", action="store_true")
    parser.add_argument("--no-llm", action="store_true")
    parser.add_argument(
        "--llm-model",
        default=_DEFAULT_LLM_MODEL,
        help="Модель для LLM-коррекции (как в Pipline_1_New)",
    )
    parser.add_argument(
        "--llm-base-url",
        default="https://ai.gov.by/api/openai/v1",
        help="Базовый URL OpenAI-совместимого API (если задан — выставляет OPENAI_BASE_URL на время запуска)",
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
            print("Файл не выбран.", file=sys.stderr)
            return 2
        args.audio = picked

    audio_path = str(Path(args.audio).expanduser().resolve())
    ap = Path(audio_path)
    out_path = args.out or ap.with_name(ap.stem + "_diarization_asr_v2.json")
    if out_path.suffix.lower() != ".json":
        out_path = out_path.with_suffix(".json")

    num_speakers = args.num_speakers
    if num_speakers is None:
        num_speakers = _prompt_num_speakers()
    if num_speakers is None:
        env_ns = (os.getenv("NUM_SPEAKERS") or "").strip()
        if env_ns.isdigit():
            num_speakers = int(env_ns)

    openai_key = (os.getenv("OPENAI_API_KEY") or "").strip()
    auto_llm = bool(openai_key) and _truthy_env("USE_LLM", True)
    use_llm = (auto_llm or bool(args.use_llm)) and (not bool(args.no_llm))
    if use_llm and not openai_key:
        print("LLM: OPENAI_API_KEY не найден — LLM выключено", file=sys.stderr, flush=True)
        use_llm = False

    llm_model = (os.getenv("LLM_MODEL") or "").strip() or None

    if use_llm and num_speakers is None and sys.stdin.isatty():
        try:
            s = input(
                "Сколько говорящих ожидается (целое число)? [Enter = неизвестно]: "
            ).strip()
        except EOFError:
            s = ""
        if s.isdigit():
            num_speakers = int(s)

    if use_llm:
        llm_preview = (
            llm_model
            or str(args.llm_model).strip()
            or _DEFAULT_LLM_MODEL
        ).strip()
        print(
            "LLM: включено ("
            f"USE_LLM={os.getenv('USE_LLM')!r}, "
            f"model={llm_preview!r}"
            ")",
            flush=True,
        )
    else:
        print("LLM: выключено", flush=True)

    llm_base = (args.llm_base_url or "").strip()
    if llm_base:
        os.environ["OPENAI_BASE_URL"] = llm_base

    dev_arg = str(args.device).strip() or "auto"
    try:
        seg_rows, emb_X = run_multimodal_pipeline(
            audio_path,
            device=dev_arg,
            asr_backend=args.asr,
            giga_model_name=args.giga_model,
            whisper_model_size=args.whisper_model,
            clusterer=args.clusterer,
            num_speakers=num_speakers,
            agglomerative_threshold=args.agg_threshold,
            hdbscan_min_cluster_size=args.hdbscan_min_size,
            joint_alpha=args.joint_alpha,
            joint_beta=args.joint_beta,
            joint_gamma=args.joint_gamma,
            joint_temporal_sigma_sec=args.joint_temporal_sigma,
            smooth_median_window=args.smooth_window,
            micro_switch_sec=args.micro_switch_sec,
            use_hmm_smooth=bool(args.hmm_smooth),
        )
    except Exception as e:
        print(f"Ошибка: {e}", file=sys.stderr)
        return 1

    rows: list[dict[str, Any]] = [dict(r) for r in seg_rows]

    if args.pre_llm_centroid_merge and rows:
        _apply_pre_llm_centroid_merge(rows, emb_X, float(args.pre_llm_centroid_cos))

    n_before_llm = len(rows)
    if use_llm:
        from Pipline_2_New.llm_post import llm_speaker_correction

        llm_model_final = (
            llm_model or str(args.llm_model).strip() or _DEFAULT_LLM_MODEL
        ).strip()
        llm_base_url_final = (
            (str(args.llm_base_url).strip() or None)
            if args.llm_base_url is not None
            else None
        )
        try:
            rows = llm_speaker_correction(
                rows,
                model=llm_model_final,
                num_speakers=num_speakers,
                temperature=_llm_temperature(),
                timeout_sec=_llm_timeout_sec(),
                second_pass=_llm_second_pass_default(),
                base_url=llm_base_url_final,
            )
        except Exception as e:
            print(f"LLM: ошибка ({e}) — оставляем без LLM.", file=sys.stderr)

    if (
        args.post_llm_centroid_merge
        and rows
        and len(rows) == n_before_llm
        and len(rows) == int(emb_X.shape[0])
    ):
        rows = merge_speaker_labels_by_embedding(
            rows, emb_X, cos_thresh=float(args.post_llm_centroid_cos)
        )
    elif args.post_llm_centroid_merge and len(rows) != n_before_llm:
        _log("post-LLM centroid merge пропущен: LLM изменил число сегментов")

    rows = build_turns(rows, refine_merge_gap=float(args.refine_gap))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Записано: {out_path} ({len(rows)} сегментов)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
