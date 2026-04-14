"""
Пайплайн: VAD (Silero) → pyannote segmentation-3.0 → чанки по времени → ECAPA (SpeechBrain)
→ кластеризация (Agglomerative / HDBSCAN) → опциональное сглаживание → GigaAM ASR.

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
import subprocess
import sys
import tempfile
import warnings
import wave
from pathlib import Path

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
from pyannote.audio import Audio, Inference
from pyannote.audio.pipelines.utils import get_model
from pyannote.audio.pipelines.utils.diarization import SpeakerDiarizationMixin
from pyannote.audio.utils.signal import binarize
from pyannote.core import Annotation, SlidingWindowFeature

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


def _ensure_utf8_stdio() -> None:
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
            sys.stderr.reconfigure(encoding="utf-8")
        except (AttributeError, OSError):
            pass


def _load_dotenv() -> None:
    load_dotenv(REPO_ROOT / ".env")


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
    min_silence_ms: int = 100,
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
        return_seconds=True,
    )
    return [(float(x["start"]), float(x["end"])) for x in ts]


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
) -> list[dict]:
    """(7) ASR по сегментам; (3) длинные режем по MAX_ASR_CHUNK_SEC."""
    total = wav_mono.shape[-1]
    rows: list[dict] = []
    for t0, t1, spk in turns:
        if t1 - t0 < MIN_UTTERANCE_SEC:
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


def run_pipeline(
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
) -> list[dict]:
    hf = _hf_token()
    if not hf:
        raise RuntimeError("Нужен HF_TOKEN в окружении или в .env для pyannote/segmentation-3.0")

    wav = load_waveform_16k_mono(audio_path)
    wav = wav.to(torch.device(device))

    # (1) VAD
    vad_intervals = silero_speech_intervals(wav.cpu(), SAMPLE_RATE)

    # (2) segmentation-3.0
    seg_model = get_model("pyannote/segmentation-3.0", token=hf)
    seg_model.eval()
    seg_model.to(torch.device(device))
    specs = seg_model.specifications
    spec0 = specs[0] if isinstance(specs, tuple) else specs
    duration = spec0.duration
    step = max(duration * 0.01, segmentation_step_ratio * duration)
    seg_inf = Inference(
        seg_model,
        duration=duration,
        step=step,
        batch_size=8,
        device=torch.device(device),
    )
    file_dict: dict = {"uri": Path(audio_path).stem, "waveform": wav, "sample_rate": SAMPLE_RATE}
    segmentations = seg_inf(file_dict)

    if spec0.powerset:
        binarized = segmentations
    else:
        binarized = binarize(segmentations, onset=0.5, initial_state=False)

    receptive_field = seg_model.receptive_field
    count = SpeakerDiarizationMixin.speaker_count(
        binarized,
        receptive_field,
        warm_up=(0.0, 0.0),
    )

    if float(np.nanmax(count.data)) == 0.0:
        return []

    # (4) ECAPA embeddings по маскам (аналог pyannote get_embeddings, но SpeechBrain)
    from speechbrain.inference.classifiers import EncoderClassifier
    from speechbrain.utils.fetching import LocalStrategy

    sb_dir = REPO_ROOT / "Pipline_1_New" / "_ecapa_pretrained"
    sb_dir.mkdir(parents=True, exist_ok=True)
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

    num_chunks, num_frames, local_spk = binarized.data.shape
    embeddings = np.full((num_chunks, local_spk, emb_dim), np.nan, dtype=np.float32)
    _audio = Audio(sample_rate=SAMPLE_RATE, mono="downmix")

    for c, (chunk, masks) in enumerate(binarized):
        waveform, _ = _audio.crop(file_dict, chunk, mode="pad")
        w = waveform.float().to(device)
        chunk_len_sec = float(w.shape[-1] / SAMPLE_RATE)
        rel_centers = (np.arange(num_frames, dtype=np.float64) + 0.5) * (
            chunk_len_sec / max(num_frames, 1)
        )

        for s_idx in range(local_spk):
            mask = np.nan_to_num(masks[:, s_idx], nan=0.0).astype(np.float32)
            if np.sum(mask > 0.25) < max(3, int(0.05 * num_frames)):
                continue
            rel = np.linspace(0.0, chunk_len_sec, w.shape[-1], endpoint=False)
            w_np = w.squeeze(0).detach().cpu().numpy()
            mask_audio = np.clip(np.interp(rel, rel_centers, mask), 0.0, 1.0)
            x = torch.from_numpy(w_np * mask_audio).float().unsqueeze(0).to(device)
            if float(torch.sqrt(torch.mean(x**2))) < 1e-4:
                continue
            with torch.inference_mode():
                emb = encoder.encode_batch(x)
            vec = emb.squeeze(0).detach().cpu().numpy().reshape(-1)
            embeddings[c, s_idx] = vec.astype(np.float32)

    # (5) clustering → hard_clusters (num_chunks, local_spk)
    active_cs: list[tuple[int, int]] = []
    flat: list[np.ndarray] = []
    for c in range(num_chunks):
        for s in range(local_spk):
            v = embeddings[c, s]
            if np.any(np.isnan(v)):
                continue
            flat.append(v)
            active_cs.append((c, s))
    flat_arr = np.stack(flat, axis=0) if flat else np.zeros((0, emb_dim), dtype=np.float32)

    hard_clusters = np.full((num_chunks, local_spk), -2, dtype=np.int64)
    if flat_arr.shape[0] == 0:
        return []

    norms = np.linalg.norm(flat_arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    X = flat_arr / norms

    if clusterer == "hdbscan":
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
    else:
        from sklearn.cluster import AgglomerativeClustering

        n = X.shape[0]
        if n == 1:
            labels = np.zeros((1,), dtype=np.int64)
        else:
            agg = AgglomerativeClustering(
                n_clusters=None,
                distance_threshold=agglomerative_threshold,
                linkage="average",
                metric="cosine",
            )
            labels = agg.fit_predict(X).astype(np.int64)

    for i, (c, s) in enumerate(active_cs):
        hard_clusters[c, s] = int(labels[i])

    inactive = np.sum(binarized.data, axis=1) == 0
    hard_clusters[inactive] = -2
    if int(np.max(hard_clusters)) < 0:
        return []

    count_exc = SlidingWindowFeature(
        np.minimum(count.data, 1).astype(np.int8),
        count.sliding_window,
    )
    discrete_exc = _reconstruct_discrete(segmentations, hard_clusters, count_exc)
    ann = SpeakerDiarizationMixin.to_annotation(
        discrete_exc,
        min_duration_on=0.0,
        min_duration_off=0.1,
    )

    turns = _annotation_to_turns(ann)
    lab_map = _speaker_label_chronological(
        [(a, b, str(int(x)) if str(x).isdigit() else str(x)) for a, b, x in turns]
    )
    turns_named: list[tuple[float, float, str]] = []
    for a, b, x in turns:
        k = str(int(x)) if str(x).isdigit() else str(x)
        turns_named.append((a, b, lab_map.get(k, k)))

    if refine_merge_gap > 0:
        turns_named = _merge_short_gaps(turns_named, refine_merge_gap)

    if not skip_vad_trim and vad_intervals:
        clipped: list[tuple[float, float, str]] = []
        for a, b, spk in turns_named:
            for ca, cb in _clip_segment_to_vad(a, b, vad_intervals):
                clipped.append((ca, cb, spk))
        turns_named = clipped

    try:
        import gigaam
    except ImportError as e:
        raise RuntimeError(
            "Установите gigaam (см. requirements_gigaam.txt в корне проекта)"
        ) from e

    gmodel = gigaam.load_model(
        giga_model_name,
        device=device,
        fp16_encoder=False if device == "cpu" else True,
        use_flash=False,
    )
    return transcribe_segments_gigaam(
        gmodel, wav.cpu(), SAMPLE_RATE, turns_named, word_timestamps=False
    )


def main() -> int:
    _ensure_utf8_stdio()
    _load_dotenv()
    parser = argparse.ArgumentParser(description="Диаризация (Silero + segmentation-3.0 + ECAPA + ASR GigaAM)")
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
        "--agg-threshold",
        type=float,
        default=0.55,
        help="Порог distance_threshold для Agglomerative (cosine), меньше — больше спикеров",
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
    args = parser.parse_args()
    if not args.audio or not str(args.audio).strip():
        picked = pick_audio_path()
        if not picked:
            print("Файл не выбран — выход.", file=sys.stderr)
            return 2
        args.audio = picked
    audio_path = str(Path(args.audio).expanduser().resolve())
    print(f"Аудио: {audio_path}", flush=True)
    ap = Path(audio_path)
    out_path = args.out or ap.with_name(ap.stem + "_diarization_asr.json")
    if out_path.suffix.lower() != ".json":
        out_path = out_path.with_suffix(".json")

    try:
        rows = run_pipeline(
            audio_path,
            device=args.device,
            giga_model_name=args.giga_model,
            clusterer=args.clusterer,
            agglomerative_threshold=args.agg_threshold,
            hdbscan_min_cluster_size=args.hdbscan_min_size,
            refine_merge_gap=args.refine_gap,
            skip_vad_trim=args.skip_vad_trim,
            segmentation_step_ratio=args.seg_step_ratio,
        )
    except Exception as e:
        print(f"Ошибка: {e}", file=sys.stderr)
        return 1

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Записано: {out_path} ({len(rows)} сегментов)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
