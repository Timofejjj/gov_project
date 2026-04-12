"""Пример: задайте ASSEMBLYAI_API_KEY. Запуск без аргументов открывает выбор аудио в системе.

При передаче пути в командной строке диалог не показывается:
  python3 run_diarization_example.py path/to/file.wav

Переносимый запуск (ищет .venv/venv, иначе python3 из PATH):
  ./run_diarization_example.sh
  PYTHON=/opt/homebrew/bin/python3.13 ./run_diarization_example.sh

На macOS у Python из Homebrew часто нет модуля _tkinter; тогда используется системный
диалог через osascript или запрос пути в терминале.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from audio_diarization import run_diarization


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
        title="Выберите аудиофайл",
        filetypes=[
            ("Аудио", "*.wav *.mp3 *.m4a *.flac *.ogg *.webm"),
            ("WAV", "*.wav"),
            ("Все файлы", "*.*"),
        ],
    )
    root.destroy()
    return path if path else None


def _pick_audio_path_applescript() -> str | None:
    """Нативный диалог выбора файла на macOS (без tkinter)."""
    script = r"""
try
    tell application "System Events" to activate
    set f to choose file with prompt "Выберите аудиофайл (WAV и др.)"
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
        "Tkinter недоступен. Введите полный путь к WAV/аудиофайлу и нажмите Enter:",
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

    return _pick_audio_path_stdin()


def main() -> None:
    if len(sys.argv) >= 2:
        path = sys.argv[1]
    else:
        path = pick_audio_path()
        if not path:
            print("Файл не выбран.", file=sys.stderr)
            sys.exit(1)

    result = run_diarization(path)
    if result.get("error"):
        print("Error:", result["error"], file=sys.stderr)
        sys.exit(2)
    print(result.get("transcript_text", ""))
    print("\n--- utterances (speaker diarization) ---")
    print(json.dumps(result.get("utterances") or [], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
