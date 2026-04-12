"""Пример: задайте ASSEMBLYAI_API_KEY. Запуск без аргументов открывает выбор аудио в системе.

При передаче пути в командной строке диалог не показывается:
  python run_diarization_example.py path\\to\\file.wav

Переносимый запуск (ищет .venv/venv, иначе python из PATH):

  macOS/Linux:
    ./run_diarization_example.sh
    PYTHON=/opt/homebrew/bin/python3.13 ./run_diarization_example.sh

  Windows (из каталога проекта):
    run_diarization_example.bat
    powershell -ExecutionPolicy Bypass -File .\\run_diarization_example.ps1
    .venv\\Scripts\\python.exe run_diarization_example.py

  Переменная PYTHON (полный путь к интерпретатору) поддерживается в .sh / .ps1 / .bat.

  Первый запуск (langgraph, assemblyai и др.): install_requirements.bat или
  install_requirements.ps1, либо: выбранный_python.exe -m pip install -r requirements.txt

Если нет tkinter: на macOS используется osascript; на Windows — диалог через PowerShell
(WinForms); иначе запрашивается путь в терминале.
"""

from __future__ import annotations

import base64
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


def _pick_audio_path_powershell() -> str | None:
    """Нативный диалог выбора файла на Windows (без tkinter), через WinForms."""
    ps = r"""
Add-Type -AssemblyName System.Windows.Forms
$d = New-Object System.Windows.Forms.OpenFileDialog
$d.Title = 'Выберите аудиофайл'
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

    if sys.platform == "win32":
        path = _pick_audio_path_powershell()
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
