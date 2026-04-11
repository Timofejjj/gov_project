"""Пример: задайте ASSEMBLYAI_API_KEY. Запуск без аргументов открывает выбор аудио в системе.

При передаче пути в командной строке диалог не показывается:
  python run_diarization_example.py path/to/file.wav
"""

from __future__ import annotations

import json
import sys
import tkinter as tk
from tkinter import filedialog

from audio_diarization import run_diarization


def pick_audio_path() -> str | None:
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
