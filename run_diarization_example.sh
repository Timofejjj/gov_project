#!/usr/bin/env bash
# Запуск без жёсткого пути к python3.12: подойдёт любой python3 или venv проекта.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

for candidate in "${PYTHON:-}" "$ROOT/.venv/bin/python" "$ROOT/venv/bin/python"; do
  if [[ -n "$candidate" && -x "$candidate" ]]; then
    exec "$candidate" "$ROOT/run_diarization_example.py" "$@"
  fi
done

if command -v python3 >/dev/null 2>&1; then
  exec python3 "$ROOT/run_diarization_example.py" "$@"
fi

echo "Не найден интерпретатор. Установите Python 3, создайте venv (python3 -m venv .venv)" >&2
echo "или укажите: PYTHON=/path/to/python $0" >&2
exit 1
