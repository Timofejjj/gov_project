@echo off
setlocal
cd /d "%~dp0"

if defined PYTHON if exist "%PYTHON%" (
  "%PYTHON%" -m pip install -r "%~dp0requirements.txt"
  exit /b %ERRORLEVEL%
)
if exist "%~dp0.venv\Scripts\python.exe" (
  "%~dp0.venv\Scripts\python.exe" -m pip install -r "%~dp0requirements.txt"
  exit /b %ERRORLEVEL%
)
if exist "%~dp0venv\Scripts\python.exe" (
  "%~dp0venv\Scripts\python.exe" -m pip install -r "%~dp0requirements.txt"
  exit /b %ERRORLEVEL%
)
where py >nul 2>&1 && (
  py -3 -m pip install -r "%~dp0requirements.txt"
  exit /b %ERRORLEVEL%
)
where python >nul 2>&1 && (
  python -m pip install -r "%~dp0requirements.txt"
  exit /b %ERRORLEVEL%
)

echo Не найден интерпретатор. Установите Python 3, создайте venv: python -m venv .venv >&2
echo или задайте PYTHON=путь\к\python.exe >&2
exit /b 1
