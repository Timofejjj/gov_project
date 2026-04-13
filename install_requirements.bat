@echo off
setlocal
cd /d "%~dp0"

set "REQ=%~dp0requirements.txt"
set "REQG=%~dp0requirements_gigaam.txt"
set "REQPY=%~dp0requirements_pyannote.txt"
set "REQAM2=%~dp0requirements_giga_am2.txt"

if defined PYTHON if exist "%PYTHON%" (
  "%PYTHON%" -m pip install -r "%REQ%"
  if errorlevel 1 exit /b %ERRORLEVEL%
  if exist "%REQG%" "%PYTHON%" -m pip install -r "%REQG%"
  if errorlevel 1 exit /b %ERRORLEVEL%
  if exist "%REQPY%" "%PYTHON%" -m pip install -r "%REQPY%"
  if errorlevel 1 exit /b %ERRORLEVEL%
  if exist "%REQAM2%" "%PYTHON%" -m pip install -r "%REQAM2%"
  exit /b %ERRORLEVEL%
)
if exist "%~dp0.venv\Scripts\python.exe" (
  "%~dp0.venv\Scripts\python.exe" -m pip install -r "%REQ%"
  if errorlevel 1 exit /b %ERRORLEVEL%
  if exist "%REQG%" "%~dp0.venv\Scripts\python.exe" -m pip install -r "%REQG%"
  if errorlevel 1 exit /b %ERRORLEVEL%
  if exist "%REQPY%" "%~dp0.venv\Scripts\python.exe" -m pip install -r "%REQPY%"
  if errorlevel 1 exit /b %ERRORLEVEL%
  if exist "%REQAM2%" "%~dp0.venv\Scripts\python.exe" -m pip install -r "%REQAM2%"
  exit /b %ERRORLEVEL%
)
if exist "%~dp0venv\Scripts\python.exe" (
  "%~dp0venv\Scripts\python.exe" -m pip install -r "%REQ%"
  if errorlevel 1 exit /b %ERRORLEVEL%
  if exist "%REQG%" "%~dp0venv\Scripts\python.exe" -m pip install -r "%REQG%"
  if errorlevel 1 exit /b %ERRORLEVEL%
  if exist "%REQPY%" "%~dp0venv\Scripts\python.exe" -m pip install -r "%REQPY%"
  if errorlevel 1 exit /b %ERRORLEVEL%
  if exist "%REQAM2%" "%~dp0venv\Scripts\python.exe" -m pip install -r "%REQAM2%"
  exit /b %ERRORLEVEL%
)
where py >nul 2>&1 && (
  py -3 -m pip install -r "%REQ%"
  if errorlevel 1 exit /b %ERRORLEVEL%
  if exist "%REQG%" py -3 -m pip install -r "%REQG%"
  if errorlevel 1 exit /b %ERRORLEVEL%
  if exist "%REQPY%" py -3 -m pip install -r "%REQPY%"
  if errorlevel 1 exit /b %ERRORLEVEL%
  if exist "%REQAM2%" py -3 -m pip install -r "%REQAM2%"
  exit /b %ERRORLEVEL%
)
where python >nul 2>&1 && (
  python -m pip install -r "%REQ%"
  if errorlevel 1 exit /b %ERRORLEVEL%
  if exist "%REQG%" python -m pip install -r "%REQG%"
  if errorlevel 1 exit /b %ERRORLEVEL%
  if exist "%REQPY%" python -m pip install -r "%REQPY%"
  if errorlevel 1 exit /b %ERRORLEVEL%
  if exist "%REQAM2%" python -m pip install -r "%REQAM2%"
  exit /b %ERRORLEVEL%
)

echo Не найден интерпретатор. Установите Python 3, создайте venv: python -m venv .venv >&2
echo или задайте PYTHON=путь\к\python.exe >&2
exit /b 1
