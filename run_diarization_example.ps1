# Запуск с интерпретатором из venv или из PATH (аналог run_diarization_example.sh).
# Запуск из каталога скрипта:
#   powershell -ExecutionPolicy Bypass -File .\run_diarization_example.ps1
#   powershell -ExecutionPolicy Bypass -File .\run_diarization_example.ps1 D:\audio\file.wav
$ErrorActionPreference = "Stop"
$Root = $PSScriptRoot
Set-Location -LiteralPath $Root

$candidates = @()
if ($env:PYTHON) { $candidates += $env:PYTHON }
$candidates += @(
    (Join-Path $Root ".venv\Scripts\python.exe"),
    (Join-Path $Root "venv\Scripts\python.exe")
)

$script = Join-Path $Root "run_diarization_example.py"
foreach ($c in $candidates) {
    if ($c -and (Test-Path -LiteralPath $c)) {
        & $c $script @args
        exit $LASTEXITCODE
    }
}

if (Get-Command py -ErrorAction SilentlyContinue) {
    py -3 $script @args
    exit $LASTEXITCODE
}
if (Get-Command python -ErrorAction SilentlyContinue) {
    & python $script @args
    exit $LASTEXITCODE
}

Write-Error "Не найден интерпретатор. Установите Python 3, создайте venv: python -m venv .venv, либо задайте переменную окружения PYTHON."
exit 1
