# pip install -r requirements.txt тем же интерпретатором, что и run_diarization_example.ps1
#   powershell -ExecutionPolicy Bypass -File .\install_requirements.ps1
$ErrorActionPreference = "Stop"
$Root = $PSScriptRoot
Set-Location -LiteralPath $Root

$req = Join-Path $Root "requirements.txt"
if (-not (Test-Path -LiteralPath $req)) {
    Write-Error "Не найден requirements.txt в $Root"
    exit 1
}

$candidates = @()
if ($env:PYTHON) { $candidates += $env:PYTHON }
$candidates += @(
    (Join-Path $Root ".venv\Scripts\python.exe"),
    (Join-Path $Root "venv\Scripts\python.exe")
)

foreach ($c in $candidates) {
    if ($c -and (Test-Path -LiteralPath $c)) {
        & $c -m pip install -r $req
        exit $LASTEXITCODE
    }
}

if (Get-Command py -ErrorAction SilentlyContinue) {
    py -3 -m pip install -r $req
    exit $LASTEXITCODE
}
if (Get-Command python -ErrorAction SilentlyContinue) {
    & python -m pip install -r $req
    exit $LASTEXITCODE
}

Write-Error "Не найден интерпретатор. Установите Python 3, создайте venv: python -m venv .venv, либо задайте переменную окружения PYTHON."
exit 1
