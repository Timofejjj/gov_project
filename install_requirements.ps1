# pip install -r requirements.txt, requirements_gigaam.txt и requirements_pyannote.txt
# тем же интерпретатором,
# что и run_diarization_example.ps1 / Giga_AM/run_transcribe.py
#   powershell -ExecutionPolicy Bypass -File .\install_requirements.ps1
$ErrorActionPreference = "Stop"
$Root = $PSScriptRoot
Set-Location -LiteralPath $Root

$req = Join-Path $Root "requirements.txt"
$reqGiga = Join-Path $Root "requirements_gigaam.txt"
$reqPyan = Join-Path $Root "requirements_pyannote.txt"
$reqAm2 = Join-Path $Root "requirements_giga_am2.txt"
if (-not (Test-Path -LiteralPath $req)) {
    Write-Error "Не найден requirements.txt в $Root"
    exit 1
}

function Install-ReqForPython {
    param([Parameter(Mandatory = $true)][string]$PythonExe)
    & $PythonExe -m pip install -r $req
    if ($LASTEXITCODE -ne 0) { return $LASTEXITCODE }
    if (Test-Path -LiteralPath $reqGiga) {
        & $PythonExe -m pip install -r $reqGiga
        if ($LASTEXITCODE -ne 0) { return $LASTEXITCODE }
    }
    if (Test-Path -LiteralPath $reqPyan) {
        & $PythonExe -m pip install -r $reqPyan
        if ($LASTEXITCODE -ne 0) { return $LASTEXITCODE }
    }
    if (Test-Path -LiteralPath $reqAm2) {
        & $PythonExe -m pip install -r $reqAm2
    }
    return $LASTEXITCODE
}

$candidates = @()
if ($env:PYTHON) { $candidates += $env:PYTHON }
$candidates += @(
    (Join-Path $Root ".venv\Scripts\python.exe"),
    (Join-Path $Root "venv\Scripts\python.exe")
)

foreach ($c in $candidates) {
    if ($c -and (Test-Path -LiteralPath $c)) {
        $code = Install-ReqForPython -PythonExe $c
        exit $code
    }
}

if (Get-Command py -ErrorAction SilentlyContinue) {
    py -3 -m pip install -r $req
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    if (Test-Path -LiteralPath $reqGiga) {
        py -3 -m pip install -r $reqGiga
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    }
    if (Test-Path -LiteralPath $reqPyan) {
        py -3 -m pip install -r $reqPyan
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    }
    if (Test-Path -LiteralPath $reqAm2) { py -3 -m pip install -r $reqAm2 }
    exit $LASTEXITCODE
}
if (Get-Command python -ErrorAction SilentlyContinue) {
    $code = Install-ReqForPython -PythonExe "python"
    exit $code
}

Write-Error "Не найден интерпретатор. Установите Python 3, создайте venv: python -m venv .venv, либо задайте переменную окружения PYTHON."
exit 1
