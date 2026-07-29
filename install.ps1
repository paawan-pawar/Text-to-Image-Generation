$ErrorActionPreference = "Stop"

Write-Host "Setting up Text-to-Image Generation System..."

if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    throw "Python is not installed or not available on PATH."
}

$pythonVersion = & python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
if ($pythonVersion -ne "3.14") {
    throw @"
Unsupported Python version: $pythonVersion

This project is pinned to package versions that work with Python 3.14.
Install Python 3.14, remove the existing venv, and rerun this script.
"@
}

if (Test-Path "venv") {
    Remove-Item -Recurse -Force "venv"
}

python -m venv venv

$venvPython = Join-Path $PWD "venv\Scripts\python.exe"
& $venvPython -m pip install --upgrade pip
& $venvPython -m pip install -r requirements.txt

Write-Host "Setup complete! Run .\venv\Scripts\Activate.ps1 and then python app.py to start the application."