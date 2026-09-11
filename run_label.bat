@echo off
setlocal
cd /d "%~dp0"
set "SHELLSHOCK_PYTHON=%~dp0.venv\Scripts\python.exe"
if not exist "%SHELLSHOCK_PYTHON%" set "SHELLSHOCK_PYTHON=%~dp0..\..\.venv\Scripts\python.exe"
if not exist "%SHELLSHOCK_PYTHON%" (
  echo Python environment not found. Create .venv or run with your Python environment.
  pause
  exit /b 1
)
"%SHELLSHOCK_PYTHON%" "label_yolo_captures.py" --all-images %*
pause
