@echo off
cd /d "%~dp0"
".\.venv\Scripts\python.exe" "detect_shellshock.py" --resolution 3840x2160 --output-dir output
pause
