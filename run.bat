@echo off
cd /d "%~dp0"
".\.venv\Scripts\python.exe" "detect_shellshock_yolo.py"
pause
