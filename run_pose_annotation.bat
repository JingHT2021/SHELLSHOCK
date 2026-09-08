@echo off
cd /d "%~dp0"
".venv\Scripts\python.exe" annotate_enemies.py --all-images --barrel-length 35
pause
