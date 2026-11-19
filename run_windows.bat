@echo off
rem Starts the app on Windows: native window first, browser as a fallback.
rem Run from the repository root with the conda environment active:  run_windows.bat
python -m app.main
if errorlevel 1 (
  echo.
  echo The native window did not start. Opening the app in the browser instead.
  python app\serve.py
)
