@echo off
rem ============================================================
rem  CIS RCWA simulator - one-click run for Windows
rem  (first run creates a venv and installs deps; then just runs)
rem  Requires: Python 3.9+  (from https://python.org, check "Add to PATH")
rem  ASCII-only on purpose: Korean text in a .bat breaks cmd (codepage).
rem  Korean guide is in the .txt files next to this launcher.
rem ============================================================
cd /d "%~dp0"
where python >nul 2>nul
if errorlevel 1 (
  echo [ERROR] python not found. Install from https://python.org
  echo         and check "Add Python to PATH" during setup.
  pause & exit /b 1
)
if not exist .venv (
  echo [SETUP] creating venv + installing packages... (a few minutes, first time only)
  python -m venv .venv || (echo venv creation failed & pause & exit /b 1)
  .venv\Scripts\python -m pip install --upgrade pip
  .venv\Scripts\python -m pip install -r requirements.txt || (echo install failed & pause & exit /b 1)
)
echo [RUN] opening http://127.0.0.1:8787 in your browser...
.venv\Scripts\python app.py
pause
