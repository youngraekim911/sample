@echo off
rem ============================================================
rem  CIS RCWA - "leave it running and go home" server (Windows)
rem
rem  For long DOE / surrogate sweeps you start and then walk away.
rem  * The computation runs in THIS server, not in the browser.
rem  * Even if your browser is closed by company policy, it keeps going.
rem  * Results auto-save to out\surrogate_cache when a run finishes.
rem  * Reopen the browser later: Settings page auto-reconnects; finished
rem    results appear in the saved-surrogate list to load instantly.
rem
rem  !! Do NOT close THIS window while a run is in progress !!
rem  (Minimizing is fine. Finished/cached results survive closing.)
rem
rem  Device: uses your current python's torch if all deps are present
rem  (keeps your CUDA/GPU), else a local CPU-only .venv. For GPU in the
rem  venv, run install_gpu_windows.bat.
rem  ASCII-only; Korean guide: see the .txt files in this folder.
rem ============================================================
cd /d "%~dp0"
where python >nul 2>nul
if errorlevel 1 (
  echo [ERROR] python not found. Install from https://python.org
  echo         and check "Add Python to PATH" during setup.
  pause & exit /b 1
)

rem --- keep the PC awake during a long run (ignored if blocked by policy) ---
powercfg -change -monitor-timeout-ac 0 >nul 2>nul
powercfg -change -standby-timeout-ac 0 >nul 2>nul

rem --- pick python: current one if it already has deps, else .venv ---
set "PYEXE=python"
python -c "import torch,numpy,yaml,matplotlib" >nul 2>nul
if not errorlevel 1 goto run
if not exist .venv (
  echo [SETUP] creating venv + installing packages... (a few minutes, first time only)
  echo         NOTE: this installs CPU-only torch. For GPU run install_gpu_windows.bat.
  python -m venv .venv || (echo venv creation failed & pause & exit /b 1)
  .venv\Scripts\python -m pip install --upgrade pip
  .venv\Scripts\python -m pip install -r requirements.txt || (echo install failed & pause & exit /b 1)
)
set "PYEXE=.venv\Scripts\python"

:run
echo.
echo ============================================================
echo  Server: http://127.0.0.1:8787
%PYEXE% -c "import torch;print('  torch.cuda.is_available =', torch.cuda.is_available())"
echo  - A browser opens. Start the DOE Run and just leave it.
echo  - You may close the browser; reopen it later to see progress.
echo  * Do NOT close THIS window (minimize is OK).
echo ============================================================
echo.
start "" http://127.0.0.1:8787
%PYEXE% app.py --no-browser
echo.
echo [server stopped] any in-progress run was interrupted.
pause
