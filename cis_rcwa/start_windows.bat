@echo off
rem ============================================================
rem  CIS Simulator - ONE launcher for Windows (double-click me)
rem
rem  Normal use : double-click  ->  loading screen opens in the
rem               browser, checks/installs packages by itself,
rem               then switches to the start page automatically.
rem
rem  Leave-running mode ("run and go home"):
rem               start_windows.bat bg
rem               (keeps PC awake, no auto browser; computation
rem                continues even if the browser is closed)
rem
rem  Device (CPU/GPU): the app auto-uses CUDA when the running
rem  python's torch sees a GPU. This launcher prefers a python
rem  that already has torch (so your CUDA install keeps working).
rem  Missing packages are installed by the loading screen itself
rem  via pip - no separate installer needed.
rem
rem  ASCII-only on purpose: Korean text in a .bat breaks cmd.
rem  Korean guide: see the .txt files in this folder.
rem ============================================================
cd /d "%~dp0"

where python >nul 2>nul
if errorlevel 1 (
  echo [ERROR] python not found. Install from https://python.org
  echo         and check "Add Python to PATH" during setup.
  pause & exit /b 1
)

rem --- prefer a python that already has torch (keeps CUDA/GPU) ---
set "PYEXE=python"
python -c "import torch" >nul 2>nul
if not errorlevel 1 goto picked
if exist .venv\Scripts\python.exe (
  .venv\Scripts\python -c "import torch" >nul 2>nul
  if not errorlevel 1 set "PYEXE=.venv\Scripts\python"
)
:picked

if /i "%~1"=="bg" goto background

echo [RUN] %PYEXE% app.py   (loading screen opens in your browser)
%PYEXE% app.py
pause
exit /b 0

:background
rem --- keep the PC awake during a long run (ignored if blocked) ---
powercfg -change -monitor-timeout-ac 0 >nul 2>nul
powercfg -change -standby-timeout-ac 0 >nul 2>nul
echo.
echo ============================================================
echo  Server: http://127.0.0.1:8787   (leave-running mode)
echo  - Open the address, start your Run, then just leave.
echo  - Closing the BROWSER is fine; results auto-save.
echo  * Do NOT close THIS window (minimize is OK).
echo ============================================================
echo.
start "" http://127.0.0.1:8787
%PYEXE% app.py --no-browser
echo.
echo [server stopped] any in-progress run was interrupted.
pause
