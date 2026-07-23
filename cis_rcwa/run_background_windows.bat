@echo off
rem ============================================================
rem  CIS RCWA - "leave it running and go home" server (Windows)
rem
rem  For long DOE / surrogate sweeps you start and then walk away.
rem  * The computation runs in THIS server, not in the browser.
rem  * Even if your browser (Chrome/Edge/IE) is closed by company
rem    policy, the computation keeps going.
rem  * Results are auto-saved to out\surrogate_cache the moment a run
rem    finishes (a small on-disk DB).
rem  * When you reopen the browser later, the Settings page auto-
rem    reconnects, and finished results appear in the saved-surrogate
rem    list to load instantly.
rem
rem  !! Do NOT close THIS window while a run is in progress !!
rem  (Minimizing is fine. Finished/cached results survive closing.)
rem
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

rem --- keep the PC awake during a long run (ignored if blocked by policy) ---
powercfg -change -monitor-timeout-ac 0 >nul 2>nul
powercfg -change -standby-timeout-ac 0 >nul 2>nul

echo.
echo ============================================================
echo  Server: http://127.0.0.1:8787
echo  - A browser opens. Start the DOE Run and just leave it.
echo  - You may close the browser; reopen it later to see progress.
echo  - When a run finishes, the result is saved automatically.
echo  * Do NOT close THIS window (minimize is OK).
echo ============================================================
echo.
start "" http://127.0.0.1:8787
.venv\Scripts\python app.py --no-browser
echo.
echo [server stopped] any in-progress run was interrupted.
pause
