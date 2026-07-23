@echo off
rem ============================================================
rem  CIS RCWA simulator - one-click run for Windows
rem  ASCII-only on purpose: Korean in a .bat breaks cmd (codepage).
rem  Korean guide: see the .txt files in this folder.
rem
rem  Device (CPU/GPU):
rem   - The solver auto-uses CUDA if the running Python's torch sees a GPU.
rem   - This launcher FIRST tries your current 'python'. If it already has
rem     all deps (e.g. your existing CUDA torch), it runs with THAT, so the
rem     GPU keeps working. Only if deps are missing does it build a local
rem     .venv (whose 'pip install torch' is CPU-only).
rem   - To put GPU torch into the .venv: run install_gpu_windows.bat.
rem ============================================================
cd /d "%~dp0"
where python >nul 2>nul
if errorlevel 1 (
  echo [ERROR] python not found. Install from https://python.org
  echo         and check "Add Python to PATH" during setup.
  pause & exit /b 1
)

rem --- prefer current python if it already has every dependency ---
python -c "import torch,numpy,yaml,matplotlib" >nul 2>nul
if errorlevel 1 goto setup_venv
echo [RUN] using current python (deps already present)...
python -c "import torch;print('       torch.cuda.is_available =', torch.cuda.is_available())"
python app.py
goto done

:setup_venv
if not exist .venv (
  echo [SETUP] creating venv + installing packages... (a few minutes, first time only)
  echo         NOTE: this installs CPU-only torch. For GPU run install_gpu_windows.bat.
  python -m venv .venv || (echo venv creation failed & pause & exit /b 1)
  .venv\Scripts\python -m pip install --upgrade pip
  .venv\Scripts\python -m pip install -r requirements.txt || (echo install failed & pause & exit /b 1)
)
echo [RUN] using .venv python...
.venv\Scripts\python -c "import torch;print('       torch.cuda.is_available =', torch.cuda.is_available())"
.venv\Scripts\python app.py

:done
pause
