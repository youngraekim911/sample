@echo off
rem ============================================================
rem  CIS RCWA - install CUDA (GPU) build of PyTorch into the venv
rem
rem  Why: the default "pip install torch" on Windows is CPU-ONLY,
rem  so the app runs on CPU even with an NVIDIA GPU. This installs
rem  the CUDA build so torch.cuda.is_available() becomes True and
rem  the solver auto-switches to GPU (no code change needed).
rem
rem  Requires: an NVIDIA GPU + up-to-date NVIDIA driver.
rem  (AMD/Intel/Apple GPUs are not CUDA and will stay on CPU.)
rem
rem  CUDA wheel tag below is cu121 (works with driver R530+).
rem  If your driver is older/newer, change CUDA_TAG to match
rem  the selector at https://pytorch.org  (e.g. cu118, cu124, cu126).
rem ============================================================
cd /d "%~dp0"
set CUDA_TAG=cu121

if not exist .venv (
  echo [ERROR] .venv not found. Run run_windows.bat once first
  echo         to create the environment, then run this.
  pause & exit /b 1
)

echo [INFO] checking for NVIDIA GPU...
where nvidia-smi >nul 2>nul
if errorlevel 1 (
  echo [WARN] nvidia-smi not found - you may not have an NVIDIA GPU / driver.
  echo        Continuing anyway; if there is no NVIDIA GPU this will not help.
) else (
  nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
)

echo.
echo [STEP] removing CPU-only torch, installing CUDA (%CUDA_TAG%) build...
.venv\Scripts\python -m pip uninstall -y torch
.venv\Scripts\python -m pip install torch --index-url https://download.pytorch.org/whl/%CUDA_TAG%
if errorlevel 1 (
  echo [ERROR] install failed. Open https://pytorch.org and copy the exact
  echo         command for your CUDA version, or try a different CUDA_TAG.
  pause & exit /b 1
)

echo.
echo [VERIFY]
.venv\Scripts\python -c "import torch; print('torch', torch.__version__); print('cuda available:', torch.cuda.is_available()); print('gpu:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"
echo.
echo If 'cuda available: True' above, GPU is ready. Start the app and the
echo server line will show (device=cuda). DOE/QE will then run on the GPU.
pause
