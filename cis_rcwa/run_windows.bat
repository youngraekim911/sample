@echo off
rem ============================================================
rem  CIS RCWA 시뮬레이터 — 윈도우 원클릭 실행
rem  (처음 실행 시 가상환경 생성 + 의존성 자동 설치, 이후엔 바로 실행)
rem  필요: Python 3.9+ (https://python.org 설치, "Add to PATH" 체크)
rem ============================================================
cd /d "%~dp0"
where python >nul 2>nul
if errorlevel 1 (
  echo [오류] python 을 찾을 수 없습니다. https://python.org 에서 설치 후
  echo        설치 시 "Add Python to PATH" 를 체크하세요.
  pause & exit /b 1
)
if not exist .venv (
  echo [설치] 가상환경 생성 + 패키지 설치중... (수 분 소요, 최초 1회)
  python -m venv .venv || (echo venv 생성 실패 & pause & exit /b 1)
  .venv\Scripts\python -m pip install --upgrade pip
  .venv\Scripts\python -m pip install -r requirements.txt || (echo 설치 실패 & pause & exit /b 1)
)
echo [실행] http://127.0.0.1:8787 브라우저가 곧 열립니다...
.venv\Scripts\python app.py
pause
