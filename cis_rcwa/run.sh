#!/usr/bin/env bash
# CIS RCWA 시뮬레이터 — mac/linux 원클릭 실행
# (처음 실행 시 가상환경 생성 + 의존성 자동 설치, 이후엔 바로 실행)
set -e
cd "$(dirname "$0")"
PY=python3; command -v $PY >/dev/null || PY=python
if [ ! -d .venv ]; then
  echo "[설치] 가상환경 생성 + 패키지 설치중... (수 분 소요, 최초 1회)"
  $PY -m venv .venv
  .venv/bin/python -m pip install --upgrade pip
  .venv/bin/python -m pip install -r requirements.txt
fi
echo "[실행] http://127.0.0.1:8787 브라우저가 곧 열립니다..."
exec .venv/bin/python app.py
