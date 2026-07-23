#!/usr/bin/env bash
# ============================================================
#  CIS RCWA — "걸어두고 퇴근" 백그라운드 서버 (mac/linux)
#
#  오래 걸리는 DOE/surrogate 스윕을 걸어두고 자리를 뜰 때 사용.
#  * 계산은 브라우저가 아니라 이 파이썬 서버에서 돕니다.
#  * 브라우저가 꺼져도 계산은 계속되고, 끝나면 out/surrogate_cache 에
#    자동 저장됩니다. 다시 브라우저를 열면 '설정'에서 자동 재접속됩니다.
#  * nohup 으로 띄우므로 터미널을 닫아도 서버가 살아있습니다.
#    로그: server.log, 종료: kill $(cat server.pid)
# ============================================================
set -e
cd "$(dirname "$0")"
PY=python3; command -v $PY >/dev/null || PY=python
if [ ! -d .venv ]; then
  echo "[설치] 가상환경 생성 + 패키지 설치중... (수 분 소요, 최초 1회)"
  $PY -m venv .venv
  .venv/bin/python -m pip install --upgrade pip
  .venv/bin/python -m pip install -r requirements.txt
fi

# 절전 방지(best-effort): mac=caffeinate, linux=systemd-inhibit 있으면 사용
PREFIX=""
if command -v caffeinate >/dev/null 2>&1; then PREFIX="caffeinate -s";
elif command -v systemd-inhibit >/dev/null 2>&1; then PREFIX="systemd-inhibit --what=sleep:idle --why=cis-rcwa"; fi

echo "[서버] http://127.0.0.1:8787 (백그라운드, --no-browser)"
nohup $PREFIX .venv/bin/python app.py --no-browser >server.log 2>&1 &
echo $! >server.pid
sleep 1
echo "  PID $(cat server.pid) · 로그: server.log · 종료: kill \$(cat server.pid)"
echo "  브라우저에서 http://127.0.0.1:8787 열고 DOE Run 후 그냥 두고 가세요."
echo "  (브라우저는 닫아도 됨 — 다시 열면 자동 재접속, 끝난 결과는 💾 목록에서 로드)"
