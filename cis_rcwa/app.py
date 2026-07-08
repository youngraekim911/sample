# -*- coding: utf-8 -*-
"""CIS 구조 위저드 + RCWA QE 해석 — 한 줄 실행 진입점.

    pip install torch numpy pyyaml
    python3 app.py                 # http://127.0.0.1:8787 브라우저 자동 오픈

브라우저에서: step1~8 로 구조 설정 -> 상단 [▶ QE 해석] -> 파장범위/nG 설정 -> Run
-> QE/R/A 스펙트럼 확인 + CSV 저장.  (구조 npy 추출은 [⤓ npy + yaml] 그대로)
"""
import argparse

from src.api.server import serve

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="CIS wizard + RCWA QE server")
    ap.add_argument("--port", type=int, default=8787)
    ap.add_argument("--no-browser", action="store_true")
    a = ap.parse_args()
    serve(port=a.port, open_browser=not a.no_browser)
