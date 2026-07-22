# -*- coding: utf-8 -*-
"""CIS 구조 위저드 + RCWA QE 해석 — 한 줄 실행 진입점.

    python app.py                  # http://127.0.0.1:8787 브라우저 자동 오픈
    (의존성이 없으면 traceback 대신 설치 안내 + 선택 자동설치)

브라우저에서: step1~8 로 구조 설정 -> 상단 [▶ QE 해석] -> 파장범위/nG 설정 -> Run
-> QE/R/A 스펙트럼 확인 + CSV 저장.  (구조 npy 추출은 [⤓ npy + yaml] 그대로)
"""
import argparse
import importlib.util
import os
import subprocess
import sys

REQUIRED = [("torch", "torch"), ("numpy", "numpy"),
            ("yaml", "pyyaml"), ("matplotlib", "matplotlib")]


def _preflight():
    """의존성 검사 — 없으면 traceback 대신 설치 안내(+선택 자동설치).
    단독 exe(frozen)면 의존성이 번들돼 있으므로 건너뜀."""
    if getattr(sys, "frozen", False):
        return
    missing = [pkg for mod, pkg in REQUIRED
               if importlib.util.find_spec(mod) is None]
    if not missing:
        return
    base = os.path.dirname(os.path.abspath(
        globals().get("__file__", sys.argv[0] or ".")))
    req = os.path.join(base, "requirements.txt")
    py = os.path.basename(sys.executable) or "python"
    print("=" * 62)
    print("[cis-rcwa] 필요한 파이썬 패키지가 설치되어 있지 않습니다:")
    print("           " + ", ".join(missing))
    print()
    print("  설치 (아래 한 줄):")
    print(f"    {py} -m pip install " + " ".join(missing))
    if os.path.exists(req):
        print("  또는:")
        print(f"    {py} -m pip install -r requirements.txt")
    print("=" * 62)
    try:
        ans = input("지금 자동으로 설치할까요? [y/N] ").strip().lower()
    except EOFError:
        ans = ""
    if ans == "y":
        r = subprocess.call([sys.executable, "-m", "pip", "install"] + missing)
        if r == 0 and all(importlib.util.find_spec(m) for m, _ in REQUIRED):
            print("[cis-rcwa] 설치 완료 — 계속 실행합니다.\n")
            return
        print("[cis-rcwa] 자동 설치 실패 — 위 명령을 직접 실행해 주세요.")
    sys.exit(1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="CIS wizard + RCWA QE server")
    ap.add_argument("--port", type=int, default=8787)
    ap.add_argument("--no-browser", action="store_true")
    a = ap.parse_args()
    _preflight()
    # torch 첫 import 가 10~30초 걸릴 수 있어, 멈춘 것처럼 보이지 않게 먼저 알림
    print("[cis-rcwa] 로딩중... (torch/numpy import — 첫 실행은 10~30초 걸릴 수 있음)",
          flush=True)
    from src.api.server import serve
    serve(port=a.port, open_browser=not a.no_browser)
