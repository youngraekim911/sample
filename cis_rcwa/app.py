# -*- coding: utf-8 -*-
"""CIS 구조 위저드 + RCWA QE 해석 — 한 줄 실행 진입점.

    python app.py                  # http://127.0.0.1:8787 브라우저 자동 오픈

실행 즉시 가벼운 '부트 서버'가 떠서 로딩 화면(RGGB 스피너 + 환경 체크)을
보여주고, 백그라운드에서 의존성 확인/설치와 엔진(torch) 로딩을 진행한 뒤
같은 주소에서 본 서버로 자연스럽게 전환된다. 사내망 등으로 설치가 불가하면
무엇이 없는지/어떻게 요청할지 화면에 그대로 안내한다.

    python app.py --no-browser     # 브라우저 자동 오픈 없이 (배경 실행용)
    환경변수 CIS_NO_INSTALL=1      # pip 자동 설치 시도 자체를 끔 (완전 오프라인)
"""
import argparse
import importlib.util
import json
import os
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# exe(PyInstaller) 실행이면 번들 자원 폴더(_MEIPASS)에서 boot.html 을 찾는다
BASE = getattr(sys, "_MEIPASS",
               os.path.dirname(os.path.abspath(globals().get("__file__", sys.argv[0] or "."))))

# (모듈명, pip 패키지명, 필수 여부)
REQUIRED = [("numpy", "numpy", True), ("yaml", "pyyaml", True),
            ("matplotlib", "matplotlib", True), ("torch", "torch", True)]

BOOT = {"ready": False, "steps": [], "error": None}
_SRV = {}                      # 로딩 완료된 본 서버 모듈


def _step(name, state, note=""):
    for s in BOOT["steps"]:
        if s["name"] == name:
            s["state"], s["note"] = state, note
            return
    BOOT["steps"].append({"name": name, "state": state, "note": note})


def _boot_page_bytes():
    p = os.path.join(BASE, "editors", "boot.html")
    try:
        with open(p, "rb") as f:
            return f.read()
    except OSError:            # boot.html 이 없어도 최소한의 안내는 띄움
        return ("<meta charset='utf-8'><body style='font-family:sans-serif'>"
                "<h3>CIS Simulator 준비 중…</h3>"
                "<p>잠시 후 자동으로 시작됩니다. (editors/boot.html 없음)</p>"
                "<script>setInterval(()=>fetch('/api/ping').then(r=>{if(r.ok)location.href='/';}"
                ").catch(()=>{}),700)</script>").encode("utf-8")


class _BootHandler(BaseHTTPRequestHandler):
    def log_message(self, *a):                     # 조용히
        pass

    def do_GET(self):
        if self.path.startswith("/api/boot/status"):
            body = json.dumps(BOOT, ensure_ascii=False).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
        elif self.path.startswith("/api/"):
            # 본 서버 API 는 아직 없음 — 200 을 주면 로딩 화면이 '준비 완료'로
            # 오판하고 너무 일찍 이동하므로 반드시 404
            self.send_response(404)
            self.end_headers()
            return
        else:                                       # 그 외 경로는 로딩 화면
            body = _boot_page_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def _check_and_load(boot_httpd):
    """의존성 확인(없으면 설치 시도) → 엔진(torch/서버) 로딩 → 부트 서버 종료."""
    frozen = getattr(sys, "frozen", False)
    no_install = frozen or os.environ.get("CIS_NO_INSTALL")
    py = f"Python {sys.version_info.major}.{sys.version_info.minor}"
    _step(py, "ok", os.path.basename(sys.executable) or "")

    missing = []
    for mod, pkg, req in REQUIRED:
        _step(pkg, "run", "확인중")
        if importlib.util.find_spec(mod) is not None:
            _step(pkg, "ok", "")
            continue
        if no_install:
            _step(pkg, "bad", "없음 (자동 설치 꺼짐)")
            missing.append(pkg)
            continue
        _step(pkg, "run", "설치 시도중… (수 분 걸릴 수 있음)")
        try:
            r = subprocess.run([sys.executable, "-m", "pip", "install",
                                "--no-input", pkg],
                               capture_output=True, timeout=900)
            ok = (r.returncode == 0 and importlib.util.find_spec(mod) is not None)
        except Exception:
            ok = False
        if ok:
            _step(pkg, "ok", "설치됨")
        else:
            _step(pkg, "bad", "설치 실패 — 사내망/오프라인이면 IT 에 요청")
            missing.append(pkg)

    if missing:
        BOOT["error"] = ("필수 패키지가 없어 시작할 수 없습니다: "
                         + ", ".join(missing) + "\n\n"
                         "사내 PC 라 외부 설치가 막혀 있다면 IT/관리자에게 아래 설치를 요청하세요:\n"
                         f"  {os.path.basename(sys.executable) or 'python'} -m pip install "
                         + " ".join(missing) + "\n"
                         "(오프라인 배포용 wheel 파일을 받아 설치하는 방법도 가능합니다)")
        return                                     # 부트 서버 유지 — 안내 화면 계속 표시

    _step("엔진 로딩 (torch)", "run", "첫 실행은 10~30초")
    try:
        from src.api import server as srv          # 무거운 import (torch 포함)
        _SRV["mod"] = srv
    except Exception as e:
        _step("엔진 로딩 (torch)", "bad", str(e)[:120])
        BOOT["error"] = f"엔진 로딩 실패: {type(e).__name__}: {e}"
        return
    _step("엔진 로딩 (torch)", "ok", "")
    try:
        import torch
        _step("GPU (CUDA)", "ok",
              "사용" if torch.cuda.is_available() else "없음 — CPU 모드로 동작")
    except Exception:
        pass

    BOOT["ready"] = True
    time.sleep(1.5)                                # 화면이 최종 상태를 폴링할 여유
    boot_httpd.shutdown()                          # → 메인 스레드가 본 서버로 교체
    # (로딩 화면은 마지막 상태를 캐시해 체크 애니메이션을 끝까지 보여준 뒤 넘어옴)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="CIS wizard + RCWA QE server")
    ap.add_argument("--port", type=int, default=8787)
    ap.add_argument("--no-browser", action="store_true")
    a = ap.parse_args()

    url = f"http://127.0.0.1:{a.port}/"
    boot_httpd = ThreadingHTTPServer(("127.0.0.1", a.port), _BootHandler)
    print(f"[cis-rcwa] 로딩 화면: {url}  (환경 체크 → 자동으로 본 화면 전환)", flush=True)
    threading.Thread(target=_check_and_load, args=(boot_httpd,), daemon=True).start()
    if not a.no_browser:
        try:
            import webbrowser
            threading.Timer(0.4, lambda: webbrowser.open(url)).start()
        except Exception:
            pass
    try:
        boot_httpd.serve_forever()                 # 준비 완료 시 worker 가 shutdown
    except KeyboardInterrupt:
        sys.exit(0)
    boot_httpd.server_close()
    if BOOT["error"] or "mod" not in _SRV:
        sys.exit(1)
    # 같은 포트에서 본 서버 시작 (브라우저의 로딩 화면이 자동으로 넘어옴)
    _SRV["mod"].serve(port=a.port, open_browser=False)
