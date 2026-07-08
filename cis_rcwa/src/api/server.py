# -*- coding: utf-8 -*-
"""로컬 QE 해석 서버 — structure(위저드) -> RCWA -> QE 를 한 화면 flow 로 연결.

    python3 app.py            # 서버 시작 + 브라우저 자동 오픈
    python3 app.py --port 8788 --no-browser

블록 구성 (프론트는 이 API 만 알면 됨):
    GET  /                    위저드 HTML (editors/structure_wizard.html)
    POST /api/qe              {yaml, lam0, lam1, n, nG, downsample, theta} -> {job}
    GET  /api/qe/status?job=  {state, progress, note, rows:[[lam,R,QE,A],..], error}
    POST /api/qe/cancel       {job} -> 남은 파장 중단
    GET  /api/ping            {ok, device}

내부: 요청 yaml 을 out/jobs/<job>.yaml 로 저장 -> RCWAPlaneWaveSimulator (스키마
자동인식, materials 폴더 자동 연동) -> 파장별 TE/TM 평균 -> rows 누적(부분 결과
폴링 가능) -> csv 저장.
"""
import os
import json
import time
import uuid
import threading
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
JOBS_DIR = os.path.join(ROOT, "out", "jobs")

JOBS = {}          # job id -> dict(state, progress, note, rows, error, cancel)
_LOCK = threading.Lock()


# --------------------------------------------------------------- QE 작업
def _run_job(jid, cfg_path, p):
    job = JOBS[jid]
    try:
        from ..sim.simulator import RCWAPlaneWaveSimulator
        job["note"] = "구조 생성 + 층 스택 준비중..."
        sim = RCWAPlaneWaveSimulator(cfg_path, nG=p["nG"], downsample=p["downsample"])
        job["note"] = (f"device={sim.device} · grid {sim.grid_ny}×{sim.grid_nx} · "
                       f"layers {len(sim.layer_stack)} · nG {sim.nG if hasattr(sim,'nG') else p['nG']}")
        lams = np.linspace(p["lam0"], p["lam1"], p["n"])
        for i, lam in enumerate(lams):
            if job.get("cancel"):
                job["state"] = "cancelled"
                return
            t0 = time.time()
            o_te = sim.run(float(lam), theta=p["theta"], pol_te=1.0, pol_tm=0.0)
            o_tm = sim.run(float(lam), theta=p["theta"], pol_te=0.0, pol_tm=1.0)
            R = 0.5 * (o_te["R"] + o_tm["R"])
            QE = 0.5 * (o_te["QE"] + o_tm["QE"])
            A = 0.5 * (o_te["A_stack"] + o_tm["A_stack"])
            job["rows"].append([round(float(lam), 5), round(float(R), 5),
                                round(float(QE), 5), round(float(A), 5)])
            job["progress"] = (i + 1) / len(lams)
            job["note"] = f"λ={lam*1000:.0f}nm  QE={QE:.3f}  ({time.time()-t0:.1f}s/λ, TE+TM)"
        # csv 저장
        csv_path = os.path.join(JOBS_DIR, jid + "_qe.csv")
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write("lambda_um,R,QE_Si,A_stack\n")
            for r in job["rows"]:
                f.write(",".join(str(x) for x in r) + "\n")
        job["csv"] = os.path.relpath(csv_path, ROOT)
        job["state"] = "done"
    except Exception as e:
        job["error"] = f"{type(e).__name__}: {e}"
        job["trace"] = traceback.format_exc()[-2000:]
        job["state"] = "error"


def start_job(yaml_text, p):
    os.makedirs(JOBS_DIR, exist_ok=True)
    jid = uuid.uuid4().hex[:12]
    cfg_path = os.path.join(JOBS_DIR, jid + ".yaml")
    with open(cfg_path, "w", encoding="utf-8") as f:
        f.write(yaml_text)
    JOBS[jid] = {"state": "running", "progress": 0.0, "note": "시작중...",
                 "rows": [], "error": None, "cancel": False, "params": p}
    th = threading.Thread(target=_run_job, args=(jid, cfg_path, p), daemon=True)
    th.start()
    return jid


# --------------------------------------------------------------- HTTP
class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):          # 조용히
        pass

    def _json(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _file(self, path, ctype):
        try:
            with open(path, "rb") as f:
                body = f.read()
        except OSError:
            self.send_response(404); self.end_headers(); return
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        u = urlparse(self.path)
        if u.path in ("/", "/index.html", "/wizard"):
            self._file(os.path.join(ROOT, "editors", "structure_wizard.html"),
                       "text/html; charset=utf-8")
        elif u.path == "/api/ping":
            self._json({"ok": True,
                        "device": "cuda" if torch.cuda.is_available() else "cpu"})
        elif u.path == "/api/qe/status":
            jid = (parse_qs(u.query).get("job") or [""])[0]
            job = JOBS.get(jid)
            if not job:
                self._json({"error": "unknown job"}, 404); return
            self._json({k: job[k] for k in
                        ("state", "progress", "note", "rows", "error", "csv")
                        if k in job})
        else:
            self.send_response(404); self.end_headers()

    def do_POST(self):
        u = urlparse(self.path)
        try:
            n = int(self.headers.get("Content-Length") or 0)
            data = json.loads(self.rfile.read(n) or b"{}")
        except Exception:
            self._json({"error": "bad json"}, 400); return
        if u.path == "/api/qe":
            yaml_text = data.get("yaml") or ""
            if not yaml_text.strip():
                self._json({"error": "yaml 이 비었습니다"}, 400); return
            p = {"lam0": float(data.get("lam0", 0.40)),
                 "lam1": float(data.get("lam1", 0.70)),
                 "n": max(1, int(data.get("n", 7))),
                 "nG": max(9, int(data.get("nG", 101))),
                 "downsample": max(1, int(data.get("downsample", 2))),
                 "theta": float(data.get("theta", 0.0))}
            jid = start_job(yaml_text, p)
            self._json({"job": jid})
        elif u.path == "/api/qe/cancel":
            job = JOBS.get(data.get("job") or "")
            if job:
                job["cancel"] = True
            self._json({"ok": bool(job)})
        else:
            self.send_response(404); self.end_headers()


def serve(port=8787, open_browser=True):
    os.chdir(ROOT)                              # materials 폴더 자동탐색 기준
    httpd = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    url = f"http://127.0.0.1:{port}/"
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[cis-rcwa] wizard + QE server: {url}  (device={dev}, Ctrl+C 종료)")
    if open_browser:
        try:
            import webbrowser
            threading.Timer(0.6, lambda: webbrowser.open(url)).start()
        except Exception:
            pass
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[cis-rcwa] bye")


if __name__ == "__main__":
    serve()
