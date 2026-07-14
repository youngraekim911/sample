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
def _auto_nG(job, cfg_path, p):
    """대표 파장에서 nG 를 올려가며 QE 수렴(Δ<0.5%p) 탐색 -> 'QE real' 용 nG.

    셀이 클수록(작은 pitch × 여러 픽셀) 필요한 차수가 커진다 — 고정 nG 는
    과소평가 위험. 101→145→201→257 순으로 확인, 연속 두 값의 컬러별 QE
    최대 변화가 0.5%p 미만이면 수렴으로 판단.
    """
    from ..sim.simulator import RCWAPlaneWaveSimulator
    lam0, lam1 = p["lam0"], p["lam1"]
    lam_cal = 0.55 if lam0 - 1e-9 <= 0.55 <= lam1 + 1e-9 else 0.5 * (lam0 + lam1)
    seq = [101, 145, 201, 257]
    prev = None
    chosen = seq[-1]
    hist = []
    for nG in seq:
        if job.get("cancel"):
            return chosen
        job["note"] = f"nG auto: nG={nG} 수렴 확인중 (λ={lam_cal*1000:.0f}nm)..."
        sim = RCWAPlaneWaveSimulator(cfg_path, nG=nG, downsample=p["downsample"])
        o1 = sim.run(lam_cal, theta=p["theta"], pol_te=1.0, pol_tm=0.0)
        o2 = sim.run(lam_cal, theta=p["theta"], pol_te=0.0, pol_tm=1.0)
        rgb = o1.get("QE_rgb") or {}
        q = ({c: 0.5 * (o1["QE_rgb"][c] + o2["QE_rgb"][c]) for c in rgb}
             if rgb else {"QE": 0.5 * (o1["QE"] + o2["QE"])})
        chosen = nG
        if prev is not None:
            d = max(abs(q[c] - prev[c]) for c in q)
            hist.append(f"{nG}(Δ{d*100:.1f}%p)")
            if d < 0.005:
                break
        else:
            hist.append(str(nG))
        prev = q
    job["nG_auto"] = " → ".join(hist) + f"  채택 nG={chosen}"
    return chosen


def _run_job(jid, cfg_path, p):
    job = JOBS[jid]
    try:
        from ..sim.simulator import RCWAPlaneWaveSimulator
        job["note"] = "구조 생성 + 층 스택 준비중..."
        if p.get("nG") == "auto":
            p["nG"] = _auto_nG(job, cfg_path, p)
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
            w = 1.0 if p.get("pol") == "sum" else 0.5    # 평균(비편광 표준) | 합산(참조 호환, x2)
            R = w * (o_te["R"] + o_tm["R"])
            QE = w * (o_te["QE"] + o_tm["QE"])
            A = w * (o_te["A_stack"] + o_tm["A_stack"])
            row = [round(float(lam), 5), round(float(R), 5),
                   round(float(QE), 5), round(float(A), 5)]
            note_rgb = ""
            if o_te.get("QE_rgb"):                       # 픽셀(색)별 QE: CF 하부 Si 창 흡수
                for c in "RGB":
                    v = w * (o_te["QE_rgb"].get(c, 0) + o_tm["QE_rgb"].get(c, 0))
                    row.append(round(float(v), 5))
                note_rgb = f"  R/G/B={row[4]:.3f}/{row[5]:.3f}/{row[6]:.3f}"
                note_rgb += f"  (전체 Si흡수, 심부 {w*(o_te.get('QE_deep',0)+o_tm.get('QE_deep',0)):.3f} 포함)"
            job["rows"].append(row)
            job["progress"] = (i + 1) / len(lams)
            job["note"] = f"λ={lam*1000:.0f}nm{note_rgb}  ({time.time()-t0:.1f}s/λ, TE+TM)"
        # csv 저장
        csv_path = os.path.join(JOBS_DIR, jid + "_qe.csv")
        with open(csv_path, "w", encoding="utf-8") as f:
            hdr = "lambda_um,R,QE_Si_total,A_stack"
            if job["rows"] and len(job["rows"][0]) >= 7:
                hdr += ",QE_R,QE_G,QE_B"
            f.write(hdr + "\n")
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
        # 위저드 업데이트가 항상 반영되도록 캐시 금지 (구버전 UI 잔존 방지)
        self.send_header("Cache-Control", "no-store, must-revalidate")
        self.send_header("Pragma", "no-cache")
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
                        ("state", "progress", "note", "rows", "error", "csv", "nG_auto")
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
        if u.path == "/api/qe/diag":
            # 진단: 물질 n,k 점검 + 경계 투과 프로파일 (단일 λ, 동기 실행)
            yaml_text = data.get("yaml") or ""
            if not yaml_text.strip():
                self._json({"error": "yaml 이 비었습니다"}, 400); return
            try:
                os.makedirs(JOBS_DIR, exist_ok=True)
                cfg_path = os.path.join(JOBS_DIR, "diag_" + uuid.uuid4().hex[:8] + ".yaml")
                with open(cfg_path, "w", encoding="utf-8") as f:
                    f.write(yaml_text)
                from ..sim.simulator import RCWAPlaneWaveSimulator
                sim = RCWAPlaneWaveSimulator(cfg_path,
                                             nG=min(61, max(9, int(data.get("nG", 61)))),
                                             downsample=max(4, int(data.get("downsample", 4))))
                d = sim.diagnose(float(data.get("lam", 0.55)),
                                 theta=float(data.get("theta", 0.0)))
                self._json(d)
            except Exception as e:
                self._json({"error": f"{type(e).__name__}: {e}",
                            "trace": traceback.format_exc()[-1500:]}, 500)
            return
        if u.path == "/api/qe":
            yaml_text = data.get("yaml") or ""
            if not yaml_text.strip():
                self._json({"error": "yaml 이 비었습니다"}, 400); return
            ng_req = data.get("nG", 101)
            p = {"lam0": float(data.get("lam0", 0.40)),
                 "lam1": float(data.get("lam1", 0.70)),
                 "n": max(1, int(data.get("n", 7))),
                 "nG": "auto" if str(ng_req) == "auto" else max(9, int(ng_req)),
                 "downsample": max(1, int(data.get("downsample", 2))),
                 "theta": float(data.get("theta", 0.0)),
                 "pol": str(data.get("pol", "avg"))}
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
