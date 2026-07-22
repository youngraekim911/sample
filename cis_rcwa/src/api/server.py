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
import sys
import json
import time
import uuid
import shutil
import threading
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

import numpy as np
import torch

# PyInstaller(단독 exe) 대응: 읽기전용 번들자원(ASSET_ROOT)과 쓰기/사용자자원(APP_DIR) 분리.
#  - 소스 실행: 둘 다 리포 루트
#  - exe 실행: ASSET_ROOT=번들(_MEIPASS, 읽기전용), APP_DIR=exe 폴더(물질 편집·out 저장)
FROZEN = getattr(sys, "frozen", False)
if FROZEN:
    ASSET_ROOT = getattr(sys, "_MEIPASS", os.path.dirname(sys.executable))
    APP_DIR = os.path.dirname(os.path.abspath(sys.executable))
else:
    ASSET_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    APP_DIR = ASSET_ROOT
ROOT = ASSET_ROOT                              # 하위호환(html 등 읽기자원 기준)
JOBS_DIR = os.path.join(APP_DIR, "out", "jobs")   # 쓰기 — exe 옆

JOBS = {}          # job id -> dict(state, progress, note, rows, error, cancel)
_LOCK = threading.Lock()


def _materials_dir():
    """물질 폴더: exe 옆(사용자 편집분) 우선, 없으면 번들 기본값."""
    user = os.path.join(APP_DIR, "data", "materials")
    return user if os.path.isdir(user) else os.path.join(ASSET_ROOT, "data", "materials")


MATERIALS_DIR = _materials_dir()


def _read_materials_folder():
    """data/materials/*.txt 를 파싱해 {name: [[wl_um, n, k], ...]} 반환.

    브라우저 parseNK 와 동일 규약: 3열(파장 n k), '#' 주석 무시, 파장 nm(>100)
    → µm 로 환산, k 는 |k|. **매 호출마다 폴더를 새로 읽어** 항상 최신(캐시 없음)
    — 위저드가 열릴 때/매 run 직전에 이걸 받아 folderLib 를 갱신한다.
    """
    out = {}
    mdir = _materials_dir()                       # 매 호출 재평가(사용자 폴더 생기면 즉시 반영)
    try:
        files = sorted(os.listdir(mdir))
    except OSError:
        return out
    for fn in files:
        if not fn.lower().endswith(".txt"):
            continue
        rows = []
        try:
            with open(os.path.join(mdir, fn), encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line[0] == "#":
                        continue
                    parts = line.replace(",", " ").split()
                    try:
                        v = [float(x) for x in parts[:3]]
                    except ValueError:
                        continue
                    if len(v) >= 3:
                        rows.append(v[:3])
        except OSError:
            continue
        if not rows:
            continue
        rows.sort(key=lambda r: r[0])
        sc = 0.001 if max(r[0] for r in rows) > 100 else 1.0
        out[fn[:-4]] = [[round(r[0] * sc, 4), round(r[1], 4), round(abs(r[2]), 5)]
                        for r in rows]
    return out


# --------------------------------------------------------------- QE 작업
def _recommend_mesh(cfg_path, quality="std"):
    """구조에서 (nG, downsample) 즉시 추천 — 계산 없이 0초.

    근거: 필요한 회절 차수는 셀 크기(pitch × n_pixels)와 최소 가로 피처
    (DTI 폭, grid 폭)가 정한다. 기준점: 2µm 셀·100nm 피처에서 nG=101 검증.
      nG ∝ (셀 면적) × (100nm/최소피처)  → 품질 배율/시간 캡 적용.
    시간 추정: eig ≈ nG³, FFT ≈ 격자² — nG=101·200² ≈ 12s/λ(TE+TM, CPU) 기준.
    """
    from ..config.loader import load_config
    cfg = load_config(cfg_path)
    g = cfg.get("grid", {})
    span = float(g.get("pixel_pitch_um", 1.0)) * int(g.get("n_pixels", 2))
    st = cfg.get("stack", {})
    feats = [float((st.get("dti") or {}).get("width_um", 0) or 0),
             float((st.get("grid") or {}).get("width_um", 0) or 0)]
    feat = min([f for f in feats if f > 0] or [0.10])
    base = 101.0 * (span / 2.0) ** 2 * min(max(0.10 / feat, 0.7), 2.0)

    def est_s(nG, ds):                          # s/λ 추정 (TE+TM)
        nlat = min(max(int(round(span / (0.005 * ds))), 128), 2400)
        return 12.0 * (nG / 101.0) ** 3 * (nlat / 200.0) ** 2

    if quality == "fast":                       # 미리보기: 수 초/λ
        nG, ds = 61, 4
    elif quality == "converged":                # 수렴: 구조에 맞춘 수렴 nG 중심 (여러 nG 평균)
        from ..sim.converge import recommend_center_nG
        nG = recommend_center_nG(span, feat)
        return nG, 2, est_s(nG, 2)              # 캡 없음 — 수렴 우선
    elif quality == "high":                     # 수렴 지향: 추천값 그대로 (느림 감수)
        nG, ds = int(base * 1.3), 2
    else:                                       # 표준: 추천값을 ~40s/λ 로 시간 캡
        nG, ds = int(base), 2
        while est_s(nG, ds) > 40 and nG > 101:
            nG = int(nG * 0.9)
    nG = max(41, min(257, nG | 1))              # 홀수화 + 범위
    return nG, ds, est_s(nG, ds)


def _run_converged(jid, job, cfg_path, p, lams):
    """수렴 모드: 각 λ 를 center nG 주변 여러 nG 창에서 돌려 평균 -> 참값 + 불확도밴드.

    단일 nG(미수렴) 대신 여러 nG 를 평균해 진동을 상쇄 -> 구조에 무관한 재현가능
    참값. row 에 ±밴드(색채널 nG-std 최대)를 8번째로 부가.
    """
    from ..sim.simulator import RCWAPlaneWaveSimulator
    from ..sim.converge import converged_qe
    center = int(p["nG"]); ds = int(p["downsample"])
    n_samples = int(p.get("conv_samples", 4))
    make_sim = lambda ng: RCWAPlaneWaveSimulator(cfg_path, nG=ng, downsample=ds)
    for i, lam in enumerate(lams):
        if job.get("cancel"):
            job["state"] = "cancelled"; return
        t0 = time.time()
        res = converged_qe(make_sim, [float(lam)], center, n_samples=n_samples,
                           span_frac=0.5, pol=("sum" if p.get("pol") == "sum" else "avg"),
                           channels="RGB")
        q = res["qe"][float(lam)]
        row = [round(float(lam), 5), round(q["refl"]["mean"], 5),
               round(q["QE_tot"]["mean"], 5), round(q["A_stack"]["mean"], 5)]
        for c in "RGB":
            row.append(round(q[c]["mean"], 5))
        row.append(round(res["band"], 5))            # 8번째: ±수렴 불확도 밴드
        job["rows"].append(row)
        job["progress"] = (i + 1) / len(lams)
        job["note"] = (f"λ={lam*1000:.0f}nm  R/G/B={row[4]:.3f}/{row[5]:.3f}/{row[6]:.3f} "
                       f"±{res['band']*100:.1f}%p (수렴평균 nG={res['nGs']}, {time.time()-t0:.0f}s)")
    csv_path = os.path.join(JOBS_DIR, jid + "_qe.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("lambda_um,R,QE_Si_total,A_stack,QE_R,QE_G,QE_B,conv_band_pp\n")
        for r in job["rows"]:
            f.write(",".join(str(x) for x in r) + "\n")
    job["csv"] = os.path.relpath(csv_path, ROOT)
    job["state"] = "done"


def _run_job(jid, cfg_path, p):
    job = JOBS[jid]
    try:
        from ..sim.simulator import RCWAPlaneWaveSimulator
        job["note"] = "구조 생성 + 층 스택 준비중..."
        if p.get("nG") == "auto" or p.get("downsample") == "auto":
            nG, ds, est = _recommend_mesh(cfg_path, p.get("quality", "std"))
            if p.get("nG") == "auto":
                p["nG"] = nG
            if p.get("downsample") == "auto":
                p["downsample"] = ds
            job["nG_auto"] = (f"자동 추천 (품질 {p.get('quality','std')}): "
                              f"nG={p['nG']}, downsample={p['downsample']} · 예상 ~{est:.0f}s/λ")
        lams = np.linspace(p["lam0"], p["lam1"], p["n"])
        if p.get("quality") == "converged":
            _run_converged(jid, job, cfg_path, p, lams)
            return
        sim = RCWAPlaneWaveSimulator(cfg_path, nG=p["nG"], downsample=p["downsample"])
        job["note"] = (f"device={sim.device} · grid {sim.grid_ny}×{sim.grid_nx} · "
                       f"layers {len(sim.layer_stack)} · nG {sim.nG if hasattr(sim,'nG') else p['nG']}")
        for i, lam in enumerate(lams):
            if job.get("cancel"):
                job["state"] = "cancelled"
                return
            t0 = time.time()
            o_te = sim.run(float(lam), theta=p["theta"], pol_te=1.0, pol_tm=0.0)
            if job.get("cancel"):                        # 편광 사이에도 반응 (체감 지연 ↓)
                job["state"] = "cancelled"
                return
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


# --------------------------------------------------------------- DOE job
def _run_doe_job(jid, cfg_path, p):
    """DOE 스윕 + surrogate 피팅 (백그라운드 스레드)."""
    job = JOBS[jid]
    try:
        from ..config.loader import load_config
        from ..sim.simulator import RCWAPlaneWaveSimulator
        from ..sim import doe as doe_mod
        cfg = load_config(cfg_path)
        RCWAPlaneWaveSimulator._auto_model_defaults(cfg)   # QE 경로와 동일 자동 모델
        n = max(2, int(p.get("n", 16)))
        lam0, lam1 = float(p.get("lam0", 0.40)), float(p.get("lam1", 0.70))
        waves = [round(lam0 * 1000 + i * (lam1 - lam0) * 1000 / (n - 1))
                 for i in range(n)]
        total = len(doe_mod.doe_points(p["mode"]))
        job["note"] = f"{total}개 조건 × {n}λ — 1번째 조건 완료 후 예상시간 표시"
        job["doe_total"] = total
        job["doe_done"] = 0

        def prog(done, tot, eta, point):
            job["doe_done"] = done
            job["progress"] = done / tot
            job["eta_s"] = round(eta)
            job["note"] = (f"{done}/{tot} 조건 · 남은시간 ~{int(eta//60)}분"
                           f"{int(eta % 60)}초 · 현재 {list(point)}")

        res = doe_mod.run_doe(cfg, waves, mode=p["mode"], nG=p["nG"],
                              downsample=p["downsample"], lateral_n=p.get("lateral_n", 256),
                              progress=prog, cancel=lambda: job["cancel"])
        csv_path = os.path.join(JOBS_DIR, jid + "_doe.csv")
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write(doe_mod.doe_csv(res))
        job["csv"] = csv_path
        if not res.get("cancelled") and p["mode"] != "axis":
            sur = doe_mod.fit_surrogate(res)
            sur_path = os.path.join(JOBS_DIR, jid + "_surrogate.json")
            with open(sur_path, "w", encoding="utf-8") as f:
                json.dump(sur, f)
            job["surrogate"] = sur_path
            # 대표 R² (그린 채널 중앙 파장) 리포트
            ws = sur["wavelengths_nm"]
            job["r2_G_mid"] = sur["r2"]["G"][int(ws[len(ws) // 2])]
        job["elapsed_s"] = res.get("elapsed_s")
        job["state"] = "cancelled" if res.get("cancelled") else "done"
        job["note"] = (f"{'중단' if res.get('cancelled') else '완료'} · "
                       f"{job['doe_done']}/{total} 조건 · {res.get('elapsed_s', 0)}s")
    except Exception as e:
        job["error"] = f"{type(e).__name__}: {e}"
        job["trace"] = traceback.format_exc()[-2000:]
        job["state"] = "error"


def start_doe_job(yaml_text, p):
    os.makedirs(JOBS_DIR, exist_ok=True)
    jid = "doe" + uuid.uuid4().hex[:9]
    cfg_path = os.path.join(JOBS_DIR, jid + ".yaml")
    with open(cfg_path, "w", encoding="utf-8") as f:
        f.write(yaml_text)
    JOBS[jid] = {"state": "running", "progress": 0.0, "note": "시작중...",
                 "error": None, "cancel": False, "params": p}
    th = threading.Thread(target=_run_doe_job, args=(jid, cfg_path, p), daemon=True)
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
        elif u.path == "/api/materials":
            # data/materials 폴더를 매번 새로 읽어 반환 (위저드가 열릴 때/run 직전 갱신용)
            self._json(_read_materials_folder())
        elif u.path == "/api/qe/status":
            jid = (parse_qs(u.query).get("job") or [""])[0]
            job = JOBS.get(jid)
            if not job:
                self._json({"error": "unknown job"}, 404); return
            self._json({k: job[k] for k in
                        ("state", "progress", "note", "rows", "error", "csv", "nG_auto")
                        if k in job})
        elif u.path == "/api/doe/status":
            jid = (parse_qs(u.query).get("job") or [""])[0]
            job = JOBS.get(jid)
            if not job:
                self._json({"error": "unknown job"}, 404); return
            self._json({k: job[k] for k in
                        ("state", "progress", "note", "error", "eta_s", "elapsed_s",
                         "doe_done", "doe_total", "r2_G_mid")
                        if k in job})
        elif u.path == "/api/doe/file":
            q = parse_qs(u.query)
            jid = (q.get("job") or [""])[0]
            kind = (q.get("kind") or ["csv"])[0]
            job = JOBS.get(jid)
            key = "surrogate" if kind == "surrogate" else "csv"
            if not job or key not in job:
                self.send_response(404); self.end_headers(); return
            self._file(job[key], "application/json" if kind == "surrogate"
                       else "text/csv; charset=utf-8")
        else:
            self.send_response(404); self.end_headers()

    def do_POST(self):
        u = urlparse(self.path)
        try:
            n = int(self.headers.get("Content-Length") or 0)
            data = json.loads(self.rfile.read(n) or b"{}")
        except Exception:
            self._json({"error": "bad json"}, 400); return
        if u.path == "/api/lint":
            # 구조 사전 점검 — 위저드/사용자가 run 전에 문제를 미리 확인
            try:
                import yaml as _yaml
                from ..structure.lint import lint_wizard_cfg
                cfg = _yaml.safe_load(data.get("yaml") or "") or {}
                names = set(_read_materials_folder().keys())
                self._json(lint_wizard_cfg(cfg, material_names=names))
            except Exception as e:
                self._json({"errors": [f"{type(e).__name__}: {e}"], "warnings": []})
            return
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
            # 빈칸/NaN 입력(프론트가 null 로 직렬화)에도 안전하게 — 기본값으로 코어스
            def _num(key, dflt, cast=float):
                v = data.get(key, dflt)
                try:
                    return cast(v) if v is not None else dflt
                except (TypeError, ValueError):
                    return dflt
            try:
                ng_req = data.get("nG", "auto")
                ds_req = data.get("downsample", "auto")
                p = {"lam0": _num("lam0", 0.40),
                     "lam1": _num("lam1", 0.70),
                     "n": max(1, _num("n", 7, int)),
                     "nG": "auto" if str(ng_req) == "auto"
                           else (lambda v: v if v % 2 else v + 1)(max(9, int(ng_req))),
                     "downsample": "auto" if str(ds_req) == "auto" else max(1, int(ds_req)),
                     "quality": str(data.get("quality", "std")),
                     "theta": _num("theta", 0.0),
                     "pol": str(data.get("pol", "avg"))}
            except (TypeError, ValueError) as e:
                self._json({"error": f"파라미터 오류: {e}"}, 400); return
            jid = start_job(yaml_text, p)
            self._json({"job": jid})
        elif u.path == "/api/qe/cancel":
            job = JOBS.get(data.get("job") or "")
            if job:
                job["cancel"] = True
            self._json({"ok": bool(job)})
        elif u.path == "/api/doe":
            yaml_text = data.get("yaml") or ""
            if not yaml_text.strip():
                self._json({"error": "yaml 이 비었습니다"}, 400); return
            try:
                mode = str(data.get("mode", "surrogate"))
                assert mode in ("axis", "surrogate", "full"), "mode"
                ng = max(9, int(data.get("nG", 151)))
                p = {"mode": mode, "nG": ng if ng % 2 else ng + 1,
                     "downsample": max(1, int(data.get("downsample", 2))),
                     "n": max(2, int(data.get("n", 16))),
                     "lam0": float(data.get("lam0", 0.40)),
                     "lam1": float(data.get("lam1", 0.70))}
            except (TypeError, ValueError, AssertionError) as e:
                self._json({"error": f"파라미터 오류: {e}"}, 400); return
            jid = start_doe_job(yaml_text, p)
            from ..sim.doe import doe_points
            self._json({"job": jid, "total": len(doe_points(p["mode"]))})
        elif u.path == "/api/doe/cancel":
            job = JOBS.get(data.get("job") or "")
            if job:
                job["cancel"] = True
            self._json({"ok": bool(job)})
        else:
            self.send_response(404); self.end_headers()


def _seed_user_assets():
    """exe 실행 시, 사용자가 편집할 자원(data/·conf/)을 exe 옆에 최초 1회 복사.
    이미 있으면 건드리지 않음(사용자 편집 보존). 소스 실행이면 no-op."""
    if not FROZEN or os.path.abspath(APP_DIR) == os.path.abspath(ASSET_ROOT):
        return
    for sub in ("data", "conf"):
        src, dst = os.path.join(ASSET_ROOT, sub), os.path.join(APP_DIR, sub)
        if os.path.isdir(src) and not os.path.isdir(dst):
            try:
                shutil.copytree(src, dst)
                print(f"[cis-rcwa] 초기 자원 복사: {dst}")
            except Exception as e:
                print(f"[cis-rcwa] 자원 복사 경고({sub}): {e}")


def serve(port=8787, open_browser=True):
    _seed_user_assets()
    os.makedirs(JOBS_DIR, exist_ok=True)
    os.chdir(APP_DIR)                           # materials 폴더 자동탐색 기준(사용자 옆)
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
