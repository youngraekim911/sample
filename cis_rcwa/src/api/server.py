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
import math
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
DEFAULT_CACHE_DIR = os.path.join(APP_DIR, "out", "surrogate_cache")   # 기본 DB 폴더


def _cache_dir():
    """활성 surrogate DB 폴더 — 환경변수/포인터파일(사용자 지정)로 바꿀 수 있음."""
    from ..sim import surrogate_cache as sc
    return sc.resolve_dir(APP_DIR, DEFAULT_CACHE_DIR)

JOBS = {}          # job id -> dict(state, progress, note, rows, error, cancel)
_LOCK = threading.Lock()


def _dir_size_bytes(d):
    """폴더 내 파일 총 바이트 (얕게)."""
    tot = 0
    try:
        for name in os.listdir(d):
            fp = os.path.join(d, name)
            if os.path.isfile(fp):
                tot += os.path.getsize(fp)
    except OSError:
        pass
    return tot


def _doe_points_count(mode, k, nsamples=None):
    """실제 실행 조건 수(열거 없이 공식으로 — full 5^k 폭증 방지)."""
    from math import comb
    if mode == "lhs":
        return int(nsamples or 0)
    if mode == "axis":
        return 1 + 4 * k
    if mode == "surrogate":
        return 1 + 4 * k + 4 * comb(k, 2)
    if mode == "full":
        return 5 ** k
    return 0


def _doe_estimate(data):
    """용량/시간 계량기 — 실행 전 예상 파일크기·소요시간 + 현재 DB 총용량.

    파일크기: doe.csv(지배적) + surrogate.json 을 실측 보정계수로 추정.
    시간: solve 1회 기준 per_solve_s(측정치 있으면 사용) × 조건수 × 파장 × 2편광.
    """
    mode = str(data.get("mode", "lhs"))
    axes = data.get("axes") or []
    k = len(axes) if axes else int(data.get("naxes", 6))
    nwl = max(1, int(data.get("n", 16)))
    nsamples = data.get("nsamples")
    model = str(data.get("model", "auto"))
    pts = _doe_points_count(mode, k, nsamples)

    # ── 파일 크기 (실측 보정: csv ≈ 행×(k+4)×8.7B, 행=pts×nwl) ──
    rows = pts * nwl
    csv_b = rows * (k + 4) * 8.7
    feat = 1 + 2 * k + (k * (k - 1)) // 2
    poly_json_b = nwl * 3 * feat * 8 + 400
    rbf_json_b = nwl * 3 * pts * 11 + pts * k * 9 + 800   # rbf 는 표본 저장으로 큼
    json_b = (max(poly_json_b, rbf_json_b) if model in ("auto", "rbf")
              else poly_json_b + (nwl * 3 * k * 8 if model == "cubic" else 0))
    total_b = csv_b + json_b

    # ── 시간 ──
    # 정확도 우선순위: (1) 사용자 구조/기기에서 측정된 per_wl_s(파장당, 2편광 포함) →
    # (2) per_solve_s → (3) nG 로 러프(harmonic 지배 O(nG^~2.2), 실측 nG41=0.76s/solve 기준).
    nG = int(data.get("nG", 151))
    per_wl = data.get("per_wl_s")
    if per_wl is not None:                                # 가장 정확(기기·구조 반영)
        est_s = float(per_wl) * pts * nwl
        per_solve = float(per_wl) / 2.0
        basis = "measured"
    else:
        per_solve = data.get("per_solve_s")
        if per_solve is None:
            per_solve = 0.76 * (nG / 41.0) ** 2.2        # raster 아닌 harmonic 지배
            basis = "rough(nG)"
        else:
            basis = "per_solve"
        per_solve = float(per_solve)
        est_s = per_solve * pts * nwl * 2                 # 2편광

    db = _cache_dir()
    return {"points": pts, "rows": rows, "naxes": k,
            "csv_bytes": int(csv_b), "json_bytes": int(json_b),
            "total_bytes": int(total_b),
            "est_seconds_lo": int(est_s / 4.0),          # GPU 여지
            "est_seconds": int(est_s),
            "est_seconds_hi": int(est_s * 2.5),          # 고품질/CPU 여지
            "time_basis": basis,
            "db_dir": db, "db_total_bytes": _dir_size_bytes(db),
            "per_solve_s": round(float(per_solve), 3)}


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
            o_te = sim.run(float(lam), theta=p["theta"], phi=p.get("phi", 0.0),
                           pol_te=1.0, pol_tm=0.0)
            if job.get("cancel"):                        # 편광 사이에도 반응 (체감 지연 ↓)
                job["state"] = "cancelled"
                return
            o_tm = sim.run(float(lam), theta=p["theta"], phi=p.get("phi", 0.0),
                           pol_te=0.0, pol_tm=1.0)
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


LAST_QE_JID = None        # 페이지 이동/새로고침 후에도 마지막 QE 결과를 복구(reattach)


def start_job(yaml_text, p):
    global LAST_QE_JID
    os.makedirs(JOBS_DIR, exist_ok=True)
    jid = uuid.uuid4().hex[:12]
    cfg_path = os.path.join(JOBS_DIR, jid + ".yaml")
    with open(cfg_path, "w", encoding="utf-8") as f:
        f.write(yaml_text)
    JOBS[jid] = {"state": "running", "progress": 0.0, "note": "시작중...",
                 "rows": [], "error": None, "cancel": False, "params": p}
    LAST_QE_JID = jid
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
        from ..sim import surrogate_cache as sc
        cfg = load_config(cfg_path)
        RCWAPlaneWaveSimulator._auto_model_defaults(cfg)   # QE 경로와 동일 자동 모델
        n = max(2, int(p.get("n", 16)))
        lam0, lam1 = float(p.get("lam0", 0.40)), float(p.get("lam1", 0.70))
        waves = [round(lam0 * 1000 + i * (lam1 - lam0) * 1000 / (n - 1))
                 for i in range(n)]
        # 사용자가 축을 골랐으면 그 축(리스트), 아니면 기본 6축. run_doe/apply_point 는
        # (key,label,step,unit) 순서만 쓰므로 JSON 리스트 그대로 사용 가능.
        axes = p.get("axes") or doe_mod.AXES_DEFAULT
        job["naxes"] = len(axes)
        total = len(doe_mod.doe_points(p["mode"], naxes=len(axes),
                                       nsamples=p.get("nsamples")))
        lateral_n = p.get("lateral_n", 256)
        mdir = _materials_dir()
        is_lhs = p["mode"] == "lhs"

        # ── surrogate 캐시(DB) 조회: 구조·조건·물질이 동일하면 재계산 생략 ──
        key = None
        if p["mode"] != "axis":
            try:
                conds = {"waves": waves, "mode": p["mode"],
                         "nG": p["nG"], "downsample": p["downsample"],
                         "lateral_n": lateral_n}
                if is_lhs:                                    # 샘플링·모델까지 키에 반영
                    conds.update({"nsamples": p["nsamples"], "bound": p["bound"],
                                  "seed": p["seed"], "model": p["model"],
                                  "cv_folds": p["cv_folds"]})
                key = sc.compute_key(cfg, conds, axes, mdir)
                job["cache_key"] = key
            except Exception:
                key = None
        if key and not p.get("force"):
            rec = sc.load(_cache_dir(), key)
            if rec:
                sur = rec["surrogate"]
                sur_path = os.path.join(JOBS_DIR, jid + "_surrogate.json")
                with open(sur_path, "w", encoding="utf-8") as f:
                    json.dump(sur, f)
                job["surrogate"] = sur_path
                if rec.get("csv"):
                    csv_path = os.path.join(JOBS_DIR, jid + "_doe.csv")
                    with open(csv_path, "w", encoding="utf-8") as f:
                        f.write(rec["csv"])
                    job["csv"] = csv_path
                ws = sur["wavelengths_nm"]
                job["r2_G_mid"] = doe_mod.r2_at(sur, "G", ws[len(ws) // 2])
                if sur.get("type") == "emulator":      # 넓은 에뮬레이터: 검증지표 복원
                    m = sur.get("metrics", {})
                    job["model"] = sur.get("model")
                    job["r2_cv_mean"] = m.get("r2_cv_mean")
                    job["r2_insample_mean"] = m.get("r2_insample_mean")
                    job["per_model_cv"] = m.get("per_model_cv")
                job["doe_total"] = total
                job["doe_done"] = total
                job["progress"] = 1.0
                job["elapsed_s"] = 0
                job["cached"] = True
                job["state"] = "done"
                job["note"] = (f"✓ 캐시 적중 — 동일 구조·조건, 재계산 생략 "
                               f"(key {key[:8]})")
                return

        job["note"] = f"{total}개 조건 × {n}λ — 1번째 조건 완료 후 예상시간 표시"
        job["doe_total"] = total
        job["doe_done"] = 0

        dev = "cuda(GPU)" if torch.cuda.is_available() else "cpu"
        job["device"] = dev

        # ── 체크포인트: 매 조건 완료마다 <key>.part.json 저장 + 같은 키로 재실행
        #    시 이어하기 — 장시간 스윕이 취소/크래시/정전으로 죽어도 완료분 보존 ──
        resume = None
        if key:
            ck = sc.load_checkpoint(_cache_dir(), key)
            if ck and not p.get("force"):
                resume = (ck.get("doe") or {}).get("points") or None
                if resume:
                    job["resumed"] = len(resume)
                    job["note"] = (f"🔄 이어서 계산 — 중간 저장된 {len(resume)}/{total}"
                                   f"개 조건 재사용")
        ck_meta = {"mode": p["mode"], "product": str(cfg.get("product", "")),
                   "total": total}

        def on_point_fn(out):
            # 저장 성공/실패를 job 에 기록 — 진행 문구가 '실제 상태'를 말하게 한다
            cdir = _cache_dir()
            try:
                sc.save_checkpoint(cdir, key, out, ck_meta)
                job["ckpt_n"] = len(out["points"])
                job["ckpt_dir"] = cdir
                job.pop("ckpt_error", None)
            except Exception as e:
                job["ckpt_error"] = f"{type(e).__name__}: {e}"
        on_point = on_point_fn if key else None

        def prog(done, tot, eta, point):
            job["doe_done"] = done
            job["progress"] = done / tot
            job["eta_s"] = round(eta)
            if job.get("ckpt_error"):
                ck = f" · ⚠ 중간저장 실패: {job['ckpt_error'][:60]}"
            elif job.get("ckpt_n"):
                ck = f" · 💾 {job['ckpt_n']}개 저장됨 → {job.get('ckpt_dir','')}"
            elif not key:
                ck = " · (이 모드는 중간저장 없음)"
            else:
                ck = ""
            job["note"] = (f"[{dev}] {done}/{tot} 조건 · 남은시간 ~{int(eta//60)}분"
                           f"{int(eta % 60)}초 · 현재 {list(point)}" + ck)

        res = doe_mod.run_doe(cfg, waves, mode=p["mode"], nG=p["nG"],
                              downsample=p["downsample"], lateral_n=p.get("lateral_n", 256),
                              axes=axes, nsamples=p.get("nsamples"),
                              bound=p.get("bound", 2.0), seed=p.get("seed", 0),
                              progress=prog, cancel=lambda: job["cancel"],
                              resume_points=resume, on_point=on_point)
        csv_path = os.path.join(JOBS_DIR, jid + "_doe.csv")
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write(doe_mod.doe_csv(res))
        job["csv"] = csv_path
        if p["mode"] == "axis" and not res.get("cancelled"):
            # 스크리닝의 산출물 = '노브 중요도 랭킹' (모델/캐시 없음 — 화면에 표시)
            try:
                job["screening"] = doe_mod.screening_from_axis(res)
            except Exception:
                pass
        if not res.get("cancelled") and p["mode"] != "axis":
            if is_lhs:            # 넓은 공간 에뮬레이터(다중모델 + 교차검증 자동선택)
                from ..sim.emulator import fit_emulator
                sur = fit_emulator(res, model=p["model"], cv_folds=p["cv_folds"],
                                   base_cfg=cfg)      # 탐색 페이지용 기준 구조 동봉
                job["model"] = sur["model"]
                job["r2_cv_mean"] = sur["metrics"]["r2_cv_mean"]
                job["r2_insample_mean"] = sur["metrics"]["r2_insample_mean"]
                job["per_model_cv"] = sur["metrics"]["per_model_cv"]
            else:                 # 국소 2차 RSM (기존 호환)
                sur = doe_mod.fit_surrogate(res)
            sur_path = os.path.join(JOBS_DIR, jid + "_surrogate.json")
            with open(sur_path, "w", encoding="utf-8") as f:
                json.dump(sur, f)
            job["surrogate"] = sur_path
            # 대표 R² (그린 채널 중앙 파장) 리포트
            ws = sur["wavelengths_nm"]
            job["r2_G_mid"] = doe_mod.r2_at(sur, "G", ws[len(ws) // 2])
            # ── 캐시(DB) 저장: 다음에 같은 구조·조건이면 재계산 생략 ──
            if key:
                try:
                    meta = {"mode": p["mode"], "nG": p["nG"],
                            "downsample": p["downsample"], "wavelengths_nm": list(ws),
                            "product": str(cfg.get("product", "")),
                            "r2_G_mid": job["r2_G_mid"],
                            "elapsed_s": res.get("elapsed_s"),
                            # 구조 지문 — 같은 구조를 조건만 바꿔 돌린 모델을 묶는 키
                            "struct_key": sc.compute_struct_key(cfg, mdir)[:12],
                            "naxes": len(axes),
                            "axis_keys": [a[0] for a in axes]}
                    if is_lhs:
                        meta.update({"model": job.get("model"),
                                     "r2_cv_mean": job.get("r2_cv_mean"),
                                     "n_samples": p.get("nsamples")})
                    sc.save(_cache_dir(), key, sur, meta,
                            csv_text=doe_mod.doe_csv(res))
                    sc.clear_checkpoint(_cache_dir(), key)   # 완주 → 중간 저장 정리
                except Exception:
                    pass
        # ── 중단(취소)돼도: 체크포인트는 유지(이어하기용) + 완료분으로 부분 모델 ──
        partial_note = ""
        if res.get("cancelled") and key:
            npts = len(res.get("points") or [])
            if job.get("ckpt_error"):
                partial_note = f" · ⚠ 중간저장 실패: {job['ckpt_error'][:80]}"
            elif job.get("ckpt_n"):
                partial_note = (f" · 💾 {job['ckpt_n']}개 조건 저장됨 "
                                f"({job.get('ckpt_dir','')}) — 같은 조건으로 다시 "
                                f"Run 하면 이어서 계산")
            else:
                partial_note = " · (완료된 조건이 없어 중간저장 없음)"
            if is_lhs and npts >= max(10, len(axes) + 2):
                try:                                   # 완료분만으로도 임시 모델 제공
                    from ..sim.emulator import fit_emulator
                    sur = fit_emulator(res, model=p["model"],
                                       cv_folds=min(p["cv_folds"], max(2, npts // 4)),
                                       base_cfg=cfg)
                    sur_path = os.path.join(JOBS_DIR, jid + "_surrogate.json")
                    with open(sur_path, "w", encoding="utf-8") as f:
                        json.dump(sur, f)
                    job["surrogate"] = sur_path
                    job["model"] = sur["model"]
                    job["r2_cv_mean"] = sur["metrics"]["r2_cv_mean"]
                    job["r2_G_mid"] = doe_mod.r2_at(sur, "G",
                                                    sur["wavelengths_nm"][len(sur["wavelengths_nm"]) // 2])
                    partial_note += f" · 부분 모델 생성(CV R²={job['r2_cv_mean']})"
                except Exception:
                    pass
        job["elapsed_s"] = res.get("elapsed_s")
        job["state"] = "cancelled" if res.get("cancelled") else "done"
        if is_lhs and not res.get("cancelled"):
            job["note"] = (
                f"완료 · 넓은 에뮬레이터[{job.get('model')}] {job['doe_done']}/{total}샘플 · "
                f"검증 CV R²={job.get('r2_cv_mean')} (in-sample "
                f"{job.get('r2_insample_mean')}) · {res.get('elapsed_s', 0)}s")
        else:
            job["note"] = (f"{'중단' if res.get('cancelled') else '완료'} · "
                           f"{job['doe_done']}/{total} 조건 · {res.get('elapsed_s', 0)}s"
                           + partial_note)
    except Exception as e:
        job["error"] = f"{type(e).__name__}: {e}"
        job["trace"] = traceback.format_exc()[-2000:]
        job["state"] = "error"


# --------------------------------------------------------------- 사광 이미지 시뮬
LAST_IMG_JID = None


def _cra_spec_path(product):
    return os.path.join(APP_DIR, "data", "cra", str(product).strip() + ".yaml")


def _same_color_diff(units, labels, waves, threshold_pct=30.0):
    """동컬러 픽셀 간 diff — 사광의 핵심 판정 지표.

    shrink 로 초점을 되돌려도 빗각 자체 때문에 같은 색 픽셀들(tetra 는 2×2 quad
    4개, bayer 는 Gr/Gb)이 서로 다른 QE 를 받는다. 이 채널내 편차가 크면 디모자익
    후 미로(maze)/격자 아티팩트가 생기므로 스펙: diff ≤ threshold(기본 30%).

    diff_pct = (max−min)/mean × 100, 같은 라벨 픽셀끼리(필드·파장별).
    컬러별 최종 판정은 on-band 파장(그 컬러 평균 QE 가 최대치의 50% 이상인 λ)
    에서만 — off-band(예: 450nm 의 R)는 QE 자체가 0 근처라 상대 diff 가
    의미없이 튀기 때문. 반환 dict 는 결과 JSON 에 그대로 실린다.
    """
    L = np.asarray(labels, dtype=object)
    colors = sorted({str(v) for v in L.flat})
    # 컬러×파장 평균 QE (전 필드) → on-band 파장 집합
    cmean = {c: {} for c in colors}
    for w in waves:
        acc = {c: [] for c in colors}
        for u in units.values():
            arr = np.asarray(u["pixels"][int(w)], float)
            for c in colors:
                acc[c].append(float(arr[L == c].mean()))
        for c in colors:
            cmean[c][int(w)] = float(np.mean(acc[c]))
    onband = {}
    for c in colors:
        top = max(cmean[c].values())
        ob = [w for w in cmean[c] if cmean[c][w] >= 0.5 * top]
        onband[c] = ob if ob else [int(w) for w in waves]

    per_field, worst = {}, {}                        # worst[(wl,c)] = 최악 필드
    for (x, y), u in units.items():
        fkey = f"{x},{y}"
        per_field[fkey] = {}
        for w in waves:
            arr = np.asarray(u["pixels"][int(w)], float)
            d = {}
            for c in colors:
                v = arr[L == c]
                mn, mx, mu = float(v.min()), float(v.max()), float(v.mean())
                dp = (mx - mn) / mu * 100.0 if mu > 1e-12 else 0.0
                d[c] = {"min": round(mn, 5), "max": round(mx, 5),
                        "mean": round(mu, 5), "n": int(v.size),
                        "diff_pct": round(dp, 2)}
                k = (int(w), c)
                if k not in worst or dp > worst[k]["diff_pct"]:
                    worst[k] = {"diff_pct": round(dp, 2), "field": fkey,
                                "r": round(math.hypot(x, y), 3)}
            per_field[fkey][int(w)] = d
    worst_wl = {}
    for (w, c), rec in worst.items():
        worst_wl.setdefault(str(w), {})[c] = rec
    overall = {}                                     # 컬러별 최종(on-band 한정)
    for c in colors:
        for w in onband[c]:
            rec = worst.get((int(w), c))
            if rec and (c not in overall
                        or rec["diff_pct"] > overall[c]["diff_pct"]):
                overall[c] = dict(rec, wl=int(w))
    return {"threshold_pct": float(threshold_pct),
            "metric": "(max-min)/mean x100 · 같은 라벨 픽셀 · 필드/파장별",
            "onband_nm": {c: sorted(onband[c]) for c in colors},
            "per_field": per_field, "worst": worst_wl, "overall": overall,
            "pass": bool(overall) and all(v["diff_pct"] <= threshold_pct
                                          for v in overall.values())}


def _run_image_job(jid, cfg_path, p):
    """센서 전면 QE 이미지: 옥탄트 필드 격자 → 필드별 (CRA shift + F# 원뿔 적분 +
    편광평균) unit QE → 8-fold 대칭 확장 조립 → json/png/npz 저장."""
    job = JOBS[jid]
    try:
        from ..config.loader import load_config
        from ..sim.simulator import RCWAPlaneWaveSimulator
        from ..structure.blocks import ir_from_wizard_cfg
        from ..sim import cra as cra_mod
        from ..sim import octant as oc
        cfg = load_config(cfg_path)
        RCWAPlaneWaveSimulator._auto_model_defaults(cfg)
        product = str(cfg.get("product", "")).strip()
        spec = p.get("spec")
        if not spec:
            path = _cra_spec_path(product)
            if not os.path.isfile(path):
                raise ValueError(f"CRA 스펙 파일이 없습니다: data/cra/{product}.yaml "
                                 f"— 제품 CRA yaml 을 그 위치에 넣어주세요 "
                                 f"(예시: data/cra/qcell_demo.yaml)")
            spec = cra_mod.load_cra_spec(path)
        n = max(1, int(p["n"]))
        waves = [round(p["lam0"] * 1000 + i * (p["lam1"] - p["lam0"]) * 1000
                       / max(n - 1, 1)) for i in range(n)]
        fields = oc.make_xy_fields_set(p["field_step"], p.get("region", "octant"))
        ts, ps = int(p["theta_samples"]), int(p["phi_samples"])
        total = len(fields) * len(waves) * ts * ps * 2
        job["doe_total"] = len(fields)
        job["doe_done"] = 0
        base_ir = ir_from_wizard_cfg(cfg, p.get("lateral_n", 256))
        dev = "cuda(GPU)" if torch.cuda.is_available() else "cpu"
        job["device"] = dev
        units = {}
        t0 = time.time()
        solves = {"n": 0}

        def on_solve():
            solves["n"] += 1
            el = time.time() - t0
            eta = el / solves["n"] * (total - solves["n"])
            job["progress"] = solves["n"] / total
            job["note"] = (f"[{dev}] 필드 {min(job['doe_done']+1, len(fields))}"
                           f"/{len(fields)} · RCWA {solves['n']}/{total} · "
                           f"남은 ~{int(eta//60)}분{int(eta % 60)}초")

        for i, (x, y) in enumerate(fields):
            if job["cancel"]:
                break
            u = cra_mod.unit_qe_at_field(
                base_ir, spec, x, y, waves, nG=p["nG"], downsample=p["downsample"],
                theta_samples=ts, phi_samples=ps,
                on_solve=on_solve, cancel=lambda: job["cancel"])
            if u is None:
                break
            units[(x, y)] = u
            job["doe_done"] = i + 1
        if not units:
            raise ValueError("계산된 필드가 없습니다 (시작 직후 중단됨)")
        labels = units[next(iter(units))]["labels"]
        images = {}
        meta = None
        for w in waves:
            uw = {f: units[f]["pixels"][int(w)] for f in units}
            img, meta = oc.assemble_image(uw, p["field_step"], labels)
            images[int(w)] = img
        mean = np.nanmean(np.stack([images[int(w)] for w in waves]), axis=0)
        diff = _same_color_diff(units, labels, waves,
                                float(p.get("diff_threshold_pct", 30.0)))

        def j2(a):                                  # NaN -> None (JS JSON 호환)
            return [[(None if not np.isfinite(v) else round(float(v), 5))
                     for v in row] for row in np.asarray(a)]
        res = {"wavelengths_nm": [int(w) for w in waves],
               "field_step": p["field_step"], "meta": meta, "labels": labels,
               "product": product, "region": p.get("region", "octant"),
               "theta_samples": ts, "phi_samples": ps,
               "spec": {k: spec[k] for k in
                        ("f_number", "ml_um_per_deg", "cf_um_per_deg")},
               "fields": {f"{x},{y}": {"cra_deg": units[(x, y)]["cra_deg"],
                                       "shift_ml_um": units[(x, y)]["shift_ml_um"],
                                       "rgb": units[(x, y)]["rgb"]}
                          for (x, y) in units},
               "images": {str(int(w)): j2(images[int(w)]) for w in waves},
               "mean": j2(mean), "diff": diff}
        jp = os.path.join(JOBS_DIR, jid + "_image.json")
        with open(jp, "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False)
        job["image"] = jp
        try:                                        # npz (수치 후처리용)
            zp = os.path.join(JOBS_DIR, jid + "_image.npz")
            np.savez(zp, mean=mean,
                     **{f"wl{int(w)}": images[int(w)] for w in waves})
            job["npz"] = zp
        except Exception:
            pass
        try:                                        # png (mean 히트맵)
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(7, 5.4))
            im = ax.imshow(mean, cmap="viridis")
            ax.set_title(f"QE map (mean of {len(waves)} wavelengths)")
            fig.colorbar(im, ax=ax, shrink=0.85)
            pp = os.path.join(JOBS_DIR, jid + "_image.png")
            fig.savefig(pp, dpi=120, bbox_inches="tight")
            plt.close(fig)
            job["png"] = pp
        except Exception:
            pass
        job["image_ready"] = True
        mn, mx = float(np.nanmin(mean)), float(np.nanmax(mean))
        job["elapsed_s"] = round(time.time() - t0, 1)
        job["state"] = "cancelled" if job["cancel"] else "done"
        dtxt = ""
        if diff.get("overall"):
            parts = [f"{c} {diff['overall'][c]['diff_pct']:.0f}%"
                     for c in sorted(diff["overall"])]
            dtxt = (" · 동컬러 diff " + " ".join(parts)
                    + (" ✓" if diff["pass"] else f" ⚠>{diff['threshold_pct']:.0f}%"))
        job["note"] = (f"{'중단(부분 조립)' if job['cancel'] else '완료'} · "
                       f"필드 {job['doe_done']}/{len(fields)} · QE {mn:.3f}~{mx:.3f}"
                       f"{dtxt} · {job['elapsed_s']}s")
    except Exception as e:
        job["error"] = f"{type(e).__name__}: {e}"
        job["trace"] = traceback.format_exc()[-2000:]
        job["state"] = "error"


# --------------------------------------------------------------- BO(최적 조건 찾기)
def _run_bo_job(jid, p):
    """베이지안 최적화 1라운드: 저장된 모델(키)의 원자료 -> GP+EI 로 다음 실험
    추천 -> RCWA 실측 -> 데이터에 합쳐 모델 재학습 -> 같은 키로 저장.
    목표는 유저 언어 그대로: {ch, wl, goal(max/min)}."""
    job = JOBS[jid]
    try:
        from ..sim import bo as bo_mod
        from ..sim import doe as doe_mod
        from ..sim import surrogate_cache as sc
        from ..sim.emulator import fit_emulator
        from ..sim.simulator import RCWAPlaneWaveSimulator
        from ..structure.blocks import ir_from_wizard_cfg

        cdir = _cache_dir()
        rec = sc.load(cdir, p["key"])
        if not rec:
            raise ValueError("모델을 DB 에서 찾지 못했습니다 (key)")
        sur = rec["surrogate"]
        axes = [list(a) for a in sur["axes"]]
        k = len(axes)
        base_cfg = sur.get("base_cfg")
        if not base_cfg:
            raise ValueError("구조가 동봉되지 않은 모델 — 새로 만든 모델로 시도하세요")
        ws = sorted(int(w) for w in sur["wavelengths_nm"])
        wl = min(ws, key=lambda w: abs(w - int(p["wl"])))    # 모델 파장 중 가장 가까운 것
        ch, goal = p["ch"], p["goal"]
        pts = bo_mod.points_from_csv(rec.get("csv", ""), k)
        if len(pts) < 5:
            raise ValueError("모델의 원자료(CSV)가 부족해 BO 를 시작할 수 없습니다")

        def val(qe):
            q = qe.get(wl) or qe.get(str(wl)) or {}
            return float(q.get(ch, 0.0))
        sgn = 1.0 if goal == "max" else -1.0
        X = [pp["steps"] for pp in pts]
        y = [sgn * val(pp["qe"]) for pp in pts]
        base_best = max(y)                                    # 지금까지의 최고(부호 적용)

        lo = [v * 2 for v in sur["box"]["lo"]]
        hi = [v * 2 for v in sur["box"]["hi"]]
        n = int(p["n"])
        sugg = bo_mod.suggest(X, y, lo, hi, n_pick=n, seed=len(pts))
        job["doe_total"] = n
        job["doe_done"] = 0
        meta = dict(rec.get("meta") or {})
        nG = int(meta.get("nG", 101))
        dsamp = int(meta.get("downsample", 2))
        dev = "cuda(GPU)" if torch.cuda.is_available() else "cpu"
        obj_txt = f"{ch}@{wl}nm {'최대' if goal == 'max' else '최소'}"
        evaluated = []
        t0 = time.time()
        for i, sp in enumerate(sugg):
            if job["cancel"]:
                break
            c = doe_mod.apply_point(base_cfg, sp, axes)
            sim = RCWAPlaneWaveSimulator(ir_from_wizard_cfg(c, 256), nG=nG,
                                         downsample=dsamp)
            qe = {}
            for w in ws:
                if job["cancel"]:
                    break
                acc = {"R": 0.0, "G": 0.0, "B": 0.0}
                for pol in ((1.0, 0.0), (0.0, 1.0)):
                    o = sim.run(w / 1000.0, pol_te=pol[0], pol_tm=pol[1])
                    for L in "RGB":
                        acc[L] += 0.5 * o["QE_rgb"][L]
                qe[w] = {L: round(acc[L], 5) for L in "RGB"}
            if job["cancel"] or len(qe) < len(ws):
                break
            pts.append({"steps": [round(float(v), 6) for v in sp], "qe": qe})
            v = sgn * val(qe)
            evaluated.append(round(val(qe), 5))
            X.append(sp); y.append(v)
            job["doe_done"] = i + 1
            el = time.time() - t0
            eta = el / (i + 1) * (n - i - 1)
            cur_best = max(y)
            job["progress"] = (i + 1) / n
            job["note"] = (f"[{dev}] 자동실험 {i+1}/{n} · {obj_txt} 지금까지 최고 "
                           f"{sgn*cur_best:.4f} · 남은 ~{int(eta//60)}분{int(eta%60)}초")

        done_n = job["doe_done"]
        if done_n:                        # 하나라도 실측했으면 모델 갱신 + 저장
            res = {"axes": axes, "mode": "lhs", "nG": nG,
                   "bound": float(sur.get("bound", 2.0)), "seed": 0,
                   "wavelengths_nm": ws, "points": pts}
            sur2 = fit_emulator(res, model="auto",
                                cv_folds=min(5, max(2, len(pts) // 4)),
                                base_cfg=base_cfg)
            meta.update({"n_samples": len(pts),
                         "bo_rounds": int(meta.get("bo_rounds", 0)) + 1,
                         "r2_cv_mean": sur2["metrics"]["r2_cv_mean"],
                         "model": sur2["model"]})
            sc.save(cdir, p["key"], sur2, meta, csv_text=doe_mod.doe_csv(res))
            job["cache_key"] = p["key"]
            job["r2_cv_mean"] = sur2["metrics"]["r2_cv_mean"]
        bi = int(np.argmax(y))
        best_steps = [round(float(v), 3) for v in X[bi]]
        job["bo"] = {
            "objective": obj_txt, "n_done": done_n,
            "base_value": round(sgn * base_best, 5),
            "best_value": round(sgn * max(y), 5),
            "improved": bool(max(y) > base_best + 1e-9),
            "delta": round(sgn * (max(y) - base_best), 5),
            "best_steps": best_steps,
            "best_params": [{"label": a[1], "unit": a[3],
                             "delta": round(float(best_steps[i]) * float(a[2]), 4)}
                            for i, a in enumerate(axes)],
            "saved": bool(done_n)}
        job["elapsed_s"] = round(time.time() - t0, 1)
        job["state"] = "cancelled" if job["cancel"] else "done"
        job["note"] = (f"{'중단' if job['cancel'] else '완료'} · 자동실험 {done_n}/{n} · "
                       f"{obj_txt}: {job['bo']['base_value']} → {job['bo']['best_value']}"
                       + (f" (+{abs(job['bo']['delta'])})" if job["bo"]["improved"]
                          else " (개선 없음 — 라운드를 더 돌리거나 목표를 바꿔보세요)"))
    except Exception as e:
        job["error"] = f"{type(e).__name__}: {e}"
        job["trace"] = traceback.format_exc()[-2000:]
        job["state"] = "error"


LAST_DOE_JID = None       # 브라우저가 꺼졌다 다시 열려도 재접속(reattach)할 최근 DOE 작업


def start_doe_job(yaml_text, p):
    global LAST_DOE_JID
    os.makedirs(JOBS_DIR, exist_ok=True)
    jid = "doe" + uuid.uuid4().hex[:9]
    cfg_path = os.path.join(JOBS_DIR, jid + ".yaml")
    with open(cfg_path, "w", encoding="utf-8") as f:
        f.write(yaml_text)
    JOBS[jid] = {"state": "running", "progress": 0.0, "note": "시작중...",
                 "error": None, "cancel": False, "params": p}
    LAST_DOE_JID = jid
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
        elif u.path in ("/explorer", "/model", "/model_explorer.html"):
            # 모델 탐색 — 학습된 에뮬레이터로 RCWA 없이 즉석 QE (front user 용)
            self._file(os.path.join(ROOT, "editors", "model_explorer.html"),
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
        elif u.path == "/api/qe/last":
            # 페이지 이동/새로고침 후 마지막 QE 실행 복구 — rows(결과) + 실행 당시
            # yaml(원인분석이 그 구조를 그대로 재사용)까지 돌려준다
            if not LAST_QE_JID or LAST_QE_JID not in JOBS:
                self._json({"job": None}); return
            job = JOBS[LAST_QE_JID]
            out = {k: job[k] for k in
                   ("state", "progress", "note", "rows", "error", "nG_auto")
                   if k in job}
            out["job"] = LAST_QE_JID
            try:
                with open(os.path.join(JOBS_DIR, LAST_QE_JID + ".yaml"),
                          encoding="utf-8") as f:
                    out["yaml"] = f.read()
            except OSError:
                pass
            self._json(out)
        elif u.path == "/api/doe/status":
            jid = (parse_qs(u.query).get("job") or [""])[0]
            job = JOBS.get(jid)
            if not job:
                self._json({"error": "unknown job"}, 404); return
            self._json({k: job[k] for k in
                        ("state", "progress", "note", "error", "eta_s", "elapsed_s",
                         "doe_done", "doe_total", "r2_G_mid", "cached", "cache_key", "device",
                         "model", "r2_cv_mean", "r2_insample_mean", "per_model_cv", "naxes", "resumed",
                         "ckpt_n", "ckpt_dir", "ckpt_error", "screening", "bo", "image_ready")
                        if k in job})
        elif u.path == "/api/doe/last":
            # 브라우저 재시작 후 재접속: 이 서버가 마지막으로 시작한 DOE 작업 상태.
            # (서버 프로세스가 살아있으면 브라우저가 꺼져도 계산은 계속 → 다시 물림)
            if not LAST_DOE_JID or LAST_DOE_JID not in JOBS:
                self._json({"job": None}); return
            job = JOBS[LAST_DOE_JID]
            out = {k: job[k] for k in
                   ("state", "progress", "note", "error", "eta_s", "elapsed_s",
                    "doe_done", "doe_total", "r2_G_mid", "cached", "cache_key", "device",
                    "model", "r2_cv_mean", "r2_insample_mean", "per_model_cv", "naxes", "resumed",
                         "ckpt_n", "ckpt_dir", "ckpt_error", "screening", "bo", "image_ready")
                   if k in job}
            out["job"] = LAST_DOE_JID
            self._json(out)
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
        elif u.path == "/api/boot/status":
            # 부트(로딩) 화면이 본 서버 전환을 감지할 때 사용 — steps 없이 ready 만
            self._json({"ready": True})
        elif u.path == "/api/cra/spec":
            # 제품 CRA 스펙 조회 — data/cra/<제품명>.yaml (파일명=제품명 자동 바인딩)
            prod = (parse_qs(u.query).get("product") or [""])[0].strip()
            path = _cra_spec_path(prod)
            cra_dir = os.path.join(APP_DIR, "data", "cra")
            avail = sorted(fn[:-5] for fn in (os.listdir(cra_dir)
                                              if os.path.isdir(cra_dir) else [])
                           if fn.endswith(".yaml"))
            if not prod or not os.path.isfile(path):
                self._json({"found": False, "product": prod,
                            "path": f"data/cra/{prod or '<제품명>'}.yaml",
                            "available": avail})
                return
            try:
                from ..sim.cra import load_cra_spec
                sp = load_cra_spec(path)
                self._json({"found": True, "product": prod,
                            "path": f"data/cra/{prod}.yaml", "available": avail,
                            "f_number": sp["f_number"],
                            "ml_um_per_deg": sp["ml_um_per_deg"],
                            "cf_um_per_deg": sp["cf_um_per_deg"],
                            "cra_corner_deg": sp["module"][-1][1],
                            "n_rows": len(sp["module"]),
                            "sensor_same": sp["module"] == sp["sensor"]})
            except Exception as e:
                self._json({"found": False, "product": prod, "available": avail,
                            "error": f"파일 파싱 실패: {e}"})
        elif u.path == "/api/image/file":
            q = parse_qs(u.query)
            jid = (q.get("job") or [""])[0]
            kind = (q.get("kind") or ["json"])[0]
            job = JOBS.get(jid)
            keymap = {"json": ("image", "application/json"),
                      "png": ("png", "image/png"),
                      "npz": ("npz", "application/octet-stream")}
            k, ct = keymap.get(kind, keymap["json"])
            if not job or k not in job:
                self.send_response(404); self.end_headers(); return
            self._file(job[k], ct)
        elif u.path == "/api/image/last":
            # 페이지 이동/재시작 후 마지막 이미지 잡 재접속
            if not LAST_IMG_JID or LAST_IMG_JID not in JOBS:
                self._json({"job": None}); return
            job = JOBS[LAST_IMG_JID]
            out = {k: job[k] for k in
                   ("state", "progress", "note", "error", "elapsed_s",
                    "doe_done", "doe_total", "image_ready", "device")
                   if k in job}
            out["job"] = LAST_IMG_JID
            self._json(out)
        elif u.path == "/api/doe/cache":
            # 저장된 surrogate 목록 (DB) — 최신순 메타
            from ..sim import surrogate_cache as sc
            self._json({"items": sc.index(_cache_dir())})
        elif u.path == "/api/doe/cache/config":
            # 현재 DB 위치 + 기본값 + 항목 수 (프론트가 표시/편집)
            from ..sim import surrogate_cache as sc
            d = _cache_dir()
            self._json({"dir": d, "default_dir": DEFAULT_CACHE_DIR,
                        "is_default": os.path.abspath(d) == os.path.abspath(DEFAULT_CACHE_DIR),
                        "env_override": bool(os.environ.get("CIS_SURROGATE_DB")),
                        "count": len(sc.list_keys(d))})
        elif u.path == "/api/doe/cache/file":
            # 캐시 키로 surrogate JSON 직접 로드 (재계산 0회 — 바로 역설계/민감도)
            from ..sim import surrogate_cache as sc
            k = (parse_qs(u.query).get("key") or [""])[0]
            rec = sc.load(_cache_dir(), k) if k else None
            if not rec:
                self.send_response(404); self.end_headers(); return
            self._json(rec["surrogate"])
        else:
            self.send_response(404); self.end_headers()

    def do_POST(self):
        u = urlparse(self.path)
        try:
            n = int(self.headers.get("Content-Length") or 0)
            data = json.loads(self.rfile.read(n) or b"{}")
        except Exception:
            self._json({"error": "bad json"}, 400); return
        if u.path == "/api/attribute":
            # A. 에너지 귀속 — 한 파장 QE 가 왜 그 값인지 (빛이 어디로 갔나). 동기.
            try:
                os.makedirs(JOBS_DIR, exist_ok=True)
                cfg_path = os.path.join(JOBS_DIR, "attr_" + uuid.uuid4().hex[:8] + ".yaml")
                with open(cfg_path, "w", encoding="utf-8") as f:
                    f.write(data.get("yaml") or "")
                from ..sim.simulator import RCWAPlaneWaveSimulator
                from ..sim.attribute import energy_attribution
                ng = max(9, int(data.get("nG", 101)))
                sim = RCWAPlaneWaveSimulator(cfg_path, nG=ng if ng % 2 else ng + 1,
                                             downsample=max(1, int(data.get("downsample", 2))))
                self._json(energy_attribution(sim, float(data.get("wavelength_nm", 525))))
            except Exception as e:
                self._json({"error": f"{type(e).__name__}: {e}",
                            "trace": traceback.format_exc()[-1500:]}, 500)
            return
        if u.path == "/api/sensitivity":
            # B. 파라미터 민감도. surrogate 있으면 즉시(B-1), 없으면 유한차분(B-2).
            try:
                from ..sim import sensitivity as sens
                wl = float(data.get("wavelength_nm", 525))
                ch = str(data.get("channel", "G"))
                sur = data.get("surrogate")
                if sur and sur.get("type") == "emulator":
                    from ..sim.emulator import sensitivity as em_sens
                    self._json(em_sens(sur, wl, ch))
                elif sur:
                    self._json(sens.sensitivity_from_surrogate(sur, wl, ch))
                else:
                    import yaml as _yaml
                    cfg = _yaml.safe_load(data.get("yaml") or "") or {}
                    from ..sim.simulator import RCWAPlaneWaveSimulator
                    RCWAPlaneWaveSimulator._auto_model_defaults(cfg)
                    self._json(sens.local_sensitivity(
                        cfg, wl, ch, nG=max(9, int(data.get("nG", 81))),
                        downsample=max(1, int(data.get("downsample", 3)))))
            except Exception as e:
                self._json({"error": f"{type(e).__name__}: {e}",
                            "trace": traceback.format_exc()[-1500:]}, 500)
            return
        if u.path == "/api/inverse":
            # ②. 역설계 — 목표 QE 곡선 -> 근접 구조 (surrogate 필요). 즉시.
            try:
                sur = data.get("surrogate")
                if not sur:
                    self._json({"error": "surrogate 가 필요합니다 (먼저 DOE 실행)."}, 400)
                    return
                chs = tuple(data.get("channels") or ("R", "G", "B"))
                if sur.get("type") == "emulator":
                    from ..sim.emulator import invert as em_invert
                    self._json(em_invert(sur, data.get("target") or {},
                                         weights=data.get("weights"), channels=chs))
                else:
                    from ..sim.inverse import invert
                    self._json(invert(sur, data.get("target") or {},
                                      weights=data.get("weights"), channels=chs))
            except Exception as e:
                self._json({"error": f"{type(e).__name__}: {e}",
                            "trace": traceback.format_exc()[-1500:]}, 500)
            return
        if u.path == "/api/doe/cache/config":
            # surrogate DB 위치 지정 (공유 폴더/드라이브). 빈 값이면 기본으로 복귀.
            try:
                from ..sim import surrogate_cache as sc
                if os.environ.get("CIS_SURROGATE_DB"):
                    self._json({"error": "환경변수 CIS_SURROGATE_DB 가 설정되어 있어 "
                                         "여기서 바꿀 수 없습니다."}, 400); return
                newd = sc.set_dir(APP_DIR, data.get("dir") or "")
                d = _cache_dir()
                self._json({"ok": True, "dir": d,
                            "is_default": os.path.abspath(d) == os.path.abspath(DEFAULT_CACHE_DIR),
                            "count": len(sc.list_keys(d))})
            except Exception as e:
                self._json({"error": f"{type(e).__name__}: {e}"}, 500)
            return
        if u.path == "/api/doe/cache/sync":
            # 로컬 DB ↔ 공유 폴더 양방향 병합 (서로 없는 항목만 복사)
            try:
                from ..sim import surrogate_cache as sc
                shared = (data.get("shared_dir") or "").strip()
                if not shared:
                    self._json({"error": "공유 폴더 경로(shared_dir)가 필요합니다."}, 400); return
                self._json(sc.sync(_cache_dir(), shared))
            except Exception as e:
                self._json({"error": f"{type(e).__name__}: {e}",
                            "trace": traceback.format_exc()[-800:]}, 500)
            return
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
                     "phi": _num("phi", 0.0),
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
                assert mode in ("axis", "surrogate", "full", "lhs"), "mode"
                ng = max(9, int(data.get("nG", 151)))
                p = {"mode": mode, "nG": ng if ng % 2 else ng + 1,
                     "downsample": max(1, int(data.get("downsample", 2))),
                     "n": max(2, int(data.get("n", 16))),
                     "lam0": float(data.get("lam0", 0.40)),
                     "lam1": float(data.get("lam1", 0.70)),
                     "force": bool(data.get("force", False))}
                if mode == "lhs":
                    # 공간채움 넓은 에뮬레이터: 샘플수·상자·시드·모델선택
                    p["nsamples"] = max(8, int(data.get("nsamples", 80)))
                    p["bound"] = float(data.get("bound", 2.0))
                    p["seed"] = int(data.get("seed", 0))
                    p["model"] = str(data.get("model", "auto"))
                    assert p["model"] in ("auto", "quadratic", "cubic", "rbf"), "model"
                    p["cv_folds"] = max(2, int(data.get("cv_folds", 5)))
                # 사용자 선택 축: [[key,label,step,unit], ...] (연속축만). 없으면 기본 6축.
                ax = data.get("axes")
                if ax:
                    axn = []
                    for a in ax:
                        if isinstance(a, (list, tuple)) and len(a) >= 4:
                            axn.append([str(a[0]), str(a[1]), float(a[2]), str(a[3])])
                    assert axn, "axes"
                    p["axes"] = axn
            except (TypeError, ValueError, AssertionError) as e:
                self._json({"error": f"파라미터 오류: {e}"}, 400); return
            jid = start_doe_job(yaml_text, p)
            from ..sim.doe import doe_points
            total = len(doe_points(p["mode"], naxes=len(p.get("axes") or [0] * 6),
                                   nsamples=p.get("nsamples")))
            self._json({"job": jid, "total": total})
        elif u.path == "/api/doe/axes":
            # 이 구조에서 흔들 수 있는 축 카탈로그 + 중심값·추천 step
            try:
                import yaml as _yaml
                from ..sim.doe import axis_catalog
                cfg = _yaml.safe_load(data.get("yaml") or "") or {}
                from ..sim.simulator import RCWAPlaneWaveSimulator
                RCWAPlaneWaveSimulator._auto_model_defaults(cfg)
                self._json({"axes": axis_catalog(cfg)})
            except Exception as e:
                self._json({"error": f"{type(e).__name__}: {e}"}, 500)
        elif u.path == "/api/doe/estimate":
            # 용량 계량기: 이 설정으로 돌리면 예상 파일크기·시간 + 현재 DB 총용량
            try:
                self._json(_doe_estimate(data))
            except Exception as e:
                self._json({"error": f"{type(e).__name__}: {e}"}, 500)
        elif u.path == "/api/image":
            # 🌈 사광·이미지 시뮬 시작 — 필드 격자 × 원뿔 적분 × 파장
            yaml_text = data.get("yaml") or ""
            if not yaml_text.strip():
                self._json({"error": "yaml 이 비었습니다"}, 400); return
            try:
                ng = max(9, int(data.get("nG", 101)))
                ip = {"field_step": float(data.get("field_step", 0.2)),
                      "region": str(data.get("region", "octant")),
                      "lam0": float(data.get("lam0", 0.45)),
                      "lam1": float(data.get("lam1", 0.65)),
                      "n": max(1, min(8, int(data.get("n", 3)))),
                      "nG": ng if ng % 2 else ng + 1,
                      "downsample": max(1, int(data.get("downsample", 2))),
                      "theta_samples": max(1, min(7, int(data.get("theta_samples", 3)))),
                      "phi_samples": max(1, min(7, int(data.get("phi_samples", 3)))),
                      "lateral_n": max(64, int(data.get("lateral_n", 256))),
                      "diff_threshold_pct":
                          max(1.0, float(data.get("diff_threshold_pct", 30.0)))}
                assert ip["field_step"] in (0.05, 0.1, 0.2), "field_step"
                assert ip["region"] in ("octant", "full"), "region"
            except (TypeError, ValueError, AssertionError) as e:
                self._json({"error": f"파라미터 오류: {e}"}, 400); return
            global LAST_IMG_JID
            os.makedirs(JOBS_DIR, exist_ok=True)
            jid = "img" + uuid.uuid4().hex[:9]
            cfg_path = os.path.join(JOBS_DIR, jid + ".yaml")
            with open(cfg_path, "w", encoding="utf-8") as f:
                f.write(yaml_text)
            JOBS[jid] = {"state": "running", "progress": 0.0,
                         "note": "구조/스펙 준비중…", "error": None, "cancel": False,
                         "params": ip}
            LAST_IMG_JID = jid
            threading.Thread(target=_run_image_job, args=(jid, cfg_path, ip),
                             daemon=True).start()
            self._json({"job": jid})
        elif u.path == "/api/bo":
            # 🎯 최적 조건 찾기(BO) 시작 — 저장된 모델 키 + 유저 언어 목표
            try:
                bp = {"key": str(data.get("key") or ""),
                      "ch": str(data.get("ch", "G")),
                      "wl": int(data.get("wl", 550)),
                      "goal": str(data.get("goal", "max")),
                      "n": max(3, min(30, int(data.get("n", 10))))}
                assert bp["key"], "key"
                assert bp["ch"] in ("R", "G", "B"), "ch"
                assert bp["goal"] in ("max", "min"), "goal"
            except (TypeError, ValueError, AssertionError) as e:
                self._json({"error": f"파라미터 오류: {e}"}, 400); return
            jid = "bo" + uuid.uuid4().hex[:9]
            JOBS[jid] = {"state": "running", "progress": 0.0, "note": "다음 실험 고르는 중…",
                         "error": None, "cancel": False, "params": bp}
            threading.Thread(target=_run_bo_job, args=(jid, bp), daemon=True).start()
            self._json({"job": jid, "total": bp["n"]})
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
    # device 진단: cpu 로 떴는데 GPU 를 기대했다면 아래 세 줄로 원인 파악
    #  (다른 파이썬? CPU 전용 torch? CUDA_VISIBLE_DEVICES 로 GPU 숨김?)
    _cudab = getattr(torch.version, "cuda", None)
    print(f"           python={sys.executable}")
    print(f"           torch={torch.__version__} cuda_build={_cudab} "
          f"is_available={torch.cuda.is_available()} "
          f"device_count={torch.cuda.device_count() if torch.cuda.is_available() else 0}"
          + (f" CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']!r}"
             if 'CUDA_VISIBLE_DEVICES' in os.environ else ""))
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
