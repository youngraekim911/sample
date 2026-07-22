# -*- coding: utf-8 -*-
"""DOE(실험계획) 스윕 + 2차 다항 surrogate 추출.

구조 설계 노브 4축을 현 셋업 중심 5레벨(-2..+2)로 흔들어 QE 스펙트럼을 수집하고,
2차 반응표면(quadratic response surface) surrogate 를 최소자승 피팅한다.

축 (스텝 크기는 axes 설정으로 조절):
    cf_dA     : CF R/G/B 두께 동시 증감 (Å 단위 스텝, 기본 300Å -> ±600Å)
    planar_um : ML 평탄층 두께 증감 (µm 스텝, 기본 0.025 -> ±0.05µm)
    ml_h_um   : ML 돔 두께(height) 증감 (µm 스텝, 기본 0.025 -> ±0.05µm)
    ml_scale  : ML radius 배율 증감 (스텝, 기본 0.025 -> ±0.05)

모드 (실행 횟수):
    axis      : 중심 1 + 축별 ±1,±2 (4×4)            = 17  (주효과 스크리닝)
    surrogate : axis 17 + 2인자 (±1,±1) 조합 C(4,2)×4 = 41  (2차 모델 15계수 피팅, 권장)
    full      : 5^4 전조합                             = 625 (GPU 권장)

surrogate: 응답 y(파장, 채널) 별로
    y ≈ c0 + Σ ci·xi + Σ cii·xi² + Σ cij·xi·xj   (x = 스텝지수 -2..+2 정규화 /2)
계수 15개 최소자승 -> JSON. 예측은 predict() 또는 JSON 계수로 즉석 계산.
"""
import itertools
import json
import time

import numpy as np

AXES_DEFAULT = [
    # (키, 라벨, 스텝 크기, 단위)  — 스텝지수 s∈{-2,-1,0,1,2}, 변화량 = s*step
    ("cf_dA",     "CF RGB 두께",   300.0,  "A"),
    ("planar_um", "ML 평탄층",     0.025,  "um"),
    ("ml_h_um",   "ML 두께",       0.025,  "um"),
    ("ml_scale",  "ML radius 배율", 0.025, "x"),
]


def doe_points(mode):
    """모드 -> 스텝지수 튜플 리스트 [(s0,s1,s2,s3), ...]  (si ∈ -2..2)."""
    if mode == "full":
        return [p for p in itertools.product(range(-2, 3), repeat=4)]
    pts = [(0, 0, 0, 0)]
    for ax in range(4):                                  # 축별 ±1, ±2
        for s in (-2, -1, 1, 2):
            p = [0] * 4
            p[ax] = s
            pts.append(tuple(p))
    if mode == "axis":
        return pts
    if mode == "surrogate":                              # + 2인자 상호작용 (±1,±1)
        for a, b in itertools.combinations(range(4), 2):
            for sa, sb in itertools.product((-1, 1), (-1, 1)):
                p = [0] * 4
                p[a], p[b] = sa, sb
                pts.append(tuple(p))
        return pts
    raise ValueError(f"unknown mode {mode}")


def apply_point(cfg, steps, axes=AXES_DEFAULT):
    """cfg(dict, 원본 훼손 없음) 에 스텝지수 적용 -> 새 cfg."""
    import copy
    c = copy.deepcopy(cfg)
    st = c["stack"]
    for (key, _lb, step, _u), s in zip(axes, steps):
        if s == 0:
            continue
        d = s * step
        if key == "cf_dA":
            for col in ("R", "G", "B"):
                t = st["cf"][col]
                t["thickness_um"] = max(0.05, round(t["thickness_um"] + d * 1e-4, 5))
        elif key == "planar_um":
            st["ml"]["planar_um"] = max(0.0, round(st["ml"]["planar_um"] + d, 5))
        elif key == "ml_h_um":
            st["ml"]["height_um"] = max(0.05, round(st["ml"]["height_um"] + d, 5))
        elif key == "ml_scale":
            ml = st["ml"]
            if ml.get("quads"):
                for row in ml["quads"]:
                    for q in row:
                        q["scale"] = round(float(q.get("scale", 1.0)) + d, 5)
            if ml.get("lenses"):
                for L in ml["lenses"]:
                    if "scale" in L:
                        L["scale"] = round(float(L["scale"]) + d, 5)
    return c


def run_doe(cfg, wavelengths_nm, mode="surrogate", nG=151, downsample=2,
            lateral_n=256, materials_dir=None, axes=AXES_DEFAULT,
            progress=None, cancel=None):
    """DOE 실행. progress(done,total,eta_s,point) 콜백, cancel() -> bool 중단.

    반환: {"axes":[...], "mode", "points": [{"steps":[...], "qe": {nm: {R,G,B}}}, ...],
           "wavelengths_nm": [...], "elapsed_s": float}
    """
    from ..structure.blocks import ir_from_wizard_cfg
    from .simulator import RCWAPlaneWaveSimulator

    pts = doe_points(mode)
    total = len(pts)
    out = {"axes": [list(a) for a in axes], "mode": mode, "nG": nG,
           "wavelengths_nm": list(wavelengths_nm), "points": []}
    t0 = time.time()
    for i, p in enumerate(pts):
        if cancel and cancel():
            out["cancelled"] = True
            break
        c = apply_point(cfg, p, axes)
        ir = ir_from_wizard_cfg(c, lateral_n)
        sim = RCWAPlaneWaveSimulator(ir, nG=nG, downsample=downsample,
                                     materials_dir=materials_dir)
        qe = {}
        for w in wavelengths_nm:
            acc = {"R": 0.0, "G": 0.0, "B": 0.0}
            for pol in ((1.0, 0.0), (0.0, 1.0)):
                o = sim.run(w / 1000.0, pol_te=pol[0], pol_tm=pol[1])
                for L in "RGB":
                    acc[L] += 0.5 * o["QE_rgb"][L]
            qe[int(w)] = {L: round(acc[L], 5) for L in "RGB"}
        out["points"].append({"steps": list(p), "qe": qe})
        if progress:
            done = i + 1
            el = time.time() - t0
            eta = el / done * (total - done)
            progress(done, total, eta, p)
    out["elapsed_s"] = round(time.time() - t0, 1)
    return out


# ------------------------------------------------------------------ surrogate
def _feat(x):
    """x(4,) 정규화 스텝(-1..1) -> 2차 특징 15개."""
    f = [1.0] + list(x) + [v * v for v in x]
    for a, b in itertools.combinations(range(4), 2):
        f.append(x[a] * x[b])
    return f


def fit_surrogate(doe_result):
    """DOE 결과 -> 파장×채널별 2차 계수 (최소자승). 반환 dict (JSON 직렬화 가능).

    특징 순서: [1, x1..x4, x1²..x4², x12,x13,x14,x23,x24,x34]  (x = 스텝지수/2)
    """
    pts = [p for p in doe_result["points"]]
    X = np.array([_feat([s / 2.0 for s in p["steps"]]) for p in pts])
    ws = doe_result["wavelengths_nm"]
    coefs, r2s = {}, {}
    for L in "RGB":
        coefs[L], r2s[L] = {}, {}
        for w in ws:
            y = np.array([p["qe"][int(w)][L] for p in pts])
            c, *_ = np.linalg.lstsq(X, y, rcond=None)
            yh = X @ c
            ss = float(((y - y.mean()) ** 2).sum())
            r2 = 1.0 - float(((y - yh) ** 2).sum()) / ss if ss > 1e-12 else 1.0
            coefs[L][int(w)] = [round(float(v), 6) for v in c]
            r2s[L][int(w)] = round(r2, 4)
    return {"type": "quadratic_rsm", "axes": doe_result["axes"],
            "feature_order": "1,x1..x4,x1^2..x4^2,x12,x13,x14,x23,x24,x34 (x=step/2)",
            "wavelengths_nm": list(ws), "coef": coefs, "r2": r2s}


def predict(surrogate, steps, wavelength_nm, channel):
    """surrogate JSON + 스텝지수(-2..2) -> QE 예측."""
    x = [s / 2.0 for s in steps]
    c = surrogate["coef"][channel][int(wavelength_nm)]
    return float(np.dot(_feat(x), c))


def doe_csv(doe_result):
    """DOE 결과 -> CSV 문자열 (한 줄 = 한 조건×파장)."""
    axes = doe_result["axes"]
    hdr = [a[0] for a in axes] + ["nm", "R", "G", "B"]
    lines = [",".join(hdr)]
    for p in doe_result["points"]:
        for w in doe_result["wavelengths_nm"]:
            q = p["qe"][int(w)]
            lines.append(",".join([str(s) for s in p["steps"]] +
                                  [str(int(w)), str(q["R"]), str(q["G"]), str(q["B"])]))
    return "\n".join(lines) + "\n"
