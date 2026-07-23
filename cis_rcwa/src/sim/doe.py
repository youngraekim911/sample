# -*- coding: utf-8 -*-
"""DOE(실험계획) 스윕 + 2차 다항 surrogate 추출.

구조 설계 노브 k축을 현 셋업 중심 5레벨(-2..+2)로 흔들어 QE 스펙트럼을 수집하고,
2차 반응표면(quadratic response surface) surrogate 를 최소자승 피팅한다.
(축 개수 k 는 axes 정의에서 자동 — 4축이든 6축이든 일반화되어 동작)

기본 축 (스텝 크기는 axes 설정으로 조절):
    cf_R_dA   : CF Red   두께 증감 (Å 단위 스텝, 기본 300Å -> ±600Å)
    cf_G_dA   : CF Green 두께 증감 (Å 단위 스텝, 기본 300Å -> ±600Å)
    cf_B_dA   : CF Blue  두께 증감 (Å 단위 스텝, 기본 300Å -> ±600Å)
    planar_um : ML 평탄층 두께 증감 (µm 스텝, 기본 0.025 -> ±0.05µm)
    ml_h_um   : ML 돔 두께(height) 증감 (µm 스텝, 기본 0.025 -> ±0.05µm)
    ml_scale  : ML radius 배율 증감 (스텝, 기본 0.025 -> ±0.05)

모드 (실행 횟수, k=축 개수):
    axis      : 중심 1 + 축별 ±1,±2 (4k)                = 1+4k   (주효과 스크리닝)
    surrogate : axis + 2인자 (±1,±1) 조합 C(k,2)×4       = 권장 (2차 모델 피팅)
    full      : 5^k 전조합                               = (GPU 권장, k 크면 폭증 주의)
    lhs       : 공간채움(Latin Hypercube) n샘플, 상자 [-bound..bound] 연속       ★넓은 에뮬레이터
      · k=4: axis 17 / surrogate 41 / full 625
      · k=6: axis 25 / surrogate 85 / full 15625
      · lhs: nsamples 지정(권장 8·k~15·k) — 중심 별모양과 달리 상자 전체를 고르게 덮음.
        국소 2차가 아닌 '넓은 공간 에뮬레이터'(RBF/고차)를 학습하려면 lhs 로 수집.

surrogate: 응답 y(파장, 채널) 별로
    y ≈ c0 + Σ ci·xi + Σ cii·xi² + Σ cij·xi·xj   (x = 스텝지수 -2..+2 정규화 /2)
계수 (1+2k+C(k,2))개 최소자승 -> JSON. 예측은 predict() 또는 JSON 계수로 즉석 계산.
"""
import itertools
import json
import time

import numpy as np

AXES_DEFAULT = [
    # (키, 라벨, 스텝 크기, 단위)  — 스텝지수 s∈{-2,-1,0,1,2}, 변화량 = s*step
    ("cf_R_dA",   "CF Red 두께",    300.0,  "A"),
    ("cf_G_dA",   "CF Green 두께",  300.0,  "A"),
    ("cf_B_dA",   "CF Blue 두께",   300.0,  "A"),
    ("planar_um", "ML 평탄층",      0.025,  "um"),
    ("ml_h_um",   "ML 두께",        0.025,  "um"),
    ("ml_scale",  "ML radius 배율",  0.025, "x"),
]

# CF 색상축 키 -> 해당 CF 채널 (per-color CF 두께 스윕)
_CF_AXIS_COL = {"cf_R_dA": "R", "cf_G_dA": "G", "cf_B_dA": "B"}


def lhs_design(k, n, bound=2.0, seed=0):
    """Latin Hypercube 공간채움 설계 — 상자 [-bound,bound]^k 를 n점으로 고르게 덮음.

    각 축을 n개 균등층으로 나눠 층마다 정확히 1점(중복 없는 사영) + 층 내 무작위 지터,
    축별 독립 셔플. 같은 seed 면 결정적(재현 가능). 반환: [(x0,..,x_{k-1}), ...] (실수 스텝).
    별모양(axis) 설계와 달리 상자 내부·모서리까지 데이터가 퍼져 넓은 공간 학습에 적합.
    """
    n = int(n)
    rng = np.random.default_rng(int(seed))
    cuts = np.arange(n) / n                                   # 층 하단
    pts = np.empty((n, k))
    for d in range(k):
        jit = rng.uniform(0.0, 1.0 / n, size=n)              # 층 내 지터
        col = cuts + jit
        rng.shuffle(col)                                     # 축별 독립 셔플
        pts[:, d] = col
    pts = (pts * 2.0 - 1.0) * float(bound)                   # [0,1]->[-bound,bound]
    return [tuple(round(float(v), 6) for v in row) for row in pts]


def doe_points(mode, naxes=None, nsamples=None, bound=2.0, seed=0):
    """모드 -> 스텝지수 튜플 리스트 [(s0,..,s_{k-1}), ...].

    axis/surrogate/full: si ∈ -2..2 정수. lhs: 실수 스텝(공간채움, nsamples 필요).
    naxes(k) 미지정 시 기본 축 개수 사용. 축 개수와 반드시 일치해야 함.
    """
    k = int(naxes) if naxes else len(AXES_DEFAULT)
    if mode == "lhs":
        if not nsamples:
            raise ValueError("lhs 모드는 nsamples(샘플 수)가 필요합니다.")
        return lhs_design(k, nsamples, bound=bound, seed=seed)
    if mode == "full":
        return [p for p in itertools.product(range(-2, 3), repeat=k)]
    pts = [tuple([0] * k)]
    for ax in range(k):                                  # 축별 ±1, ±2
        for s in (-2, -1, 1, 2):
            p = [0] * k
            p[ax] = s
            pts.append(tuple(p))
    if mode == "axis":
        return pts
    if mode == "surrogate":                              # + 2인자 상호작용 (±1,±1)
        for a, b in itertools.combinations(range(k), 2):
            for sa, sb in itertools.product((-1, 1), (-1, 1)):
                p = [0] * k
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
        if key == "cf_dA":                               # (구버전) CF 3색 동시
            for col in ("R", "G", "B"):
                t = st["cf"][col]
                t["thickness_um"] = max(0.05, round(t["thickness_um"] + d * 1e-4, 5))
        elif key in _CF_AXIS_COL:                         # CF 색상별 독립 (R/G/B 각각)
            col = _CF_AXIS_COL[key]
            t = st["cf"][col]
            t["thickness_um"] = max(0.05, round(t["thickness_um"] + d * 1e-4, 5))
        elif key == "planar_um":
            st["ml"]["planar_um"] = max(0.0, round(st["ml"]["planar_um"] + d, 5))
        elif key == "ml_h_um":
            # 전역 + quad별/렌즈별 개별 돔두께 모두 증감 (개별값은 전역보다 우선이라
            # 전역만 바꾸면 개별 설정 quad 는 스윕에서 빠짐 -> 함께 이동)
            ml = st["ml"]
            ml["height_um"] = max(0.05, round(float(ml.get("height_um", 0)) + d, 5))
            for row in (ml.get("quads") or []):
                for q in row:
                    if float(q.get("height_um", 0) or 0) > 0:
                        q["height_um"] = max(0.05, round(q["height_um"] + d, 5))
            for L in (ml.get("lenses") or []):
                hk = "h" if "h" in L else ("height_um" if "height_um" in L else None)
                if hk and float(L[hk] or 0) > 0:
                    L[hk] = max(0.05, round(float(L[hk]) + d, 5))
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
            nsamples=None, bound=2.0, seed=0, progress=None, cancel=None):
    """DOE 실행. progress(done,total,eta_s,point) 콜백, cancel() -> bool 중단.

    lhs 모드는 nsamples/bound/seed 로 공간채움 샘플 수·상자·시드 지정.
    반환: {"axes":[...], "mode", "points": [{"steps":[...], "qe": {nm: {R,G,B}}}, ...],
           "wavelengths_nm": [...], "elapsed_s": float}
    """
    from ..structure.blocks import ir_from_wizard_cfg
    from .simulator import RCWAPlaneWaveSimulator

    pts = doe_points(mode, len(axes), nsamples=nsamples, bound=bound, seed=seed)
    total = len(pts)
    out = {"axes": [list(a) for a in axes], "mode": mode, "nG": nG,
           "bound": float(bound), "seed": int(seed),
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
            # 파장/편광 단위로도 중단 반응 (조건 하나가 수 분일 수 있음 —
            # 미완 조건은 버리고 즉시 종료, 완료된 조건까지만 결과에 남김)
            if cancel and cancel():
                out["cancelled"] = True
                break
            acc = {"R": 0.0, "G": 0.0, "B": 0.0}
            for pol in ((1.0, 0.0), (0.0, 1.0)):
                if cancel and cancel():
                    out["cancelled"] = True
                    break
                o = sim.run(w / 1000.0, pol_te=pol[0], pol_tm=pol[1])
                for L in "RGB":
                    acc[L] += 0.5 * o["QE_rgb"][L]
            if out.get("cancelled"):
                break
            qe[int(w)] = {L: round(acc[L], 5) for L in "RGB"}
        if out.get("cancelled"):
            break
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
    """x(k,) 정규화 스텝(-1..1) -> 2차 특징 (1 + 2k + C(k,2))개.

    순서: [1, x1..xk, x1²..xk², x_i·x_j (i<j 사전순)].
    """
    x = list(x)
    f = [1.0] + x + [v * v for v in x]
    for a, b in itertools.combinations(range(len(x)), 2):
        f.append(x[a] * x[b])
    return f


def fit_surrogate(doe_result):
    """DOE 결과 -> 파장×채널별 2차 계수 (최소자승). 반환 dict (JSON 직렬화 가능).

    특징 순서(k=축 개수): [1, x1..xk, x1²..xk², x_i·x_j (i<j)]  (x = 스텝지수/2)
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
    k = len(doe_result["axes"])
    return {"type": "quadratic_rsm", "axes": doe_result["axes"], "naxes": k,
            "feature_order": f"1, x1..x{k}, x1^2..x{k}^2, x_i*x_j (i<j) (x=step/2)",
            "wavelengths_nm": list(ws), "coef": coefs, "r2": r2s}


def coef_at(surrogate, channel, wavelength_nm):
    """surrogate coef 접근 — JSON 왕복 후 파장 키가 str 이어도 안전하게 조회."""
    cmap = surrogate["coef"][channel]
    w = int(wavelength_nm)
    if w in cmap:
        return cmap[w]
    if str(w) in cmap:
        return cmap[str(w)]
    raise KeyError(f"surrogate 에 파장 {w}nm 계수가 없습니다 (있는 값: "
                   f"{list(cmap.keys())})")


def r2_at(surrogate, channel, wavelength_nm):
    """surrogate r2 접근 — 키 str/int 모두 허용, 없으면 None."""
    rmap = surrogate.get("r2", {}).get(channel, {})
    w = int(wavelength_nm)
    return rmap.get(w, rmap.get(str(w)))


def predict(surrogate, steps, wavelength_nm, channel):
    """surrogate JSON + 스텝지수(-2..2) -> QE 예측."""
    x = [s / 2.0 for s in steps]
    return float(np.dot(_feat(x), coef_at(surrogate, channel, wavelength_nm)))


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
