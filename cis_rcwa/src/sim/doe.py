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


def _apply_axis(st, key, d):
    """스택(st)에 축 key 의 물리 변화량 d 를 in-place 적용. 알 수 없는 키는 무시(방어)."""
    if key == "cf_dA":                                   # (구버전) CF 3색 두께 동시(Å)
        for col in ("R", "G", "B"):
            t = st["cf"][col]
            t["thickness_um"] = max(0.05, round(t["thickness_um"] + d * 1e-4, 5))
    elif key in _CF_AXIS_COL:                            # CF 색별 두께(Å)
        t = st["cf"][_CF_AXIS_COL[key]]
        t["thickness_um"] = max(0.05, round(t["thickness_um"] + d * 1e-4, 5))
    elif key == "si_um":                                 # Si 광다이오드 두께(µm)
        si = st.setdefault("si", {})
        si["thickness_um"] = max(0.2, round(float(si.get("thickness_um", 3.0)) + d, 5))
    elif key[:3] == "cf_" and key.endswith("_curv_um"):  # CF 색별 곡률(µm)
        t = st["cf"][key[3]]
        t["curvature_um"] = round(float(t.get("curvature_um", 0) or 0) + d, 5)
    elif key[:3] == "cf_" and key.endswith("_cham_um"):  # CF 색별 상부모서리 챔퍼(µm)
        t = st["cf"][key[3]]
        t["chamfer_um"] = max(0.0, round(float(t.get("chamfer_um", 0) or 0) + d, 5))
    elif key[:3] == "cf_" and key.endswith("_scale"):    # CF 색별 가로세로 사이즈(배율)
        t = st["cf"][key[3]]
        for kk in ("scale_x", "scale_y"):
            t[kk] = max(0.1, round(float(t.get(kk, 1.0) or 1.0) + d, 5))
    elif key == "grid_w_um":                             # Grid 격벽 폭(µm)
        g = st["grid"]
        g["width_um"] = max(0.02, round(float(g["width_um"]) + d, 5))
    elif key == "grid_coat_um":                          # Grid 측벽 코팅(µm)
        g = st["grid"]
        g["coat_um"] = max(0.0, round(float(g.get("coat_um", 0) or 0) + d, 5))
    elif key == "grid_dz_um":                            # Grid 교차점 deadzone(µm)
        g = st["grid"]
        g["deadzone_um"] = max(0.0, min(0.25, round(float(g.get("deadzone_um", 0) or 0) + d, 5)))
    elif key[:7] == "gridstk" and key.endswith("_um"):   # Grid 스택 층별 두께(µm)
        L = st["grid"]["stack"][int(key[7:-3])]
        L["height_um"] = max(0.02, round(float(L["height_um"]) + d, 5))
    elif key[:4] == "barl" and key.endswith("_um"):      # BARL/ARL 하부 층별 두께(µm)
        L = st["barl"][int(key[4:-3])]
        L["thickness_um"] = max(0.005, round(float(L["thickness_um"]) + d, 5))
    elif key == "arltop_um":                             # ARL top 두께(µm)
        a = st.setdefault("arl_top", {})
        a["thickness_um"] = max(0.0, round(float(a.get("thickness_um", 0) or 0) + d, 5))
    elif key == "dti_w_um":                              # DTI 폭(µm)
        dd = st.setdefault("dti", {})
        dd["width_um"] = max(0.01, round(float(dd.get("width_um", 0.09) or 0.09) + d, 5))
    elif key == "planar_um":                             # ML 평탄층(µm)
        st["ml"]["planar_um"] = max(0.0, round(st["ml"].get("planar_um", 0) + d, 5))
    elif key == "ml_h_um":                               # ML 돔 두께(전역+quad+lens)(µm)
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
    elif key == "ml_scale":                              # ML radius 배율
        ml = st["ml"]
        for row in (ml.get("quads") or []):
            for q in row:
                q["scale"] = round(float(q.get("scale", 1.0)) + d, 5)
        for L in (ml.get("lenses") or []):
            if "scale" in L:
                L["scale"] = round(float(L["scale"]) + d, 5)
    elif key in ("ml_shift_x_um", "ml_shift_y_um"):      # ML 전체 위치 이동(µm)
        ml = st["ml"]
        kk = "shift_x_um" if key.endswith("x_um") else "shift_y_um"
        ml[kk] = round(float(ml.get(kk, 0) or 0) + d, 5)
    elif key[:4] == "ml_q" and key.endswith("_h_um"):    # ML quad별 돔 두께(µm)
        a, b = int(key[4]), int(key[5])
        ml = st["ml"]
        q = ml["quads"][a][b]
        base = float(q.get("height_um", 0) or 0) or float(ml.get("height_um", 0) or 0.4)
        q["height_um"] = max(0.05, round(base + d, 5))
    elif key[:4] == "ml_q" and key[-6:] in ("_dx_um", "_dy_um"):   # ML quad별 위치(µm)
        a, b = int(key[4]), int(key[5])
        q = st["ml"]["quads"][a][b]
        kk = "dx_um" if key.endswith("_dx_um") else "dy_um"
        q[kk] = round(float(q.get(kk, 0) or 0) + d, 5)
    elif key[:3] == "cf_" and key.endswith("_chang_deg"):  # CF 색별 챔퍼 각도(°)
        t = st["cf"][key[3]]
        t["chamfer_angle"] = max(5.0, min(85.0,
            round(float(t.get("chamfer_angle", 45) or 45) + d, 3)))
    elif key == "dti_open_um":                           # DTI center open 크기(µm, x·y 동시)
        dd = st.setdefault("dti", {})
        for kk in ("center_gap_x_um", "center_gap_y_um"):
            dd[kk] = max(0.02, round(float(dd.get(kk, 0.28) or 0.28) + d, 5))


def apply_point(cfg, steps, axes=AXES_DEFAULT):
    """cfg(dict, 원본 훼손 없음) 에 스텝지수 적용 -> 새 cfg. delta = 스텝지수 s × step."""
    import copy
    c = copy.deepcopy(cfg)
    st = c["stack"]
    for (key, _lb, step, _u), s in zip(axes, steps):
        if s == 0:
            continue
        _apply_axis(st, key, s * step)
    return c


# ------------------------------------------------------------ 축 카탈로그
# 축별 추천 step = max(floor, frac × |중심값|)  — 구조 중심값에서 자동 스케일.
# 기본 상자 ±2 = 정상 설계범위, ±3 = outlier 여유. (spec: key, 라벨, 단위, frac, floor)
def _step_reco(center, frac, floor):
    return round(max(floor, frac * abs(float(center or 0))), 6)


def axis_catalog(cfg):
    """이 구조에서 흔들 수 있는 모든 축 + 중심값·추천 step 목록.

    반환: [{key,label,unit,center,step,group,kind("continuous"/"discrete"),present,note}, ...]
    step 은 중심값 기반 자동 추천(사용자가 UI 에서 수정 가능). ±2step=정상, ±3step=outlier.
    group 은 구조 섹션(Si·DTI / BARL·ARL / Grid / CF / ML) — UI 시각 분류용.
    """
    st = (cfg.get("stack") or {})
    out = []

    def add(group, key, label, unit, center, frac, floor, note=""):
        out.append({"key": key, "label": label, "unit": unit, "group": group,
                    "center": round(float(center or 0), 5),
                    "step": _step_reco(center, frac, floor),
                    "kind": "continuous", "present": True, "note": note})

    si = st.get("si") or {}
    add("Si·DTI", "si_um", "Si 두께", "um", si.get("thickness_um", 3.0), 0.08, 0.05)
    dti = st.get("dti") or {}
    dti_on = dti and (dti.get("mode") or "").lower() not in ("", "none")
    if dti_on:
        add("Si·DTI", "dti_w_um", "DTI 폭", "um", dti.get("width_um", 0.09), 0.1, 0.005)
        if "open" in str(dti.get("mode", "")) or dti.get("center_open"):
            add("Si·DTI", "dti_open_um", "DTI center open", "um",
                dti.get("center_gap_x_um", dti.get("center_gap_um", 0.28)), 0.1, 0.02)
    for i, L in enumerate(st.get("barl") or []):
        add("BARL·ARL", f"barl{i}_um", f"BARL[{i}] {L.get('material','')} 두께", "um",
            L.get("thickness_um", 0.05), 0.1, 0.005)
    arl = st.get("arl_top") or {}
    if arl:
        add("BARL·ARL", "arltop_um", "ARL top 두께", "um",
            arl.get("thickness_um", 0), 0.1, 0.005)
    g = st.get("grid") or {}
    add("Grid", "grid_w_um", "Grid 폭", "um", g.get("width_um", 0.15), 0.1, 0.01)
    if g.get("coat_um"):
        add("Grid", "grid_coat_um", "Grid 측벽코팅", "um", g.get("coat_um", 0), 0.15, 0.005)
    if float(g.get("deadzone_um", g.get("dz_um", 0)) or 0) > 0:
        add("Grid", "grid_dz_um", "Grid 교차점 deadzone", "um",
            g.get("deadzone_um", g.get("dz_um", 0)), 0.2, 0.02)
    for i, L in enumerate(g.get("stack") or []):
        add("Grid", f"gridstk{i}_um", f"Grid스택[{i}] {L.get('material','')} 두께", "um",
            L.get("height_um", 0.1), 0.1, 0.01)
    cf = st.get("cf") or {}
    for cclr in "RGB":
        f = cf.get(cclr) or {}
        add("CF", f"cf_{cclr}_dA", f"CF {cclr} 두께", "A",
            float(f.get("thickness_um", 0.6)) * 1e4, 0.08, 150.0)
        add("CF", f"cf_{cclr}_curv_um", f"CF {cclr} 곡률", "um",
            f.get("curvature_um", 0), 0.15, 0.02)
        add("CF", f"cf_{cclr}_cham_um", f"CF {cclr} 챔퍼 reach", "um",
            f.get("chamfer_um", 0), 0.15, 0.02)
        add("CF", f"cf_{cclr}_chang_deg", f"CF {cclr} 챔퍼 각도", "deg",
            f.get("chamfer_angle", 45), 0.0, 5.0)
        add("CF", f"cf_{cclr}_scale", f"CF {cclr} 사이즈(가로세로)", "x",
            f.get("scale_x", 1.0), 0.05, 0.02)
    ml = st.get("ml") or {}
    _pp = float((cfg.get("grid") or {}).get("pixel_pitch_um", 1.0) or 1.0)
    add("ML", "planar_um", "ML 평탄층", "um", ml.get("planar_um", 0.1), 0.1, 0.01)
    add("ML", "ml_h_um", "ML 두께(전역)", "um", ml.get("height_um", 0.5), 0.08, 0.02)
    add("ML", "ml_scale", "ML radius 배율", "x", 1.0, 0.0, 0.025)
    add("ML", "ml_shift_x_um", "ML 전체 위치 X", "um", ml.get("shift_x_um", 0),
        0.0, round(_pp * 0.04, 4))
    add("ML", "ml_shift_y_um", "ML 전체 위치 Y", "um", ml.get("shift_y_um", 0),
        0.0, round(_pp * 0.04, 4))
    # ML quad(2×2)별 개별 두께·위치 — quads 모드일 때만
    quads = ml.get("quads")
    if isinstance(quads, list) and len(quads) == 2 and not ml.get("lenses"):
        gh = float(ml.get("height_um", 0) or 0.4)
        for a in range(2):
            for b in range(2):
                q = (quads[a] or [{}, {}])[b] or {}
                qh = float(q.get("height_um", 0) or 0) or gh
                add("ML", f"ml_q{a}{b}_h_um", f"ML({a},{b}) 두께", "um", qh, 0.08, 0.02)
                add("ML", f"ml_q{a}{b}_dx_um", f"ML({a},{b}) 위치 X", "um",
                    q.get("dx_um", 0), 0.0, round(_pp * 0.04, 4))
                add("ML", f"ml_q{a}{b}_dy_um", f"ML({a},{b}) 위치 Y", "um",
                    q.get("dy_um", 0), 0.0, round(_pp * 0.04, 4))

    # 이산(카테고리) — LHS 로 못 흔듦, 후보별 별도 실행·비교
    if dti:
        out.append({"key": "dti_liner", "label": "DTI liner(물질)", "unit": "cat",
                    "group": "Si·DTI", "kind": "discrete", "present": True,
                    "note": "이산 — 후보 물질별로 따로 돌려 비교(LHS 제외)"})
        out.append({"key": "dti_center_open", "label": "DTI center open(켬/끔)",
                    "unit": "cat", "group": "Si·DTI", "kind": "discrete", "present": True,
                    "note": "이산 — 켬/끔 각각 돌려 비교(LHS 제외)"})
    return out


def run_doe(cfg, wavelengths_nm, mode="surrogate", nG=151, downsample=2,
            lateral_n=256, materials_dir=None, axes=AXES_DEFAULT,
            nsamples=None, bound=2.0, seed=0, progress=None, cancel=None,
            resume_points=None, on_point=None):
    """DOE 실행. progress(done,total,eta_s,point) 콜백, cancel() -> bool 중단.

    lhs 모드는 nsamples/bound/seed 로 공간채움 샘플 수·상자·시드 지정.
    resume_points: 이전 체크포인트의 points — 설계점 프리픽스와 일치하면 그만큼
                   건너뛰고 이어서 계산 (설계점은 seed 기반 결정적이라 재현됨).
    on_point(out): 조건 하나가 끝날 때마다 호출 — 중간 저장(체크포인트)용.
    반환: {"axes":[...], "mode", "points": [{"steps":[...], "qe": {nm: {R,G,B}}}, ...],
           "wavelengths_nm": [...], "elapsed_s": float, ["resumed": n]}
    """
    from ..structure.blocks import ir_from_wizard_cfg
    from .simulator import RCWAPlaneWaveSimulator

    pts = doe_points(mode, len(axes), nsamples=nsamples, bound=bound, seed=seed)
    total = len(pts)
    out = {"axes": [list(a) for a in axes], "mode": mode, "nG": nG,
           "bound": float(bound), "seed": int(seed),
           "wavelengths_nm": list(wavelengths_nm), "points": []}
    # ── 체크포인트 이어하기: 저장분 steps 가 설계점 프리픽스와 일치할 때만 ──
    if resume_points:
        okr = len(resume_points) <= total and all(
            len(rp.get("steps", [])) == len(pts[i]) and
            all(abs(float(a) - float(b)) < 1e-6
                for a, b in zip(rp["steps"], pts[i]))
            for i, rp in enumerate(resume_points))
        if okr:
            # JSON 왕복 시 파장 키가 str 로 바뀌므로 int 로 정규화 (새 점과 타입 통일)
            out["points"] = [{"steps": [float(s) for s in rp["steps"]],
                              "qe": {int(w): {ch: float(v) for ch, v in q.items()}
                                     for w, q in (rp.get("qe") or {}).items()}}
                             for rp in resume_points]
            out["resumed"] = len(resume_points)
    start_i = len(out["points"])
    t0 = time.time()
    for i, p in enumerate(pts):
        if i < start_i:                                  # 이미 계산된 조건 건너뜀
            continue
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
        if on_point:                                   # 매 조건 완료 즉시 중간 저장
            try:
                on_point(out)
            except Exception:
                pass                                    # 저장 실패가 계산을 멈추면 안 됨
        if progress:
            done = i + 1
            el = time.time() - t0
            new_done = done - start_i                   # eta 는 새로 계산한 것 기준
            eta = el / max(new_done, 1) * (total - done)
            progress(done, total, eta, p)
    out["elapsed_s"] = round(time.time() - t0, 1)
    return out


def screening_from_axis(res):
    """axis(스크리닝) 결과 -> 노브 중요도 랭킹.

    각 축의 ±1스텝 중심차분 ΔQE 를 파장 전체에서 훑어, 채널별 최대(부호 유지)와
    종합 점수(=채널 최대의 절대값 최대)를 계산. 점수 내림차순 정렬.
    반환: [{axis,label,unit,step,d_R,wl_R,d_G,wl_G,d_B,wl_B,score}, ...]
    """
    axes = res["axes"]
    pts = res["points"]
    ws = [int(w) for w in res["wavelengths_nm"]]
    k = len(axes)

    def qe_at(steps):
        for p in pts:
            if len(p["steps"]) == k and all(abs(float(a) - float(b)) < 1e-9
                                            for a, b in zip(p["steps"], steps)):
                return p["qe"]
        return None

    rows = []
    for i, ax in enumerate(axes):
        sp, sm = [0] * k, [0] * k
        sp[i], sm[i] = 1, -1
        qp, qm = qe_at(sp), qe_at(sm)
        if not qp or not qm:
            continue
        ent = {"axis": ax[0], "label": ax[1], "unit": ax[3], "step": ax[2]}
        best = 0.0
        for ch in "RGB":
            dmax, wat = 0.0, ws[0]
            for w in ws:
                d = (float(qp[w][ch]) - float(qm[w][ch])) / 2.0
                if abs(d) > abs(dmax):
                    dmax, wat = d, w
            ent["d_" + ch] = round(dmax, 5)
            ent["wl_" + ch] = wat
            best = max(best, abs(dmax))
        ent["score"] = round(best, 5)
        rows.append(ent)
    rows.sort(key=lambda r: -r["score"])
    return rows


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
