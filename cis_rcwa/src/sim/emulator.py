# -*- coding: utf-8 -*-
"""넓은 공간 에뮬레이터 — DOE 샘플로 QE(λ,채널)의 '모델'을 학습.

기존 doe.fit_surrogate 는 중심 주변 ±2스텝의 *국소 2차*(quadratic RSM)만 담는다.
설계상자 전체를 lhs 로 넓게 샘플하면 QE 는 공진성·비단조라 2차로는 부족 — 이 모듈은

  1) 표현력이 다른 모델 후보를 함께 피팅:
       quadratic : 1 + 2k + C(k,2)                 (doe._feat 와 동일 = 하위호환)
       cubic     : quadratic + x_i³ (포화/S자 곡선)
       rbf       : 방사기저함수 보간 (공진성·강한 비선형 — 넓은 공간에 강함)
  2) k-fold 교차검증으로 *진짜 예측오차*(out-of-sample R²/RMSE)를 보고
       → in-sample r²(=훈련오차)가 1.0 이어도 CV 가 낮으면 '외운 것'을 잡아냄.
  3) model="auto" 는 CV R² 로 최선 모델을 자동 선택(후보 지표 모두 리포트).
  4) 예측 좌표가 샘플 상자 밖이면 외삽 경고(모델 신뢰구간 밖).

핵심 계약(doe 와 호환): x = step/2 정규화. quadratic 계수·특징순서는 doe._feat 와 동일하므로
quadratic 에뮬레이터는 기존 sensitivity/inverse(2차 계수 기반)와 그대로 호환된다.

emulator dict 는 JSON 직렬화 가능(파장 키는 int 로 저장, 조회는 문자열도 허용).
"""
import itertools

import numpy as np

from .doe import _feat as _feat_quad     # [1, x_i, x_i², x_i·x_j]  (x=step/2)


# ---------------------------------------------------------------- 특징(다항)
def _feat_cubic(x):
    """quadratic 특징 + 순수 3차 x_i³ (포화/변곡 표현). 순서: quad ++ [x_i³]."""
    return _feat_quad(x) + [v ** 3 for v in x]


def _poly_features(model):
    return _feat_cubic if model == "cubic" else _feat_quad


# ---------------------------------------------------------------- RBF
def _pairwise(A, B):
    """유클리드 거리행렬 (len(A), len(B)). (외부 의존성 없이 numpy 로.)"""
    A = np.asarray(A, float)
    B = np.asarray(B, float)
    d2 = (A ** 2).sum(1)[:, None] + (B ** 2).sum(1)[None, :] - 2.0 * A @ B.T
    return np.sqrt(np.maximum(d2, 0.0))


def _rbf_phi(r, eps, kernel):
    if kernel == "gaussian":
        return np.exp(-(r / eps) ** 2)
    if kernel == "thinplate":
        rr = np.where(r < 1e-12, 1e-12, r)
        return rr ** 2 * np.log(rr)
    return np.sqrt((r / eps) ** 2 + 1.0)          # multiquadric(기본): 외삽시 붕괴X


def _rbf_epsilon(X):
    """형상계수 = 표본 간 거리의 중앙값 (스케일 자동)."""
    if len(X) < 2:
        return 1.0
    D = _pairwise(X, X)
    off = D[~np.eye(len(X), dtype=bool)]
    m = float(np.median(off[off > 0])) if np.any(off > 0) else 1.0
    return m if m > 1e-9 else 1.0


def _rbf_fit(X, y, eps, kernel, ridge):
    """단일 타깃 y 에 대한 RBF 가중치 (평균제거 + 최소자승 안정화).

    주의(중요): multiquadric·thinplate 은 '조건부' 양정치 커널이라 Φ 에 음의
    고유값이 존재한다(21점 예: 20개). 여기에 큰 ridge(1e-3)를 더하면 고유값이
    0 근처로 밀려 해가 폭발한다 — 보간기인데 in-sample R² 가 -13 까지 떨어졌던
    실제 원인. ridge 는 '아주 작게'만 쓰고, 안정화는 SVD 최소자승(lstsq)이
    담당한다(rank 결손·악조건에서도 최소노름 해를 준다).
    """
    mu = float(np.mean(y))
    Phi = _rbf_phi(_pairwise(X, X), eps, kernel)
    A = Phi + float(ridge) * np.eye(len(X))
    w, *_ = np.linalg.lstsq(A, np.asarray(y, float) - mu, rcond=None)
    return w, mu


def _rbf_predict(Xq, X, w, mu, eps, kernel):
    return _rbf_phi(_pairwise(Xq, X), eps, kernel) @ w + mu


# ---------------------------------------------------------------- CV 지표
def _r2_rmse(y, yh):
    y = np.asarray(y, float)
    yh = np.asarray(yh, float)
    ss = float(((y - y.mean()) ** 2).sum())
    sse = float(((y - yh) ** 2).sum())
    r2 = 1.0 - sse / ss if ss > 1e-12 else 1.0
    rmse = float(np.sqrt(sse / len(y)))
    return r2, rmse


def _kfold(n, folds, seed=0):
    """결정적 k-fold 인덱스 분할."""
    idx = np.arange(n)
    np.random.default_rng(int(seed)).shuffle(idx)
    folds = max(2, min(int(folds), n))
    return [idx[i::folds] for i in range(folds)]


# ---------------------------------------------------------------- 모델 후보
def _fit_one_model(model, Xsteps, Y, folds, seed, rbf_kernel, ridge):
    """한 모델을 (파장×채널) 전체에 피팅 + CV.

    Xsteps: (N,k) 스텝(-bound..bound).  Y: {ch: {wl: y[N]}}.
    반환: (param, insample{ch:{wl:r2}}, cv_r2{ch:{wl}}, cv_rmse{ch:{wl}})
      param(quad/cubic): {"coef": {ch:{wl:[...]}}}
      param(rbf): {"kernel","eps","ridge","weights":{ch:{wl:[...]}}, "mean":{ch:{wl:..}}}
    """
    x = np.asarray(Xsteps, float) / 2.0                       # x=step/2 (doe 계약)
    N = len(x)
    folds_idx = _kfold(N, folds, seed)
    ins, cvr, cve = {}, {}, {}
    if model in ("quadratic", "cubic"):
        featf = _poly_features(model)
        F = np.array([featf(row) for row in x])               # (N, P)
        param = {"coef": {}}
        for ch, wmap in Y.items():
            param["coef"][ch] = {}
            ins[ch], cvr[ch], cve[ch] = {}, {}, {}
            for wl, y in wmap.items():
                y = np.asarray(y, float)
                c, *_ = np.linalg.lstsq(F, y, rcond=None)
                param["coef"][ch][int(wl)] = [round(float(v), 6) for v in c]
                r2i, _ = _r2_rmse(y, F @ c)
                ins[ch][int(wl)] = round(r2i, 4)
                # CV
                yh = np.empty(N)
                for te in folds_idx:
                    tr = np.setdiff1d(np.arange(N), te)
                    cc, *_ = np.linalg.lstsq(F[tr], y[tr], rcond=None)
                    yh[te] = F[te] @ cc
                r2c, rmc = _r2_rmse(y, yh)
                cvr[ch][int(wl)] = round(r2c, 4)
                cve[ch][int(wl)] = round(rmc, 5)
        return param, ins, cvr, cve

    if model == "rbf":
        eps = _rbf_epsilon(x)
        param = {"kernel": rbf_kernel, "eps": round(float(eps), 6),
                 "ridge": float(ridge), "weights": {}, "mean": {}}
        for ch, wmap in Y.items():
            param["weights"][ch], param["mean"][ch] = {}, {}
            ins[ch], cvr[ch], cve[ch] = {}, {}, {}
            for wl, y in wmap.items():
                y = np.asarray(y, float)
                w, mu = _rbf_fit(x, y, eps, rbf_kernel, ridge)
                param["weights"][ch][int(wl)] = [round(float(v), 8) for v in w]
                param["mean"][ch][int(wl)] = round(float(mu), 8)
                r2i, _ = _r2_rmse(y, _rbf_predict(x, x, w, mu, eps, rbf_kernel))
                ins[ch][int(wl)] = round(r2i, 4)
                yh = np.empty(N)
                for te in folds_idx:
                    tr = np.setdiff1d(np.arange(N), te)
                    epstr = _rbf_epsilon(x[tr])
                    wtr, mutr = _rbf_fit(x[tr], y[tr], epstr, rbf_kernel, ridge)
                    yh[te] = _rbf_predict(x[te], x[tr], wtr, mutr, epstr, rbf_kernel)
                r2c, rmc = _r2_rmse(y, yh)
                cvr[ch][int(wl)] = round(r2c, 4)
                cve[ch][int(wl)] = round(rmc, 5)
        return param, ins, cvr, cve

    raise ValueError(f"unknown model {model}")


def _mean_metric(m):
    """{ch:{wl:v}} -> 전체 평균(대표 지표)."""
    vals = [v for ch in m.values() for v in ch.values()]
    return round(float(np.mean(vals)), 4) if vals else None


# ---------------------------------------------------------------- 공개 API
def fit_emulator(doe_result, model="auto", cv_folds=5,
                 rbf_kernel="multiquadric", ridge=1e-8, base_cfg=None):
    """DOE 결과 -> 넓은 공간 에뮬레이터(JSON dict) + 교차검증 지표.

    model: "quadratic"|"cubic"|"rbf"|"auto"(CV R²로 자동선택).
    반환 dict:
      type="emulator", model, axes, naxes, wavelengths_nm, bound, box{lo,hi},
      metrics{r2_insample_mean, r2_cv_mean, rmse_cv_mean, per_model_cv{...}},
      r2/r2_cv/rmse_cv: {ch:{wl}},  + (poly) coef  또는 (rbf) rbf{...}, X
    """
    pts = doe_result["points"]
    Xsteps = np.array([p["steps"] for p in pts], float)       # (N,k)
    ws = [int(w) for w in doe_result["wavelengths_nm"]]
    Y = {ch: {w: np.array([p["qe"][w][ch] for p in pts], float) for w in ws}
         for ch in "RGB"}
    # ── 사광 채널: 점마다 obl(빗각 QE)·diff(동컬러 %)가 있으면 같은 모델로 학습.
    #    oR/oG/oB = 대표 필드 빗각 QE. diff 는 CIS 채널별(R/Gr/Gb/B — G 는
    #    Gr/Gb 로 분리) dR/dGr/dGb/dB 로 각각 학습 (% 0~100 그대로).
    has_obl = bool(pts) and all(p.get("obl") and p.get("diff") for p in pts)
    if has_obl:
        for L in "RGB":
            Y["o" + L] = {w: np.array([p["obl"][w][L] for p in pts], float)
                          for w in ws}
        dcols = sorted({c for p in pts for m in p["diff"].values() for c in m})
        for c in dcols:
            Y["d" + c] = {w: np.array([float(p["diff"][w].get(c, 0.0))
                                       for p in pts], float) for w in ws}
    k = Xsteps.shape[1]

    candidates = (["quadratic", "cubic", "rbf"] if model == "auto"
                  else [model])
    fitted = {}
    for mdl in candidates:
        param, ins, cvr, cve = _fit_one_model(
            mdl, Xsteps, Y, cv_folds, doe_result.get("seed", 0), rbf_kernel, ridge)
        fitted[mdl] = {"param": param, "ins": ins, "cvr": cvr, "cve": cve,
                       "cv_mean": _mean_metric(cvr)}

    # 자동선택: CV R² 평균 최대
    chosen = (max(fitted, key=lambda m: (fitted[m]["cv_mean"] or -1e9))
              if model == "auto" else model)
    F = fitted[chosen]

    # 대표 지표는 직광 R/G/B 만으로 (기존과 비교 가능하게) — 사광 채널은 별도 평균
    direct = lambda m: {c: m[c] for c in "RGB" if c in m}
    out = {"type": "emulator", "model": chosen, "axes": doe_result["axes"],
           "naxes": k, "wavelengths_nm": ws,
           "bound": float(doe_result.get("bound", 2.0)),
           "box": {"lo": [round(float(v), 4) for v in Xsteps.min(0)],
                   "hi": [round(float(v), 4) for v in Xsteps.max(0)]},
           "n_samples": len(pts), "channels": list(Y.keys()),
           "r2": F["ins"], "r2_cv": F["cvr"], "rmse_cv": F["cve"],
           "metrics": {
               "r2_insample_mean": _mean_metric(direct(F["ins"])),
               "r2_cv_mean": _mean_metric(direct(F["cvr"])),
               "rmse_cv_mean": _mean_metric(direct(F["cve"])),
               "per_model_cv": {m: fitted[m]["cv_mean"] for m in fitted}}}
    if has_obl:
        # on-band(각 컬러 평균 QE 가 최대치의 50% 이상인 λ) — diff 판정은 이 λ에서만.
        onband = {}
        for L in "RGB":
            mus = {w: float(np.mean(Y[L][w])) for w in ws}
            top = max(mus.values())
            ob = [w for w in ws if mus[w] >= 0.5 * top]
            onband[L] = ob if ob else list(ws)
        ob_meta = dict(doe_result.get("oblique") or {})
        ob_meta.update({"onband_nm": onband, "threshold_pct": 30.0,
                        "r2_cv_mean": _mean_metric(
                            {c: F["cvr"][c] for c in F["cvr"] if c not in "RGB"})})
        out["oblique"] = ob_meta
    if chosen in ("quadratic", "cubic"):
        out["coef"] = F["param"]["coef"]
        out["feature_order"] = ("1, x1..xk, x1^2..xk^2, x_i*x_j(i<j)"
                                + (", x1^3..xk^3" if chosen == "cubic" else "")
                                + " (x=step/2)")
    else:
        out["rbf"] = {kk: F["param"][kk] for kk in
                      ("kernel", "eps", "ridge", "weights", "mean")}
        out["X"] = [[round(float(v), 6) for v in row] for row in (Xsteps / 2.0)]
    # 기준 구조를 함께 담아 '파일 하나로 완결' — 모델 탐색 페이지가 서버 없이도
    # 구조 미리보기를 그리고, 슬라이더 변화를 실제 치수로 환산할 수 있다.
    if base_cfg is not None:
        out["base_cfg"] = base_cfg
    return out


def _wl_key(m, wl):
    w = int(wl)
    if w in m:
        return w
    if str(w) in m:
        return str(w)
    raise KeyError(f"에뮬레이터에 {w}nm 없음 (있는 값: {list(m.keys())})")


def predict_emulator(emu, steps, wavelength_nm, channel):
    """에뮬레이터 + 스텝(-bound..bound) -> QE 예측 (모델 종류 자동)."""
    x = np.array([s / 2.0 for s in steps], float)
    mdl = emu.get("model", "quadratic")
    if mdl in ("quadratic", "cubic"):
        cmap = emu["coef"][channel]
        coef = cmap[_wl_key(cmap, wavelength_nm)]
        featf = _poly_features(mdl)
        return float(np.dot(featf(x), coef))
    rb = emu["rbf"]
    X = np.asarray(emu["X"], float)
    wmap = rb["weights"][channel]
    wk = _wl_key(wmap, wavelength_nm)
    w = np.asarray(wmap[wk], float)
    mu = rb["mean"][channel][wk] if wk in rb["mean"][channel] \
        else rb["mean"][channel][int(wk)]
    return float(_rbf_predict(x[None, :], X, w, mu, rb["eps"], rb["kernel"])[0])


def _mean_lookup(m, wk):
    if wk in m:
        return m[wk]
    return m[int(wk)] if int(wk) in m else m[str(wk)]


def predict_batch(emu, Xsteps, wavelength_nm, channel):
    """여러 스텝(M,k) 동시 예측 (M,). 모델 종류 자동."""
    x = np.asarray(Xsteps, float) / 2.0
    mdl = emu.get("model", "quadratic")
    if mdl in ("quadratic", "cubic"):
        featf = _poly_features(mdl)
        F = np.array([featf(r) for r in x])
        cmap = emu["coef"][channel]
        coef = np.asarray(cmap[_wl_key(cmap, wavelength_nm)], float)
        return F @ coef
    rb = emu["rbf"]
    X = np.asarray(emu["X"], float)
    wmap = rb["weights"][channel]
    wk = _wl_key(wmap, wavelength_nm)
    w = np.asarray(wmap[wk], float)
    mu = _mean_lookup(rb["mean"][channel], wk)
    return _rbf_predict(x, X, w, mu, rb["eps"], rb["kernel"])


def invert(emu, target, weights=None, channels=("R", "G", "B"),
           grid=11, refine_iters=40):
    """모델-무관 역설계 — 목표 QE 에 가장 근접한 스텝(구조)을 격자+좌표하강으로.

    target: {wl_nm:{R,G,B}} (일부만 줘도 됨). 상자는 에뮬레이터 샘플 상자[lo,hi].
    반환: inverse.invert 와 같은 계약(steps, params, predicted, residual_rms, ...).
    """
    axes = emu["axes"]
    k = emu["naxes"]
    ws = emu["wavelengths_nm"]
    lo = np.array(emu["box"]["lo"]) * 2.0                     # step 단위 상자
    hi = np.array(emu["box"]["hi"]) * 2.0

    tgt, wt, key = [], [], []
    for wl in ws:
        wl = int(wl)
        tk = next((kk for kk in target if int(kk) == wl), None)
        if tk is None:
            continue
        for ch in channels:
            if ch in target[tk]:
                tgt.append(float(target[tk][ch]))
                wv = 1.0
                if weights:
                    wk2 = next((kk for kk in weights if int(kk) == wl), None)
                    if wk2 is not None:
                        wv = float(weights[wk2].get(ch, 1.0))
                wt.append(wv)
                key.append((wl, ch))
    if not tgt:
        raise ValueError("target 에 유효한 (파장,채널) 이 없습니다.")
    tgt = np.array(tgt)
    wt = np.array(wt)

    def sse(X):
        e = np.zeros(len(X))
        for (wl, ch), t, w_ in zip(key, tgt, wt):
            e += w_ * (predict_batch(emu, X, wl, ch) - t) ** 2
        return e

    # 1) 격자 스캔 (상자 안, 축 개수에 맞춰 해상도 자동 축소)
    cap = 60000
    gk = grid if grid ** k <= cap else max(3, int(cap ** (1.0 / k)))
    axgrids = [np.linspace(lo[d], hi[d], gk) for d in range(k)]
    G = np.array(list(itertools.product(*axgrids)))
    best = G[int(np.argmin(sse(G)))].astype(float)
    if gk < grid:                                            # 고차원 무작위 다중시작 보강
        rng = np.random.default_rng(0)
        Rnd = lo + rng.uniform(size=(6000, k)) * (hi - lo)
        e = sse(Rnd)
        j = int(np.argmin(e))
        if e[j] < sse(best[None])[0]:
            best = Rnd[j].astype(float)
    # 2) 좌표하강 정밀화
    step = (hi - lo) / max(gk - 1, 1)
    for _ in range(refine_iters):
        improved = False
        for d in range(k):
            cand = np.tile(best, (7, 1))
            cand[:, d] = np.clip(best[d] + np.linspace(-step[d], step[d], 7),
                                 lo[d], hi[d])
            e = sse(cand)
            j = int(np.argmin(e))
            if e[j] < sse(best[None])[0] - 1e-12:
                best = cand[j]
                improved = True
        step *= 0.6
        if not improved and float(np.max(step)) < 1e-4:
            break

    predflat = np.array([predict_batch(emu, best[None], wl, ch)[0]
                         for (wl, ch) in key])
    pred, per_point = {}, []
    for (wl, ch), pp, t in zip(key, predflat, tgt):
        pred.setdefault(wl, {})[ch] = round(float(pp), 5)
        per_point.append({"wl": wl, "ch": ch, "target": round(t, 5),
                          "predicted": round(float(pp), 5), "err": round(float(pp - t), 5)})
    rms = float(np.sqrt((wt * (predflat - tgt) ** 2).sum() / wt.sum()))
    at_edge = [bool(best[d] <= lo[d] + 1e-6 or best[d] >= hi[d] - 1e-6) for d in range(k)]
    return {"axes": axes, "steps": [round(float(v), 3) for v in best],
            "params": [{"axis": ax[0], "label": ax[1],
                        "delta": round(float(best[i] * ax[2]), 4), "unit": ax[3]}
                       for i, ax in enumerate(axes)],
            "predicted": pred, "target": target, "residual_rms": round(rms, 5),
            "per_point": per_point, "at_edge": at_edge, "edge_warning": any(at_edge)}


def extrapolation(emu, steps):
    """예측 좌표가 샘플 상자(관측 min/max) 밖인지 축별 판정."""
    x = [s / 2.0 for s in steps]
    lo, hi = emu["box"]["lo"], emu["box"]["hi"]
    flags = [bool(xi < lo[i] - 1e-6 or xi > hi[i] + 1e-6) for i, xi in enumerate(x)]
    return {"per_axis": flags, "any": any(flags)}


def sensitivity(emu, wavelength_nm, channel, at=None, eps=0.1):
    """모델-무관 국소 민감도 ∂QE/∂step (중심차분). at=기준 스텝(기본 중심 0)."""
    k = emu["naxes"]
    at = list(at) if at is not None else [0.0] * k
    rows = []
    for i, ax in enumerate(emu["axes"]):
        sp, sm = list(at), list(at)
        sp[i] += eps
        sm[i] -= eps
        d1 = (predict_emulator(emu, sp, wavelength_nm, channel)
              - predict_emulator(emu, sm, wavelength_nm, channel)) / (2 * eps)
        rows.append({"axis": ax[0], "label": ax[1], "unit": ax[3], "step": ax[2],
                     "dqe_per_step": round(float(d1), 5), "abs": abs(float(d1))})
    rows.sort(key=lambda r: -r["abs"])
    return {"wavelength_nm": int(wavelength_nm), "channel": channel,
            "method": f"emulator:{emu.get('model')}",
            "ranking": [{kk: v for kk, v in r.items() if kk != "abs"} for r in rows]}
