# -*- coding: utf-8 -*-
"""②. 역설계 — 목표 QE 곡선 -> 그 곡선에 가장 근접한 구조.

surrogate(2차 다항, forward)를 거꾸로 풀어, 목표 T(λ,채널)에 대해
    min_x  Σ w·(surrogate(x) − T)²
를 만족하는 '하나의 구조 x'(4축 스텝) 를 찾는다. surrogate 가 다항이라
격자 스캔+좌표하강으로 밀리초에 수렴 (외부 의존성 없음).

정직한 출력: '달성 가능한 것 중 가장 가까운' 구조 + 목표대비 잔차 + 상자경계 경고
(x 가 ±2 한계에 붙으면 탐색 상자 밖일 수 있음 = surrogate 외삽 주의).
"""
import itertools

import numpy as np


def _feat_matrix(X):
    """X:(M,k) 스텝지수 -> 2차 특징 (M, 1+2k+C(k,2)). x=step/2 정규화.
    순서는 doe._feat 과 동일: [1, x_i, x_i², x_i·x_j(i<j)]."""
    x = X / 2.0
    k = x.shape[1]
    cols = [np.ones(len(X))]
    cols += [x[:, i] for i in range(k)]
    cols += [x[:, i] ** 2 for i in range(k)]
    for a, b in itertools.combinations(range(k), 2):
        cols.append(x[:, a] * x[:, b])
    return np.stack(cols, axis=1)


def invert(surrogate, target, weights=None, channels=("R", "G", "B"),
           grid=13, bound=2.0, refine_iters=40):
    """target: {wl_nm: {R,G,B}} (일부 채널/파장만 줘도 됨). weights: 같은 구조.

    반환: {steps:[k], axes, predicted:{wl:{R,G,B}}, target, residual_rms,
           per_point:[...], at_edge:[bool×k], edge_warning}   (k=축 개수)
    """
    axes = surrogate["axes"]
    ws = surrogate["wavelengths_nm"]
    coef = surrogate["coef"]

    # 목표를 (파장,채널) 리스트로 평탄화 + 가중치
    tgt, wt, key = [], [], []
    for wl in ws:
        for ch in channels:
            if int(wl) in {int(k) for k in target} and ch in target[_k(target, wl)]:
                tgt.append(float(target[_k(target, wl)][ch]))
                wv = 1.0
                if weights and _k(weights, wl, True) is not None:
                    wv = float(weights.get(_k(weights, wl, True), {}).get(ch, 1.0))
                wt.append(wv)
                key.append((int(wl), ch))
    if not tgt:
        raise ValueError("target 에 유효한 (파장,채널) 이 없습니다.")
    tgt = np.array(tgt)
    wt = np.array(wt)
    from .doe import coef_at
    C = np.array([coef_at(surrogate, ch, wl) for (wl, ch) in key])   # (P,15)

    def sse(X):
        F = _feat_matrix(X)                                 # (M,15)
        pred = F @ C.T                                      # (M,P)
        return (wt * (pred - tgt) ** 2).sum(axis=1)         # (M,)

    k = len(axes)
    # 1) 격자 스캔 — grid^k 가 폭증하지 않게 축 개수에 맞춰 격자 해상도 자동 축소
    cap = 60000
    gk = grid if grid ** k <= cap else max(3, int(cap ** (1.0 / k)))
    g = np.linspace(-bound, bound, gk)
    G = np.array(list(itertools.product(g, repeat=k)))      # (gk^k, k)
    best = G[np.argmin(sse(G))].astype(float)
    # 1b) 고차원(격자 성긴 경우)은 무작위 다중시작으로 basin 보강 (결정적 seed)
    if gk < grid:
        rng = np.random.default_rng(0)
        Rnd = rng.uniform(-bound, bound, size=(6000, k))
        e = sse(Rnd)
        j = int(np.argmin(e))
        if e[j] < sse(best[None])[0]:
            best = Rnd[j].astype(float)
    # 2) 좌표하강 정밀화
    step = (2 * bound) / (gk - 1)
    for _ in range(refine_iters):
        improved = False
        for d in range(k):
            cand = np.tile(best, (7, 1))
            cand[:, d] = np.clip(best[d] + np.linspace(-step, step, 7), -bound, bound)
            e = sse(cand)
            j = int(np.argmin(e))
            if e[j] < sse(best[None])[0] - 1e-12:
                best = cand[j]
                improved = True
        step *= 0.6
        if not improved and step < 1e-4:
            break

    F = _feat_matrix(best[None])
    predflat = (F @ C.T)[0]
    pred = {}
    per_point = []
    for (wl, ch), p, t in zip(key, predflat, tgt):
        pred.setdefault(wl, {})[ch] = round(float(p), 5)
        per_point.append({"wl": wl, "ch": ch, "target": round(t, 5),
                          "predicted": round(float(p), 5),
                          "err": round(float(p - t), 5)})
    rms = float(np.sqrt((wt * (predflat - tgt) ** 2).sum() / wt.sum()))
    at_edge = [bool(abs(v) > bound - 1e-3) for v in best]
    return {"axes": axes, "steps": [round(float(v), 3) for v in best],
            "params": _steps_to_params(axes, best),
            "predicted": pred, "target": target,
            "residual_rms": round(rms, 5), "per_point": per_point,
            "at_edge": at_edge,
            "edge_warning": any(at_edge)}


def _steps_to_params(axes, steps):
    """스텝지수 -> 사람이 읽는 실변화량 (예: CF +450Å, 평탄 -0.03µm)."""
    out = []
    for ax, s in zip(axes, steps):
        d = s * ax[2]
        out.append({"axis": ax[0], "label": ax[1],
                    "delta": round(float(d), 4), "unit": ax[3]})
    return out


def _k(dct, wl, allow_none=False):
    """dct 의 키가 int/str 섞여도 wl 매칭."""
    for k in dct:
        if int(k) == int(wl):
            return k
    return None if allow_none else wl
