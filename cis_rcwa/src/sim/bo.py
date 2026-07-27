# -*- coding: utf-8 -*-
"""베이지안 최적화(BO) — '다음에 돌려볼 조건'을 모델이 스스로 고른다.

왜: RBF/2차 서로게이트는 측정점 사이를 보간할 뿐 '어디가 불확실한지' 모른다.
한 번 뽑은 데이터로 최적점을 믿으면 성긴 구역의 가짜 봉우리에 속을 수 있다.
BO 는 GP(가우시안 프로세스: RBF 의 사촌, 예측값+불확실성을 함께 출력) 위에서
EI(Expected Improvement) 획득함수로 "유망하거나(활용) 아직 모르는(탐사)" 지점을
다음 실험으로 추천하고, 실측 후 모델을 갱신하는 피드백 루프를 돈다 —
시작점/국소함정 문제를 설계로 회피.

외부 의존성 없음(numpy). N(표본) 수백 이하 스케일 전용.
"""
import math

import numpy as np

_erf = np.vectorize(math.erf)


def _sqdist(A, B):
    A = np.asarray(A, float)
    B = np.asarray(B, float)
    return np.maximum((A ** 2).sum(1)[:, None] + (B ** 2).sum(1)[None, :]
                      - 2.0 * A @ B.T, 0.0)


def _phi(z):
    return np.exp(-0.5 * z * z) / math.sqrt(2 * math.pi)


def _Phi(z):
    return 0.5 * (1.0 + _erf(z / math.sqrt(2)))


def gp_fit(X, y, ls_grid=(0.5, 1.0, 2.0)):
    """간단 GP 회귀 — RBF 커널, lengthscale 은 중앙값 휴리스틱×그리드 중
    로그우도 최대를 선택(초모수 자동). 반환 dict 는 gp_predict 입력."""
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    mu = float(y.mean())
    yc = y - mu
    sf2 = float(yc.var()) or 1e-6
    D = np.sqrt(_sqdist(X, X))
    off = D[D > 0]
    med = float(np.median(off)) if off.size else 1.0
    sn2 = max(1e-8, 1e-4 * sf2)                    # 작은 nugget (수치 안정 + 노이즈)
    best = None
    for f in ls_grid:
        ls = med * f
        K = sf2 * np.exp(-0.5 * _sqdist(X, X) / ls ** 2) + (sn2 + 1e-8) * np.eye(len(X))
        try:
            L = np.linalg.cholesky(K)
        except np.linalg.LinAlgError:
            continue
        a = np.linalg.solve(L.T, np.linalg.solve(L, yc))
        ll = -0.5 * float(yc @ a) - float(np.log(np.diag(L)).sum())
        if best is None or ll > best[0]:
            best = (ll, ls, L, a)
    _, ls, L, a = best
    return {"X": X, "mu": mu, "sf2": sf2, "ls": ls, "L": L, "a": a}


def gp_predict(g, Xq):
    """평균 m 과 표준편차 s — s 가 '모델이 자신 없는 정도'."""
    Ks = g["sf2"] * np.exp(-0.5 * _sqdist(np.asarray(Xq, float), g["X"]) / g["ls"] ** 2)
    m = g["mu"] + Ks @ g["a"]
    v = np.linalg.solve(g["L"], Ks.T)
    var = np.maximum(g["sf2"] - (v ** 2).sum(0), 1e-12)
    return m, np.sqrt(var)


def suggest(X, y, lo, hi, n_pick=10, n_cand=6000, seed=0, xi=0.005):
    """다음 실험 추천 — EI 최대 후보 n_pick 개 (서로 겹치지 않게 diversity 보장).

    X: (N,k) 지금까지의 스텝, y: (N,) 목표값(클수록 좋게 정규화해 넣을 것),
    lo/hi: 탐색 상자(스텝 단위). 반환: [steps 리스트]*n_pick.
    """
    rng = np.random.default_rng(int(seed))
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    lo = np.asarray(lo, float)
    hi = np.asarray(hi, float)
    k = len(lo)
    g = gp_fit(X, y)
    C = lo + rng.uniform(size=(n_cand, k)) * (hi - lo)          # 전역 후보(탐사)
    bi = int(np.argmax(y))                                       # 현 최고점 주변(활용)
    C2 = np.clip(X[bi] + rng.normal(scale=0.08, size=(max(1, n_cand // 4), k))
                 * (hi - lo), lo, hi)
    C = np.vstack([C, C2])
    m, s = gp_predict(g, C)
    best = float(y.max())
    z = (m - best - xi) / s
    ei = (m - best - xi) * _Phi(z) + s * _phi(z)
    order = np.argsort(-ei)
    picks = []
    minsep = 0.08 * float(np.linalg.norm(hi - lo)) / math.sqrt(k)
    for idx in order:
        c = C[idx]
        if all(float(np.linalg.norm(c - p)) > minsep for p in picks):
            picks.append(c)
            if len(picks) >= n_pick:
                break
    return [[float(v) for v in p] for p in picks]


def points_from_csv(csv_text, k):
    """doe_csv 역파싱 — 캐시에 저장된 원자료를 points 리스트로 복원.

    형식: 헤더(k개 축키 + nm,R,G,B [+ obl_R..B, diff_*]), 행 = 조건×파장.
    사광 열(obl_*/diff_*)이 있으면 point 에 obl/diff 도 복원 — BO 재학습 때
    사광 채널이 소실되지 않게 한다.
    """
    lines = [ln for ln in csv_text.strip().splitlines() if ln.strip()]
    if not lines:
        return []
    hdr = lines[0].split(",")
    col = {name: i for i, name in enumerate(hdr)}
    obl_cols = [h for h in hdr if h.startswith("obl_")]
    diff_cols = [h for h in hdr if h.startswith("diff_")]
    pts, obls, diffs = {}, {}, {}
    order = []
    for ln in lines[1:]:
        c = ln.split(",")
        if len(c) < k + 4:
            continue
        steps = tuple(round(float(x), 6) for x in c[:k])
        if steps not in pts:
            pts[steps], obls[steps], diffs[steps] = {}, {}, {}
            order.append(steps)
        wl = int(float(c[k]))
        pts[steps][wl] = {"R": float(c[k + 1]), "G": float(c[k + 2]),
                          "B": float(c[k + 3])}
        if obl_cols and len(c) > max(col[h] for h in obl_cols):
            obls[steps][wl] = {h[4:]: float(c[col[h]]) for h in obl_cols
                               if c[col[h]] != ""}
            diffs[steps][wl] = {h[5:]: float(c[col[h]]) for h in diff_cols
                                if len(c) > col[h] and c[col[h]] != ""}
    out = []
    for s in order:
        pt = {"steps": list(s), "qe": pts[s]}
        if obls[s]:
            pt["obl"] = obls[s]
            pt["diff"] = diffs[s]
        out.append(pt)
    return out
