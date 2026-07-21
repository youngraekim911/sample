# -*- coding: utf-8 -*-
"""nG 수렴 추정 — 어떤 구조든 '참(수렴) QE' 를 재현가능하게 준다.

FMM/RCWA 는 푸리에 차수 nG 의 유한 절단 때문에 QE 가 nG 에 따라 진동한다
(특히 셀이 크고 피처가 작을수록 심함). 따라서 **단일 nG 는 미수렴**이라 구조마다
오차가 다르고, nG 를 '타겟이 나오는 값'에 고정하면 그 구조에만 맞는 오버피팅이 된다.

올바른 방향(수렴):
  1) nG 중심을 구조 크기/최소 피처에 맞춰 충분히 크게 잡고(recommend_center_nG)
  2) 그 주변 여러 nG 창에서 돌려 진동을 상쇄(평균) -> 수렴값 + 불확도(±std)
  => 타겟을 맞추는 게 아니라 그 구조·그 n,k 의 물리 참값을 준다. 다른 구조에 그대로 적용됨.

QE 가 타겟과 다르면 그건 nG 로 맞출 게 아니라 n,k / 구조 / 빠진 물리(심부 수집효율 등)
의 문제다.
"""
import numpy as np


def recommend_center_nG(span_um, feat_um):
    """수렴 목표 nG 중심 추천 — 셀 클수록·최소피처 작을수록 크게.

    기준점: 2µm 셀·100nm 피처에서 nG~300 이 수렴 근처 (수렴 스터디 기반).
    nG ∝ (셀면적) × (100nm/최소피처).  범위 [151, 701] 로 캡.
    """
    base = 300.0 * (float(span_um) / 2.0) ** 2 * min(max(0.10 / float(feat_um), 0.5), 3.0)
    ng = int(round(base))
    return int(max(151, min(701, ng | 1)))


def _odd_grid(lo, hi, n):
    gs = np.linspace(int(lo), int(hi), int(n))
    return sorted({int(2 * (round(g) // 2) + 1) for g in gs})   # 홀수 유일값


def converged_qe(make_sim, lam_list, center_nG, n_samples=4, span_frac=0.5,
                 pol="avg", channels="RGB"):
    """center_nG 주변 n_samples 개 홀수 nG 에서 QE 를 돌려 채널별 수렴 추정.

    make_sim(nG) -> 시뮬레이터 (같은 IR, nG 만 다르게).
    반환: {"nGs":[...], "qe": {lam: {ch: {"mean","std","min","max"}}}, "band": max_std}
      mean = 수렴 추정값, std = 수렴 불확도(작을수록 잘 수렴).
    """
    lo = center_nG * (1.0 - span_frac / 2.0)
    hi = center_nG * (1.0 + span_frac / 2.0)
    nGs = _odd_grid(lo, hi, n_samples)
    # 집계 키는 채널명(R/G/B)과 충돌 안 나게 별칭 사용: refl/QE_tot/A_stack
    aggmap = {"refl": "R", "QE_tot": "QE", "A_stack": "A_stack"}
    keys = list(channels) + list(aggmap)
    acc = {lam: {k: [] for k in keys} for lam in lam_list}
    for ng in nGs:
        sim = make_sim(ng)
        for lam in lam_list:
            if pol == "avg":
                o1 = sim.run(lam, pol_te=1.0, pol_tm=0.0)
                o2 = sim.run(lam, pol_te=0.0, pol_tm=1.0)
                o = {a: 0.5 * (o1[s] + o2[s]) for a, s in aggmap.items()}
                q = {c: 0.5 * (o1["QE_rgb"][c] + o2["QE_rgb"][c]) for c in channels}
            else:
                r = sim.run(lam)
                o = {a: r[s] for a, s in aggmap.items()}
                q = {c: r["QE_rgb"][c] for c in channels}
            for c in channels:
                acc[lam][c].append(float(q[c]))
            for a in aggmap:
                acc[lam][a].append(float(o[a]))
    qe = {}
    band = 0.0
    for lam in lam_list:
        qe[lam] = {}
        for k in keys:
            a = np.array(acc[lam][k])
            st = float(a.std())
            if k in channels:
                band = max(band, st)                   # 밴드는 색채널 진동만 (집계는 제외)
            qe[lam][k] = {"mean": float(a.mean()), "std": st,
                          "min": float(a.min()), "max": float(a.max())}
    return {"nGs": list(nGs), "qe": qe, "band": band}
