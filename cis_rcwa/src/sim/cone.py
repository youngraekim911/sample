# -*- coding: utf-8 -*-
"""원뿔 입사광 샘플링 — F-number 로 정해지는 광선 원뿔의 (θ,φ) 점 + solid-angle 가중.

마이크로렌즈 위 광학계는 F# 원뿔(반각 θ_cone = atan(1/(2·F#))) 안의 모든 방향에서
빛을 보낸다. 단일 CRA 각 1점 계산은 원뿔 평균을 놓치므로,
    QE = Σ_i w_i · QE(θ_i, φ_i),  Σw=1
로 가중 평균한다. 이 모듈은 그 (θ_i, φ_i, w_i) 를 만드는 순수 기하 함수.

수식(구면 기하): 중심각 θ0 로 기울어진 원뿔을 θ=const 소원들로 자르면, 각 θ 에서
원뿔 내부에 드는 방위각 반폭 γ(θ) 는
    cos γ = (tan²θ0 − tan²θc + tan²θ) / (2·tanθ0·tanθ)
이고 solid-angle 요소는 dΩ ∝ γ(θ)·tanθ/cos²θ. θ 는 [θmin,θmax] 중점 샘플,
φ 는 φ0±γ(θ) 를 등분(중점) 샘플한다.
"""
import cmath
import math

import numpy as np


def calc_mra(theta_deg, phi_deg, theta_samples, phi_samples, f_number=2.0,
             include_edge=False):
    """원뿔 각도 샘플 생성.

    theta_deg/phi_deg: 원뿔 중심(주광선 = CRA/방위각). f_number: 렌즈 F#.
    theta_samples×phi_samples 개 점 반환 — {"theta_deg":[], "phi_deg":[], "weight":[]},
    sum(weight)=1. 1×1 이면 중심각 1점(가중 1) = 원뿔 무시(기존 동작).
    """
    if theta_samples <= 1 and phi_samples <= 1:
        return {"theta_deg": [float(theta_deg)], "phi_deg": [float(phi_deg)],
                "weight": [1.0]}
    th0 = math.radians(float(theta_deg))
    ph0 = math.radians(float(phi_deg))
    thc = math.atan(1.0 / (2.0 * float(f_number)))       # 원뿔 반각

    # θ 범위: 원뿔이 광축과 만드는 최소/최대 각 (tan 공간에서 ±tanθc)
    th_max = math.atan(math.tan(th0) + math.tan(thc))
    th_min = 0.0 if th0 < thc else math.atan(math.tan(th0) - math.tan(thc))
    th_list = np.linspace(th_min, th_max, theta_samples * 2 + 1)[1::2]  # 중점 샘플

    gamma = np.empty(len(th_list))
    w_th = np.empty(len(th_list))
    t0, tc = math.tan(th0), math.tan(thc)
    for i, th in enumerate(th_list):
        t = math.tan(th)
        denom = 2.0 * t0 * t
        if abs(denom) < 1e-12:                           # 축상(θ0≈0) 특이점: 원뿔이
            g = math.pi if t <= tc + 1e-12 else 0.0      # 광축 포함 → 전 방위각
        else:
            g = cmath.acos((t0 * t0 - tc * tc + t * t) / denom).real
        gamma[i] = g
        w_th[i] = g * t / (math.cos(th) ** 2)            # dΩ ∝ γ·tanθ/cos²θ

    if include_edge:
        ratios = np.linspace(-1.0, 1.0, phi_samples)
    else:
        ratios = np.linspace(-1.0, 1.0, phi_samples * 2 + 1)[1::2]

    th_out, ph_out, w_out = [], [], []
    for r in ratios:
        for i, th in enumerate(th_list):
            th_out.append(math.degrees(th))
            ph_out.append(math.degrees(r * gamma[i] + ph0))
            w_out.append(w_th[i])
    w = np.asarray(w_out, float)
    s = float(w.sum())
    w = (w / s) if s > 0 else np.full(len(w), 1.0 / len(w))
    return {"theta_deg": [float(v) for v in th_out],
            "phi_deg": [float(v) for v in ph_out],
            "weight": [float(v) for v in w]}
