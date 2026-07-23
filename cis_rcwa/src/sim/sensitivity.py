# -*- coding: utf-8 -*-
"""B. 파라미터 민감도 — 이 QE 를 '가장 크게 움직이는' 구조 노브 랭킹.

B-1: DOE surrogate(2차 다항)의 1차 계수 = 국소 민감도 ∂QE/∂노브 (추가 RCWA 0회).
B-2: surrogate 없이 현 구조에서 각 축 ±1 스텝 유한차분 (그 파장만, 소량 RCWA).

랭킹은 '의미있는 1스텝당 ΔQE' 로 정규화해 단위 다른 노브를 공정 비교.
(스텝 크기는 DOE axes 정의: CF 300Å, 평탄/ML두께 0.025µm, 배율 0.025)
"""
import itertools


def _feat_index(k):
    """surrogate feature 순서(k축): [1, x1..xk, x1²..xk², x_i·x_j (i<j)].
    -> (선형계수 인덱스[k], 제곱계수 인덱스[k], 교차쌍 리스트)."""
    lin = list(range(1, 1 + k))
    sq = list(range(1 + k, 1 + 2 * k))
    cross = list(itertools.combinations(range(k), 2))
    return lin, sq, cross


def sensitivity_from_surrogate(surrogate, wavelength_nm, channel):
    """B-1: surrogate 계수에서 축별 국소 민감도. x=step/2 정규화이므로
    '1 스텝당 ΔQE' = 0.5·(1차계수). 2차(곡률)도 함께 보고."""
    from .doe import coef_at, r2_at
    axes = surrogate["axes"]
    w = int(wavelength_nm)
    coef = coef_at(surrogate, channel, w)
    lin, sq, _ = _feat_index(len(axes))
    rows = []
    for i, ax in enumerate(axes):
        d1 = 0.5 * float(coef[lin[i]])          # ∂QE/∂step (중심)
        d2 = 0.25 * float(coef[sq[i]])          # 0.5²·2차계수 (스텝²당 곡률/2)
        rows.append({"axis": ax[0], "label": ax[1], "unit": ax[3],
                     "step": ax[2], "dqe_per_step": round(d1, 5),
                     "curvature": round(d2, 5), "abs": abs(d1)})
    rows.sort(key=lambda r: -r["abs"])
    r2 = r2_at(surrogate, channel, w)
    return {"wavelength_nm": w, "channel": channel, "method": "surrogate",
            "r2": r2, "ranking": [{k: v for k, v in r.items() if k != "abs"}
                                  for r in rows]}


def local_sensitivity(cfg, wavelength_nm, channel, nG=101, downsample=2,
                      lateral_n=256, materials_dir=None, axes=None):
    """B-2: surrogate 없이 유한차분 (그 파장만). 각 축 ±1 스텝 중심차분."""
    from ..structure.blocks import ir_from_wizard_cfg
    from .simulator import RCWAPlaneWaveSimulator
    from .doe import AXES_DEFAULT, apply_point
    axes = axes or AXES_DEFAULT

    k = len(axes)

    def qe_at(steps):
        c = apply_point(cfg, steps, axes)
        sim = RCWAPlaneWaveSimulator(ir_from_wizard_cfg(c, lateral_n), nG=nG,
                                     downsample=downsample, materials_dir=materials_dir)
        a = 0.0
        for pol in ((1.0, 0.0), (0.0, 1.0)):
            a += 0.5 * sim.run(wavelength_nm / 1000.0, pol_te=pol[0],
                               pol_tm=pol[1])["QE_rgb"][channel]
        return a

    rows = []
    for i, ax in enumerate(axes):
        sp, sm = [0] * k, [0] * k
        sp[i], sm[i] = 1, -1
        d1 = 0.5 * (qe_at(tuple(sp)) - qe_at(tuple(sm)))    # 중심차분 ∂QE/∂step
        rows.append({"axis": ax[0], "label": ax[1], "unit": ax[3], "step": ax[2],
                     "dqe_per_step": round(d1, 5), "abs": abs(d1)})
    rows.sort(key=lambda r: -r["abs"])
    return {"wavelength_nm": int(wavelength_nm), "channel": channel,
            "method": "finite_diff",
            "ranking": [{k: v for k, v in r.items() if k != "abs"} for r in rows]}
