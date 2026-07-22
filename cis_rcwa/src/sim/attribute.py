# -*- coding: utf-8 -*-
"""A. 에너지 귀속 — 한 파장 QE 가 그 값인 '물리적' 이유 (빛이 어디로 갔나).

입사광 1 을 반사 + 층별 흡수(비-Si) + DTI 트렌치 흡수 + Si 재결합손실 +
심부투과 + Si 수집(=QE) 으로 완전 분해. 합 ≈ 1 (에너지 보존).

RCWA 1회(편광평균 2회)로 계산 — 추가 근사 없음. QE 스펙트럼의 한 점을 클릭하면
"이 QE 는 반사 3% + DTI흡수 14% + CF흡수 6% + ... 때문" 을 즉답.
"""
import numpy as np


def energy_attribution(sim, wavelength_nm, pol="avg"):
    """sim(RCWAPlaneWaveSimulator) 의 wavelength 에서 에너지 귀속.

    반환 dict:
        wavelength_nm, reflect, transmit_deep,
        above_si: {tag: frac}   # 비-Si 손실층(CF/grid/BARL/ML/코팅) 흡수
        dti_trench, si_recomb, qe_collected(=Si수집=QE),
        by_channel: {R,G,B: {qe, recomb, trench}}  # 색 픽셀별
        checksum   # 합 (1 이면 완전 분해)
    """
    lam = wavelength_nm / 1000.0
    pols = ((1.0, 0.0), (0.0, 1.0)) if pol == "avg" else \
        (((1.0, 0.0),) if pol == "te" else ((0.0, 1.0),))
    det = sim.ir.detector
    tags = getattr(sim.ir, "layer_tags", None) or \
        [str(i) for i in range(len(sim.layer_stack))]
    M = len(sim.layer_stack)
    nS, nB = sim.n_si_layers, sim.n_below_band
    band = set(range(M - nS - nB, M - nB))
    r0 = float(getattr(det, "collect_r0", 0.0) or 0.0)
    ld = float(getattr(det, "collect_ld_um", 0.0) or 0.0)
    use_coll = r0 > 0 and ld > 0
    deep_is_det = bool(getattr(det, "deep_is_detector", True))
    pixidx = det.pixel_map
    labels = det.pixel_labels or []
    excl = det.exclude_mask if det is not None and det.exclude_mask is not None \
        else np.zeros(sim.ir.grid_shape, dtype=bool)

    acc = {"reflect": 0.0, "transmit_deep": 0.0, "dti_trench": 0.0,
           "si_recomb": 0.0, "qe_collected": 0.0}
    above = {}
    chan = {c: {"qe": 0.0, "recomb": 0.0, "trench": 0.0} for c in "RGB"}

    for pw in pols:
        o = sim.run(lam, pol_te=pw[0], pol_tm=pw[1], pixel_qe="diag")
        solver = o["_solver"]
        w = 1.0 / len(pols)
        acc["reflect"] += w * float(o["R"])
        T_deep = float(o["QE"])                       # diag 모드: 심부투과(=solver T)
        # 심부투과가 검출기(반무한 기판=Si)로 흡수되면 QE 로 수집(η≈1),
        # 아니면 손실(광손실). 시뮬레이터 규약(deep_is_detector)과 일치시킴.
        if deep_is_det:
            acc["qe_collected"] += w * T_deep
            if pixidx is not None:
                Sz = solver.transmitted_flux_map(sim.grid_ny, sim.grid_nx)
                Sz = Sz.detach().cpu().numpy()
                ng = Sz.size
                for p, L in enumerate(labels):
                    if L in "RGB":
                        chan[L]["qe"] += w * float(Sz[pixidx == p].sum()) / ng
        else:
            acc["transmit_deep"] += w * T_deep
        maps, C = solver.absorption_maps(sim.grid_ny, sim.grid_nx)
        # 밴드층 상단 기준 절대깊이(η(z)용)
        z_off, zc = {}, 0.0
        for li in sorted(band):
            z_off[li] = zc
            zc += float(sim.layer_stack[li][1])
        for li, dens_t in maps.items():
            dens = dens_t.detach().cpu().numpy() * C
            if li in band:
                zres, _C = solver.absorption_maps_zresolved(sim.grid_ny, sim.grid_nx, [li])
                zl, slices = zres.get(li, ([], []))
                for zloc, s_t in zip(zl, slices):
                    s = s_t.detach().cpu().numpy() * C
                    e = 1.0 - r0 * np.exp(-(z_off[li] + zloc) / ld) if use_coll else 1.0
                    acc["dti_trench"] += w * float(s[excl].sum())
                    act = s[~excl]
                    opt = float(act.sum())
                    acc["qe_collected"] += w * opt * e
                    acc["si_recomb"] += w * opt * (1.0 - e)
                    if pixidx is not None:
                        for p, L in enumerate(labels):
                            if L in "RGB":
                                m = (pixidx == p) & (~excl)
                                col = float(s[m].sum())
                                chan[L]["qe"] += w * col * e
                                chan[L]["recomb"] += w * col * (1.0 - e)
                        for p, L in enumerate(labels):
                            if L in "RGB":
                                chan[L]["trench"] += w * float(s[(pixidx == p) & excl].sum())
            else:
                tg = tags[li] if li < len(tags) else str(li)
                above[tg] = above.get(tg, 0.0) + w * float(dens.sum())

    out = {"wavelength_nm": int(wavelength_nm),
           "reflect": round(acc["reflect"], 5),
           "transmit_deep": round(acc["transmit_deep"], 5),
           "above_si": {k: round(v, 5) for k, v in sorted(above.items(),
                                                          key=lambda x: -x[1]) if v > 1e-6},
           "dti_trench": round(acc["dti_trench"], 5),
           "si_recomb": round(acc["si_recomb"], 5),
           "qe_collected": round(acc["qe_collected"], 5),
           "by_channel": {c: {k: round(v, 5) for k, v in d.items()}
                          for c, d in chan.items()}}
    out["checksum"] = round(out["reflect"] + out["transmit_deep"] + out["dti_trench"]
                            + out["si_recomb"] + out["qe_collected"]
                            + sum(out["above_si"].values()), 5)
    return out
