# -*- coding: utf-8 -*-
"""CRA(주광선각) 렌즈-shift 필드 스윕.

센서 필드(0=중심 ~ 1.0=코너)마다 CRA 타겟이 있고, 빗각 입사로 초점이 틀어져 QE 가
떨어진다. 상부 구조(ML, CF+grid)를 빛 오는 쪽으로 shrink(=shift) 시켜 초점을 Si 중심
으로 되돌린다. shift = shrink_ratio(µm/deg) × CRA(field). ML 이 CF+grid 보다 크게 이동.

Si/DTI/BARL/검출기는 고정. shift 는 주기 wrap(np.roll)로 옆 unit 침범을 정확히 표현.
azimuth=0 -> 1D 반경(+x), 45 -> 2D 대각(코너). 방향은 빛 오는 쪽(마중); 부호는
shift_sign 으로 뒤집을 수 있다(실측으로 교정).
"""
import math
import numpy as np


def cra_of_field(field, field_cra_deg):
    """field(0~1) -> CRA(deg). field_cra_deg 는 field 0..1 등간격 테이블."""
    tab = np.asarray(field_cra_deg, dtype=float)
    xs = np.linspace(0.0, 1.0, len(tab))
    return float(np.interp(field, xs, tab))


def field_qe_sweep(base_ir, cra_cfg, wavelengths, fields=None,
                   materials_dir=None, nG=161, downsample=2, verbose=True):
    """필드별 QE 스윕. base_ir 는 ir_from_wizard_cfg 산출(layer_tags 필수).

    cra_cfg 키: field_cra_deg[list], shrink_ml_um_per_deg, shrink_cfgrid_um_per_deg,
                azimuth_deg(0=1D/45=대각), shift_sign(+1/-1).
    반환: {field: {"cra_deg":.., "shift_ml_um":.., "shift_cf_um":..,
                   "qe": {wl_nm: {"R":..,"G":..,"B":..}}}}
    """
    from .simulator import RCWAPlaneWaveSimulator
    from ..structure.blocks import apply_cra_shift

    if fields is None:
        fields = [round(0.1 * i, 1) for i in range(11)]
    tab = cra_cfg["field_cra_deg"]
    s_ml = float(cra_cfg.get("shrink_ml_um_per_deg", 0.0))
    s_cf = float(cra_cfg.get("shrink_cfgrid_um_per_deg", 0.0))
    az = math.radians(float(cra_cfg.get("azimuth_deg", 0.0)))
    sign = float(cra_cfg.get("shift_sign", 1.0))
    # 입사 규약: kx0=sin(θ)cos(φ) -> φ 방향으로 '전파'(=광원은 반대편 -φ쪽).
    # shrink 는 광원 쪽으로 마중(빛 오는 방향). 따라서 shift 는 전파의 반대방향.
    # sign=+1(기본)=광원쪽(물리적 정답), -1=반대(교정용).
    ux, uy = -math.cos(az), -math.sin(az)                # 광원 방향 단위벡터

    out = {}
    for f in fields:
        cra = cra_of_field(f, tab)
        mag_ml = sign * s_ml * cra
        mag_cf = sign * s_cf * cra
        shift_ml = (mag_ml * ux, mag_ml * uy)
        shift_cf = (mag_cf * ux, mag_cf * uy)
        ir_f = apply_cra_shift(base_ir, shift_ml_um=shift_ml, shift_cfgrid_um=shift_cf)
        sim = RCWAPlaneWaveSimulator(ir_f, nG=nG, downsample=downsample,
                                     materials_dir=materials_dir)
        qe = {}
        for w in wavelengths:
            acc = {"R": 0.0, "G": 0.0, "B": 0.0}
            for pol in ((1.0, 0.0), (0.0, 1.0)):         # 편광평균
                o = sim.run(w / 1000.0, theta=cra, phi=math.degrees(az),
                            pol_te=pol[0], pol_tm=pol[1])
                for L in "RGB":
                    acc[L] += 0.5 * o["QE_rgb"][L]
            qe[w] = {L: round(acc[L], 4) for L in "RGB"}
        out[f] = {"cra_deg": round(cra, 2), "shift_ml_um": round(mag_ml, 4),
                  "shift_cf_um": round(mag_cf, 4), "qe": qe}
        if verbose:
            g = {w: qe[w]["G"] for w in wavelengths}
            print(f"field {f:.1f}  CRA={cra:5.2f}°  shift ML={mag_ml:+.3f} "
                  f"CF={mag_cf:+.3f}µm  QE_G={g}", flush=True)
    return out
