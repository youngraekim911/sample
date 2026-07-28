# -*- coding: utf-8 -*-
"""CRA(주광선각) 렌즈-shift 필드 스윕.

센서 필드(0=중심 ~ 1.0=코너)마다 CRA 타겟이 있고, 빗각 입사로 초점이 틀어져 QE 가
떨어진다. 상부 구조(ML, CF+grid)를 빛 오는 쪽으로 shrink(=shift) 시켜 초점을 Si 중심
으로 되돌린다. shift = shrink_ratio(µm/deg) × CRA(field). ML 이 CF+grid 보다 크게 이동.

Si/DTI/BARL/검출기는 고정. shift 는 주기 wrap(np.roll)로 옆 unit 침범을 정확히 표현.
azimuth=0 -> 1D 반경(+x), 45 -> 2D 대각(코너). 방향은 빛 오는 쪽(마중); 부호는
shift_sign 으로 뒤집을 수 있다(필요시 교정).
"""
import math
import numpy as np


def cra_of_field(field, field_cra_deg):
    """field(0~1) -> CRA(deg). field_cra_deg 는 field 0..1 등간격 테이블."""
    tab = np.asarray(field_cra_deg, dtype=float)
    xs = np.linspace(0.0, 1.0, len(tab))
    return float(np.interp(field, xs, tab))


def _interp_table(field, table):
    """table = [[field, cra_deg], ...] (비등간격 허용) -> 선형보간 CRA."""
    t = np.asarray(table, float)
    return float(np.interp(field, t[:, 0], t[:, 1]))


def load_cra_spec(path):
    """제품 CRA 스펙 yaml 로드 — 사내 제품 파일 형식을 그대로 지원.

    형식(예):
        project_id: 제품명
        spec:
          module_cra: {columns:[field, image_height_um, cra_deg], data:[[...],...]}
          sensor_cra: *module_cra          # 다르면 별도 표
          ml_shrink_ratio: 0.0188e-6       # [m/deg]
          cf_shrink_ratio: 0.0085e-6
          f_number: 2.08
    module_cra = 빛의 실제 입사각(θ,φ 계산용), sensor_cra = ML/CF shift 크기용.
    sensor 가 없으면 module 로 대체. 보간은 선형(np.interp — 단조 표에 안전;
    poly6th 지정은 무시하고 선형 사용).
    반환: {module:[[f,cra]..], sensor:[[f,cra]..], ml_um_per_deg, cf_um_per_deg,
           f_number, product}
    """
    import yaml
    with open(path, encoding="utf-8") as f:
        doc = yaml.safe_load(f) or {}
    sp = doc.get("spec") or doc                       # spec 없이 평평해도 허용

    def table(node):
        if node is None:
            return None
        cols = [str(c).strip() for c in (node.get("columns") or [])]
        data = node.get("data") or []
        fi = cols.index("field") if "field" in cols else 0
        ci = cols.index("cra_deg") if "cra_deg" in cols else len(cols) - 1
        rows = [[float(r[fi]), float(r[ci])] for r in data]
        # field 가 촘촘(0.02 간격 등)하거나 행이 많아도/순서가 섞여도 안전:
        # np.interp 는 오름차순 필수 → 정렬 + 중복 field 는 마지막 값 사용
        rows.sort(key=lambda r: r[0])
        dedup = []
        for f, c in rows:
            if dedup and abs(dedup[-1][0] - f) < 1e-12:
                dedup[-1][1] = c
            else:
                dedup.append([f, c])
        return dedup

    mod = table(sp.get("module_cra"))
    if mod is None and sp.get("field_cra_deg"):       # 간이 형식(리스트)도 허용
        tab = [float(v) for v in sp["field_cra_deg"]]
        mod = [[i / (len(tab) - 1), v] for i, v in enumerate(tab)]
    if not mod:
        raise ValueError("module_cra 표가 없습니다 (columns/data 형식)")
    sen = table(sp.get("sensor_cra")) or mod          # 없으면 module 공용

    def ratio(*keys, default=0.0):
        for k in keys:
            if sp.get(k) is not None:
                v = float(sp[k])
                return v * 1e6 if abs(v) < 1e-3 else v   # m/deg -> µm/deg 자동 인식
        return default
    return {"product": str(doc.get("project_id", "")),
            "module": mod, "sensor": sen,
            "ml_um_per_deg": ratio("ml_shrink_ratio", "shrink_ml_um_per_deg"),
            "cf_um_per_deg": ratio("cf_shrink_ratio", "shrink_cfgrid_um_per_deg"),
            "f_number": float(sp.get("f_number", 2.0)),
            "shift_sign": float(sp.get("shift_sign", 1.0))}


def unit_qe_at_field(base_ir, spec, x_field, y_field, wavelengths_nm,
                     nG=101, downsample=2, materials_dir=None,
                     theta_samples=3, phi_samples=3,
                     on_solve=None, cancel=None):
    """센서 필드 (x,y) 한 지점의 unit QE — CRA shift + F# 원뿔 적분 + 편광 평균.

    x∈[-0.8,0.8], y∈[-0.6,0.6] 정규화 좌표(코너 r=1). 흐름:
      r=√(x²+y²) → module CRA(입사 θ)·sensor CRA(shift 크기) → ML/CF 를 빛 쪽으로
      shift → calc_mra 원뿔점 (θ_i,φ_i,w_i) → 각 점 TE/TM RCWA → 가중합.
    반환: {"cra_deg","azimuth_deg","shift_ml_um","shift_cf_um",
           "pixels":{wl: npx×npx list}, "rgb":{wl:{R,G,B}}, "labels": npx×npx}
    """
    from .cone import calc_mra
    from .simulator import RCWAPlaneWaveSimulator
    from ..structure.blocks import apply_cra_shift

    r = math.hypot(float(x_field), float(y_field))
    az = math.degrees(math.atan2(float(y_field), float(x_field))) if r > 1e-12 else 0.0
    cra_mod = _interp_table(r, spec["module"])        # 실제 입사각
    cra_sen = _interp_table(r, spec["sensor"])        # shift 산출용
    sign = float(spec.get("shift_sign", 1.0))
    mag_ml = sign * float(spec["ml_um_per_deg"]) * cra_sen
    mag_cf = sign * float(spec["cf_um_per_deg"]) * cra_sen
    azr = math.radians(az)
    ux, uy = -math.cos(azr), -math.sin(azr)           # 광원 쪽(전파 반대)으로 마중
    ir_f = apply_cra_shift(base_ir, shift_ml_um=(mag_ml * ux, mag_ml * uy),
                           shift_cfgrid_um=(mag_cf * ux, mag_cf * uy))
    sim = RCWAPlaneWaveSimulator(ir_f, nG=nG, downsample=downsample,
                                 materials_dir=materials_dir)
    det = ir_f.detector
    npx = int(round(math.sqrt(len(det.pixel_labels))))
    labels = [[det.pixel_labels[rr * npx + cc] for cc in range(npx)]
              for rr in range(npx)]

    mra = calc_mra(cra_mod, az, theta_samples, phi_samples,
                   f_number=float(spec.get("f_number", 2.0)))
    out_pix, out_rgb = {}, {}
    for w in wavelengths_nm:
        acc = np.zeros((npx, npx))
        rgb = {"R": 0.0, "G": 0.0, "B": 0.0}
        for th, ph, wt in zip(mra["theta_deg"], mra["phi_deg"], mra["weight"]):
            for pol in ((1.0, 0.0), (0.0, 1.0)):
                if cancel and cancel():
                    return None
                o = sim.run(w / 1000.0, theta=th, phi=ph,
                            pol_te=pol[0], pol_tm=pol[1])
                acc += 0.5 * wt * np.asarray(o["QE_pixels"], float).reshape(npx, npx)
                for L in "RGB":
                    rgb[L] += 0.5 * wt * float(o["QE_rgb"].get(L, 0.0))
                if on_solve:
                    on_solve()
        out_pix[int(w)] = [[round(float(v), 5) for v in row] for row in acc]
        out_rgb[int(w)] = {L: round(rgb[L], 5) for L in "RGB"}
    return {"cra_deg": round(cra_mod, 3), "azimuth_deg": round(az, 2),
            "shift_ml_um": round(mag_ml, 4), "shift_cf_um": round(mag_cf, 4),
            "pixels": out_pix, "rgb": out_rgb, "labels": labels}


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
        # F# 원뿔 적분 (theta/phi_samples 미지정=1×1 → 기존 단일각과 동일)
        from .cone import calc_mra
        mra = calc_mra(cra, math.degrees(az),
                       int(cra_cfg.get("theta_samples", 1)),
                       int(cra_cfg.get("phi_samples", 1)),
                       f_number=float(cra_cfg.get("f_number", 2.0)))
        qe = {}
        for w in wavelengths:
            acc = {"R": 0.0, "G": 0.0, "B": 0.0}
            for th, ph, wt in zip(mra["theta_deg"], mra["phi_deg"], mra["weight"]):
                for pol in ((1.0, 0.0), (0.0, 1.0)):     # 편광평균
                    o = sim.run(w / 1000.0, theta=th, phi=ph,
                                pol_te=pol[0], pol_tm=pol[1])
                    for L in "RGB":
                        acc[L] += 0.5 * wt * o["QE_rgb"][L]
            qe[w] = {L: round(acc[L], 4) for L in "RGB"}
        out[f] = {"cra_deg": round(cra, 2), "shift_ml_um": round(mag_ml, 4),
                  "shift_cf_um": round(mag_cf, 4), "qe": qe}
        if verbose:
            g = {w: qe[w]["G"] for w in wavelengths}
            print(f"field {f:.1f}  CRA={cra:5.2f}°  shift ML={mag_ml:+.3f} "
                  f"CF={mag_cf:+.3f}µm  QE_G={g}", flush=True)
    return out
