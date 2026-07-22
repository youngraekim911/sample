# -*- coding: utf-8 -*-
"""구조 린터 — 누가 구조(yaml)를 바꿔도 RCWA 로 안전하게 연결되도록,
블록 조립 전에 흔한 실수를 사람이 읽을 수 있는 메시지로 잡는다.

errors  : IR 조립/RCWA 가 깨지거나 무의미한 결과가 되는 치명 문제 (실행 차단)
warnings: 계산은 되지만 의도와 다를 수 있는 것 (실행 허용, 안내)

lint_wizard_cfg(cfg) -> {"errors": [...], "warnings": [...]}
assert_wizard_cfg(cfg) : errors 있으면 StructureError 로 즉시 중단
"""


class StructureError(ValueError):
    """구조 정의가 RCWA 로 연결 불가 — 메시지에 무엇을/어디를 고칠지 담는다."""


_COLORS = {"R", "G", "B"}


def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def lint_wizard_cfg(cfg, material_names=None):
    """wizard-v3 스키마 구조를 점검. material_names(set) 주면 물질 존재도 검사."""
    E, W = [], []
    err = E.append
    warn = W.append

    if not isinstance(cfg, dict):
        return {"errors": ["구조가 dict(yaml) 가 아닙니다"], "warnings": []}

    g = cfg.get("grid") or {}
    pitch = _num(g.get("pixel_pitch_um"))
    npx = g.get("n_pixels")
    if not pitch or pitch <= 0:
        err("grid.pixel_pitch_um 이 없거나 0 이하입니다 (픽셀 피치 µm).")
    try:
        npx = int(npx)
        assert npx >= 1
    except (TypeError, ValueError, AssertionError):
        err("grid.n_pixels 가 1 이상의 정수가 아닙니다.")
        npx = None

    bayer = cfg.get("bayer")
    if npx:
        if not (isinstance(bayer, list) and len(bayer) == npx
                and all(isinstance(r, list) and len(r) == npx for r in bayer)):
            err(f"bayer 는 {npx}×{npx} 배열이어야 합니다 (grid.n_pixels 와 일치).")
        else:
            bad = {c for row in bayer for c in row} - _COLORS
            if bad:
                err(f"bayer 에 알 수 없는 색: {sorted(bad)} (R/G/B 만 허용).")

    st = cfg.get("stack") or {}
    if not st:
        err("stack 이 없습니다.")
        return {"errors": E, "warnings": W}

    # --- Si ---
    si = st.get("si") or {}
    if not si.get("material"):
        err("stack.si.material 이 없습니다 (기판 물질).")
    if not (_num(si.get("thickness_um")) or 0) > 0:
        err("stack.si.thickness_um 이 0 이하입니다 (광다이오드 두께).")

    # --- DTI (선택) ---
    d = st.get("dti")
    if d and (d.get("mode") or "").lower() not in ("", "none"):
        wdti = _num(d.get("width_um")) or 0
        if wdti <= 0:
            err("stack.dti.width_um 이 0 이하입니다.")
        elif pitch and wdti >= pitch:
            err(f"stack.dti.width_um({wdti}) 가 픽셀 피치({pitch}) 이상 — 픽셀이 사라집니다.")

    # --- BARL (선택, 다층) ---
    for i, l in enumerate(st.get("barl") or []):
        if not l.get("material"):
            err(f"stack.barl[{i}].material 이 없습니다.")
        if not (_num(l.get("thickness_um")) or 0) > 0:
            err(f"stack.barl[{i}].thickness_um 이 0 이하입니다.")

    # --- Grid ---
    gr = st.get("grid") or {}
    gw = _num(gr.get("width_um")) or 0
    if gw <= 0:
        err("stack.grid.width_um 이 0 이하입니다 (격벽 폭).")
    elif pitch and gw >= pitch * (gr.get("pitch", 1) or 1):
        warn(f"stack.grid.width_um({gw}) 가 셀 폭에 가깝습니다 — CF 공간이 매우 좁아집니다.")
    gstack = gr.get("stack") or []
    if not gstack:
        err("stack.grid.stack 이 비었습니다 (격벽 물질 층 최소 1개 필요).")
    for i, l in enumerate(gstack):
        if not l.get("material"):
            err(f"stack.grid.stack[{i}].material 이 없습니다.")
        if not (_num(l.get("height_um")) or 0) > 0:
            err(f"stack.grid.stack[{i}].height_um 이 0 이하입니다.")

    # --- CF ---
    cf = st.get("cf") or {}
    for c in ("R", "G", "B"):
        f = cf.get(c) or {}
        if not f.get("material"):
            err(f"stack.cf.{c}.material 이 없습니다.")
        if not (_num(f.get("thickness_um")) or 0) > 0:
            err(f"stack.cf.{c}.thickness_um 이 0 이하입니다.")
        for k in ("scale_x", "scale_y"):
            s = _num(f.get(k))
            if s is not None and not (0 < s <= 1.5):
                warn(f"stack.cf.{c}.{k}={s} 가 (0,1.5] 밖 — 0~1 권장(1=셀 가득).")
    if isinstance(bayer, list):
        used_c = {c for row in bayer for c in (row if isinstance(row, list) else [])}
        for c in used_c & _COLORS:
            if c not in cf:
                err(f"bayer 는 {c} 를 쓰는데 stack.cf.{c} 정의가 없습니다.")

    # --- ML ---
    ml = st.get("ml") or {}
    if not ml.get("material"):
        err("stack.ml.material 이 없습니다 (마이크로렌즈/평탄층 물질).")
    if (_num(ml.get("planar_um")) or 0) < 0:
        err("stack.ml.planar_um 이 음수입니다.")
    q = ml.get("quads")
    has_h = (_num(ml.get("height_um")) or 0) > 0
    has_lens = bool(ml.get("lenses"))
    if not has_lens:
        if not (isinstance(q, list) and len(q) == 2
                and all(isinstance(r, list) and len(r) == 2 for r in q)):
            err("stack.ml.quads 가 2×2 배열이 아닙니다 (또는 lenses 를 직접 지정).")
        else:
            ok_sh = {"1x1", "2x2", "2x1", "1x2"}
            for a in range(2):
                for b in range(2):
                    sp = q[a][b] or {}
                    if sp.get("shape") not in ok_sh:
                        err(f"stack.ml.quads[{a}][{b}].shape={sp.get('shape')!r} "
                            f"— {sorted(ok_sh)} 중 하나여야 합니다.")
        if not has_h and not (_num(ml.get("hr")) or 0) > 0 \
                and not (_num(ml.get("sag_height_um")) or 0) > 0:
            warn("ML 돔 높이 근거(height_um/hr/sag_height_um)가 모두 없음 — 기본 hr 로 추정.")

    # --- ARL top (선택) ---
    arl = st.get("arl_top")
    if arl and not arl.get("material") and (_num(arl.get("thickness_um")) or 0) > 0:
        err("stack.arl_top.thickness_um>0 인데 material 이 없습니다.")

    # --- 물질 존재 (folder/embedded) ---
    if material_names is not None:
        names = set(material_names)
        embedded = set((cfg.get("materials") or {}).keys())
        avail = names | embedded | {cfg.get("ambient") or "air", "air"}
        need = set()
        need.add(si.get("material"))
        if d:
            need.update([d.get("liner"), d.get("fill")])
            for l in (d.get("liners") or []):
                need.add(l.get("material"))
        for l in st.get("barl") or []:
            need.add(l.get("material"))
        need.add(gr.get("coat_material"))
        for l in gstack:
            need.add(l.get("material"))
        for c in ("R", "G", "B"):
            need.add((cf.get(c) or {}).get("material"))
        need.add(ml.get("material"))
        if arl:
            need.add(arl.get("material"))
        missing = sorted(n for n in need if n and n not in avail)
        if missing:
            err(f"물질 폴더/정의에 없는 물질: {missing} — data/materials/<이름>.txt 로 넣으세요.")

    return {"errors": E, "warnings": W}


def assert_wizard_cfg(cfg, material_names=None):
    r = lint_wizard_cfg(cfg, material_names)
    if r["errors"]:
        raise StructureError(
            "구조 정의 오류 (RCWA 연결 불가):\n  - " + "\n  - ".join(r["errors"]))
    return r["warnings"]
