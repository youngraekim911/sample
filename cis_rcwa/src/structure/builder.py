#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CIS qcell 구조 빌더
    YAML 정의 -> 3D voxel 구조 -> npy 저장 (+ 복소굴절률, meta)

구조(아래->위):
    Si(PD) -> Metal grid(DTI) -> ARC -> Color Filter(격벽 사이 채움)
           -> ML spacer(평탄화) -> Microlens

배열 규약:
    matid  shape = [nz, ny, nx]   (z: 아래=0, 위로 증가 / 빛은 위에서 입사)
    index  shape = [nz, ny, nx]   complex64  (옵션)
"""
import os
import json
import argparse
import numpy as np
import yaml


# ----------------------------------------------------------------------------
# 유틸
# ----------------------------------------------------------------------------
def _um2vox(t_um, d_um):
    """두께(um) -> voxel 개수 (반올림, 최소 1)."""
    return max(1, int(round(t_um / d_um)))


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ----------------------------------------------------------------------------
# 빌더
# ----------------------------------------------------------------------------
class QcellBuilder:
    def __init__(self, cfg, base_dir="."):
        self.cfg = cfg
        self.base_dir = base_dir
        g = cfg["grid"]
        self.pitch = float(g["pixel_pitch_um"])
        self.n_pix = int(g["n_pixels"])
        self.dxy = float(g["dxy_um"])
        self.dz = float(g["dz_um"])
        self.air_margin = float(g.get("air_margin_um", 0.2))

        # 물질 id / 굴절률 테이블
        self.mat = cfg["materials"]
        self.id_of = {name: int(m["id"]) for name, m in self.mat.items()}
        self.nk_of = {int(m["id"]): (float(m["n"]), float(m["k"]))
                      for m in self.mat.values()}

        # 색이름 -> CF material 이름
        self.color2mat = {"R": "cf_red", "G": "cf_green", "B": "cf_blue"}

        # 가로/세로 격자
        self.span = self.pitch * self.n_pix           # 전체 lateral 크기 (um)
        self.nx = _um2vox(self.span, self.dxy)
        self.ny = _um2vox(self.span, self.dxy)
        # voxel 중심 좌표 (um)
        self.xc = (np.arange(self.nx) + 0.5) * self.dxy
        self.yc = (np.arange(self.ny) + 0.5) * self.dxy
        self.XX, self.YY = np.meshgrid(self.xc, self.yc, indexing="xy")  # [ny,nx]

        self.layer_bounds = []   # (name, z0_um, z1_um)

    # --- 격벽(grid wall) 마스크: 주기적 픽셀 경계에 폭 w 벽 ---
    def _wall_mask(self, w):
        u = np.mod(self.XX, self.pitch)
        v = np.mod(self.YY, self.pitch)
        half = w / 2.0
        mx = (u < half) | (u > self.pitch - half)
        my = (v < half) | (v > self.pitch - half)
        return mx | my   # [ny,nx] bool

    # --- 픽셀 (row,col) index 맵 ---
    def _pixel_index(self):
        ix = np.clip((self.XX / self.pitch).astype(int), 0, self.n_pix - 1)
        iy = np.clip((self.YY / self.pitch).astype(int), 0, self.n_pix - 1)
        return iy, ix   # each [ny,nx]

    # --- Color Filter 색상 맵 (픽셀별) ---
    def _cf_color_map(self):
        pat = self.cfg["bayer"]["pattern"]
        iy, ix = self._pixel_index()
        cmap = np.empty((self.ny, self.nx), dtype=np.uint8)
        for r in range(self.n_pix):
            for c in range(self.n_pix):
                color = pat[r][c]
                matname = self.color2mat[color]
                sel = (iy == r) & (ix == c)
                cmap[sel] = self.id_of[matname]
        return cmap

    # --- B(theta): 정규화 경계 반경 (이상형=1.0), reflow 변형 프로파일 ---
    @staticmethod
    def _eval_btheta(theta, b_samples):
        """theta[-pi..pi] 에서 주기적 선형보간으로 B(theta) 평가."""
        b = np.asarray(b_samples, dtype=float)
        n = len(b)
        ang = np.linspace(0.0, 2 * np.pi, n, endpoint=False)
        xp = np.concatenate([ang, [2 * np.pi]])
        fp = np.concatenate([b, [b[0]]])
        return np.interp(np.mod(theta, 2 * np.pi), xp, fp)

    @staticmethod
    def _ctrl_to_btheta(ctrl, n=128):
        """control point [[angle_deg, radius_mult], ...] -> B(theta) n-samples."""
        c = sorted(((float(np.deg2rad(a)) % (2 * np.pi)), float(r)) for a, r in ctrl)
        angs = np.array([a for a, _ in c])
        rs = np.array([r for _, r in c])
        # 주기 확장 후 보간
        xp = np.concatenate([angs - 2 * np.pi, angs, angs + 2 * np.pi])
        fp = np.concatenate([rs, rs, rs])
        out = np.linspace(0.0, 2 * np.pi, n, endpoint=False)
        return np.interp(out, xp, fp)

    def _load_deformations(self):
        """quads 레이아웃 + 렌즈별 B(theta) 변형 로드 (YAML inline + 외부 JSON)."""
        ml = self.cfg["microlens_layout"]
        quads = ml["quads"]
        deform_raw = dict(ml.get("deformations", {}) or {})

        # 외부 파일 (HTML 에디터 export). quads 도 포함하면 우선 적용.
        f = ml.get("deform_file")
        if f:
            path = f if os.path.isabs(f) else os.path.join(self.base_dir, f)
            with open(path, encoding="utf-8") as fh:
                data = json.load(fh)
            if data.get("quads"):
                quads = data["quads"]
            deform_raw.update(data.get("deformations", {}) or {})

        # 각 항목을 B(theta) 배열로 정규화
        deform = {}
        for lid, spec in deform_raw.items():
            if not spec:
                continue
            if "b_theta" in spec:
                deform[lid] = np.asarray(spec["b_theta"], dtype=float)
            elif "ctrl" in spec:
                deform[lid] = self._ctrl_to_btheta(spec["ctrl"])
        return quads, deform

    # --- 렌즈 목록 열거: (id, cx, cy, ax, ay) / 이상형 반축 ---
    def _lens_list(self, quads):
        p = self.pitch
        lenses = []
        for qy in range(2):
            for qx in range(2):
                spec = quads[qy][qx]
                shape = spec["shape"]
                orient = spec.get("orient", "h")
                x0, y0 = qx * 2 * p, qy * 2 * p
                cx0, cy0 = x0 + p, y0 + p
                if shape == "2x2":
                    lenses.append((f"q{qy}{qx}_2x2_0", cx0, cy0, p, p))
                elif shape == "1x1":
                    for dyp in (0, 1):
                        for dxp in (0, 1):
                            k = dyp * 2 + dxp
                            lenses.append((f"q{qy}{qx}_1x1_{k}",
                                           x0 + (dxp + 0.5) * p,
                                           y0 + (dyp + 0.5) * p, p / 2, p / 2))
                elif shape == "2x1":
                    if orient == "h":     # 가로 타원 2p x p, 위/아래 2개
                        for dyp in (0, 1):
                            lenses.append((f"q{qy}{qx}_2x1_{dyp}",
                                           cx0, y0 + (dyp + 0.5) * p, p, p / 2))
                    else:                 # 세로 타원 p x 2p, 좌/우 2개
                        for dxp in (0, 1):
                            lenses.append((f"q{qy}{qx}_2x1_{dxp}",
                                           x0 + (dxp + 0.5) * p, cy0, p / 2, p))
                else:
                    raise ValueError(f"unknown ML shape: {shape}")
        return lenses

    # --- ML dome 표면(sag) 계산: 변형 base + volume 보존 ---
    def _build_microlens(self):
        h0 = float(self.cfg["layers"]["microlens"]["sag_height_um"])
        quads, deform = self._load_deformations()
        lenses = self._lens_list(quads)
        sag = np.zeros((self.ny, self.nx), dtype=np.float64)
        dA = self.dxy * self.dxy
        info = []
        for (lid, cx, cy, ax, ay) in lenses:
            U = (self.XX - cx) / ax
            V = (self.YY - cy) / ay
            rho = np.sqrt(U * U + V * V)
            theta = np.arctan2(V, U)
            if lid in deform:
                B = self._eval_btheta(theta, deform[lid])
            else:
                B = np.ones_like(rho)
            inside = rho <= B
            prof = np.zeros_like(rho)
            ratio = rho[inside] / B[inside]
            prof[inside] = np.sqrt(np.clip(1.0 - ratio * ratio, 0.0, 1.0))
            # volume 보존: 이상 반타원체 부피 V0 = h0 * (2/3)*pi*ax*ay
            unit_vol = prof.sum() * dA           # 단위높이(h=1) footprint 부피
            V0 = h0 * (2.0 / 3.0) * np.pi * ax * ay
            h = (V0 / unit_vol) if unit_vol > 0 else h0
            np.maximum(sag, prof * h, out=sag)
            info.append({"id": lid, "cx": round(float(cx), 4), "cy": round(float(cy), 4),
                         "ax": ax, "ay": ay, "height_um": round(float(h), 4),
                         "deformed": lid in deform})
        return sag, float(sag.max()), info

    # ------------------------------------------------------------------
    def build(self):
        L = self.cfg["layers"]

        # z 두께 계산
        nz_si     = _um2vox(L["substrate_si"]["thickness_um"], self.dz)
        nz_grid   = _um2vox(L["metal_grid"]["thickness_um"], self.dz)
        nz_arc    = _um2vox(L["arc"]["thickness_um"], self.dz)
        nz_cf     = _um2vox(L["color_filter"]["thickness_um"], self.dz)
        nz_spacer = _um2vox(L["ml_spacer"]["thickness_um"], self.dz)
        sag, h_max, self.ml_info = self._build_microlens()
        nz_ml     = _um2vox(h_max, self.dz)
        nz_air    = _um2vox(self.air_margin, self.dz)
        nz = nz_si + nz_grid + nz_arc + nz_cf + nz_spacer + nz_ml + nz_air

        matid = np.full((nz, self.ny, self.nx), self.id_of["air"], dtype=np.uint8)

        # 미리 계산
        wall_grid = self._wall_mask(float(L["metal_grid"]["wall_width_um"]))
        wall_cf   = self._wall_mask(float(L["color_filter"]["wall_width_um"]))
        cf_color  = self._cf_color_map()

        z = 0
        zb = []  # (name, z0, z1) in voxel

        # 1) Si substrate
        matid[z:z+nz_si] = self.id_of[L["substrate_si"]["fill"]]
        zb.append(("substrate_si", z, z+nz_si)); z += nz_si

        # 2) Metal grid (DTI): 벽=metal, 내부=fill
        blk = np.where(wall_grid, self.id_of[L["metal_grid"]["wall"]],
                       self.id_of[L["metal_grid"]["fill"]]).astype(np.uint8)
        matid[z:z+nz_grid] = blk[None, :, :]
        zb.append(("metal_grid", z, z+nz_grid)); z += nz_grid

        # 3) ARC (균일)
        matid[z:z+nz_arc] = self.id_of[L["arc"]["fill"]]
        zb.append(("arc", z, z+nz_arc)); z += nz_arc

        # 4) Color Filter: 벽=metal, 사이=CF 색상
        blk = np.where(wall_cf, self.id_of[L["color_filter"]["wall"]],
                       cf_color).astype(np.uint8)
        matid[z:z+nz_cf] = blk[None, :, :]
        zb.append(("color_filter", z, z+nz_cf)); z += nz_cf

        # 5) ML spacer (평탄화, 균일)
        matid[z:z+nz_spacer] = self.id_of[L["ml_spacer"]["fill"]]
        zb.append(("ml_spacer", z, z+nz_spacer)); z += nz_spacer

        # 6) Microlens: sag 표면 아래를 ml 로 채움
        ml_id  = self.id_of[L["microlens"]["material"]]
        z_ml0  = z
        # 각 (y,x) 에서 렌즈 높이 만큼 z 채우기
        sag_vox = np.round(sag / self.dz).astype(int)      # [ny,nx]
        sag_vox = np.clip(sag_vox, 0, nz_ml)
        # z 인덱스 벡터화 채움
        zloc = np.arange(nz_ml)[:, None, None]             # [nz_ml,1,1]
        fill = zloc < sag_vox[None, :, :]                  # [nz_ml,ny,nx]
        sub = matid[z_ml0:z_ml0+nz_ml]
        sub[fill] = ml_id
        matid[z_ml0:z_ml0+nz_ml] = sub
        zb.append(("microlens", z, z+nz_ml)); z += nz_ml

        # 7) air margin (이미 air)
        zb.append(("air_margin", z, z+nz_air)); z += nz_air

        self.layer_bounds = zb
        self.matid = matid
        self.nz = nz
        return matid

    # ------------------------------------------------------------------
    def to_index(self, dtype=np.complex64):
        """matid -> 복소굴절률 n + ik 배열."""
        idx = np.zeros(self.matid.shape, dtype=dtype)
        for mid, (n, k) in self.nk_of.items():
            idx[self.matid == mid] = complex(n, k)
        return idx

    def meta(self):
        return {
            "shape_zyx": [int(self.nz), int(self.ny), int(self.nx)],
            "voxel_um": {"dz": self.dz, "dy": self.dxy, "dx": self.dxy},
            "lateral_span_um": self.span,
            "pixel_pitch_um": self.pitch,
            "n_pixels": self.n_pix,
            "materials": {name: {"id": int(m["id"]), "n": m["n"], "k": m["k"]}
                          for name, m in self.mat.items()},
            "layer_bounds_vox": [
                {"name": n, "z0": int(a), "z1": int(b),
                 "z0_um": round(a*self.dz, 4), "z1_um": round(b*self.dz, 4)}
                for (n, a, b) in self.layer_bounds],
            "microlenses": getattr(self, "ml_info", []),
            "axis_note": "matid[z,y,x]; z=0 bottom(Si), light incident from top(+z)",
        }


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="CIS qcell -> npy builder")
    ap.add_argument("-c", "--config", default="conf/qcell_config.yaml")
    ap.add_argument("-o", "--outdir", default=".")
    args = ap.parse_args()

    cfg = load_config(args.config)
    b = QcellBuilder(cfg, base_dir=os.path.dirname(os.path.abspath(args.config)))
    matid = b.build()
    ex = cfg["export"]
    os.makedirs(args.outdir, exist_ok=True)

    p_mat = os.path.join(args.outdir, ex["matid_npy"])
    np.save(p_mat, matid)
    print(f"[saved] {p_mat}  shape={matid.shape} dtype={matid.dtype} "
          f"({matid.nbytes/1e6:.1f} MB)")

    meta = b.meta()
    p_meta = os.path.join(args.outdir, ex["meta_json"])
    with open(p_meta, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"[saved] {p_meta}")

    if ex.get("write_index", False):
        idx = b.to_index()
        p_idx = os.path.join(args.outdir, ex["index_npy"])
        np.save(p_idx, idx)
        print(f"[saved] {p_idx}  shape={idx.shape} dtype={idx.dtype} "
              f"({idx.nbytes/1e6:.1f} MB)")

    # 요약
    print("\n[layer z-bounds]")
    for lb in meta["layer_bounds_vox"]:
        print(f"  {lb['name']:14s}  z {lb['z0_um']:.3f} -> {lb['z1_um']:.3f} um")
    uniq, cnt = np.unique(matid, return_counts=True)
    id2name = {int(m['id']): n for n, m in cfg['materials'].items()}
    print("\n[material voxel fraction]")
    tot = matid.size
    for u, c in zip(uniq, cnt):
        print(f"  {id2name.get(int(u), '?'):9s} id={int(u)}  {100*c/tot:5.1f}%")


if __name__ == "__main__":
    main()
