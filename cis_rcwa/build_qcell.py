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
    with open(path, "r") as f:
        return yaml.safe_load(f)


# ----------------------------------------------------------------------------
# 빌더
# ----------------------------------------------------------------------------
class QcellBuilder:
    def __init__(self, cfg):
        self.cfg = cfg
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

    # --- 각 quad ML dome 표면(sag) 계산 -> [ny,nx] sag(um), <=0 는 렌즈 없음 ---
    def _microlens_sag(self):
        h = float(self.cfg["layers"]["microlens"]["sag_height_um"])
        quads = self.cfg["microlens_layout"]["quads"]
        sag = np.zeros((self.ny, self.nx), dtype=np.float64)
        p = self.pitch

        def add_dome(cx, cy, ax, ay):
            rho2 = ((self.XX - cx) / ax) ** 2 + ((self.YY - cy) / ay) ** 2
            inside = rho2 <= 1.0
            s = np.zeros_like(sag)
            s[inside] = h * np.sqrt(np.clip(1.0 - rho2[inside], 0.0, 1.0))
            np.maximum(sag, s, out=sag)

        for qy in range(2):
            for qx in range(2):
                spec = quads[qy][qx]
                shape = spec["shape"]
                orient = spec.get("orient", "h")
                # quad 좌하단 픽셀 origin, quad 는 2x2 pixel = 2p x 2p
                x0 = qx * 2 * p
                y0 = qy * 2 * p
                cx0, cy0 = x0 + p, y0 + p   # quad 중심

                if shape == "2x2":
                    add_dome(cx0, cy0, p, p)                       # 원형 R=p
                elif shape == "1x1":
                    for dyp in (0, 1):
                        for dxp in (0, 1):
                            cx = x0 + (dxp + 0.5) * p
                            cy = y0 + (dyp + 0.5) * p
                            add_dome(cx, cy, p / 2, p / 2)         # 원형 R=p/2
                elif shape == "2x1":
                    if orient == "h":   # 가로 타원 2p x p, 위/아래 2개
                        for dyp in (0, 1):
                            cy = y0 + (dyp + 0.5) * p
                            add_dome(cx0, cy, p, p / 2)
                    else:               # 세로 타원 p x 2p, 좌/우 2개
                        for dxp in (0, 1):
                            cx = x0 + (dxp + 0.5) * p
                            add_dome(cx, cy0, p / 2, p)
                else:
                    raise ValueError(f"unknown ML shape: {shape}")
        return sag, h

    # ------------------------------------------------------------------
    def build(self):
        L = self.cfg["layers"]

        # z 두께 계산
        nz_si     = _um2vox(L["substrate_si"]["thickness_um"], self.dz)
        nz_grid   = _um2vox(L["metal_grid"]["thickness_um"], self.dz)
        nz_arc    = _um2vox(L["arc"]["thickness_um"], self.dz)
        nz_cf     = _um2vox(L["color_filter"]["thickness_um"], self.dz)
        nz_spacer = _um2vox(L["ml_spacer"]["thickness_um"], self.dz)
        sag, h    = self._microlens_sag()
        nz_ml     = _um2vox(h, self.dz)
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
            "axis_note": "matid[z,y,x]; z=0 bottom(Si), light incident from top(+z)",
        }


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="CIS qcell -> npy builder")
    ap.add_argument("-c", "--config", default="qcell_config.yaml")
    ap.add_argument("-o", "--outdir", default=".")
    args = ap.parse_args()

    cfg = load_config(args.config)
    b = QcellBuilder(cfg)
    matid = b.build()
    ex = cfg["export"]
    os.makedirs(args.outdir, exist_ok=True)

    p_mat = os.path.join(args.outdir, ex["matid_npy"])
    np.save(p_mat, matid)
    print(f"[saved] {p_mat}  shape={matid.shape} dtype={matid.dtype} "
          f"({matid.nbytes/1e6:.1f} MB)")

    meta = b.meta()
    p_meta = os.path.join(args.outdir, ex["meta_json"])
    with open(p_meta, "w") as f:
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
