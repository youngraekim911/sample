# -*- coding: utf-8 -*-
"""위저드가 뽑은 npy 구조 확인 뷰어.

    python3 view_npy.py -i <product>_matid.npy [-m <product>_meta.json] [-o view.png]

npy 규약: uint8 물질 id, shape [nz, ny, nx] — z=0 이 바닥(Si), 빛은 +z 위에서 입사.
meta.json 의 materials 로 id -> 이름/색을 매핑해
XZ 단면 / YZ 단면 / z-레벨별 XY 평면 / 층 점유율을 한 장에 그림.
"""
import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

# 위저드(structure_wizard.html matColor)와 동일 팔레트
COLORS = {
    "air": "#e9f1fb", "si": "#3b3b3b", "oxide": "#bcd4e6", "grid_metal": "#141414",
    "arl_hi": "#8b6fd0", "arl_lo": "#a9c7d6", "cf_red": "#e24a4a", "cf_green": "#3fb34f",
    "cf_blue": "#3f6fd0", "ml": "#f2c14e", "ml_arl": "#d9b3ff",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--npy", required=True, help="<product>_matid.npy")
    ap.add_argument("-m", "--meta", default=None, help="<product>_meta.json (기본: npy 옆 _meta.json)")
    ap.add_argument("-o", "--out", default=None, help="출력 png (기본: npy 옆 _view.png)")
    a = ap.parse_args()

    vol = np.load(a.npy)                       # [nz, ny, nx]
    meta_path = a.meta or a.npy.replace("_matid.npy", "_meta.json")
    names = {}
    dz = dxy = None
    if os.path.exists(meta_path):
        meta = json.load(open(meta_path))
        names = {int(m["id"]): m["name"] for m in meta["materials"]}
        dz = meta["voxel_um"]["dz"]; dxy = meta["voxel_um"]["dx"]
    else:
        names = {i: f"id{i}" for i in range(int(vol.max()) + 1)}
    nz, ny, nx = vol.shape
    ez = [0, nx * (dxy or 1), 0, nz * (dz or 1)]     # 단면 extent (um)
    exy = [0, nx * (dxy or 1), 0, ny * (dxy or 1)]

    nid = int(max(names)) + 1
    cmap = ListedColormap([COLORS.get(names.get(i, ""), "#888888") for i in range(nid)])

    # 물질이 등장하는 z 대역 (air 제외) -> XY 대표 z 3곳
    occ = [(z, set(np.unique(vol[z]))) for z in range(nz)]
    z_solid = [z for z, s in occ if s - {0}]
    z_lo, z_hi = z_solid[0], z_solid[-1]
    z_picks = [int(z_lo + f * (z_hi - z_lo)) for f in (0.25, 0.62, 0.9)]

    # 단면 위치: 정확히 중앙이면 grid 벽/DTI 를 따라 잘려 metal 만 보일 수 있음
    # -> 중앙 부근에서 '물질 종류가 가장 다양한' 슬라이스 자동 선택
    def best_slice(axis):
        n = ny if axis == "y" else nx
        lo, hi = n // 4, 3 * n // 4
        cand = range(lo, hi, max(1, (hi - lo) // 64))
        def nuniq(c):
            sl = vol[:, c, :] if axis == "y" else vol[:, :, c]
            return len(np.unique(sl))
        return max(cand, key=nuniq)

    yc, xc = best_slice("y"), best_slice("x")
    fig, ax = plt.subplots(2, 3, figsize=(15, 9))
    ax[0, 0].imshow(vol[:, yc, :], origin="lower", cmap=cmap, vmin=0, vmax=nid - 1,
                    extent=ez, aspect="auto", interpolation="nearest")
    ax[0, 0].set_title(f"XZ section @ y={yc * (dxy or 1):.2f}um   shape z{nz}×y{ny}×x{nx}")
    ax[0, 1].imshow(vol[:, :, xc], origin="lower", cmap=cmap, vmin=0, vmax=nid - 1,
                    extent=ez, aspect="auto", interpolation="nearest")
    ax[0, 1].set_title(f"YZ section @ x={xc * (dxy or 1):.2f}um")
    # 층 점유율 스택 (z-프로파일): 전 구조가 들어있는지 한눈에
    fr = np.zeros((nz, nid))
    for z in range(nz):
        ids, cnt = np.unique(vol[z], return_counts=True)
        fr[z, ids] = cnt / (ny * nx)
    zs = (np.arange(nz) + 0.5) * (dz or 1)
    bottom = np.zeros(nz)
    for i in range(nid):
        if fr[:, i].max() <= 0:
            continue
        ax[0, 2].fill_betweenx(zs, bottom, bottom + fr[:, i],
                               color=COLORS.get(names.get(i, ""), "#888"), lw=0)
        bottom += fr[:, i]
    ax[0, 2].set_title("z-occupancy per material")
    ax[0, 2].set_xlabel("fraction"); ax[0, 2].set_ylabel("z (um)")
    ax[0, 2].set_xlim(0, 1); ax[0, 2].set_ylim(0, zs[-1])
    for k, zp in enumerate(z_picks):
        ax[1, k].imshow(vol[zp], origin="lower", cmap=cmap, vmin=0, vmax=nid - 1,
                        extent=exy, interpolation="nearest")
        mats = ", ".join(sorted(names.get(i, "?") for i in np.unique(vol[zp])))
        ax[1, k].set_title(f"XY @ z={zp * (dz or 1):.2f}um  [{mats}]", fontsize=9)
    for axx in ax.flat:
        axx.tick_params(labelsize=8)
    used = sorted(set(np.unique(vol)))
    fig.legend(handles=[Patch(color=COLORS.get(names.get(i, ""), "#888"), label=f"{i}:{names.get(i, '?')}")
                        for i in used], loc="lower center", ncol=min(len(used), 11), fontsize=8)
    fig.suptitle(os.path.basename(a.npy), fontsize=11)
    fig.tight_layout(rect=[0, 0.06, 1, 0.97])
    out = a.out or a.npy.replace("_matid.npy", "_view.png")
    fig.savefig(out, dpi=110)
    print(f"[view_npy] {out}")
    # 검사 요약
    print(f"  materials in npy: {[names.get(i, i) for i in used]}")
    for z, s in occ:
        pass
    bands = []
    prev = None
    for z in range(nz):
        sig = tuple(sorted(np.unique(vol[z]).tolist()))
        if sig != prev:
            bands.append((z, sig)); prev = sig
    for z, sig in bands:
        print(f"  z={z:4d} ({z * (dz or 1):6.2f} um): " + ", ".join(names.get(i, str(i)) for i in sig))


if __name__ == "__main__":
    main()
