#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
qcell npy 구조 시각 확인
    - XZ / YZ 수직 단면 (렌즈 dome, 층, 격벽 확인)
    - ML 표면 top-view (1x1 / 2x1 / 2x2 배치 확인)
    - CF 색 배치 top-view
"""
import os
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm


def load(outdir, meta_name="qcell_meta.json", mat_name="qcell_matid.npy"):
    with open(os.path.join(outdir, meta_name)) as f:
        meta = json.load(f)
    matid = np.load(os.path.join(outdir, mat_name))
    return matid, meta


# 물질 id -> 표시색
MAT_COLORS = {
    0: ("#eaf2ff", "air"),
    1: ("#404040", "Si"),
    2: ("#111111", "metal"),
    3: ("#bcd4e6", "oxide"),
    4: ("#e24a4a", "CF-R"),
    5: ("#3fb34f", "CF-G"),
    6: ("#3f6fd0", "CF-B"),
    7: ("#f2c14e", "ML"),
}


def _cmap(meta):
    ids = sorted(int(m["id"]) for m in meta["materials"].values())
    maxid = max(ids)
    colors = [MAT_COLORS.get(i, ("#ff00ff", str(i)))[0] for i in range(maxid + 1)]
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(np.arange(-0.5, maxid + 1.5, 1), cmap.N)
    return cmap, norm


def plot_sections(matid, meta, outpng):
    nz, ny, nx = matid.shape
    dz = meta["voxel_um"]["dz"]; dxy = meta["voxel_um"]["dx"]
    cmap, norm = _cmap(meta)
    ext_xz = [0, nx*dxy, 0, nz*dz]

    # 대표 단면: 픽셀 '중앙'을 지나도록 (경계=격벽 위를 피함)
    pitch_vox = int(round(meta["pixel_pitch_um"] / dxy))
    y_mid = pitch_vox // 2          # 첫 픽셀 행 중앙
    x_mid = pitch_vox // 2          # 첫 픽셀 열 중앙

    fig, axs = plt.subplots(2, 2, figsize=(12, 10))

    # XZ (y 고정)
    ax = axs[0, 0]
    im = ax.imshow(matid[:, y_mid, :], origin="lower", aspect="auto",
                   cmap=cmap, norm=norm, extent=ext_xz, interpolation="nearest")
    ax.set_title(f"XZ section @ y={y_mid*dxy:.2f}um"); ax.set_xlabel("x [um]"); ax.set_ylabel("z [um]")

    # YZ (x 고정)
    ax = axs[0, 1]
    ax.imshow(matid[:, :, x_mid], origin="lower", aspect="auto",
              cmap=cmap, norm=norm, extent=[0, ny*dxy, 0, nz*dz], interpolation="nearest")
    ax.set_title(f"YZ section @ x={x_mid*dxy:.2f}um"); ax.set_xlabel("y [um]"); ax.set_ylabel("z [um]")

    # ML top-view: ML(id=7) 이 존재하는 최고 z (표면 높이)
    ml_id = 7
    mlmask = (matid == ml_id)
    zsurf = np.where(mlmask.any(0), mlmask.shape[0] - 1 - mlmask[::-1].argmax(0), 0)
    ax = axs[1, 0]
    im2 = ax.imshow(zsurf * dz, origin="lower", cmap="viridis",
                    extent=[0, nx*dxy, 0, ny*dxy])
    ax.set_title("Microlens surface height (top view)"); ax.set_xlabel("x [um]"); ax.set_ylabel("y [um]")
    plt.colorbar(im2, ax=ax, label="z [um]", fraction=0.046)

    # CF 색 배치 top-view (color_filter 층 중앙 z 슬라이스)
    cf = next(l for l in meta["layer_bounds_vox"] if l["name"] == "color_filter")
    zc = (cf["z0"] + cf["z1"]) // 2
    ax = axs[1, 1]
    ax.imshow(matid[zc], origin="lower", cmap=cmap, norm=norm,
              extent=[0, nx*dxy, 0, ny*dxy], interpolation="nearest")
    ax.set_title(f"Color Filter layer (z={zc*dz:.2f}um)"); ax.set_xlabel("x [um]"); ax.set_ylabel("y [um]")

    # 범례
    handles = [plt.Rectangle((0, 0), 1, 1, color=MAT_COLORS[i][0])
               for i in sorted(MAT_COLORS)]
    labels = [MAT_COLORS[i][1] for i in sorted(MAT_COLORS)]
    fig.legend(handles, labels, loc="lower center", ncol=8, frameon=False)
    fig.suptitle("CIS qcell structure check", fontsize=14)
    fig.tight_layout(rect=[0, 0.04, 1, 0.97])
    fig.savefig(outpng, dpi=120)
    print(f"[saved] {outpng}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--outdir", default="out")
    ap.add_argument("--png", default=None)
    args = ap.parse_args()
    matid, meta = load(args.outdir)
    png = args.png or os.path.join(args.outdir, "qcell_view.png")
    plot_sections(matid, meta, png)
