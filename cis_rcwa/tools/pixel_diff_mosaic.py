# -*- coding: utf-8 -*-
"""픽셀별 diff 모자이크 PNG — 사광 이미지 잡 결과(_image.json)에서 생성.

    PYTHONPATH=. python3 tools/pixel_diff_mosaic.py out/jobs/<jid>_image.json [λ|mean] [out.png]

단위셀 픽셀(npx×npx) 각각의 "센서 전체(field) diff 맵"을 단위셀 배열 그대로
npx×npx 모자이크로 그린다. 소맵 색 = 그 픽셀의 동채널(quad) 평균 대비 편차%.
결과 json 에 pixel_diff 가 없으면(구버전 잡) images/labels 로 즉석 재계산한다.
"""
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np                                             # noqa: E402


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    path = sys.argv[1]
    sel = sys.argv[2] if len(sys.argv) > 2 else "mean"
    out = sys.argv[3] if len(sys.argv) > 3 else \
        os.path.splitext(path)[0] + f"_pixdiff_{sel}.png"
    res = json.load(open(path, encoding="utf-8"))

    if res.get("pixel_diff", {}).get("maps"):
        maps = res["pixel_diff"]["maps"]
        key = sel if sel in maps else "mean"
        M = np.array([[np.array(m, float) for m in row] for row in maps[key]],
                     dtype=float)                      # (npx,npx,rows,cols) — None→nan
        M = np.where(np.isfinite(M), M, np.nan)
        ch = res["pixel_diff"]["channels"]
    else:                                              # 구버전 잡 — 즉석 계산
        from src.sim.octant import per_pixel_diff_image
        img = res["mean"] if sel == "mean" else res["images"][sel]
        A = np.array([[np.nan if v is None else v for v in row] for row in img])
        M, ch = per_pixel_diff_image(A, res["labels"])

    npx = len(ch)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    CC = {"R": "#d62728", "Gr": "#2ca02c", "Gb": "#1f9e63",
          "G": "#2ca02c", "B": "#1f77b4"}
    amax = float(np.nanmax(np.abs(M))) or 1e-9
    fig, axes = plt.subplots(npx, npx, figsize=(2.1 * npx, 1.9 * npx))
    im = None
    for r in range(npx):
        for c in range(npx):
            ax = axes[r][c]
            im = ax.imshow(M[r][c], cmap="RdBu_r", vmin=-amax, vmax=amax,
                           interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            col = CC.get(str(ch[r][c]), "#888")
            for sp in ax.spines.values():
                sp.set_edgecolor(col); sp.set_linewidth(2.2)
            ax.set_title(f"({r},{c}) {ch[r][c]}", fontsize=8, color=col, pad=2)
    fig.suptitle(f"per-pixel ch.diff mosaic  [{sel}]  "
                 f"(map = full sensor field, +-{amax:.1f}%)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 0.92, 0.96))
    cax = fig.add_axes([0.94, 0.15, 0.02, 0.65])
    fig.colorbar(im, cax=cax, label="diff (%)")
    fig.savefig(out, dpi=140)
    print("saved:", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
