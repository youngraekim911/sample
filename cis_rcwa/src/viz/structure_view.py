# -*- coding: utf-8 -*-
"""structure_view — StructureIR 를 이미지로 (XZ 단면 + 평면 뷰).

IR 만 소비하므로 어떤 빌더가 만든 구조든 동일하게 렌더된다.
    python3 -m src.viz.structure_view conf/wizard_config.yaml out.png
"""
import numpy as np


_PALETTE = {"air": "#f5f7fa", "ml_arl": "#bfe3f2", "ml": "#7fb2d9", "oxide": "#cfd8e8",
            "grid_lo": "#8d99ae", "cf_red": "#e05252", "cf_green": "#4caf50",
            "cf_blue": "#4472d9", "barl1": "#e8c98a", "barl2": "#d4a24e",
            "barl3": "#f0e0b8", "barl4": "#b8842e", "barl5": "#e8c98a",
            "si": "#5a5f6b", "poly": "#e8823c", "grid_metal": "#444444"}
_FALLBACK = ["#66c2a5", "#fc8d62", "#8da0cb", "#e78ac3", "#a6d854",
             "#ffd92f", "#e5c494", "#b3b3b3", "#1b9e77", "#d95f02"]


def _colors_for(ir):
    """물질 이름 -> 색. 알려진 이름은 팔레트, 나머지는 순환 배정."""
    names = ir.material_names()
    out, fi = {}, 0
    for n in names:
        if n in _PALETTE:
            out[n] = _PALETTE[n]
        else:
            out[n] = _FALLBACK[fi % len(_FALLBACK)]
            fi += 1
    return out


def render_ir(ir, out_png, xz_y_um=None, plan_z_um=None, z_crop_um=None,
              dz_render=0.0025, title=None):
    """IR -> 다중 패널 PNG.  (eps 모드는 |eps| 그레이스케일)

    xz_y_um  : XZ 단면 y 위치 리스트 (기본: 셀의 1/4, 1/2 지점)
    plan_z_um: 평면 뷰 깊이(위에서부터) 리스트 (기본: 자동 3곳)
    z_crop_um: 표시 깊이 제한 (기본: 상부 구조 + 검출 밴드 1.2µm)
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch

    ny, nx = ir.grid_shape
    sx, sy = ir.span_x, ir.span_y
    region = ir.mode == "region"
    total_h = sum(th for _, th in ir.layers)
    band = ir.detector.band_um if ir.detector else 0.0
    if z_crop_um is None:
        z_crop_um = min(total_h, (total_h - band) + min(band, 1.2) if band else total_h)
    if xz_y_um is None:
        xz_y_um = [sy * 0.25, sy * 0.5]
    if plan_z_um is None:
        upper = total_h - band
        plan_z_um = [max(0.05, upper * 0.15), upper * 0.7, upper + 0.02]

    if region:
        ids = sorted(ir.region_materials)
        id2pos = {rid: i for i, rid in enumerate(ids)}
        cols = _colors_for(ir)
        cmap = ListedColormap([cols[ir.region_materials[r]] for r in ids])
        vmax = len(ids) - 1
        conv = lambda m: np.vectorize(id2pos.get)(m)
    else:
        cmap, vmax = "viridis", None
        conv = lambda m: np.abs(m)

    def xz(iy):
        img, z = [], 0.0
        for m2d, th in ir.layers:
            if z >= z_crop_um:
                break
            take = min(th, z_crop_um - z)
            rn = max(1, int(round(take / dz_render)))
            img.extend([conv(m2d[iy, :])] * rn)
            z += th
        return np.array(img)

    def plane(zq):
        z = 0.0
        for m2d, th in ir.layers:
            if z + th >= zq:
                return conv(m2d)
            z += th
        return conv(ir.layers[-1][0])

    ncol = max(len(xz_y_um), len(plan_z_um))
    fig, axes = plt.subplots(2, ncol, figsize=(4.2 * ncol + 1, 8.6), dpi=130)
    axes = np.atleast_2d(axes)
    for k in range(ncol):
        ax = axes[0, k]
        if k < len(xz_y_um):
            iy = min(ny - 1, int(xz_y_um[k] / sy * ny))
            A = xz(iy)
            hz = A.shape[0] * dz_render
            kw = dict(vmin=0, vmax=vmax) if region else {}
            ax.imshow(A, cmap=cmap, aspect="auto", extent=[0, sx, hz, 0],
                      interpolation="nearest", **kw)
            ax.set_title(f"XZ @ y={xz_y_um[k]:.2f}µm", fontsize=10)
            ax.set_xlabel("x (µm)"); ax.set_ylabel("z from top (µm)")
            if z_crop_um < total_h - 1e-9:
                ax.text(0.02, hz - 0.05, f"cropped @ {z_crop_um:.1f}µm (total {total_h:.1f})",
                        fontsize=7, color="w")
        else:
            ax.axis("off")
        ax = axes[1, k]
        if k < len(plan_z_um):
            kw = dict(vmin=0, vmax=vmax) if region else {}
            ax.imshow(plane(plan_z_um[k]), cmap=cmap, extent=[0, sx, sy, 0],
                      interpolation="nearest", **kw)
            ax.set_title(f"top view @ z={plan_z_um[k]:.2f}µm", fontsize=10)
            ax.set_xlabel("x (µm)"); ax.set_ylabel("y (µm)")
        else:
            ax.axis("off")
    if region:
        cols = _colors_for(ir)
        handles = [Patch(fc=cols[n], ec="#888", label=n)
                   for n in ir.material_names() if n != "air"]
        fig.legend(handles=handles, loc="lower center",
                   ncol=min(8, len(handles)), fontsize=8, frameon=False)
    fig.suptitle(title or "structure (IR view)", fontsize=12)
    fig.tight_layout(rect=[0, 0.06, 1, 0.96])
    fig.savefig(out_png)
    plt.close(fig)
    return out_png


if __name__ == "__main__":
    import sys
    from ..sim.simulator import RCWAPlaneWaveSimulator
    src = sys.argv[1] if len(sys.argv) > 1 else "conf/wizard_config.yaml"
    dst = sys.argv[2] if len(sys.argv) > 2 else "structure_ir.png"
    sim = RCWAPlaneWaveSimulator(src, downsample=2)
    print("saved:", render_ir(sim.ir, dst, title=str(src)))
