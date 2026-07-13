# -*- coding: utf-8 -*-
"""analysis_view — RCWAEngine 결과(프로브 출력)의 표준 그래프.

    plot_boundary_T(res, out)  경계별 Transmittance 스펙트럼 (블록 경계 × 컬러 분기)
    plot_qe(res, out)          픽셀별/CF별 QE 스펙트럼 (+픽셀 위치 표)
"""
import numpy as np


_LC = {"R": "#d33", "G": "#2a2", "B": "#36c", "all": "#555"}


def plot_boundary_T(res, out_png, title=None):
    """각 블록 경계 통과 후 T(λ) — 서브플롯: 진입(1-R) + 경계마다 1개, RGB 분기."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    lam = np.array(res["wavelength_um"]) * 1000
    bounds = res["boundary"]
    n = len(bounds) + 1
    ncol = min(3, n)
    nrow = (n + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.6 * ncol, 3.4 * nrow),
                             dpi=130, squeeze=False)
    panels = [("entry (1-R)", res["entry"])] + \
             [(f"after {b['tag']}", b["T"]) for b in bounds]
    for i, (ttl, T) in enumerate(panels):
        ax = axes[i // ncol][i % ncol]
        for L, ys in T.items():
            ax.plot(lam, np.array(ys) * 100, "-o", ms=3,
                    color=_LC.get(L, None), label=f"{L} pixel")
        ax.set_title(ttl, fontsize=10)
        ax.set_xlabel("wavelength (nm)"); ax.set_ylabel("T (%)")
        ax.set_ylim(0, 100); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    for j in range(n, nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")
    fig.suptitle(title or "boundary transmittance per block", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_png); plt.close(fig)
    return out_png


def plot_qe(res, out_png, title=None):
    """좌: CF별 평균 QE 스펙트럼 / 우: 픽셀별 QE (위치·CF 귀속 라벨)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    lam = np.array(res["wavelength_um"]) * 1000
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2), dpi=130)
    for L, ys in res["qe_by_cf"].items():
        a1.plot(lam, np.array(ys) * 100, "-o", ms=3, color=_LC.get(L), label=f"{L} CF avg")
    a1.set_title("QE by color filter", fontsize=10)
    a1.set_xlabel("wavelength (nm)"); a1.set_ylabel("QE (%)")
    a1.set_ylim(0, 100); a1.grid(alpha=0.3); a1.legend(fontsize=8)
    for r in res["qe"]:
        a2.plot(lam, np.array(r["qe"]) * 100, "-", lw=1.2, color=_LC.get(r["cf"]),
                alpha=0.75,
                label=f"px{r['pixel']} ({r['row']},{r['col']}) @({r['x_um']:.2f},{r['y_um']:.2f})µm · {r['cf']}")
    a2.set_title("QE per pixel (position / CF)", fontsize=10)
    a2.set_xlabel("wavelength (nm)"); a2.set_ylabel("QE (%)")
    a2.set_ylim(0, 100); a2.grid(alpha=0.3); a2.legend(fontsize=7)
    fig.suptitle(title or "pixel QE", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_png); plt.close(fig)
    return out_png
