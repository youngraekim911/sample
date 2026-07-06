# -*- coding: utf-8 -*-
"""runner — 파장 sweep 으로 QE 스펙트럼 계산 + 저장/플롯.

usage:
    python3 runner.py -c qcell_config.yaml --lam0 0.40 --lam1 0.70 --n 16 --nG 61
출력:
    out/qe_spectrum.csv   (lambda, R, QE, A_stack)
    out/qe_spectrum.png
"""
import os
import csv
import time
import argparse
import numpy as np
import torch

from .simulator import RCWAPlaneWaveSimulator


def sweep(config, lam0, lam1, nlam, nG, downsample, theta, phi, device):
    sim = RCWAPlaneWaveSimulator(config, nG=nG, downsample=downsample, device=device)
    print(f"[sim] device={sim.device} grid={sim.grid_ny}x{sim.grid_nx} "
          f"layers={len(sim.layer_stack)} nG~{nG}")
    lams = np.linspace(lam0, lam1, nlam)
    rows = []
    for lam in lams:
        t0 = time.time()
        # 비편광 근사: TE/TM 평균
        o_te = sim.run(lam, theta=theta, phi=phi, pol_te=1.0, pol_tm=0.0)
        o_tm = sim.run(lam, theta=theta, phi=phi, pol_te=0.0, pol_tm=1.0)
        R = 0.5 * (o_te["R"] + o_tm["R"])
        QE = 0.5 * (o_te["QE"] + o_tm["QE"])
        A = 0.5 * (o_te["A_stack"] + o_tm["A_stack"])
        rows.append((lam, R, QE, A))
        print(f"  λ={lam*1000:5.0f}nm  R={R:.3f}  QE(Si)={QE:.3f}  "
              f"A_stack={A:.3f}  ({time.time()-t0:.1f}s)")
    return sim, np.array(rows)


def save_outputs(rows, outdir):
    os.makedirs(outdir, exist_ok=True)
    csv_path = os.path.join(outdir, "qe_spectrum.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["lambda_um", "R", "QE_Si", "A_stack"])
        for r in rows:
            w.writerow([f"{r[0]:.4f}", f"{r[1]:.5f}", f"{r[2]:.5f}", f"{r[3]:.5f}"])
    print(f"[saved] {csv_path}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        lam = rows[:, 0] * 1000
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(lam, rows[:, 2], "-o", color="#2a7", lw=2, label="QE (into Si)")
        ax.plot(lam, rows[:, 1], "-s", color="#c55", lw=1.5, label="Reflection")
        ax.plot(lam, rows[:, 3], "-^", color="#57c", lw=1.5, label="Stack absorption")
        ax.plot(lam, rows[:, 1] + rows[:, 2] + rows[:, 3], ":", color="#888",
                label="sum (=1 check)")
        ax.set_xlabel("wavelength [nm]"); ax.set_ylabel("fraction")
        ax.set_title("CIS qcell RCWA — QE spectrum")
        ax.set_ylim(0, 1.02); ax.grid(alpha=.3); ax.legend()
        png = os.path.join(outdir, "qe_spectrum.png")
        fig.tight_layout(); fig.savefig(png, dpi=120)
        print(f"[saved] {png}")
    except Exception as e:
        print(f"[warn] plot skipped: {e}")


def main():
    ap = argparse.ArgumentParser(description="RCWA QE wavelength sweep")
    ap.add_argument("-c", "--config", default="conf/qcell_config.yaml")
    ap.add_argument("-o", "--outdir", default="out")
    ap.add_argument("--lam0", type=float, default=0.40)
    ap.add_argument("--lam1", type=float, default=0.70)
    ap.add_argument("--n", type=int, default=13, dest="nlam")
    ap.add_argument("--nG", type=int, default=61)
    ap.add_argument("--downsample", type=int, default=2)
    ap.add_argument("--theta", type=float, default=0.0)
    ap.add_argument("--phi", type=float, default=0.0)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    sim, rows = sweep(args.config, args.lam0, args.lam1, args.nlam, args.nG,
                      args.downsample, args.theta, args.phi, args.device)
    save_outputs(rows, args.outdir)


if __name__ == "__main__":
    main()
