# -*- coding: utf-8 -*-
"""DOE 스윕 + surrogate 추출 CLI (GPU/헤드리스용).

  python run_doe.py <structure.yaml> [--mode surrogate] [--nG 201] [--ds 2]
                    [--n 16] [--lam0 0.40] [--lam1 0.70] [--out out/doe]
                    [--materials <folder>]

  mode: axis(17) | surrogate(41, 2차모델 피팅) | full(625)
  출력: <out>_doe.csv (원자료), <out>_surrogate.json (2차 RSM 계수+R²)
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--mode", default="surrogate", choices=["axis", "surrogate", "full"])
    ap.add_argument("--nG", type=int, default=201)
    ap.add_argument("--ds", type=int, default=2)
    ap.add_argument("--n", type=int, default=16, help="파장 스텝 수")
    ap.add_argument("--lam0", type=float, default=0.40)
    ap.add_argument("--lam1", type=float, default=0.70)
    ap.add_argument("--lateral", type=int, default=256)
    ap.add_argument("--materials", default=None)
    ap.add_argument("--out", default="out/doe")
    a = ap.parse_args()

    from src.config.loader import load_config
    from src.sim.simulator import RCWAPlaneWaveSimulator
    from src.sim import doe

    cfg = load_config(a.config)
    RCWAPlaneWaveSimulator._auto_model_defaults(cfg)     # 서버 QE 경로와 동일 자동 모델
    waves = [round(a.lam0 * 1000 + i * (a.lam1 - a.lam0) * 1000 / (a.n - 1))
             for i in range(a.n)]
    total = len(doe.doe_points(a.mode))
    print(f"[doe] mode={a.mode} 조건 {total}개 × {a.n}λ · nG={a.nG} ds={a.ds}")

    def prog(done, tot, eta, p):
        print(f"  {done}/{tot}  남은시간 ~{int(eta // 60)}분{int(eta % 60):02d}초  "
              f"현재 {list(p)}", flush=True)

    res = doe.run_doe(cfg, waves, mode=a.mode, nG=a.nG, downsample=a.ds,
                      lateral_n=a.lateral, materials_dir=a.materials, progress=prog)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out + "_doe.csv", "w", encoding="utf-8") as f:
        f.write(doe.doe_csv(res))
    print(f"[doe] CSV -> {a.out}_doe.csv  ({res['elapsed_s']}s)")
    if a.mode != "axis":
        sur = doe.fit_surrogate(res)
        with open(a.out + "_surrogate.json", "w", encoding="utf-8") as f:
            json.dump(sur, f)
        ws = sur["wavelengths_nm"]
        mid = int(ws[len(ws) // 2])
        print(f"[doe] surrogate -> {a.out}_surrogate.json  "
              f"(R² G@{mid}nm = {sur['r2']['G'][mid]})")


if __name__ == "__main__":
    main()
