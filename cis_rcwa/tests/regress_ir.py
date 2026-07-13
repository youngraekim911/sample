# -*- coding: utf-8 -*-
"""IR 리팩터링 회귀 테스트 — python3 tests/regress_ir.py (repo 루트에서).

1) v50 위저드 yaml -> QE_G@550 이 리팩터링 전 값(77.99%)과 일치
2) IR save/load(npz) 라운드트립 -> 동일 QE
3) remap_material (물질 교체, 기하 재생성 없음) -> QE 반응
4) 합성 eps 모드 IR -> 에너지 보존
5) IR 렌더러 smoke
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.sim.simulator import RCWAPlaneWaveSimulator
from src.structure.ir import StructureIR, Detector

TMP = os.environ.get("TMPDIR", "/tmp")
CONF = "conf/wizard_config.yaml"
EXPECT_G = 0.7799            # 리팩터링 직전 측정값 (nG=101, ds=2, TE+TM 평균)


def qe_g(sim):
    o1 = sim.run(0.55, pol_te=1.0, pol_tm=0.0)
    o2 = sim.run(0.55, pol_te=0.0, pol_tm=1.0)
    return 0.5 * (o1["QE_rgb"]["G"] + o2["QE_rgb"]["G"]), o1


def main():
    # ---- 1) yaml 경로 회귀 ----
    sim = RCWAPlaneWaveSimulator(CONF, nG=101, downsample=2)
    g, o1 = qe_g(sim)
    print(f"[1] yaml QE_G@550 = {g*100:.2f}%  (기대 {EXPECT_G*100:.2f}±0.15)")
    assert abs(g - EXPECT_G) < 0.0015, "yaml 회귀 실패"
    assert 0 <= o1["R"] <= 1 and abs(o1["R"] + o1["QE"] + o1["A_stack"] - 1) < 1e-6, \
        "에너지 보존 실패"

    # ---- 2) IR npz 라운드트립 ----
    p = os.path.join(TMP, "regress_ir.npz")
    sim.ir.save_npz(p)
    ir2 = StructureIR.load_npz(p)
    sim2 = RCWAPlaneWaveSimulator(ir2, nG=101, downsample=2)
    g2, _ = qe_g(sim2)
    print(f"[2] npz 라운드트립 QE_G = {g2*100:.2f}%  (Δ={abs(g2-g)*100:.4f})")
    assert abs(g2 - g) < 1e-6, "npz 라운드트립 불일치"

    # ---- 3) 물질 교체 (기하 그대로) ----
    sim3 = RCWAPlaneWaveSimulator(CONF, nG=61, downsample=4)
    gA, _ = qe_g(sim3)
    n = sim3.remap_material("cf_green", "cf_blue")      # G 자리에 B 필터
    gB, _ = qe_g(sim3)
    print(f"[3] remap cf_green->cf_blue ({n} regions): QE_G {gA*100:.1f}% -> {gB*100:.1f}%")
    assert n > 0 and gB < gA * 0.5, "remap 미반응 (B 필터는 550 강흡수)"

    # ---- 4) 합성 eps 모드 IR (구조 지식 0 으로 RCWA 실행) ----
    ny = nx = 64
    yy, xx = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
    pat = np.where((xx // 16 + yy // 16) % 2 == 0, (1.5 + 0j) ** 2, (2.0 + 0j) ** 2)
    ir_eps = StructureIR(
        span_x=1.0, span_y=1.0,
        layers=[(np.full((ny, nx), (1.47 + 0j) ** 2), 0.1),
                (pat.astype(np.complex128), 0.3)],
        region_materials={}, mode="eps", eps_lambda_um=0.55,
        ambient_eps=1.0 + 0j, substrate_eps=(4.08 + 0.028j) ** 2).validate()
    sim4 = RCWAPlaneWaveSimulator(ir_eps, nG=41)
    o = sim4.run(0.55)
    print(f"[4] eps 모드: R={o['R']:.4f} T={o['QE']:.4f} A={o['A_stack']:.4f}")
    assert abs(o["R"] + o["QE"] + o["A_stack"] - 1) < 1e-6, "eps 모드 에너지 보존 실패"

    # ---- 5) 렌더러 smoke ----
    from src.viz.structure_view import render_ir
    png = render_ir(sim.ir, os.path.join(TMP, "regress_ir.png"), title="regress")
    assert os.path.getsize(png) > 10000
    print(f"[5] 렌더러 OK: {png}")
    print("\nALL PASS")


if __name__ == "__main__":
    main()
