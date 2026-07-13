# -*- coding: utf-8 -*-
"""IR 리팩터링 회귀 테스트 — python3 tests/regress_ir.py (repo 루트에서).

1) v50 위저드 yaml -> QE_G@550 이 리팩터링 전 값(77.99%)과 일치
2) IR save/load(npz) 라운드트립 -> 동일 QE
3) remap_material (물질 교체, 기하 재생성 없음) -> QE 반응
4) 합성 eps 모드 IR -> 에너지 보존
5) IR 렌더러 smoke
6) 블록 조립 == 위저드 rcwa_layers (맵/두께/detector 완전 일치)
7) 블록 일반화: DTI liner 2겹 스택 + BARL 7층 + conformal 코팅 2겹 조합 실행
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

    # ---- 6) 블록 조립 == 위저드 rcwa_layers (기하 완전 일치) ----
    from src.config.loader import load_config
    from src.structure.wizard_builder import WizardBuilder
    from src.structure.ir import StructureIR as IRC, Detector as Det
    from src.structure.blocks import ir_from_wizard_cfg

    def canon(ir):
        rm = ir.region_materials
        n2c = {n: i for i, n in enumerate(sorted(set(rm.values())))}
        out = []
        for m, th in ir.layers:
            conv = np.vectorize(lambda x: n2c[rm[int(x)]])(m).astype(np.int16)
            if out and np.array_equal(out[-1][0], conv):
                out[-1] = (out[-1][0], out[-1][1] + th)
            else:
                out.append((conv, th))
        return out

    cfg = load_config(CONF)
    N = 200
    b = WizardBuilder(cfg).set_lateral(N)
    layers, si_n = b.rcwa_layers()                     # 레거시 경로 직접
    pixidx, colors = b.pixel_maps()
    irA = IRC(span_x=b.span, span_y=b.span, layers=layers,
              region_materials={int(i): n for n, i in b._idx.items()},
              substrate=b.s["si"]["material"],
              detector=Det(band_um=float(b.s["si"]["thickness_um"]), n_layers=si_n,
                           pixel_map=pixidx, pixel_labels=list(colors),
                           exclude_mask=b.dti_trench_mask()),
              materials=dict(cfg.get("materials", {}) or {})).validate()
    irB = ir_from_wizard_cfg(cfg, N)
    A, B = canon(irA), canon(irB)
    assert len(A) == len(B), f"canon 층 수 {len(A)} vs {len(B)}"
    for i, ((ma, ta), (mb, tb)) in enumerate(zip(A, B)):
        assert abs(ta - tb) < 1e-9 and np.array_equal(ma, mb), f"layer{i} 불일치"
    assert np.array_equal(irA.detector.pixel_map, irB.detector.pixel_map)
    assert np.array_equal(irA.detector.exclude_mask, irB.detector.exclude_mask)
    print(f"[6] 블록 조립 == rcwa_layers: canon {len(A)}층 완전 일치")

    # ---- 7) 블록 일반화 조합 (liner 2겹 + BARL 7층 + 코팅 2겹) ----
    from src.structure.blocks import (BlockContext, BlockStack, SiDtiBlock,
                                      BarlBlock, GridCfBlock, PlanarBlock,
                                      MlBlock, ConformalCoatBlock)
    ctx = BlockContext(pitch_um=1.0, n_pixels=2, lateral_n=128,
                       bayer=[["R", "G"], ["G", "B"]])
    grid = {"pitch": 1, "width_um": 0.10, "top_ratio": 0.9, "coat_on": True,
            "coat_material": "oxide", "coat_um": 0.02,
            "stack": [{"material": "grid_lo", "height_um": 0.20}]}
    cf = {c: {"material": m, "thickness_um": 0.5, "curvature_um": 0.05}
          for c, m in (("R", "cf_red"), ("G", "cf_green"), ("B", "cf_blue"))}
    st = BlockStack([
        SiDtiBlock("si", 3.0, dti={"mode": "2x2_open", "width_um": 0.10,
                                   "center_gap_x_um": 0.2, "center_gap_y_um": 0.2,
                                   "fill": "poly",
                                   "liners": [{"material": "oxide", "thickness_um": 0.02},
                                              {"material": "barl2", "thickness_um": 0.01}]}),
        BarlBlock([{"material": f"barl{1 + i % 5}", "thickness_um": 0.02}
                   for i in range(7)]),                     # 7층 gradual
        GridCfBlock(grid, cf, bg_material="ml"),
        PlanarBlock("ml", 0.10),
        MlBlock("ml", height_um=0.35,
                quads=[[{"shape": "1x1", "scale": 1}] * 2] * 2),
        ConformalCoatBlock("ml_arl", 0.08),
        ConformalCoatBlock("oxide", 0.03),                  # 코팅 2겹
    ], materials={"poly": {"n": 4.15, "k": 0.07},           # 폴더에 없는 물질은 상수로
                  "barl1": {"n": 1.77, "k": 0}, "barl2": {"n": 2.06, "k": 0},
                  "barl3": {"n": 1.46, "k": 0}, "barl4": {"n": 2.06, "k": 0},
                  "barl5": {"n": 1.77, "k": 0}})
    ir7 = st.to_ir(ctx)
    sim7 = RCWAPlaneWaveSimulator(ir7, nG=41)
    o7 = sim7.run(0.55)
    assert abs(o7["R"] + o7["QE"] + o7["A_stack"] - 1) < 1e-6
    assert set(o7["QE_rgb"]) == {"R", "G", "B"}
    print(f"[7] 블록 일반화 조합: {len(ir7.layers)}층, R={o7['R']:.3f} "
          f"QE_G={o7['QE_rgb']['G']:.3f} — OK")
    print("\nALL PASS")


if __name__ == "__main__":
    main()
