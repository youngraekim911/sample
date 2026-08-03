# -*- coding: utf-8 -*-
"""실측 QE 대조표 — 구조 yaml 을 넣으면 채널별 시뮬/실측을 나란히 출력.

    PYTHONPATH=. python3 tools/compare_measured.py [구조.yaml] [--ds 1] [--ng 101]

보정값(conf/hybrid_lens.yaml 의 measurement / back_reflector)을 바꾼 뒤
실측에 얼마나 가까워졌는지 바로 확인하는 용도.
실측 데이터는 아래 MEASURED 에 박혀 있으니 제품이 바뀌면 여기를 교체할 것.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch                                                   # noqa: E402
from src.sim.simulator import RCWAPlaneWaveSimulator           # noqa: E402

# hybrid lens 제품 실측 QE (%) — 제품이 바뀌면 이 표를 교체
MEASURED = {
    400: dict(R=7.6, G=8.5, B=37.0),   450: dict(R=1.8, G=5.7, B=59.0),
    500: dict(R=2.6, G=70.4, B=34.0),  520: dict(G=74.34),
    540: dict(G=71.5),                 550: dict(R=3.2, G=68.9, B=6.2),
    560: dict(G=64.5),                 580: dict(G=52.8),
    600: dict(R=62.9, G=37.0, B=3.7),  620: dict(R=62.1),
    640: dict(R=57.8),                 650: dict(G=13.8, B=5.1),
    660: dict(R=52.2),                 680: dict(R=47.4),
    700: dict(R=44.0, G=22.0, B=8.3),
}

ds = int(sys.argv[sys.argv.index("--ds") + 1]) if "--ds" in sys.argv else 1
nG = int(sys.argv[sys.argv.index("--ng") + 1]) if "--ng" in sys.argv else 101
_skip = set()
for f in ("--ds", "--ng"):                       # 플래그와 그 값은 위치인자에서 제외
    if f in sys.argv:
        i = sys.argv.index(f); _skip |= {i, i + 1}
args = [a for i, a in enumerate(sys.argv) if i > 0 and i not in _skip
        and not a.startswith("--")]
conf = args[0] if args else "conf/hybrid_lens.yaml"

sim = RCWAPlaneWaveSimulator(conf, nG=nG, downsample=ds)
print(f"\n구조 {conf}   nG={nG}  downsample={ds}  "
      f"대역폭={sim.qe_bandwidth_nm:.0f}nm")
print("=" * 66)
print("  λ(nm)     R 시뮬/실측        G 시뮬/실측        B 시뮬/실측")
print("  " + "-" * 62)
errs, bright, dark = [], [], []
# run_spectrum: 대역평균의 부분 파장을 이웃 중심끼리 재사용 (파장마다 run_qe 를
# 부르는 것과 값은 동일, solve 수만 감소)
nms = sorted(MEASURED)
outs = sim.run_spectrum([nm / 1000.0 for nm in nms])
for nm, (o1, o2) in zip(nms, outs):
    q = {c: 50 * (o1["QE_rgb"][c] + o2["QE_rgb"][c]) for c in "RGB"}
    line = f"   {nm}  "
    for c in "RGB":
        m = MEASURED[nm].get(c)
        if m is None:
            line += f"   {q[c]:5.1f}/  --  "
        else:
            d = abs(q[c] - m)
            errs.append(d)
            (bright if m > 20 else dark).append(d)
            line += f"   {q[c]:5.1f}/{m:5.1f}"
    print(line, flush=True)
print("  " + "-" * 62)
print(f"   평균 절대오차 {sum(errs)/len(errs):5.2f}%p   "
      f"밝은채널 {sum(bright)/len(bright):5.2f}   "
      f"어두운채널 {sum(dark)/len(dark):5.2f}   ({len(errs)}점)")
print("""
 보정값을 바꾸려면 docs/보정값_수정위치.md 참고
   measurement.bandwidth_nm / back_reflector.coverage / back_reflector.spacer_um
""")
