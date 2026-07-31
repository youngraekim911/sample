# -*- coding: utf-8 -*-
"""손실 워터폴 — '빛이 Si 에 닿기 전에 어디서 얼마나 사라지나'를 층별로 본다.

    PYTHONPATH=. python3 tools/loss_waterfall.py 내구조.yaml [파장nm] [--mat 물질폴더]

두 사람의 QE 가 다를 때, 그 차이가 '반사'인지 'CF 흡수'인지 'grid 흡수'인지
'수집효율'인지 한 눈에 가른다. 순수 광학 QE 가 실측보다 낮으면 수집효율을
아무리 만져도 실측에 못 닿으므로, 그 경우 여기부터 봐야 한다.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch                                                   # noqa: E402
from src.sim.simulator import RCWAPlaneWaveSimulator           # noqa: E402

args = [a for a in sys.argv[1:] if not a.startswith("--")]
mat = None
if "--mat" in sys.argv:
    mat = sys.argv[sys.argv.index("--mat") + 1]
conf = args[0] if args else "conf/wizard_config.yaml"
lam_nm = float(args[1]) if len(args) > 1 else 450.0
nG = int(os.environ.get("NG", "101"))
ds = int(os.environ.get("DS", "1"))

sim = RCWAPlaneWaveSimulator(conf, nG=nG, downsample=ds, materials_dir=mat)
d = sim.diagnose(lam_nm / 1000.0)

print(f"\n구조 {conf}   λ={lam_nm:.0f}nm   nG={nG}  downsample={ds}")
print("=" * 78)
print("[1] 입사면 반사 / Si 진입")
print(f"      반사 R          = {d['R']*100:6.2f} %")
er = d.get("entry_rgb") or {}
if er:
    print("      진입(1−R)       = " + "  ".join(f"{k} {100*float(v):.1f}%" for k, v in er.items()))
print(f"      Si 진입 T       = {float(d.get('T_into_si', 0))*100:6.2f} %")
print(f"      심부 투과 T     = {float(d.get('T_deep', 0))*100:6.2f} %")

print("\n[2] 층을 지나며 남는 빛 (% 단위, 100 = 입사) — 어디서 깎이는지")
print(f"      {'층 물질':34s} {'두께µm':>7s} {'R':>6s} {'G':>6s} {'B':>6s}   손실")
prev = None
for row in d.get("profile", []):
    t = row.get("T_rgb") or {}
    nm = str(row.get("mats", "?"))[:34]
    th = float(row.get("th_um", 0))
    cur = tuple(100.0 * float(t.get(c, 0.0)) for c in "RGB")
    drop = ""
    if prev:
        dd = [a - b for a, b in zip(prev, cur)]
        if max(dd) > 0.5:
            drop = "  ← " + " ".join(f"{c}-{v:.1f}" for c, v in zip("RGB", dd) if v > 0.5)
    print(f"      {nm:34s} {th:7.3f} {cur[0]:6.1f} {cur[1]:6.1f} {cur[2]:6.1f}{drop}")
    prev = cur

print("\n[3] 물질 점검 (이상 플래그가 있으면 그것부터)")
for m in d.get("materials", []):
    fl = m.get("flags") or []
    star = "  ⚠ " + " / ".join(fl) if fl else ""
    print(f"      {m['name']:10s} n={m['n']:6.3f} k={m['k']:7.4f}   {m.get('source','')}{star}")

o1 = sim.run(lam_nm / 1000.0, pol_te=1.0, pol_tm=0.0)
o2 = sim.run(lam_nm / 1000.0, pol_te=0.0, pol_tm=1.0)
qo = 50 * (o1["QE_optical"] + o2["QE_optical"])
print("\n[4] 최종")
print(f"      광학 QE(전체) = {qo:6.2f} %")
for c in "RGB":
    print(f"      QE_{c} (수집 후) = {50*(o1['QE_rgb'][c]+o2['QE_rgb'][c]):6.2f} %")
print("""
 읽는 법:
   . 순수 광학 QE 가 실측보다 낮다 -> 수집효율로는 못 메운다. [1][2] 에서
     반사/CF/grid 중 어디서 빛이 사라지는지 찾을 것.
   . [2] 에서 어느 층에서 크게 떨어지는지가 곧 범인.
   . [3] 의 n,k 가 상대와 다르면 그게 차이의 원인 — 특히 si, cf_*, tin/ti.
""")
