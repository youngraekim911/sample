# -*- coding: utf-8 -*-
"""A/B 진단 — '수집효율 수정이 이 구조에서 실제로 값을 바꾸는가'.

    PYTHONPATH=. python3 tools/ab_collection.py 내구조.yaml [파장nm ...]

같은 구조를 두 설정으로 각각 풀어서 채널별 QE 를 나란히 보여준다.
  OLD : eta0=1.0, r0=0.35, ld=0.175   (수정 전 모델)
  NEW : 엔진 자동값                    (수정 후 — 로그에 찍히는 그 값)
두 열이 같으면 수정이 적용되지 않은 것이고, 다르면 적용된 것이다.
광학 QE(수집 전)도 같이 찍어 '광학이 문제인지 수집이 문제인지' 를 가른다.
"""
import os
import sys
import copy

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import yaml                                                    # noqa: E402
import torch                                                   # noqa: E402
from src.sim.simulator import RCWAPlaneWaveSimulator           # noqa: E402

conf = sys.argv[1] if len(sys.argv) > 1 else "conf/wizard_config.yaml"
lams = [int(x) for x in sys.argv[2:]] or [400, 450, 500, 550, 600]
nG = int(os.environ.get("NG", "101"))

with open(conf, encoding="utf-8") as f:
    base = yaml.safe_load(f)
tmp = os.path.join(os.environ.get("TMPDIR", "/tmp"), "_ab_%s.yaml")


def run(tag, coll):
    cfg = copy.deepcopy(base)
    if coll is None:
        cfg.pop("collection", None)                            # 엔진 자동
    else:
        cfg["collection"] = dict(coll)
    p = tmp % tag
    with open(p, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True)
    sim = RCWAPlaneWaveSimulator(p, nG=nG, downsample=1)
    out = {}
    for nm in lams:
        o1 = sim.run(nm / 1000.0, pol_te=1.0, pol_tm=0.0)
        o2 = sim.run(nm / 1000.0, pol_te=0.0, pol_tm=1.0)
        out[nm] = ({c: 50 * (o1["QE_rgb"][c] + o2["QE_rgb"][c]) for c in "RGB"},
                   50 * (o1.get("QE_optical", o1["QE"]) + o2.get("QE_optical", o2["QE"])))
    return out


print(f"\n구조: {conf}   nG={nG}   (환경변수 NG 로 변경 가능)")
print("=" * 78)
print("[OLD] 수정 전 모델 강제 — eta0=1.0, r0=0.35, ld=0.175")
old = run("old", {"eta0": 1.0, "r0": 0.35, "ld_um": 0.175})
print("\n[NEW] 엔진 자동값 (수정 후)")
new = run("new", None)
print("\n[OPT] 순수 광학 QE (수집손실 0) — 광학이 같은지 확인용")
opt = run("opt", {"eta0": 1.0, "r0": 0.0, "ld_um": 0.1})

print("\n" + "=" * 78)
print(f"{'λ(nm)':>6} {'ch':>3} | {'OLD':>7} {'NEW':>7} {'차이':>7} | {'순수광학':>8}")
print("-" * 78)
same = True
for nm in lams:
    for c in "RGB":
        o, n_, p_ = old[nm][0][c], new[nm][0][c], opt[nm][0][c]
        d = n_ - o
        if abs(d) > 0.01:
            same = False
        print(f"{nm:>6} {c:>3} | {o:7.2f} {n_:7.2f} {d:+7.2f} | {p_:8.2f}")
print("=" * 78)
if same:
    print(""" 판정: OLD 와 NEW 가 동일 -> 수정이 이 실행에 반영되지 않았다.
   . tools/check_version.py 로 코드 교체 여부 확인
   . 실행 중인 폴더가 교체한 폴더가 맞는지 확인 (check_version.py [2] 항목)
   . __pycache__ 삭제 후 재실행, exe 사용 중이면 재빌드""")
else:
    print(""" 판정: OLD 와 NEW 가 다르다 -> 수정은 정상 적용됐다.
   기대와 값이 다르다면 '구조/물질이 내가 비교하려는 그것이 맞는지' 를 볼 차례다.
   NEW 와 순수광학 열의 차이가 곧 수집손실이다.""")
