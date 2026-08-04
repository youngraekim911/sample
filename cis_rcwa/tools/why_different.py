# -*- coding: utf-8 -*-
"""내 PC 값이 왜 다른가 — 한 방에 원인 특정.

    PYTHONPATH=. python3 tools/why_different.py [구조.yaml] [--ng 101] [--ds 1]

Green peak 이 74 가 아니라 80~81 로 나오는 원인은 지금까지 전부 아래 5개 중
하나였다. 코드/데이터/설정을 실제로 '해석된 값' 으로 확인하고, 마지막에
G@520 을 직접 계산해 기대값과 대조한다. 출력을 그대로 복사해 보내면 된다.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

OK, NG = "[ O ]", "[ X ]"
fails = []


def chk(name, cond, detail=""):
    print(f"  {OK if cond else NG} {name}" + (f"   {detail}" if detail else ""))
    if not cond:
        fails.append(name)


conf = "conf/hybrid_lens.yaml"
nG, ds = 101, 1
a = sys.argv[1:]
for i, x in enumerate(a):
    if x == "--ng" and i + 1 < len(a):
        nG = int(a[i + 1])
    elif x == "--ds" and i + 1 < len(a):
        ds = int(a[i + 1])
    elif not x.startswith("--") and (i == 0 or a[i - 1] not in ("--ng", "--ds")):
        conf = x

print("\n" + "=" * 72)
print(" 왜 값이 다른가 — 5개 확인")
print("=" * 72)
print(f" 구조 {conf}   nG={nG}  downsample={ds}")

# ── 1) 코드: Wood anomaly 해석적 회피가 들어있나 ─────────────────────────
print("\n[1] 코드 — Wood anomaly 회피 (560nm 가짜 공진 수정)")
import src.sim.simulator as S                                    # noqa: E402
src_txt = open(S.__file__, encoding="utf-8").read()
chk("_wood_free_lambda 존재", "_wood_free_lambda" in src_txt,
    "없으면 560nm 가 34.7 로 꺼져 G 곡선이 망가짐")
print(f"      import 경로 {S.__file__}")

# ── 2) 코드: 산란 바닥 (소프트웨어 보정 — src/materials/scatter_model.py) ──
print("\n[2] 코드 — 안료 산란 바닥 (scatter_model, UI ⚙ 설정 연동)")
try:
    from src.materials import scatter_model as _sc
    eff = _sc.get_floors()
    ovr = _sc.load_overrides()
    chk("scatter_model 모듈 존재", True)
    fl = eff.get("cf_green")
    chk("cf_green k530 = 0.0108 (λ^-1.5 산란 법칙)", fl is not None and abs(fl - 0.0108) < 5e-4,
        f"현재 {fl}" + (f"  (UI 재정의 {ovr} 적용 중 — ⚙ 에서 확인)" if ovr else
                        "  (기본값)"))
except ImportError as _e:
    chk("scatter_model 모듈 존재", False, f"{_e} — 패키지를 통째로 다시 푸세요")

# ── 3) 설정: yaml 이 그 파일을 가리키나 + 측정 대역폭 ────────────────────
print(f"\n[3] 설정 — {conf}")
if not os.path.exists(conf):
    chk("구조 파일 존재", False, conf)
else:
    from src.config.loader import load_config                    # noqa: E402
    cfg = load_config(conf)
    meas = cfg.get("measurement") or {}
    bw = float(meas.get("bandwidth_nm", 0) or 0)
    chk("measurement.bandwidth_nm > 0", bw > 0,
        f"현재 {bw:.0f}nm   -> 0 이면 단색 계산 (G@500 이 +5.7 높게 나옴)")
    br = ((cfg.get("stack") or {}).get("si") or {}).get("back_reflector") or {}
    print(f"      back_reflector: coverage={br.get('coverage')} "
          f"spacer_um={br.get('spacer_um')} material={br.get('material')}")

# ── 4) 해석 결과: 실제로 어떤 k 가 쓰이나 (가장 확실한 증거) ─────────────
print("\n[4] 해석된 값 — 실제 계산에 들어가는 cf_green k")
try:
    import io
    import contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        sim = S.RCWAPlaneWaveSimulator(conf, nG=nG, downsample=ds)
    print(f"      출처   {sim.res.source('cf_green')[0]}")
    kk = sim.res.nk("cf_green", 0.52)[1]
    chk("cf_green k@520 = 0.0111 (산란 법칙 @520)", abs(kk - 0.01111) < 5e-4,
        f"현재 {kk:.5f}   (0.0070 이면 바닥이 안 걸린 것)")
    print(f"      측정 대역폭 {sim.qe_bandwidth_nm:.0f}nm  ·  "
          f"층 {len(sim.layer_stack)}  ·  격자 {sim.grid_ny}×{sim.grid_nx}")
except Exception as e:
    chk("구조 로드", False, f"{type(e).__name__}: {e}")
    sim = None

# ── 5) 실제 계산: G@520 을 직접 재서 기대값과 대조 ───────────────────────
print("\n[5] 실측정 — G@520 (실측 74.3)")
# 이 저장소에서 실제로 측정한 값. 보정 적용 / 미적용 두 벌.
# 산란 바닥(scatter_kfloor.txt) 적용 상태 — conf/hybrid_lens.yaml 기준
EXPECT = {(101, 2): 75.79, (101, 1): 75.63}   # n_sub 7 기준
# 바닥 파일을 지운 상태(원본 평막 n,k 그대로)의 참고값
NO_FIX = {(101, 1): 79.47, (151, 1): 80.36, (201, 1): 81.53, (301, 1): 81.20}
if sim is not None:
    o1 = sim.run_qe(0.52, pol_te=1.0, pol_tm=0.0)
    o2 = sim.run_qe(0.52, pol_te=0.0, pol_tm=1.0)
    g = 50 * (o1["QE_rgb"]["G"] + o2["QE_rgb"]["G"])
    e = EXPECT.get((nG, ds))
    print(f"      이 PC 의 G@520 = {g:.2f}")
    if e is not None:
        chk(f"기대값 {e:.1f} 과 일치 (±1.0)", abs(g - e) <= 1.0,
            f"차이 {g-e:+.2f}")
    else:
        print(f"      (nG={nG}, ds={ds} 조합의 기대값 표가 없음 — "
              f"nG=101/ds=1 로 다시 돌려보세요)")
    nf = NO_FIX.get((nG, ds))
    if nf is not None:
        print(f"      참고: 산란 바닥이 '안' 걸린 상태의 값은 이 조건에서 {nf:.1f} 입니다")
    if g > 78:
        print("\n      >> 79~81 은 산란 바닥이 아예 안 걸린 값입니다.")
        print("         nG 를 올려서 나는 차이가 아닙니다 — [2] 의 산란 바닥이")
        print("         걸려 있으면 그런 값이 나오지 않습니다.")
        print("         위 [1]~[4] 의 [ X ] 항목이 원인입니다.")

print("\n" + "-" * 72)
print(" 조건별 기대값 (이 저장소 실측)          G@520   실측 74.3")
print("-" * 72)
print(f" {'nG':>5} {'ds':>3} | {'바닥 적용':>10} {'바닥 없음':>10}")
for k in sorted(set(EXPECT) | set(NO_FIX)):
    e = EXPECT.get(k); n_ = NO_FIX.get(k)
    print(f" {k[0]:5d} {k[1]:3d} | {('%10.2f' % e) if e else '        --'} "
          f"{('%10.2f' % n_) if n_ else '        --'}")
print(" * nG 를 올리면 (보정 여부와 무관하게) 값이 올라갑니다 — 크로스톡이")
print("   제대로 풀리면서 밝은 채널이 커지기 때문입니다. 보정은 nG 와 별개입니다.")

print("\n" + "=" * 72)
if fails:
    print(f" 실패 {len(fails)}건: {', '.join(fails)}")
    print("""
 조치
   [1] 실패 -> src/sim/simulator.py 가 교체 안 됨. 패키지를 통째로 다시 푸세요.
   [2] 실패 -> src/materials/scatter_model.py 가 없거나, UI ⚙ 설정에서
                바닥이 바뀐 상태입니다. 위저드 우측 상단 ⚙ -> '기본값 복원'.
   [3] 실패 -> 쓰시는 yaml 에 아래 블록을 넣으세요:

         measurement:
           bandwidth_nm: 20
           n_sub: 3

   [4] 실패 -> [2]/[3] 을 먼저 고치세요. 그래도 남으면 materials 폴더 경로 문제.
   [5] 만 실패 -> 코드/설정은 맞습니다. nG·downsample 차이일 수 있으니
                  nG=101, downsample=1 로 맞춰 다시 돌려보세요.
""")
else:
    print(" 전부 통과 — 이 PC 는 정합본과 같은 상태입니다.")
print("=" * 72 + "\n")
