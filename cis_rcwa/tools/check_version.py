# -*- coding: utf-8 -*-
"""설치본 자가진단 — '수정이 실제로 적용됐는가'를 30초 안에 확인.

    cis_rcwa 폴더에서:   PYTHONPATH=. python3 tools/check_version.py [구조.yaml]

yaml 을 주면 그 구조로, 안 주면 conf/wizard_config.yaml 로 확인한다.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

OK, NG = "[ O ]", "[ X ]"
fails = []


def check(name, cond, detail=""):
    print(f"  {OK if cond else NG} {name}   {detail}")
    if not cond:
        fails.append(name)


print("\n" + "=" * 70)
print(" cis_rcwa 설치본 자가진단")
print("=" * 70)

# ── 1) 코드에 수정이 들어있는가 ────────────────────────────────────────
print("\n[1] 소스 코드 확인 — 파일이 실제로 교체됐는가")
import src.sim.simulator as S
import src.structure.blocks as B
from src.structure.ir import Detector

check("수집효율 η0 상수", hasattr(S, "ETA0_DEFAULT"),
      f"ETA0_DEFAULT = {getattr(S, 'ETA0_DEFAULT', '없음')}")
check("후면 패시베이션 자동판정", hasattr(S, "_si_backside_passivated"))
check("Detector.collect_eta0 필드", hasattr(Detector(), "collect_eta0"),
      f"기본값 {getattr(Detector(), 'collect_eta0', '없음')}")
check("DTI 서브해상도 EMT", hasattr(B, "fourier_res_um"))
check("order-공간 QE 적분(가속)",
      hasattr(__import__("src.rcwa.rcwa", fromlist=["x"]).RCWASolver,
              "band_window_absorption"))

# ── 2) 실행 중인 파일이 어디인가 (다른 폴더 실행 방지) ───────────────────
print("\n[2] 지금 import 된 파일 경로 — 여기가 교체한 폴더가 맞는지 확인")
print(f"      {S.__file__}")
pyc = S.__file__.replace(".py", "")
print(f"      소스 수정시각 {os.path.getmtime(S.__file__):.0f}")

# ── 3) 구조를 넣었을 때 실제로 자동값이 걸리는가 ──────────────────────────
conf = sys.argv[1] if len(sys.argv) > 1 else "conf/wizard_config.yaml"
print(f"\n[3] 구조 적용 확인 — {conf}")
if not os.path.exists(conf):
    print(f"  {NG} 파일 없음: {conf}")
    fails.append("구조 파일")
else:
    from src.config.loader import load_config
    cfg = load_config(conf)
    had = "collection" in cfg
    S.RCWAPlaneWaveSimulator._auto_model_defaults(cfg)
    c = cfg["collection"]
    print(f"      yaml 에 collection 블록 {'있었음(명시값 우선)' if had else '없었음(전부 자동)'}")
    print(f"      -> eta0={c['eta0']}  r0={c['r0']}  ld_um={c['ld_um']}")
    st = cfg.get("stack") or {}
    barl = (st.get("barl") or [{}])
    print(f"      Si 접촉 BARL 첫 층: {barl[0].get('material','(없음)')} "
          f"-> 패시베이션 {'감지' if S._si_backside_passivated(st) else '없음'}")
    check("η0 가 1.0 이 아님 (수집손실 반영)", float(c["eta0"]) < 0.999,
          "1.0 이면 yaml 의 collection 이 eta0 를 덮어쓴 것")
    dti = st.get("dti") or {}
    if dti.get("mode"):
        check("DTI 광학 ON", dti.get("optical") is not False,
              f"optical={dti.get('optical')}")

# ── 4) 물질 데이터 확인 ────────────────────────────────────────────────
print("\n[4] 물질 데이터 — cf_green 400nm k")
p = "data/materials/cf_green.txt"
if os.path.exists(p):
    k400 = None
    for ln in open(p, encoding="utf-8"):
        f = ln.split()
        if len(f) >= 3 and f[0].startswith("400"):
            k400 = float(f[2])
    check("k@400 = 0.145 (전사오류 수정본)", k400 is not None and abs(k400 - 0.145) < 1e-6,
          f"현재 {k400}")
else:
    print(f"  {NG} {p} 없음")
    fails.append("cf_green")

print("\n" + "=" * 70)
if fails:
    print(f" 실패 {len(fails)}건: {', '.join(fails)}")
    print("""
 확인할 것:
   1) 압축을 기존 폴더 '위에' 덮어썼는지 (다른 폴더에 풀고 옛 폴더를 실행 중일 수 있음)
   2) [2] 의 경로가 교체한 폴더가 맞는지
   3) __pycache__ 폴더를 지우고 다시 실행:  find . -name __pycache__ -type d -exec rm -rf {} +
   4) 빌드된 exe 를 쓰고 있다면 build_exe.py 로 다시 빌드해야 반영됨
   5) '모델 탐색(surrogate)' 화면 값은 미리 구운 모델이라 코드 수정과 무관 —
      DOE 를 '캐시 무시'로 다시 굽어야 새 값이 나옴""")
    sys.exit(1)
print(" 전부 통과 — 수정본이 정상 적용되어 있습니다.")
print("""
 그래도 QE 가 예전과 같다면:
   . '모델 탐색' 화면인가? -> 미리 구운 surrogate 값이라 재-DOE 필요
   . QE 스펙트럼을 새로 돌렸는데 같은가? -> 실행 로그에
     '[builder] ... collection η0=0.945 r0=...' 줄이 보이는지 확인""")
sys.exit(0)
