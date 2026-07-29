# -*- coding: utf-8 -*-
"""RBF 보간 버그 수정 검증 — cis_rcwa 루트에서 실행.

    PYTHONPATH=. python3 verify_rbf_fix.py

RCWA 없이 합성 데이터로 몇 초 안에 끝납니다.
수정 전이면 1·2·4번이 실패하고, 수정 후면 전부 통과합니다.
"""
import sys
import numpy as np

from src.sim.emulator import (fit_emulator, predict_emulator,
                              _rbf_fit, _rbf_predict, _rbf_epsilon,
                              _pairwise, _rbf_phi, _r2_rmse)

OK, NG = "\033[92m✓\033[0m", "\033[91m✗\033[0m"
fails = []


def check(name, cond, detail=""):
    print(f"  {OK if cond else NG} {name}   {detail}")
    if not cond:
        fails.append(name)


# 진동(공진성) 함수 — 다항식으로는 못 잡고 보간만 잡을 수 있는 형태
def truth(s):
    return 0.58 + 0.02 * np.sin(2.1 * s) + 0.004 * s


def make_doe(steps, axes_step=0.06):
    return {"axes": [["planar_um", "test knob", axes_step, "um"]],
            "mode": "lhs", "nG": 41, "bound": 6.0, "seed": 0,
            "wavelengths_nm": [550],
            "points": [{"steps": [float(s)],
                        "qe": {550: {"R": float(truth(s)),
                                     "G": float(truth(s)),
                                     "B": float(truth(s))}}}
                       for s in steps]}


print("\n" + "=" * 64)
print(" RBF 보간 버그 수정 검증")
print("=" * 64)

train = np.linspace(-6, 6, 21)
unseen = np.linspace(-6, 6, 21)[:-1] + 0.3          # 학습에 없는 중간점 20개

# ── 1) 보간성: 학습 샘플을 그대로 재현해야 함 ────────────────────────────
print("\n[1] RBF 보간성 (in-sample R² > 0.99 — 보간기의 정의)")
e = fit_emulator(make_doe(train), model="rbf", cv_folds=5)
ins = e["metrics"]["r2_insample_mean"]
check("in-sample R²", ins > 0.99, f"= {ins:+.4f}   (수정 전에는 실패 · 실측 -13.60)")

# ── 2) 일반화: 못 본 점에서도 맞아야 함 ─────────────────────────────────
print("\n[2] 진동 함수 일반화 (미본점 R² > 0.9)")
pred = np.array([predict_emulator(e, [float(s)], 550, "G") for s in unseen])
true = truth(unseen)
r2 = 1 - ((pred - true) ** 2).sum() / ((true - true.mean()) ** 2).sum()
check("미본점 R²", r2 > 0.9, f"= {r2:+.4f}   (수정 전에는 실패 · 실측 -13.24)")

# ── 3) 다항 경로 회귀 없음 ─────────────────────────────────────────────
print("\n[3] 다항 모델 회귀 없음 (수정과 무관해야 함)")
for m, lo, hi in (("quadratic", 0.0, 0.6), ("cubic", 0.0, 0.7)):
    em = fit_emulator(make_doe(train), model=m, cv_folds=5)
    v = em["metrics"]["r2_insample_mean"]
    check(f"{m} in-sample R²", lo <= v <= hi,
          f"= {v:+.4f}   (진동 데이터라 낮은 게 정상)")

# ── 4) 커널 3종 모두 정상 동작 (모듈 '기본 ridge' 로 — 하드코딩 금지) ────
import inspect
DEFAULT_RIDGE = inspect.signature(fit_emulator).parameters["ridge"].default
print(f"\n[4] 커널 3종 보간성 — 모듈 기본 ridge={DEFAULT_RIDGE:g} 사용")
X = (train / 2.0).reshape(-1, 1)                    # x = step/2 (에뮬레이터 계약)
y = truth(train)
eps = _rbf_epsilon(X)
for kern in ("multiquadric", "gaussian", "thinplate"):
    w, mu = _rbf_fit(X, y, eps, kern, DEFAULT_RIDGE)
    r2k, _ = _r2_rmse(y, _rbf_predict(X, X, w, mu, eps, kern))
    check(f"{kern:13s} in-sample R²", r2k > 0.99, f"= {r2k:+.4f}")

# ── 진단 정보: 왜 큰 ridge 가 위험한지 (참고용 출력) ─────────────────────
print("\n[참고] Φ 행렬의 음의 고유값 개수 — ridge 가 위험한 이유")
for kern in ("multiquadric", "gaussian", "thinplate"):
    Phi = _rbf_phi(_pairwise(X, X), eps, kern)
    ev = np.linalg.eigvalsh((Phi + Phi.T) / 2)
    neg = int((ev < -1e-9).sum())
    print(f"     {kern:13s} 음의 고유값 {neg:2d}/{len(ev)}개"
          + ("   ← 조건부 양정치: 큰 ridge 금지" if neg else "   (양정치: ridge 안전)"))

print("\n" + "=" * 64)
if fails:
    print(f" 실패 {len(fails)}건: {', '.join(fails)}")
    print(" → emulator.py 의 _rbf_fit(lstsq) 와 ridge 기본값(1e-8)을 확인하세요.")
    sys.exit(1)
print(" 전부 통과 — RBF 보간이 정상 동작합니다.")
sys.exit(0)
