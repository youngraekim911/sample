# -*- coding: utf-8 -*-
"""MaterialResolver — '물질 이름 -> (n,k) @ λ' 해석 단일 창구.

우선순위 (simulator 에 흩어져 있던 규칙을 한곳으로):
  ① materials/ 폴더 파일 (항상 최신 — 폴더가 단일 진실원). materials[name].src
     지정 시 그 파일, 아니면 동명 파일.
  ② IR dispersion 테이블 (폴더에 없을 때만 폴백 — 저장 yaml 의 embedded 스냅샷은
     폴더가 있으면 무시되어, 폴더 txt 를 바꾸면 재실행 시 반영됨)
  ③ materials 상수 {n,k}
k 는 전 경로에서 |k| (음수 k 파일 방어). 파장 단위 nm/µm 자동 감지(>100 → nm).
"""
import numpy as np


class MaterialResolver:
    def __init__(self, materials=None, dispersion=None, matlib=None):
        self.materials = materials or {}
        self.dispersion = dispersion or {}
        self.matlib = matlib

    # ------------------------------------------------------------- n,k
    def nk(self, name, lam):
        mconf = self.materials.get(name, {}) or {}
        # ⓪ 혼합 물질 (mix: [[이름, 부피분율], ...]) — 유효매질. 구성 물질이 다시
        #    폴더/분산/상수 어디서 오든 상관없어 λ 분산이 그대로 따라온다.
        if mconf.get("mix"):
            return self._mix_nk(mconf["mix"], lam)
        src = mconf.get("src", name)
        # ① 폴더 우선 (단일 진실원 — embedded 스냅샷보다 항상 최신)
        if self.matlib and self.matlib.has(src):
            return self.matlib.nk(src, lam)
        if self.matlib and self.matlib.has(name):
            return self.matlib.nk(name, lam)
        # ② embedded dispersion (폴더에 없을 때만)
        disp = self.dispersion.get(name)
        if disp is not None:
            arr = np.array(disp, dtype=float)          # [[lam,n,k],...]
            arr = arr[np.argsort(arr[:, 0])]           # 파장 오름차순 (np.interp 요구)
            L = arr[:, 0]
            if L.max() > 100:                          # nm 테이블 자동 감지 (library 와 통일)
                L = L / 1000.0
            n = float(np.interp(lam, L, arr[:, 1]))
            k = abs(float(np.interp(lam, L, arr[:, 2])))
            return n, k
        # ③ 상수
        if "n" not in mconf:
            raise KeyError(f"물질 '{name}': 폴더/dispersion/상수 어디에도 정의 없음")
        return float(mconf["n"]), abs(float(mconf.get("k", 0)))

    def eps(self, name, lam):
        n, k = self.nk(name, lam)
        return complex(n, k) ** 2

    # ------------------------------------------------------------- 혼합(EMT)
    def _mix_nk(self, spec, lam):
        """부피평균 유전율 ε_eff = Σ fᵢ·εᵢ 로부터 (n,k).

        산술(병렬) 평균을 쓰는 이유: RCWA 는 ε 의 푸리에 계수를 그대로 소비하므로
        (Laurent rule), ∫ε 를 보존하는 산술평균이 해석기 정식화와 일치한다.
        분율 합이 1 이 아니면 나머지는 마지막 항목이 아니라 오류로 본다(호출측 책임).
        """
        e = 0j
        tot = 0.0
        for name, f in spec:
            f = float(f)
            e += f * self.eps(str(name), lam)
            tot += f
        if abs(tot - 1.0) > 1e-6:
            raise ValueError(f"mix 분율 합이 1 이 아님: {tot:.6f}")
        n = complex(e) ** 0.5
        if n.imag < 0:                                   # 흡수 매질 분기 고정
            n = -n
        return float(n.real), abs(float(n.imag))

    # ------------------------------------------------------------- 출처/범위 (진단용)
    def source(self, name):
        """(출처 문자열, 파장범위(µm) 또는 None) — nk() 와 동일 우선순위."""
        mconf = self.materials.get(name, {}) or {}
        if mconf.get("mix"):
            parts = " + ".join(f"{n}@{float(f):.3f}" for n, f in mconf["mix"])
            return f"유효매질(mix: {parts})", None
        src = mconf.get("src", name)
        if self.matlib and (self.matlib.has(src) or self.matlib.has(name)):
            key = src if self.matlib.has(src) else name
            return "materials 폴더", self.matlib.lam_range(key)
        disp = self.dispersion.get(name)
        if disp is not None:
            arr = np.array(disp, dtype=float)
            L = arr[:, 0] / (1000.0 if arr[:, 0].max() > 100 else 1.0)
            return "yaml dispersion(폴백)", (float(L.min()), float(L.max()))
        return "yaml 상수", None
