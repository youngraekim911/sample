# -*- coding: utf-8 -*-
"""MaterialResolver — '물질 이름 -> (n,k) @ λ' 해석 단일 창구.

우선순위 (simulator 에 흩어져 있던 규칙을 한곳으로):
  ① IR dispersion 테이블 (위저드가 브라우저 import 값을 그대로 동봉 — 화면=계산)
  ② materials/ 폴더 파일 (materials[name].src 지정 시 그 파일, 아니면 동명 파일)
  ③ materials 상수 {n,k}
k 는 전 경로에서 |k| (음수 k 파일 방어). 파장 단위 nm/µm 자동 감지.
"""
import numpy as np


class MaterialResolver:
    def __init__(self, materials=None, dispersion=None, matlib=None):
        self.materials = materials or {}
        self.dispersion = dispersion or {}
        self.matlib = matlib

    # ------------------------------------------------------------- n,k
    def nk(self, name, lam):
        disp = self.dispersion.get(name)
        if disp is not None:
            arr = np.array(disp, dtype=float)          # [[lam,n,k],...]
            L = arr[:, 0]
            if L.max() > 20:                           # nm 테이블 자동 감지
                L = L / 1000.0
            n = float(np.interp(lam, L, arr[:, 1]))
            k = abs(float(np.interp(lam, L, arr[:, 2])))
            return n, k
        mconf = self.materials.get(name, {}) or {}
        src = mconf.get("src", name)
        if self.matlib and self.matlib.has(src):
            return self.matlib.nk(src, lam)
        if self.matlib and self.matlib.has(name):
            return self.matlib.nk(name, lam)
        if "n" not in mconf:
            raise KeyError(f"물질 '{name}': dispersion/폴더/상수 어디에도 정의 없음")
        return float(mconf["n"]), abs(float(mconf.get("k", 0)))

    def eps(self, name, lam):
        n, k = self.nk(name, lam)
        return complex(n, k) ** 2

    # ------------------------------------------------------------- 출처/범위 (진단용)
    def source(self, name):
        """(출처 문자열, 파장범위(µm) 또는 None)"""
        disp = self.dispersion.get(name)
        if disp is not None:
            arr = np.array(disp, dtype=float)
            L = arr[:, 0] / (1000.0 if arr[:, 0].max() > 20 else 1.0)
            return "yaml dispersion(브라우저 테이블)", (float(L.min()), float(L.max()))
        mconf = self.materials.get(name, {}) or {}
        src = mconf.get("src", name)
        if self.matlib and (self.matlib.has(src) or self.matlib.has(name)):
            key = src if self.matlib.has(src) else name
            return "materials 폴더", self.matlib.lam_range(key)
        return "yaml 상수", None
