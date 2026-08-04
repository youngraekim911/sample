# -*- coding: utf-8 -*-
"""materials — materials/ 폴더의 물질별 txt(파장별 n,k) 자동 로드 + 보간.

txt 형식 (열):  wavelength  n  k
  - 주석/빈줄(#) 허용
  - 파장 단위 자동 감지: 최대값 > 100 이면 nm 로 보고 um 로 변환
  - 물질마다 step/range 가 달라도 각자 보간 -> 어떤 wavelength 든 커버

usage:
    lib = MaterialLibrary("materials")
    n, k = lib.nk("si", 0.55)      # um
    eps  = lib.eps("si", 0.55)     # (n+ik)^2
"""
import os
import glob
import numpy as np


SCATTER_FILE = "scatter_kfloor.txt"      # 물질별 산란 바닥 사이드카 (아래 참조)


class MaterialLibrary:
    def __init__(self, folder="materials"):
        self.folder = folder
        self.tables = {}                 # name -> (lam_um[np], n[np], k[np])
        self.floors = {}                 # name -> k_floor (산란 바닥)
        self.load()

    def load(self):
        self.tables.clear()
        self.floors.clear()
        self._load_floors()
        for path in glob.glob(os.path.join(self.folder, "*.txt")):
            name = os.path.splitext(os.path.basename(path))[0]
            if os.path.basename(path) == SCATTER_FILE:
                continue                 # 사이드카는 물질 테이블이 아님
            rows = []
            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    parts = line.replace(",", " ").split()
                    try:
                        vals = [float(x) for x in parts[:3]]
                    except ValueError:
                        continue
                    if len(vals) >= 3:
                        rows.append(vals[:3])
            if not rows:
                continue
            arr = np.array(sorted(rows, key=lambda r: r[0]), dtype=float)
            lam = arr[:, 0]
            if lam.max() > 100:          # nm -> um
                lam = lam / 1000.0
            # k 부호 규약(음수 k 파일) -> |k| 자동 변환
            self.tables[name] = (lam, arr[:, 1], np.abs(arr[:, 2]))
        return self

    def _load_floors(self):
        """materials/scatter_kfloor.txt — 안료 산란 바닥 (물질의 성질, 구조 아님).

        형식:  물질이름  k_floor   (한 줄에 하나, # 주석 허용)

        왜 여기에 있나 — 평막 투과 n,k 측정은 안료 입자 산란광이 검출기로
        들어가 '투과'로 집계되므로 산란 소광을 원리적으로 못 잡는다. 소자
        (서브µm 픽셀)에서는 그 산란이 옆 픽셀 손실이 된다. 이 빠진 몫은
        '그 안료(재료)의 성질'이므로 구조 yaml 이 아니라 물질 폴더에 둔다 —
        같은 CF 를 쓰는 어떤 구조/새 yaml 이든 자동 적용되고, 구조 파일은
        구조만 담는다. 적분구(haze) 실측이 오면 이 파일의 값을 교체할 것.
        """
        path = os.path.join(self.folder, SCATTER_FILE)
        if not os.path.exists(path):
            return
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        self.floors[parts[0]] = abs(float(parts[1]))
                    except ValueError:
                        continue

    def k_floor(self, name):
        return self.floors.get(name)

    def names(self):
        return sorted(self.tables.keys())

    def has(self, name):
        return name in self.tables

    def lam_range(self, name):
        """테이블 λ 커버 범위 (um) — 진단용."""
        if name not in self.tables:
            return None
        lam = self.tables[name][0]
        return float(lam.min()), float(lam.max())

    def nk(self, name, lam_um):
        """이름/파장(um) -> (n,k). 범위 밖은 경계값 clamp (np.interp 기본).

        k 는 |k| 로 반환한다. 측정 데이터가 n−ik 규약으로 적혀 음수 k 를 담고
        있는 경우가 흔한데(TiN/Ti 등 금속), 그대로 쓰면 이득 매질이 되어
        에너지 보존이 깨지고 R+T>1 로 발산한다. resolver 의 다른 경로들도
        동일하게 |k| 를 쓰므로 여기서만 예외가 되지 않게 맞춘다.
        """
        if name not in self.tables:
            raise KeyError(f"material '{name}' not in {self.folder}/")
        lam, n, k = self.tables[name]
        kk = abs(float(np.interp(lam_um, lam, k)))
        kf = self.floors.get(name)
        if kf:                            # 산란 바닥 (scatter_kfloor.txt) — _load_floors 참조
            kk = max(kk, kf)
        return float(np.interp(lam_um, lam, n)), kk

    def eps(self, name, lam_um):
        n, k = self.nk(name, lam_um)
        return complex(n, k) ** 2


if __name__ == "__main__":
    lib = MaterialLibrary("materials")
    print("loaded:", lib.names())
    for lam in (0.45, 0.55, 0.62):
        n, k = lib.nk("si", lam)
        print(f"  si @ {lam*1000:.0f}nm : n={n:.3f} k={k:.4f}  eps={lib.eps('si',lam):.3f}")
