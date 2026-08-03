# -*- coding: utf-8 -*-
"""materials — materials/ 폴더의 물질별 txt(파장별 n,k) 자동 로드 + 보간.

txt 형식 (열):  wavelength  n  k
  - 주석/빈줄(#) 허용
  - 파장 단위 자동 감지: 최대값 > 100 이면 nm 로 보고 um 로 변환
  - 물질마다 step/range 가 달라도 각자 보간 -> 어떤 wavelength 든 커버

지시자 (선택):  #! 키 값        ← '#' 으로 시작하므로 측정행에는 영향 없음
  #! k_floor 0.0105    이 물질의 k 하한. 물질에 딸린 성질이라 파일에 두면
                       어느 yaml(위저드 생성본 포함)로 돌려도 따라간다.
                       yaml 의 materials[이름].k_floor 가 있으면 그쪽이 우선.

usage:
    lib = MaterialLibrary("materials")
    n, k = lib.nk("si", 0.55)      # um
    eps  = lib.eps("si", 0.55)     # (n+ik)^2
"""
import os
import glob
import numpy as np


class MaterialLibrary:
    def __init__(self, folder="materials"):
        self.folder = folder
        self.tables = {}                 # name -> (lam_um[np], n[np], k[np])
        self.meta = {}                   # name -> {"k_floor": float, ...}  (#! 지시자)
        self.load()

    def load(self):
        self.tables.clear()
        self.meta.clear()
        for path in glob.glob(os.path.join(self.folder, "*.txt")):
            name = os.path.splitext(os.path.basename(path))[0]
            rows = []
            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("#!"):        # 지시자 — '#! 키 값'
                        d = line[2:].split()
                        if len(d) >= 2 and d[0] == "k_floor":
                            try:
                                self.meta.setdefault(name, {})["k_floor"] = abs(float(d[1]))
                            except ValueError:
                                pass
                        continue
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

    def names(self):
        return sorted(self.tables.keys())

    def has(self, name):
        return name in self.tables

    def k_floor(self, name):
        """파일에 '#! k_floor' 로 적힌 k 하한 (없으면 None)."""
        return self.meta.get(name, {}).get("k_floor")

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
        return float(np.interp(lam_um, lam, n)), abs(float(np.interp(lam_um, lam, k)))

    def eps(self, name, lam_um):
        n, k = self.nk(name, lam_um)
        return complex(n, k) ** 2


if __name__ == "__main__":
    lib = MaterialLibrary("materials")
    print("loaded:", lib.names())
    for lam in (0.45, 0.55, 0.62):
        n, k = lib.nk("si", lam)
        print(f"  si @ {lam*1000:.0f}nm : n={n:.3f} k={k:.4f}  eps={lib.eps('si',lam):.3f}")
