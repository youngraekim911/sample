# -*- coding: utf-8 -*-
"""RCWAPlaneWaveSimulator — qcell 구조(YAML/npy) + RCWASolver 연결.

흐름:  config -> build_qcell 구조 -> z-slice 층 스택 -> 파장별 eps -> RCWASolver
       -> R / T(=Si 결합 QE) / A_stack

QE 정의(검증된 경로)
--------------------
투과매질을 Si 반무한으로 두면 T = Si 로 결합되어 흡수될 광량 = 광학 QE.
(반사 R + 스택 흡수 A_stack(metal/CF) + Si 결합 T = 1)
※ 픽셀별 QE / 3D |E|^2 볼륨(Si 내부 필드)은 rcwa.RCWASolver 의 필드복원
  메서드(실험적) 확장으로 이어짐.
"""
import os
import numpy as np
import torch

from ..structure.builder import QcellBuilder
from ..structure.wizard_builder import WizardBuilder
from ..config.loader import load_config
from ..rcwa import RCWASolver
from ..materials.library import MaterialLibrary


class RCWAPlaneWaveSimulator:
    def __init__(self, config_path, nG=101, downsample=2, trunc="circular",
                 device=None, dtype=torch.complex128, materials_dir=None):
        self.cfg = load_config(config_path)
        self.base_dir = os.path.dirname(os.path.abspath(config_path))
        # 물질 폴더(파장별 n,k) 자동 탐색: 지정값 -> conf 옆 materials -> repo data/materials
        cands = []
        if materials_dir:
            cands.append(materials_dir if os.path.isabs(materials_dir)
                         else os.path.join(self.base_dir, materials_dir))
        cands.append(os.path.join(self.base_dir, "materials"))
        cands.append(os.path.join(self.base_dir, "..", "data", "materials"))
        cands.append(os.path.join(os.getcwd(), "data", "materials"))
        mdir = next((d for d in cands if os.path.isdir(d)), None)
        self.matlib = MaterialLibrary(mdir) if mdir else None
        if self.matlib:
            print(f"[materials] loaded {len(self.matlib.names())} from {mdir}")
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype
        self.nG = nG
        self.trunc = trunc
        self.ds = max(1, int(downsample))

        # 구조 생성 — 스키마 자동 인식: wizard v3(stack.si/cf) vs 구버전 qcell
        stack = self.cfg.get("stack") or {}
        if "si" in stack and "cf" in stack:
            b = WizardBuilder(self.cfg, base_dir=self.base_dir)
            print("[builder] wizard v3 schema")
        else:
            b = QcellBuilder(self.cfg, base_dir=self.base_dir)
        self.builder = b
        self.matid = b.build()                      # [nz,ny,nx] uint8
        self.meta = b.meta()
        self.dz = b.dz
        self.dxy = b.dxy
        self.span = b.span                          # 단위셀 주기 (um)
        self.id2name = {int(m["id"]): n for n, m in self.meta["materials"].items()}
        self.substrate = self.meta.get("substrate_material", "si")

        # 층 스택(z-slice 병합) 준비
        self._prepare_layers()

        # 픽셀별 QE 마스크 (wizard 스키마): 픽셀 1×1 의 순수 Si 창(DTI 트렌치 제외)
        self._pix = None
        if hasattr(b, "pixel_maps"):
            pixidx, colors = b.pixel_maps()
            trench = b.dti_trench_mask()
            ds = self.ds
            self._pix = {"pixidx": pixidx[::ds, ::ds], "trench": trench[::ds, ::ds],
                         "colors": colors, "npx": b.npx}

    # -----------------------------------------------------------------
    def _prepare_layers(self):
        """Si-top ~ microlens-top 구간을 z-slice 병합해 (matid2d, thickness) 리스트로."""
        bounds = {l["name"]: l for l in self.meta["layer_bounds_vox"]}
        z_si_top = bounds["substrate_si"]["z1"]     # Si 반무한(투과매질)의 위 경계
        z_top = bounds["microlens"]["z1"]           # 마이크로렌즈 꼭대기 (그 위 air=입사매질)

        mat = self.matid[z_si_top:z_top]            # [nz', ny, nx]
        ds = self.ds
        mat = mat[:, ::ds, ::ds]                    # 가로 다운샘플
        self.grid_ny, self.grid_nx = mat.shape[1], mat.shape[2]

        layers = []                                 # (matid2d, thickness_um)
        prev = None; count = 0
        for z in range(mat.shape[0]):
            sl = mat[z]
            if prev is not None and np.array_equal(sl, prev):
                count += 1
            else:
                if prev is not None:
                    layers.append((prev, count * self.dz))
                prev = sl; count = 1
        layers.append((prev, count * self.dz))
        self.layer_stack = layers                   # 위->아래는 아래 build 순
        # z-slice 는 아래(z작음)->위 순서. RCWA 는 입사(위)->투과(아래) 순으로 쌓아야 하므로 반전.
        self.layer_stack = list(reversed(layers))

    # -----------------------------------------------------------------
    def _nk_at(self, name, lam):
        """물질 name 의 (n,k) @ lam(um).  우선순위: materials/ 폴더(src|name) -> dispersion -> 상수."""
        mconf = (self.cfg.get("materials", {}) or {}).get(name, {}) or {}
        src = mconf.get("src", name)                 # 물질별 n,k 파일 지정(src) 우선
        if self.matlib and self.matlib.has(src):
            return self.matlib.nk(src, lam)
        if self.matlib and self.matlib.has(name):
            return self.matlib.nk(name, lam)
        disp = self.cfg.get("dispersion", {})
        if name in disp:
            arr = np.array(disp[name], dtype=float)     # [[lam,n,k],...]
            n = float(np.interp(lam, arr[:, 0], arr[:, 1]))
            k = float(np.interp(lam, arr[:, 0], arr[:, 2]))
            return n, k
        m = self.cfg["materials"][name]
        return float(m["n"]), float(m["k"])

    def _eps_lut(self, lam):
        """id -> 복소 eps=(n+ik)^2 @ lam."""
        lut = {}
        for mid, name in self.id2name.items():
            n, k = self._nk_at(name, lam)
            lut[mid] = complex(n, k) ** 2
        return lut

    def _eps_grid(self, matid2d, eps_lut):
        g = np.empty(matid2d.shape, dtype=np.complex128)
        for mid, eps in eps_lut.items():
            g[matid2d == mid] = eps
        return torch.as_tensor(g, dtype=self.dtype, device=self.device)

    # -----------------------------------------------------------------
    def run(self, wavelength, theta=0.0, phi=0.0, pol_te=1.0, pol_tm=0.0,
            pixel_qe=True):
        """단일 파장 RCWA -> R, QE(Si 결합 T), A_stack (+픽셀/컬러별 QE).

        픽셀별 QE 정의: 투과(Si) 계면의 공간 Poynting flux Sz(x,y) 를
        '픽셀 1×1 안의 순수 Si 창(DTI 트렌치·라이너 제외)' 마스크로 적분,
        그 픽셀 전체 면적에 입사한 파워로 정규화.
        컬러별 QE = 같은 bayer 색 픽셀들의 평균 (CF merge average).
        """
        lam = float(wavelength)
        eps_lut = self._eps_lut(lam)
        eps_inc = 1.0                                    # air (위)
        n_s, k_s = self._nk_at(self.substrate, lam)      # 기판(반무한 투과 매질)
        eps_trn = complex(n_s, k_s) ** 2

        solver = RCWASolver(lam, self.span, self.span, nG=self.nG,
                            theta=theta, phi=phi, trunc=self.trunc,
                            device=self.device, dtype=self.dtype)
        solver.setup_incidence(eps_inc, eps_trn)
        for matid2d, th in self.layer_stack:
            solver.add_layer(th, eps_grid=self._eps_grid(matid2d, eps_lut))
        o = solver.solve(pol_te=pol_te, pol_tm=pol_tm)
        out = {"wavelength": lam, "R": o["R"], "QE": o["T"],
               "A_stack": o["A"], "nG": solver.nG, "n_layers": len(self.layer_stack)}

        if pixel_qe and self._pix is not None:
            Sz = solver.transmitted_flux_map(self.grid_ny, self.grid_nx)
            Sz = Sz.detach().cpu().numpy()               # 평균 == T (calibrated)
            pixidx, trench = self._pix["pixidx"], self._pix["trench"]
            colors, npx = self._pix["colors"], self._pix["npx"]
            ngrid = Sz.size
            qe_pix = []
            for pidx in range(npx * npx):
                m = (pixidx == pidx) & (~trench)         # 픽셀 내 순수 Si 창
                # (픽셀 flux 합/전체 셀) ÷ (픽셀 면적/전체 면적 = 1/npx²)
                qe_pix.append(float(Sz[m].sum() / ngrid * (npx * npx)))
            qe_rgb = {c: float(np.mean([q for q, cc in zip(qe_pix, colors) if cc == c]))
                      for c in "RGB" if c in colors}
            out["QE_pixels"] = qe_pix
            out["QE_rgb"] = qe_rgb
            out["QE_trench"] = float(o["T"] - sum(qe_pix) / (npx * npx))  # 트렌치로 들어간 몫
        return out
