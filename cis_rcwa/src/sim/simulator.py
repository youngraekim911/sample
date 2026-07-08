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
                 device=None, dtype=torch.complex128, materials_dir=None,
                 mesh="auto", lateral_um=0.005):
        """mesh:
            "auto"  — 적응 mesh (기본): z 는 해석적 층 경계(복셀화 없음 — BARL Å,
                      20nm 코팅 등 정확), 가로는 lateral_um×downsample 미세 래스터.
            "voxel" — 구버전: npy 복셀(dz/dxy)에서 z-slice 병합.
        """
        # ---- complex64 eps 텐서 직접 입력 모드 (<name>_eps.npy + <name>_meta.json) ----
        if str(config_path).endswith(".npy"):
            self._init_eps_direct(config_path, nG, downsample, trunc, device, dtype)
            return
        self._eps_direct = False
        self.mesh = mesh
        self.lateral_um = float(lateral_um)
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
        self.dz = b.dz
        self.dxy = b.dxy
        self.span = b.span                          # 단위셀 주기 (um)
        wizard = hasattr(b, "pixel_maps")
        self._pix = None

        if wizard and self.mesh == "auto":
            # ---- 적응 mesh: z 해석적 층 경계 + 가로 미세 래스터 (npy 복셀과 독립) ----
            Nlat = int(round(self.span / (self.lateral_um * self.ds)))
            Nlat = min(max(Nlat, 128), 2400)
            bl = WizardBuilder(self.cfg, base_dir=self.base_dir).set_lateral(Nlat)
            self.layer_stack, self.n_si_layers = bl.rcwa_layers()
            self.grid_ny = self.grid_nx = Nlat
            self.si_band_um = float(self.cfg["stack"]["si"]["thickness_um"])
            self.id2name = {int(i): n for n, i in bl._idx.items()}
            self.substrate = self.cfg["stack"]["si"]["material"]
            self.matid = None; self.meta = None
            pixidx, colors = bl.pixel_maps()
            trench = bl.dti_trench_mask()
            self._pix = {"pixidx": pixidx, "trench": trench,
                         "colors": colors, "npx": bl.npx}
            print(f"[mesh] auto: lateral {Nlat}×{Nlat} ({self.span/Nlat*1000:.1f}nm)"
                  f" · layers {len(self.layer_stack)} (z 해석적 경계, 복셀화 없음)")
        else:
            # ---- 구버전 voxel mesh ----
            self.matid = b.build()                  # [nz,ny,nx] uint8
            self.meta = b.meta()
            self.id2name = {int(m["id"]): n for n, m in self.meta["materials"].items()}
            self.substrate = self.meta.get("substrate_material", "si")
            self._prepare_layers()                  # z-slice 병합
            if wizard:
                pixidx, colors = b.pixel_maps()
                trench = b.dti_trench_mask()
                ds = self.ds
                self._pix = {"pixidx": pixidx[::ds, ::ds], "trench": trench[::ds, ::ds],
                             "colors": colors, "npx": b.npx}

    # -----------------------------------------------------------------
    def _init_eps_direct(self, eps_path, nG, downsample, trunc, device, dtype):
        """complex64 eps 텐서([nz,ny,nx], (n+ik)² @ 고정 λ)를 RCWA 에 직접 입력.

        부속 meta json (<name>_meta.json, 위저드가 함께 저장):
          voxel_um(dz/dx), eps_lambda_um, substrate_material(+materials n,k),
          si_thickness_um, pixel_pitch_um, n_pixels, bayer  -> 픽셀별 QE 까지 지원.
        주의: eps 는 λ 고정 스냅샷 — sweep 시 물질 분산(파장의존 n,k)은 반영 안 됨.
        """
        import json
        self._eps_direct = True
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype
        self.nG = nG
        self.trunc = trunc
        self.ds = max(1, int(downsample))
        arr = np.load(eps_path)                      # complex64 [nz,ny,nx]
        mpath = eps_path.replace("_eps.npy", "_meta.json")
        meta = json.load(open(mpath, encoding="utf-8"))
        self.meta = meta
        self.dz = float(meta["voxel_um"]["dz"])
        self.dxy = float(meta["voxel_um"]["dx"])
        self.eps_lambda = float(meta.get("eps_lambda_um", 0) or 0)
        self.span = arr.shape[2] * self.dxy
        self.substrate = meta.get("substrate_material", "si")
        mats = {m["name"]: m for m in meta["materials"]}
        ms = mats.get(self.substrate, {"n": 4.08, "k": 0.028})
        self._eps_trn_fixed = complex(float(ms["n"]), float(ms["k"])) ** 2
        print(f"[eps-direct] {arr.shape} complex64 @ λ={self.eps_lambda}µm (분산 고정)")

        ds = self.ds
        a = arr[:, ::ds, ::ds].astype(np.complex128)
        self.grid_ny, self.grid_nx = a.shape[1], a.shape[2]
        layers = []
        prev = None; count = 0
        for z in range(a.shape[0]):
            sl = a[z]
            if prev is not None and np.array_equal(sl, prev):
                count += 1
            else:
                if prev is not None:
                    layers.append((prev, count * self.dz))
                prev = sl; count = 1
        layers.append((prev, count * self.dz))
        self.layer_stack = list(reversed(layers))    # 입사(위)->투과(아래)
        # Si 밴드(층 두께 합 == si_thickness_um) 식별 — 픽셀별 QE 용
        self.si_band_um = float(meta.get("si_thickness_um", 0) or 0)
        self.n_si_layers = 0
        acc = 0.0
        for m2d, th in reversed(self.layer_stack):
            if acc + 1e-9 >= self.si_band_um:
                break
            acc += th; self.n_si_layers += 1
        # 픽셀 마스크 — 우선순위: 옆에 저장된 <name>.yaml (정확한 기하) -> meta 근사
        self._pix = None
        ypath = eps_path.replace("_eps.npy", ".yaml")
        if self.si_band_um > 0 and os.path.exists(ypath):
            from ..structure.wizard_builder import WizardBuilder
            b = WizardBuilder(load_config(ypath))
            pixidx, colors = b.pixel_maps()
            trench = b.dti_trench_mask()
            self._pix = {"pixidx": pixidx[::ds, ::ds], "trench": trench[::ds, ::ds],
                         "colors": colors, "npx": b.npx}
        elif self.si_band_um > 0 and meta.get("bayer"):
            npx = int(meta["n_pixels"]); p = float(meta["pixel_pitch_um"])
            yy, xx = np.meshgrid(np.arange(self.grid_ny), np.arange(self.grid_nx), indexing="ij")
            # 다운샘플 격자의 원본 셀 중심: (idx*ds + 0.5)*dxy  (yaml 경로와 동일 규약)
            pr = np.clip(((yy * ds + 0.5) * self.dxy / p).astype(int), 0, npx - 1)
            pc = np.clip(((xx * ds + 0.5) * self.dxy / p).astype(int), 0, npx - 1)
            pixidx = (pr * npx + pc).astype(np.int32)
            si_sl = self.layer_stack[-1][0]          # Si 밴드 슬라이스 (마지막 층)
            eps_si = self._eps_trn_fixed
            trench = np.abs(si_sl - eps_si) > 1e-6   # 라이너만 검출 (fill=si 는 근사 한계)
            bay = meta["bayer"]
            colors = [bay[r][c] for r in range(npx) for c in range(npx)]
            self._pix = {"pixidx": pixidx, "trench": trench, "colors": colors, "npx": npx}

    # -----------------------------------------------------------------
    def _prepare_layers(self):
        """구조를 z-slice 병합해 (matid2d, thickness) 리스트로.

        wizard 스키마: Si 밴드(DTI 트렌치 포함)를 '패턴층'으로 스택에 포함
        -> DTI 벽의 반사/굴절/도파(픽셀 격리)가 광학 계산에 반영.
        투과 매질은 그 아래 균일 Si 반무한 (트렌치 바닥 아래 벌크).
        구버전 qcell 스키마: 기존대로 Si-top 위만 스택 (호환).
        """
        bounds = {l["name"]: l for l in self.meta["layer_bounds_vox"]}
        z_si_top = bounds["substrate_si"]["z1"]     # Si 밴드 위 경계
        z_top = bounds["microlens"]["z1"]           # 마이크로렌즈 꼭대기 (그 위 air=입사매질)
        self._wizard = hasattr(self.builder, "pixel_maps")
        z_lo = 0 if self._wizard else z_si_top      # wizard: Si(DTI) 밴드 포함

        mat = self.matid[z_lo:z_top]                # [nz', ny, nx]
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
        # z-slice 는 아래(z작음)->위 순서. RCWA 는 입사(위)->투과(아래) 순으로 쌓아야 하므로 반전.
        self.layer_stack = list(reversed(layers))
        # Si(DTI) 밴드 = 반전 후 마지막 층들 (z-균일 패턴이라 병합되어 보통 1층)
        self.si_band_um = (z_si_top - z_lo) * self.dz
        self.n_si_layers = 0
        if self._wizard:
            acc = 0.0
            for m2d, th in reversed(self.layer_stack):
                if acc + 1e-9 >= self.si_band_um:
                    break
                acc += th; self.n_si_layers += 1

    # -----------------------------------------------------------------
    def _nk_at(self, name, lam):
        """물질 name 의 (n,k) @ lam(um).

        우선순위: ① cfg dispersion (위저드가 브라우저 import 테이블을 yaml 에 동봉
        — 사용자가 화면에서 본 값 그대로) -> ② materials/ 폴더(src|name)
        -> ③ yaml materials 상수."""
        disp = self.cfg.get("dispersion", {}) or {}
        if name in disp:
            arr = np.array(disp[name], dtype=float)     # [[lam,n,k],...]
            L = arr[:, 0]
            if L.max() > 20:                            # nm 단위 테이블 자동 감지
                L = L / 1000.0
            n = float(np.interp(lam, L, arr[:, 1]))
            k = float(np.interp(lam, L, arr[:, 2]))
            return n, k
        mconf = (self.cfg.get("materials", {}) or {}).get(name, {}) or {}
        src = mconf.get("src", name)                 # 물질별 n,k 파일 지정(src)
        if self.matlib and self.matlib.has(src):
            return self.matlib.nk(src, lam)
        if self.matlib and self.matlib.has(name):
            return self.matlib.nk(name, lam)
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
        eps_inc = 1.0                                    # air (위)
        if self._eps_direct:
            if self.eps_lambda and abs(lam - self.eps_lambda) > 1e-9:
                print(f"[warn] eps 텐서는 λ={self.eps_lambda}µm 스냅샷 — λ={lam} 에서 분산 미반영")
            eps_trn = self._eps_trn_fixed
        else:
            eps_lut = self._eps_lut(lam)
            n_s, k_s = self._nk_at(self.substrate, lam)  # 기판(반무한 투과 매질)
            eps_trn = complex(n_s, k_s) ** 2

        solver = RCWASolver(lam, self.span, self.span, nG=self.nG,
                            theta=theta, phi=phi, trunc=self.trunc,
                            device=self.device, dtype=self.dtype)
        solver.setup_incidence(eps_inc, eps_trn)
        for matid2d, th in self.layer_stack:
            if self._eps_direct:
                solver.add_layer(th, eps_grid=torch.as_tensor(matid2d, dtype=self.dtype,
                                                              device=self.device))
            else:
                solver.add_layer(th, eps_grid=self._eps_grid(matid2d, eps_lut))
        o = solver.solve(pol_te=pol_te, pol_tm=pol_tm)
        out = {"wavelength": lam, "R": o["R"], "QE": o["T"],
               "A_stack": o["A"], "nG": solver.nG, "n_layers": len(self.layer_stack)}

        if pixel_qe == "diag":
            out["_solver"] = solver
            return out

        if pixel_qe and self._pix is not None and self.n_si_layers > 0:
            # ---- 픽셀별 QE = 픽셀 Si 볼륨(DTI 트렌치 제외) 내 3D 흡수 + 밴드 아래 심부 Si ----
            # 절대 스케일: Σ(전 손실층 흡수) = 1−R−T (에너지 보존) 로 고정 — 정확.
            maps, C = solver.absorption_maps(self.grid_ny, self.grid_nx)
            M = len(self.layer_stack)
            nS = self.n_si_layers
            T_deep = o["T"]                              # DTI 바닥 아래 심부 Si 로 가는 몫 (정확)
            pixidx, trench = self._pix["pixidx"], self._pix["trench"]
            colors, npx = self._pix["colors"], self._pix["npx"]
            ngrid = pixidx.size
            pix_abs = np.zeros(npx * npx); trench_abs = 0.0; A_band = 0.0
            for li in range(M - nS, M):                  # Si 밴드 층들 (보통 1층)
                if li not in maps:
                    continue
                dens = maps[li].detach().cpu().numpy() * C
                A_band += dens.sum()
                for pidx in range(npx * npx):
                    m = (pixidx == pidx) & (~trench)     # 픽셀 1×1 의 순수 Si 창 (라이너 안쪽)
                    pix_abs[pidx] += dens[m].sum()
                trench_abs += dens[trench].sum()         # DTI(라이너+채움) 내 흡수 = 제외분
            # 심부(트렌치 바닥 아래) Si: 픽셀 사각형 기준 배분 (DTI 없음)
            Sz = solver.transmitted_flux_map(self.grid_ny, self.grid_nx).detach().cpu().numpy()
            for pidx in range(npx * npx):
                pix_abs[pidx] += Sz[pixidx == pidx].sum() / ngrid
            qe_top = float(A_band + T_deep)              # Si 밴드 유입 총량 (A_band 는 trench 포함 전체)
            qe_pix = [float(v * npx * npx) for v in pix_abs]   # 픽셀 면적 정규화
            qe_rgb = {c: float(np.mean([q for q, cc in zip(qe_pix, colors) if cc == c]))
                      for c in "RGB" if c in colors}
            out["QE"] = qe_top                           # Si 로 들어간 총 파워 (기존 정의 유지)
            out["A_stack"] = float(1.0 - o["R"] - qe_top)
            out["QE_pixels"] = qe_pix
            out["QE_rgb"] = qe_rgb
            out["QE_trench"] = float(trench_abs)         # DTI 트렌치 내부 흡수분 (픽셀 제외)
            out["QE_deep"] = float(T_deep)
        return out

    # -----------------------------------------------------------------
    def diagnose(self, wavelength, theta=0.0):
        """QE 이상 원인 추적 — 물질 n,k 점검 + 경계 투과(T) 프로파일.

        반환 dict:
          materials: [{name, n, k, source, lam_range, flags[]}]  (사용 물질 전부)
          profile:   [{mats, th_um, A, T_after}]  air 직후(1-R)부터 Si 유입까지
                     (같은 물질 조합의 연속 층은 묶음 — ML 슬라이스 등)
          R, T_into_si, T_deep
        """
        lam = float(wavelength)
        o = self.run(lam, theta=theta, pol_te=1.0, pol_tm=0.0, pixel_qe="diag")
        solver = o.pop("_solver")
        maps, C = solver.absorption_maps(self.grid_ny, self.grid_nx)
        M = len(self.layer_stack)
        A = [float(maps[i].sum() * C) if i in maps else 0.0 for i in range(M)]
        # ---- 물질 표 + 의심 플래그 ----
        disp = self.cfg.get("dispersion", {}) or {} if not self._eps_direct else {}
        mats = []
        for mid in sorted(self.id2name):
            name = self.id2name[mid]
            n, k = self._nk_at(name, lam) if not self._eps_direct else (0, 0)
            if name in disp:
                src = "yaml dispersion(브라우저 테이블)"
                arr = np.array(disp[name], dtype=float)
                L = arr[:, 0] / (1000.0 if arr[:, 0].max() > 20 else 1.0)
                rng = (float(L.min()), float(L.max()))
            elif self.matlib and (self.matlib.has((self.cfg.get("materials", {}).get(name, {}) or {}).get("src", name))
                                  or self.matlib.has(name)):
                src = "materials 폴더"
                key = (self.cfg.get("materials", {}).get(name, {}) or {}).get("src", name)
                rng = self.matlib.lam_range(key if self.matlib.has(key) else name)
            else:
                src = "yaml 상수"
                rng = None
            flags = []
            if name != "air" and not (0.9 <= n <= 8.5):
                flags.append(f"n={n:.3g} 비정상 범위(열 순서/단위 확인)")
            if k < 0:
                flags.append("k<0 (비물리)")
            if k > 0.5 and name not in ("si",) and "metal" not in name:
                flags.append(f"k={k:.3g} 강흡수 — 흡수계수(α)를 k 로 오인했는지 확인")
            if rng and not (rng[0] - 1e-9 <= lam <= rng[1] + 1e-9):
                flags.append(f"λ={lam} 가 테이블 범위({rng[0]:.2f}~{rng[1]:.2f}µm) 밖 -> 경계값 사용")
            mats.append({"name": name, "n": round(n, 4), "k": round(k, 5),
                         "source": src, "lam_range": rng, "flags": flags})
        # ---- 색 픽셀별 워터폴: 각 색 픽셀 '기둥'의 누적 흡수를 진입값에서 차감 ----
        # 흡수 밀도 맵(에너지 보존 정합, 검증됨) 기반 — 각 색 픽셀 면적당 입사=1.
        # (기둥 사이 빛의 가로 이동은 흡수가 일어난 기둥에 귀속되는 근사)
        cmask = None
        acc = None
        if self._pix is not None:
            pixidx, colors = self._pix["pixidx"], self._pix["colors"]
            cmask = {c: np.isin(pixidx, [i for i, cc in enumerate(colors) if cc == c])
                     for c in "RGB" if c in colors}
            acc = {c: 0.0 for c in cmask}
        ngrid = self.grid_ny * self.grid_nx

        # ---- 경계 투과 프로파일 (같은 물질 조합 연속층 묶음) ----
        prof = []
        Tcur = 1.0 - o["R"]
        cur = None
        for i, (m2d, th) in enumerate(self.layer_stack):
            ids = np.unique(np.asarray(m2d)) if not self._eps_direct else []
            nm = "+".join(sorted(self.id2name[int(x)] for x in ids)) if len(ids) else f"layer{i}"
            Tcur -= A[i]
            if cmask is not None and i in maps:
                dens = maps[i].detach().cpu().numpy() * C
                for c, m in cmask.items():
                    acc[c] += float(dens[m].mean()) * ngrid   # 색 면적당 흡수
            rgb = ({c: round(1.0 - o["R"] - acc[c], 5) for c in cmask}
                   if cmask is not None else None)
            if cur and cur["mats"] == nm:
                cur["th_um"] += th; cur["A"] += A[i]; cur["T_after"] = Tcur
                cur["n_sub"] += 1; cur["T_rgb"] = rgb
            else:
                cur = {"mats": nm, "th_um": th, "A": A[i], "T_after": Tcur,
                       "n_sub": 1, "T_rgb": rgb}
                prof.append(cur)
        entry_rgb = ({c: round(1.0 - o["R"], 5) for c in cmask}
                     if cmask is not None else None)         # 진입(1-R, 균일 근사)
        for p in prof:
            p["th_um"] = round(p["th_um"], 4); p["A"] = round(p["A"], 5)
            p["T_after"] = round(p["T_after"], 5)
        nS = self.n_si_layers
        A_above = sum(A[:M - nS]) if nS else sum(A)
        return {"wavelength": lam, "R": round(o["R"], 5), "materials": mats,
                "profile": prof, "entry_rgb": entry_rgb,
                "T_into_si": round(1.0 - o["R"] - A_above, 5),   # Si 밴드 유입 = 광학 QE
                "T_deep": round(o["QE"], 5)}                     # 밴드 바닥 통과분
