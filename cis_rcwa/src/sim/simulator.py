# -*- coding: utf-8 -*-
"""RCWAPlaneWaveSimulator — StructureIR(공간분할+물질매핑) -> RCWA -> QE/진단.

블록 구조
---------
    [구조 블록]  WizardBuilder / eps npy / 임의 빌더  ──►  StructureIR
    [RCWA 블록]  이 파일 — IR 만 소비 (구조 스키마를 모름)
    [결과 블록]  QE 스펙트럼 / 픽셀·라벨별 QE / 워터폴 진단 / 구조 이미지(viz)

IR 계약 (src/structure/ir.py):
    layers            (region_map, thickness) 위->아래 — '공간' 번호
    region_materials  공간 -> 물질 이름 (물질 교체 = 이 dict 수정)
    detector          검출 밴드 + 픽셀 분할 + 제외 마스크 -> QE 집계
따라서 상부 구조(ML/CF/grid/DTI...)가 어떻게 바뀌어도, 새 빌더가 IR 만
만들어 주면 RCWA/QE 코드는 재작성이 필요 없다.

QE 정의(검증된 경로)
--------------------
전체 검출기 흡수 = 밴드 내 3D 흡수(A_band) + 밴드 아래 반무한 흡수(T_deep).
픽셀별 = 픽셀 창(제외 마스크 제거) 내 흡수 + 심부 flux 픽셀 귀속.
절대 스케일은 에너지 보존(1-R-T = Σ흡수)으로 고정.
"""
import os
import numpy as np
import torch

from ..structure.ir import StructureIR, ir_from_eps_npy, ir_from_voxels
from ..config.loader import load_config
from ..rcwa import RCWASolver
from ..materials.library import MaterialLibrary
from ..materials.resolver import MaterialResolver


class RCWAPlaneWaveSimulator:
    def __init__(self, config, nG=101, downsample=2, trunc="circular",
                 device=None, dtype=torch.complex128, materials_dir=None,
                 mesh="auto", lateral_um=0.005, fff=True):
        """config: StructureIR | 위저드 yaml 경로 | <name>_eps.npy 경로 | IR .npz 경로.

        mesh="auto"  : (yaml) z 해석적 층 경계 + 가로 미세 래스터 (권장)
        mesh="voxel" : (yaml) 구버전 복셀 z-slice 병합
        QE = 광학(내부) QE = Si 흡수율. (캐리어 수집효율은 별도 물리로 미포함.)
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype
        self.nG = nG
        self.trunc = trunc
        self.ds = max(1, int(downsample))
        self.fff = fff
        base_dir = "."

        # ---------- 구조 -> IR (세 가지 입구, 이후 코드는 IR 만 사용) ----------
        if isinstance(config, StructureIR):
            ir = config
        elif str(config).endswith(".npz"):
            ir = StructureIR.load_npz(config)
        elif str(config).endswith(".npy"):
            ir = ir_from_eps_npy(str(config), downsample=self.ds)
            print(f"[eps-direct] {len(ir.layers)} layers @ λ={ir.eps_lambda_um}µm (분산 고정)")
        else:                                        # yaml — 스키마 자동 인식
            cfg = load_config(config)
            base_dir = os.path.dirname(os.path.abspath(config))
            ir = self._ir_from_yaml(cfg, base_dir, mesh, lateral_um)
        self.ir = ir.validate()

        # ---------- 물질 해석기 (materials 폴더 자동 탐색) ----------
        cands = []
        if materials_dir:
            cands.append(materials_dir if os.path.isabs(materials_dir)
                         else os.path.join(base_dir, materials_dir))
        cands += [os.path.join(base_dir, "materials"),
                  os.path.join(base_dir, "..", "data", "materials"),
                  os.path.join(os.getcwd(), "data", "materials")]
        mdir = next((d for d in cands if os.path.isdir(d)), None)
        matlib = MaterialLibrary(mdir) if mdir else None
        if matlib:
            print(f"[materials] loaded {len(matlib.names())} from {mdir}")
        self.res = MaterialResolver(self.ir.materials, self.ir.dispersion, matlib)

        # ---------- 파생 속성 ----------
        self.layer_stack = self.ir.layers
        self.grid_ny, self.grid_nx = self.ir.grid_shape
        self.span = self.ir.span_x
        det = self.ir.detector
        self.n_si_layers = det.n_layers if det else 0
        self.n_below_band = det.n_below_band if det else 0   # 밴드 아래 비검출 층(반사경)
        self.si_band_um = det.band_um if det else 0.0

    # ------------------------------------------------------------- yaml -> IR
    @staticmethod
    def _auto_model_defaults(cfg):
        """yaml 에 없는 모델링 선택을 프로그램이 자동 판단 (명시값은 항상 우선).

        - dti.optical 미지정: 항상 광학 반영. 실제 소자에서 트렌치 경계는 저굴절
          라이너로 빛을 가둬 crosstalk 를 막는 광학 요소이므로 빼면 현실과 달라진다.
          문제였던 '서브파장 트렌치를 넣으면 오히려 crosstalk 가 커지는' 현상은 물리가
          아니라 푸리에 차수 부족(Gibbs 링잉)이었고, 이제 blocks.SiDtiBlock 이 유지
          차수에 맞는 등가 유효매질로 자동 치환해 해결한다(사용자 설정 불필요).
        - collection 미지정: 소자 QE 기본 표면 dead-layer η(z)=1-r0·exp(-z/ld) 적용.
          순수 광학 QE 를 원하면 yaml 에 collection: {r0: 0} 명시.
        """
        st = cfg.get("stack") or {}
        d = st.get("dti")
        if d and (d.get("mode") or "").lower() not in ("", "none") \
                and "optical" not in d:
            d["optical"] = True
        if "collection" not in cfg:
            cfg["collection"] = {"r0": 0.35, "ld_um": 0.175}

    def _ir_from_yaml(self, cfg, base_dir, mesh, lateral_um):
        stack = cfg.get("stack") or {}
        if "si" in stack and "cf" in stack:              # wizard v3
            g = cfg["grid"]
            span = float(g["pixel_pitch_um"]) * int(g["n_pixels"])
            if mesh == "auto":
                self._auto_model_defaults(cfg)
                # 블록 조립 경로 — dti.optical / collection 등 신규 기능 전부 반영
                # (레거시 WizardBuilder 와 기하 동등, 회귀[6] 보장). yaml 경로도 이 경로.
                from ..structure.blocks import ir_from_wizard_cfg, fourier_res_um
                Nlat = int(round(span / (float(lateral_um) * self.ds)))
                Nlat = min(max(Nlat, 128), 2400)
                res = fourier_res_um(span, self.nG, self.trunc)
                ir = ir_from_wizard_cfg(cfg, Nlat, fourier_res_um=res)
                opt = ((cfg.get("stack") or {}).get("dti") or {}).get("optical", True)
                coll = cfg.get("collection") or {}
                print(f"[builder] wizard v3 (blocks) · lateral {Nlat}×{Nlat} · "
                      f"layers {len(ir.layers)} · dti.optical={opt} · "
                      f"collection={'on' if (coll.get('r0',0) and coll.get('ld_um',0)) else 'off'}")
                e = ir.dti_emt
                if e:
                    print(f"[builder] DTI 서브해상도 자동보정 — 푸리에 해상도 "
                          f"{res:.3f}µm > 트렌치 {e['width_um']:.3f}µm → 등가 유효매질 "
                          f"폭 {e['optical_width_um']:.3f}µm "
                          f"({', '.join(f'{m} {f*100:.1f}%' for m, f in e['mix'])}) "
                          f"· 검출 제외는 실제 폭 유지")
                return ir
            # voxel 경로: 3D 복셀 -> IR (레거시 WizardBuilder, dti.optical/collection 미반영)
            from ..structure.wizard_builder import WizardBuilder
            print("[builder] wizard v3 voxel (legacy — optical/collection 미반영)")
            b = WizardBuilder(cfg, base_dir=base_dir)
            ir = b.to_ir()                                # 마스크만 재사용
            matid = b.build()[:, ::self.ds, ::self.ds]
            det = ir.detector
            if det.n_below_band:                          # voxel build 는 후면 반사경 층을
                print("[warn] voxel 경로는 back_reflector 미지원 — 반사경 무시(auto 메쉬 사용 권장)")
                det.n_below_band = 0                      # 만들지 않음 -> 밴드 인덱싱 붕괴 방지
            det.pixel_map = det.pixel_map[::self.ds, ::self.ds]
            det.exclude_mask = det.exclude_mask[::self.ds, ::self.ds]
            return ir_from_voxels(matid, b.dz, b.dxy * self.ds,
                                  {int(i): n for n, i in b._idx.items()},
                                  substrate=ir.substrate, ambient=ir.ambient,
                                  detector=det, materials=ir.materials,
                                  dispersion=ir.dispersion)
        # legacy qcell schema
        from ..structure.builder import QcellBuilder
        b = QcellBuilder(cfg, base_dir=base_dir)
        matid = b.build()
        meta = b.meta()
        bounds = {l["name"]: l for l in meta["layer_bounds_vox"]}
        z0, z1 = bounds["substrate_si"]["z1"], bounds["microlens"]["z1"]
        matid = matid[z0:z1, ::self.ds, ::self.ds]
        id2name = {int(m["id"]): n for n, m in meta["materials"].items()}
        return ir_from_voxels(matid, b.dz, b.dxy * self.ds, id2name,
                              substrate=meta.get("substrate_material", "si"),
                              materials=dict(cfg.get("materials", {}) or {}),
                              dispersion=dict(cfg.get("dispersion", {}) or {}))

    # ------------------------------------------------------------- 물질 교체
    def remap_material(self, old, new):
        """공간 기하 그대로, 물질 이름만 교체 (다음 run 부터 반영)."""
        self._solver_cache = None                        # eps 바뀜 -> 조립 캐시 무효
        return self.ir.remap_material(old, new)

    # ------------------------------------------------------------- eps 준비
    def _eps_lut(self, lam):
        """공간 번호 -> 복소 eps @ λ."""
        return {rid: self.res.eps(name, lam)
                for rid, name in self.ir.region_materials.items()}

    def _eps_grid(self, region2d, eps_lut):
        g = np.empty(region2d.shape, dtype=np.complex128)
        for rid, eps in eps_lut.items():
            g[region2d == rid] = eps
        return torch.as_tensor(g, dtype=self.dtype, device=self.device)

    def _boundary_eps(self, lam):
        """(입사 반무한 eps, 투과 반무한 eps) — 모드별."""
        if self.ir.mode == "eps":
            return self.ir.ambient_eps, self.ir.substrate_eps
        n_a, _ = self.res.nk(self.ir.ambient, lam)       # 입사측 흡수 무시 (n 만)
        return float(n_a) ** 2, self.res.eps(self.ir.substrate, lam)

    # ------------------------------------------------------------- 실행
    def run(self, wavelength, theta=0.0, phi=0.0, pol_te=1.0, pol_tm=0.0,
            pixel_qe=True, _wood_depth=0):
        """단일 파장 RCWA -> R, QE(검출기 흡수), A_stack (+픽셀/라벨별 QE)."""
        lam = float(wavelength)
        if self.ir.mode == "eps" and self.ir.eps_lambda_um and \
                abs(lam - self.ir.eps_lambda_um) > 1e-9:
            print(f"[warn] eps 텐서는 λ={self.ir.eps_lambda_um}µm 스냅샷 — "
                  f"λ={lam} 에서 분산 미반영")
        # ── 솔버 캐시: (λ,θ,φ) 동일하면 조립(eig+S-matrix, 편광 무관)을 재사용.
        #    비편광 QE(TE+TM 2회 run)가 조립을 한 번만 하게 된다 (~2배).
        key = (round(lam, 12), round(float(theta), 9), round(float(phi), 9))
        cached = getattr(self, "_solver_cache", None)
        if cached is not None and cached[0] == key:
            solver = cached[1]
        else:
            self._solver_cache = None                    # 이전 λ 조립 메모리 즉시 해제
            eps_inc, eps_trn = self._boundary_eps(lam)
            eps_lut = self._eps_lut(lam) if self.ir.mode == "region" else None
            solver = RCWASolver(lam, self.ir.span_x, self.ir.span_y, nG=self.nG,
                                theta=theta, phi=phi, trunc=self.trunc,
                                device=self.device, dtype=self.dtype, fff=self.fff)
            solver.setup_incidence(eps_inc, eps_trn)
            for m2d, th in self.layer_stack:
                if self.ir.mode == "eps":
                    if (m2d == m2d.flat[0]).all():       # 균일층 -> 해석식
                        solver.add_layer(th, eps_scalar=complex(m2d.flat[0]))
                    else:
                        solver.add_layer(th, eps_grid=torch.as_tensor(
                            m2d, dtype=self.dtype, device=self.device))
                else:
                    u = np.unique(m2d)
                    if len(u) == 1:                      # 균일층 -> 해석식
                        solver.add_layer(th, eps_scalar=eps_lut[int(u[0])])
                    else:
                        solver.add_layer(th, eps_grid=self._eps_grid(m2d, eps_lut))
            self._solver_cache = (key, solver)
        # 수치 이상 가드: λ 미세 이동 재계산
        #  (a) Wood anomaly: kz=0 차수 -> V0 특이 -> R/T 비유한
        #  (b) 공진점 고유분해 불안정(금속 grid 등): R+T>1 (유니터리티 파괴)
        #      -> 에너지 정합상수 C<0 -> QE 음수로 전파되므로 여기서 차단
        try:
            o = solver.solve(pol_te=pol_te, pol_tm=pol_tm)
            bad = not (np.isfinite(o["R"]) and np.isfinite(o["T"])) \
                or not (-1e-6 <= o["R"] <= 1 + 1e-6) \
                or not (-1e-6 <= o["T"] <= 1 + 1e-6) \
                or (o["R"] + o["T"]) > 1 + 1e-6
        except Exception:
            bad = True
        if bad:
            if _wood_depth >= 3:                        # 재귀 무한루프 가드
                raise RuntimeError(
                    f"λ={lam}µm: R/T 이상(비유한 또는 R+T>1)이 λ 미세이동 "
                    f"{_wood_depth}회 후에도 지속 — 물질 n,k/구조 문제일 수 있음")
            lam_shift = lam * (1 + 5e-4)
            self._solver_cache = None                   # 이상 조립은 캐시에 남기지 않음
            print(f"[warn] λ={lam}µm 수치 이상(Wood/유니터리티) -> λ={lam_shift:.5f}µm 로 재계산")
            return self.run(lam_shift, theta=theta, phi=phi, pol_te=pol_te,
                            pol_tm=pol_tm, pixel_qe=pixel_qe, _wood_depth=_wood_depth + 1)
        out = {"wavelength": lam, "R": o["R"], "QE": o["T"],
               "A_stack": o["A"], "nG": solver.nG, "n_layers": len(self.layer_stack)}

        if pixel_qe == "diag":
            out["_solver"] = solver
            return out
        if pixel_qe and self.ir.detector is not None and self.n_si_layers > 0:
            out.update(self._detector_qe(solver, o))
        return out

    # ------------------------------------------------------------- QE 집계
    def _detector_qe(self, solver, o):
        """IR.detector 규약으로 픽셀/라벨별 광학 QE 집계 (구조 지식 불필요).

        QE = 밴드 3D 흡수 + 심부(T) — 픽셀 창(제외 마스크 제거) 귀속,
        절대 스케일은 에너지 보존으로 고정. 라벨별 = 같은 라벨 픽셀 평균.
        (광학/내부 QE = Si 흡수율. 캐리어 수집효율은 별도 물리, 여기선 미포함.)
        """
        det = self.ir.detector
        M = len(self.layer_stack)
        nS = self.n_si_layers
        nB = self.n_below_band                           # 반사경 등 밴드 아래 층 (검출 제외)
        T_deep = o["T"] if det.deep_is_detector else 0.0
        ngrid = self.grid_ny * self.grid_nx
        pixidx = det.pixel_map if det.pixel_map is not None else \
            np.zeros((self.grid_ny, self.grid_nx), dtype=np.int32)
        labels = det.pixel_labels or [""]
        npix = len(labels)
        excl = det.exclude_mask if det.exclude_mask is not None else \
            np.zeros_like(pixidx, dtype=bool)

        # 캐리어 수집효율 η(z): 광입사 Si 표면(밴드 상단)의 dead-layer 모델.
        # r0=0(기본) -> η≡1 -> 순수 광학 QE. r0>0 -> 얕은흡수(단파장) 수집손실.
        # 밴드층은 z-분해 흡수(층 내부 슬라이스)로 절대깊이별 η 적용 — 추가 eig 없음.
        r0 = float(getattr(det, "collect_r0", 0.0) or 0.0)
        ld = float(getattr(det, "collect_ld_um", 0.0) or 0.0)
        use_coll = r0 > 0.0 and ld > 0.0
        band_lis = list(range(M - nS - nB, M - nB))      # 밴드 층 (상단->하단)
        z_off, acc = {}, 0.0                             # 각 밴드층 상단의 절대깊이(Si표면=0)
        for li in band_lis:
            z_off[li] = acc
            acc += float(self.layer_stack[li][1])
        # 창 적분을 order-공간에서 정확 계산(실공간 560² 재구성 제거 — 수치 동일).
        # windows[0]=전체(=1) 는 층 총흡수(정합 상수 C)용 규약 — 반드시 첫 번째.
        pix_i = torch.as_tensor(np.ascontiguousarray(pixidx, dtype=np.int64))
        excl_t = torch.as_tensor(np.ascontiguousarray(excl))
        ones_w = torch.ones((self.grid_ny, self.grid_nx), dtype=torch.float64)
        ex_w = excl_t.to(torch.float64)
        windows = [ones_w, ex_w]
        for p in range(npix):                            # 픽셀 창 (제외 마스크 제거)
            windows.append(((pix_i == p) & (~excl_t)).to(torch.float64))
        zws, C = solver.band_window_absorption(windows, band_lis)

        pix_abs = torch.zeros(npix, dtype=torch.float64)  # 수집(collected) 흡수
        trench_abs = 0.0
        A_band = 0.0                                     # 광학 총 밴드 흡수(에너지보존)
        A_coll = 0.0                                     # 수집 총 밴드 흡수
        for li in band_lis:                              # 검출 밴드 층들 (반사경 nB 제외)
            if li not in zws:
                continue
            zc_list, WA = zws[li]                        # WA: (nz, 2+npix)
            WA = WA.detach().to("cpu", torch.float64)
            z_abs = z_off[li] + torch.tensor(zc_list, dtype=torch.float64)
            e = (1.0 - r0 * torch.exp(-z_abs / ld)) if use_coll \
                else torch.ones(len(zc_list), dtype=torch.float64)
            A_band += float(WA[:, 0].sum())
            A_coll += float(e @ WA[:, 0])
            trench_abs += float(WA[:, 1].sum())
            pix_abs += e @ WA[:, 2:]
        pix_abs = pix_abs.numpy() * C
        trench_abs *= C
        A_band *= C
        A_coll *= C
        if det.deep_is_detector:
            # 심부 흡수: 밴드 바닥 투과 flux 를 픽셀 귀속 (그 깊이엔 구조 없음).
            # 심부는 접합 근처 -> η≈1 (수집손실 없음).
            Sz = solver.transmitted_flux_map(self.grid_ny, self.grid_nx)
            Sz = Sz.detach().cpu().numpy()
            for p in range(npix):
                pix_abs[p] += Sz[pixidx == p].sum() / ngrid
            A_coll += Sz.sum() / ngrid
        # 픽셀 면적 정규화 (pixel_map 분할 면적 기준 — 비정방 픽셀도 지원)
        area = np.array([max(1, (pixidx == p).sum()) for p in range(npix)]) / ngrid
        qe_pix = [float(v / a) for v, a in zip(pix_abs, area)]
        qe_lab = {L: float(np.mean([q for q, l in zip(qe_pix, labels) if l == L]))
                  for L in dict.fromkeys(labels)}
        qe_opt = float(A_band + T_deep)                  # 광학 QE (에너지보존)
        qe_total = float(A_coll + T_deep)                # 수집(소자) QE = device-QE 관례
        return {"QE": qe_total,
                "QE_optical": qe_opt,                    # 순수 광학 흡수 (수집전)
                "A_stack": float(1.0 - o["R"] - qe_opt),  # R+QE_optical+A_stack=1 항등
                "QE_recomb": float(qe_opt - qe_total),   # 수집손실(재결합)
                "QE_pixels": qe_pix,
                "QE_rgb": qe_lab,                        # 라벨별 (RGB 는 그 부분집합)
                "QE_trench": float(trench_abs),
                "QE_deep": float(T_deep)}

    # ------------------------------------------------------------- 진단
    def efield_xz(self, wavelength, row=1, Ny=128, Nx=128, nz_per_um=24,
                  theta=0.0, phi=0.0, xy_offset_um=0.05, xy_n=160, cancel=None):
        """XZ 단면 + Si 표면 XY 평면 |E|² 맵 (TE/TM 평균) — 초점면 시각화.

        row: 1..npx 픽셀 행 — 그 행 중심 y 로 자른 XZ 단면. 전 층을 층별 내부
        필드 재구성(layer_internal_fields)으로 샘플. 추가로 BARL–Si 경계면에서
        xy_offset_um 만큼 Si 안쪽의 XY 평면(xy_n×xy_n)을 같은 solve 에서 샘플 —
        ML→CF→BARL 을 지난 초점이 각 2×2(quad) '중심'에 꽂히는지 확인용.
        반환: {"x_um","z_um","I"(nz,Nx max=1), "boundaries", "row_labels",
               "span_um","total_um","row","peak_raw",
               "xy": {"z_um","offset_um","n","I"(xy_n,xy_n max=1)}, "bayer"}
        주의: 접선 성분 |Ex|²+|Ey|² (Ez 제외) — 초점 위치/모양 시각화 목적.
        """
        det = self.ir.detector
        npx = int(round(np.sqrt(len(det.pixel_labels)))) if det else 2
        row = max(1, min(npx, int(row)))
        iy = min(Ny - 1, int(round((row - 0.5) / npx * Ny)))
        acc, acc_xy = None, None
        zs, bounds, total, xy_z = [], [], 0.0, None
        for pol in ((1.0, 0.0), (0.0, 1.0)):
            if cancel and cancel():
                return None
            o = self.run(wavelength, theta=theta, phi=phi,
                         pol_te=pol[0], pol_tm=pol[1], pixel_qe="diag")
            sv = o["_solver"]
            sv.absorption_profile()                  # _node_ab/_flux_calib 준비
            rows_I, zc = [], []
            z0, bl, prev = 0.0, [], None
            si_z0 = None
            for i, (_m2d, th) in enumerate(self.layer_stack):
                if cancel and cancel():
                    return None
                tag = (self.ir.layer_tags[i]
                       if self.ir.layer_tags and i < len(self.ir.layer_tags) else "")
                if tag != prev:                      # 블록 경계만 기록 (ML/CF/BARL/Si…)
                    bl.append({"z_um": round(z0, 4), "tag": tag})
                    prev = tag
                if si_z0 is None and "si" in str(tag).lower():
                    si_z0 = z0                       # BARL–Si 경계(=Si 표면)
                nz = max(2, int(round(float(th) * nz_per_um)))
                zf = [(k + 0.5) / nz for k in range(nz)]
                sl = sv.layer_internal_fields(i, zf, Ny, Nx)
                for k, dslice in enumerate(sl):
                    rows_I.append(dslice["E2"][iy].detach().cpu().numpy().real)
                    zc.append(z0 + zf[k] * float(th))
                z0 += float(th)
            # ── XY 평면: Si 표면 + offset 위치의 층/깊이를 찾아 전면 샘플 ──
            if si_z0 is not None:
                zt = si_z0 + max(0.0, float(xy_offset_um))
                zz = 0.0
                for i, (_m2d, th) in enumerate(self.layer_stack):
                    if zz + float(th) >= zt - 1e-9:
                        frac = min(0.999, max(0.001, (zt - zz) / float(th)))
                        pl = sv.layer_internal_fields(i, [frac], xy_n, xy_n)
                        M = pl[0]["E2"].detach().cpu().numpy().real
                        acc_xy = M if acc_xy is None else acc_xy + M
                        xy_z = zz + frac * float(th)
                        break
                    zz += float(th)
            I = np.asarray(rows_I, float)
            acc = I if acc is None else acc + I
            zs, bounds, total = zc, bl, z0
        acc *= 0.5
        mx = float(acc.max()) or 1.0
        labels = [det.pixel_labels[(row - 1) * npx + c] for c in range(npx)] \
            if det else []
        out = {"x_um": [round(j * self.span / Nx, 4) for j in range(Nx)],
               "z_um": [round(z, 4) for z in zs],
               "I": [[round(float(v) / mx, 5) for v in r] for r in acc],
               "boundaries": bounds, "row_labels": labels,
               "span_um": round(float(self.span), 4),
               "total_um": round(total, 4), "row": row, "peak_raw": round(mx, 5)}
        if acc_xy is not None:
            acc_xy *= 0.5
            mxy = float(acc_xy.max()) or 1.0
            out["xy"] = {"z_um": round(float(xy_z), 4),
                         "offset_um": round(float(xy_offset_um), 4), "n": int(xy_n),
                         "I": [[round(float(v) / mxy, 5) for v in r]
                               for r in acc_xy]}
            if det:
                out["bayer"] = [[det.pixel_labels[r * npx + c]
                                 for c in range(npx)] for r in range(npx)]
        return out

    def diagnose(self, wavelength, theta=0.0):
        """물질 n,k 점검 + 경계 투과(T) 워터폴 — 전부 IR 기반.

        반환: materials[{name,n,k,source,lam_range,flags[]}],
              profile[{mats,th_um,A,T_after,T_rgb}], entry_rgb, R, T_into_si, T_deep
        """
        lam = float(wavelength)
        o = self.run(lam, theta=theta, pol_te=1.0, pol_tm=0.0, pixel_qe="diag")
        solver = o.pop("_solver")
        maps, C = solver.absorption_maps(self.grid_ny, self.grid_nx)
        M = len(self.layer_stack)
        nS = self.n_si_layers
        nB = self.n_below_band                            # 밴드 아래 반사경 층
        A = [float(maps[i].sum() * C) if i in maps else 0.0 for i in range(M)]

        # ---- 물질 표 + 의심 플래그 ----
        mats = []
        if self.ir.mode == "region":
            for name in self.ir.material_names():
                n, k = self.res.nk(name, lam)
                src, rng = self.res.source(name)
                flags = []
                if name != "air" and not (0.9 <= n <= 8.5):
                    flags.append(f"n={n:.3g} 비정상 범위(열 순서/단위 확인)")
                if k > 0.5 and name not in ("si",) and "metal" not in name:
                    flags.append(f"k={k:.3g} 강흡수 — 흡수계수(α)를 k 로 오인했는지 확인")
                if rng and not (rng[0] - 1e-9 <= lam <= rng[1] + 1e-9):
                    flags.append(f"λ={lam} 가 테이블 범위({rng[0]:.2f}~{rng[1]:.2f}µm) 밖 -> 경계값 사용")
                mats.append({"name": name, "n": round(n, 4), "k": round(k, 5),
                             "source": src, "lam_range": rng, "flags": flags})

        # ---- 라벨별 워터폴 마스크 ----
        det = self.ir.detector
        cmask = None
        acc = None
        if det is not None and det.pixel_map is not None:
            labels = det.pixel_labels
            cmask = {L: np.isin(det.pixel_map,
                                [i for i, l in enumerate(labels) if l == L])
                     for L in dict.fromkeys(labels)}
            acc = {L: 0.0 for L in cmask}
        ngrid = self.grid_ny * self.grid_nx

        # ---- 경계 투과 프로파일 (같은 물질 조합 연속층 묶음) ----
        # 색별 흡수 누적(셀 면적 정규화) — 마이크로렌즈 '측면 농축'은 미반영이라
        # 참고용. 농축 반영한 정확한 값은 아래 T_into_si_rgb (Si 유입 flux) 참조.
        rm = self.ir.region_materials
        prof = []
        Tcur = 1.0 - o["R"]
        cur = None
        for i, (m2d, th) in enumerate(self.layer_stack):
            if self.ir.mode == "region":
                ids = np.unique(np.asarray(m2d))
                nm = "+".join(sorted(rm[int(x)] for x in ids))
            else:
                nm = f"layer{i}"
            Tcur -= A[i]
            if cmask is not None and i in maps:
                dens = maps[i].detach().cpu().numpy() * C
                for L, m in cmask.items():
                    acc[L] += float(dens[m].sum())        # 셀 면적 기준 (색면적 스케일 제거)
            rgb = ({L: round(1.0 - o["R"] - acc[L], 5) for L in cmask}
                   if cmask is not None else None)
            if cur and cur["mats"] == nm:
                cur["th_um"] += th; cur["A"] += A[i]; cur["T_after"] = Tcur
                cur["n_sub"] += 1; cur["T_rgb"] = rgb
            else:
                cur = {"mats": nm, "th_um": th, "A": A[i], "T_after": Tcur,
                       "n_sub": 1, "T_rgb": rgb}
                prof.append(cur)
        entry_rgb = ({L: round(1.0 - o["R"], 5) for L in cmask}
                     if cmask is not None else None)
        # ---- Si 유입 (색별, flux 기반 — 마이크로렌즈 농축 지표) ----
        # 밴드 top node 하향 Poynting flux 를 셀평균=T_into_si 로 보정 후 색영역 면적정규화.
        # >100% = 그 색 픽셀로 빛이 농축됨(ML 집광). 주의: 서브파장 피치에선 Si 내부
        # 측면 회절로 색간 재분배가 있어 색별 QE 의 엄밀 상한은 아님(셀 총합만 엄밀).
        A_above = sum(A[:M - nS - nB]) if nS else sum(A)
        T_into_si = 1.0 - o["R"] - A_above
        into_rgb = None
        if cmask is not None and nS > 0:
            fmap = solver.node_flux_map_raw(M - nS - nB, self.grid_ny,
                                            self.grid_nx).detach().cpu().numpy()
            fm = float(fmap.mean())
            cal = T_into_si / fm if abs(fm) > 1e-30 else 0.0
            fmap = fmap * cal
            into_rgb = {L: round(float(fmap[m].sum() / fmap.size / max(m.mean(), 1e-9)), 5)
                        for L, m in cmask.items()}
        for p in prof:
            p["th_um"] = round(p["th_um"], 4); p["A"] = round(p["A"], 5)
            p["T_after"] = round(p["T_after"], 5)
        return {"wavelength": lam, "R": round(o["R"], 5), "materials": mats,
                "profile": prof, "entry_rgb": entry_rgb,
                "T_into_si_rgb": into_rgb,          # 색별 Si 유입 (flux, 농축반영) — QE 상한
                "T_into_si": round(T_into_si, 5),
                "T_deep": round(o["QE"], 5)}
