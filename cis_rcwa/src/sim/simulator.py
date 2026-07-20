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
    def _ir_from_yaml(self, cfg, base_dir, mesh, lateral_um):
        stack = cfg.get("stack") or {}
        if "si" in stack and "cf" in stack:              # wizard v3
            from ..structure.wizard_builder import WizardBuilder
            print("[builder] wizard v3 schema")
            b = WizardBuilder(cfg, base_dir=base_dir)
            if mesh == "auto":
                Nlat = int(round(b.span / (float(lateral_um) * self.ds)))
                Nlat = min(max(Nlat, 128), 2400)
                ir = b.to_ir(lateral_n=Nlat)
                print(f"[mesh] auto: lateral {Nlat}×{Nlat} ({b.span/Nlat*1000:.1f}nm)"
                      f" · layers {len(ir.layers)} (z 해석적 경계, 복셀화 없음)")
                return ir
            # voxel 경로: 3D 복셀 -> IR (다운샘플 포함)
            ir = b.to_ir()                                # 마스크만 재사용
            matid = b.build()[:, ::self.ds, ::self.ds]
            det = ir.detector
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
            pixel_qe=True):
        """단일 파장 RCWA -> R, QE(검출기 흡수), A_stack (+픽셀/라벨별 QE)."""
        lam = float(wavelength)
        if self.ir.mode == "eps" and self.ir.eps_lambda_um and \
                abs(lam - self.ir.eps_lambda_um) > 1e-9:
            print(f"[warn] eps 텐서는 λ={self.ir.eps_lambda_um}µm 스냅샷 — "
                  f"λ={lam} 에서 분산 미반영")
        eps_inc, eps_trn = self._boundary_eps(lam)
        eps_lut = self._eps_lut(lam) if self.ir.mode == "region" else None

        solver = RCWASolver(lam, self.ir.span_x, self.ir.span_y, nG=self.nG,
                            theta=theta, phi=phi, trunc=self.trunc,
                            device=self.device, dtype=self.dtype, fff=self.fff)
        solver.setup_incidence(eps_inc, eps_trn)
        for m2d, th in self.layer_stack:
            if self.ir.mode == "eps":
                if (m2d == m2d.flat[0]).all():           # 균일층 -> 해석식
                    solver.add_layer(th, eps_scalar=complex(m2d.flat[0]))
                else:
                    solver.add_layer(th, eps_grid=torch.as_tensor(
                        m2d, dtype=self.dtype, device=self.device))
            else:
                u = np.unique(m2d)
                if len(u) == 1:                          # 균일층 -> 해석식
                    solver.add_layer(th, eps_scalar=eps_lut[int(u[0])])
                else:
                    solver.add_layer(th, eps_grid=self._eps_grid(m2d, eps_lut))
        # Wood anomaly (kz=0 차수 -> V0 특이) 가드: λ 미세 이동 재계산
        try:
            o = solver.solve(pol_te=pol_te, pol_tm=pol_tm)
            bad = not (np.isfinite(o["R"]) and np.isfinite(o["T"]))
        except Exception:
            bad = True
        if bad:
            lam_shift = lam * (1 + 5e-4)
            print(f"[warn] λ={lam}µm Wood anomaly -> λ={lam_shift:.5f}µm 로 재계산")
            return self.run(lam_shift, theta=theta, phi=phi,
                            pol_te=pol_te, pol_tm=pol_tm, pixel_qe=pixel_qe)
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
        maps, C = solver.absorption_maps(self.grid_ny, self.grid_nx)
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

        pix_abs = np.zeros(npix)
        trench_abs = 0.0
        A_band = 0.0
        for li in range(M - nS - nB, M - nB):            # 검출 밴드 층들 (반사경 nB 제외)
            if li not in maps:
                continue
            dens = maps[li].detach().cpu().numpy() * C
            A_band += dens.sum()
            for p in range(npix):
                m = (pixidx == p) & (~excl)              # 픽셀 창 (제외분 제거)
                pix_abs[p] += dens[m].sum()
            trench_abs += dens[excl].sum()
        if det.deep_is_detector:
            # 심부 흡수: 밴드 바닥 투과 flux 를 픽셀 귀속 (그 깊이엔 구조 없음)
            Sz = solver.transmitted_flux_map(self.grid_ny, self.grid_nx)
            Sz = Sz.detach().cpu().numpy()
            for p in range(npix):
                pix_abs[p] += Sz[pixidx == p].sum() / ngrid
        # 픽셀 면적 정규화 (pixel_map 분할 면적 기준 — 비정방 픽셀도 지원)
        area = np.array([max(1, (pixidx == p).sum()) for p in range(npix)]) / ngrid
        qe_pix = [float(v / a) for v, a in zip(pix_abs, area)]
        qe_lab = {L: float(np.mean([q for q, l in zip(qe_pix, labels) if l == L]))
                  for L in dict.fromkeys(labels)}
        qe_total = float(A_band + T_deep)
        return {"QE": qe_total,
                "A_stack": float(1.0 - o["R"] - qe_total),
                "QE_pixels": qe_pix,
                "QE_rgb": qe_lab,                        # 라벨별 (RGB 는 그 부분집합)
                "QE_trench": float(trench_abs),
                "QE_deep": float(T_deep)}

    # ------------------------------------------------------------- 진단
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
