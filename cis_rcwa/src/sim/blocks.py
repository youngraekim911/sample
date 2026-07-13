# -*- coding: utf-8 -*-
"""sim.blocks — RCWA 실행의 블록화/일반화.

    [구조]  StructureIR (블록 조립 결과)
       │
    MeshPolicy   구조 mesh 최적화 정책 — 공간 크기별 자동 (z 해석적 경계 = 복셀화
       │         없음, Å층 정확 / 연속곡면만 슬라이스 / 가로 미세 래스터 / 균일층
       │         해석식으로 eig 생략) -> 10~20시간짜리를 λ당 수십 초로.
    LightSource  빛 옵션 블록 — 파장(단일/스윕), 입사각 θ/φ, 편광(avg|sum|te|tm)
       │
    RCWAEngine.run(source, probes=[...])
       ├─ QEProbe        픽셀별 QE — 위치(µm)와 '어느 CF 아래인지' 명시
       └─ BoundaryProbe  경계 Transmittance 스펙트럼 — 블록 경계마다 T(λ),
                         컬러(라벨) 분기 포함  (예: 그래프1 ARL 통과 후 T,
                         그래프2 CF 통과 후 T — R/G/B 분기)

사용:
    eng = RCWAEngine(ir, mesh=MeshPolicy(nG=101, downsample=2))
    res = eng.run(LightSource(lam=(0.40, 0.70, 13), pol="avg"),
                  probes=[QEProbe(), BoundaryProbe()])
    res["qe"]        # [{pixel,row,col,x_um,y_um,cf,qe:[...]}, ...]
    res["boundary"]  # [{tag, T:{R:[...],G:[...],B:[...]}}, ...] 위->아래 경계 순
"""
import numpy as np

from .simulator import RCWAPlaneWaveSimulator


# ==========================================================================
class MeshPolicy:
    """구조 mesh + 해상도 정책. 공간마다 크기가 다르니 mesh 는 자동 최적:

    · z: 해석적 층 경계 (IR 이 이미 블록별 정확 두께 — Å층 그대로, 복셀화 없음)
    · 연속 곡면(ML 돔/CF 응집/taper)만 슬라이스 (ml/men/taper_slices)
    · 가로: lateral_um × downsample 래스터 (구조 최소 피처 이하로)
    · 균일층은 고유분해 생략(해석식) — 층 수가 늘어도 비용 거의 불변
    · nG: 회절 차수 (금속 grid 는 FFF 로 수렴 가속)
    """

    def __init__(self, nG=101, downsample=2, lateral_um=0.005,
                 ml_slices=48, men_slices=8, taper_slices=8, fff=True):
        self.nG = nG
        self.ds = downsample
        self.lateral_um = lateral_um
        self.ml_slices = ml_slices
        self.men_slices = men_slices
        self.taper_slices = taper_slices
        self.fff = fff


# ==========================================================================
class LightSource:
    """빛 옵션 블록.

    lam : 단일 값 | 리스트 | (lam0, lam1, n) 스윕 (µm)
    theta/phi : 입사각 (deg)
    pol : "avg"(TE/TM 평균 — 비편광 표준) | "sum"(합산) | "te" | "tm"
    """

    def __init__(self, lam=0.55, theta=0.0, phi=0.0, pol="avg"):
        if isinstance(lam, tuple) and len(lam) == 3:
            self.lams = list(np.linspace(lam[0], lam[1], int(lam[2])))
        elif isinstance(lam, (list, np.ndarray)):
            self.lams = [float(x) for x in lam]
        else:
            self.lams = [float(lam)]
        self.theta = float(theta)
        self.phi = float(phi)
        self.pol = pol

    def pol_runs(self):
        """[(pol_te, pol_tm, weight), ...]"""
        if self.pol == "te":
            return [(1.0, 0.0, 1.0)]
        if self.pol == "tm":
            return [(0.0, 1.0, 1.0)]
        w = 1.0 if self.pol == "sum" else 0.5
        return [(1.0, 0.0, w), (0.0, 1.0, w)]


# ==========================================================================
class QEProbe:
    """픽셀별 QE 프로브 — 모든 픽셀의 QE 를 위치/CF 귀속과 함께.

    출력 행: {pixel, row, col, x_um, y_um, cf(위 컬러필터 라벨), qe: [λ별]}
    + 라벨 평균 {cf: [λ별]} 은 결과 "qe_by_cf".
    """
    key = "qe"

    def prepare(self, eng):
        det = eng.sim.ir.detector
        assert det is not None and det.pixel_map is not None, \
            "QEProbe: IR.detector(pixel_map) 필요"
        ny, nx = det.pixel_map.shape
        sx, sy = eng.sim.ir.span_x, eng.sim.ir.span_y
        self.rows = []
        npx_side = int(round(len(det.pixel_labels) ** 0.5)) or 1
        for k, lab in enumerate(det.pixel_labels):
            m = det.pixel_map == k
            yy, xx = np.nonzero(m)
            self.rows.append({"pixel": k, "row": k // npx_side, "col": k % npx_side,
                              "x_um": round(float((xx.mean() + 0.5) / nx * sx), 4),
                              "y_um": round(float((yy.mean() + 0.5) / ny * sy), 4),
                              "cf": lab, "qe": []})

    def collect(self, eng, lam_i, acc):
        for r, v in zip(self.rows, acc["QE_pixels"]):
            r["qe"].append(round(float(v), 5))

    def finish(self, eng, out):
        out["qe"] = self.rows
        labs = {}
        for r in self.rows:
            labs.setdefault(r["cf"], []).append(r["qe"])
        out["qe_by_cf"] = {L: [round(float(np.mean(col)), 5) for col in zip(*qs)]
                           for L, qs in labs.items()}


# ==========================================================================
class BoundaryProbe:
    """경계 Transmittance 프로브 — 각 블록 경계 통과 후 T(λ), 컬러 분기.

    빛 시작(air, 1-R 진입)부터 Si 유입까지, 블록(tag) 경계마다:
        T_after[tag][label][λ]  = 1 - R - (그 경계까지 라벨 기둥 누적 흡수)
    IR.layer_tags(블록 provenance) 기준으로 층을 묶는다.
    """
    key = "boundary"

    def prepare(self, eng):
        ir = eng.sim.ir
        tags = ir.layer_tags or [f"layer{i}" for i in range(len(ir.layers))]
        # 검출 밴드(태그 si...)는 경계 프로파일에서 '유입'으로 종결
        nS = eng.sim.n_si_layers
        self.tags = tags[:len(tags) - nS] if nS else tags
        det = ir.detector
        if det is not None and det.pixel_map is not None:
            labels = det.pixel_labels
            self.cmask = {L: np.isin(det.pixel_map,
                                     [i for i, l in enumerate(labels) if l == L])
                          for L in dict.fromkeys(labels)}
        else:
            self.cmask = {"all": np.ones(ir.grid_shape, dtype=bool)}
        # 경계 목록: 연속 동일 태그 묶음 (위->아래)
        self.groups = []
        for i, t in enumerate(self.tags):
            if self.groups and self.groups[-1][0] == t:
                self.groups[-1][1].append(i)
            else:
                self.groups.append((t, [i]))
        self.T = {t: {L: [] for L in self.cmask} for t, _ in self.groups}
        self.entry = {L: [] for L in self.cmask}
        self.R = []

    def collect(self, eng, lam_i, acc):
        R = acc["R"]
        self.R.append(round(float(R), 5))
        dens_layers = acc["_dens"]                     # {layer_i: 흡수밀도 2D}
        ngrid = eng.sim.grid_ny * eng.sim.grid_nx
        for L in self.cmask:
            self.entry[L].append(round(1.0 - R, 5))
        cum = {L: 0.0 for L in self.cmask}
        for t, idxs in self.groups:
            for i in idxs:
                if i in dens_layers:
                    d = dens_layers[i]
                    for L, m in self.cmask.items():
                        cum[L] += float(d[m].mean()) * ngrid
            for L in self.cmask:
                self.T[t][L].append(round(1.0 - R - cum[L], 5))

    def finish(self, eng, out):
        out["boundary"] = [{"tag": t, "T": self.T[t]} for t, _ in self.groups]
        out["entry"] = self.entry
        out["R"] = self.R


# ==========================================================================
class RCWAEngine:
    """IR + MeshPolicy -> 파장 루프 실행기. 프로브들이 λ마다 수집."""

    def __init__(self, ir, mesh=None, device=None):
        mesh = mesh or MeshPolicy()
        self.mesh = mesh
        self.sim = RCWAPlaneWaveSimulator(ir, nG=mesh.nG, downsample=mesh.ds,
                                          lateral_um=mesh.lateral_um,
                                          device=device, fff=mesh.fff)

    @staticmethod
    def _merge(acc, o, w):
        """편광 가중 병합 — 스칼라/리스트/딕셔너리(값=스칼라|배열) 재귀."""
        for k, v in o.items():
            if k in ("wavelength", "nG", "n_layers"):
                acc.setdefault(k, v)
            elif isinstance(v, (int, float)):
                acc[k] = acc.get(k, 0.0) + v * w
            elif isinstance(v, list) and v and isinstance(v[0], (int, float)):
                a = acc.get(k)
                acc[k] = np.array(v) * w if a is None else a + np.array(v) * w
            elif isinstance(v, dict):
                a = acc.setdefault(k, {})
                for kk, vv in v.items():
                    a[kk] = a.get(kk, 0.0) + vv * w
            else:
                acc.setdefault(k, v)

    def run(self, source, probes=(), verbose=True):
        import time
        need_dens = any(isinstance(p, BoundaryProbe) for p in probes)
        for p in probes:
            p.prepare(self)
        out = {"wavelength_um": source.lams, "theta": source.theta,
               "pol": source.pol, "nG": self.mesh.nG}
        rows = []
        for li, lam in enumerate(source.lams):
            t0 = time.time()
            acc = {}
            for te, tm, w in source.pol_runs():
                o = self.sim.run(lam, theta=source.theta, phi=source.phi,
                                 pol_te=te, pol_tm=tm,
                                 pixel_qe="diag" if need_dens else True)
                if need_dens:                      # 흡수맵 회수 + QE 재구성 (1 solve)
                    solver = o.pop("_solver")
                    maps, C = solver.absorption_maps(self.sim.grid_ny, self.sim.grid_nx)
                    o["_dens"] = {i: m.detach().cpu().numpy() * C
                                  for i, m in maps.items()}
                    if self.sim.ir.detector is not None and self.sim.n_si_layers > 0:
                        # diag 모드의 QE 는 raw T — _detector_qe 규약({"T","R"})으로 전달
                        o.update(self.sim._detector_qe(
                            solver, {"T": o["QE"], "R": o["R"]}))
                self._merge(acc, o, w)
            for p in probes:
                p.collect(self, li, acc)
            rows.append([lam, float(acc["R"]), float(acc["QE"])])
            if verbose:
                q = acc.get("QE_rgb")
                extra = ("  " + "/".join(f"{L}={q[L]:.3f}" for L in q)) if isinstance(q, dict) else ""
                print(f"  λ={lam*1000:.0f}nm R={acc['R']:.3f} QE={acc['QE']:.3f}"
                      f"{extra}  ({time.time()-t0:.1f}s)", flush=True)
        out["rows"] = rows
        for p in probes:
            p.finish(self, out)
        return out
