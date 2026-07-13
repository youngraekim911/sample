# -*- coding: utf-8 -*-
"""StructureIR — 구조 블록과 RCWA 블록을 분리하는 중간표현(계약).

개념
----
구조란 본질적으로 ① 공간 분할(boundary 로 나뉜 영역들) ② 각 공간을 채우는
물질이 무엇인가 — 두 가지 정보다. 이 모듈은 그 두 가지만 담는 순수 데이터
컨테이너를 정의한다:

    layers            : 입사(위)->투과(아래) 순서의 (region_map[ny,nx], thickness_um)
                        region_map 의 정수값 = '공간 번호' (물질이 아님)
    region_materials  : {공간 번호: 물질 이름}  ← 물질 교체는 이 dict 만 수정
    ambient/substrate : 상/하부 반무한 매질 이름
    detector          : QE 집계 규약 (검출 밴드 + 픽셀 분할 + 제외 마스크)
    materials/dispersion : 물질 이름 -> n,k (상수/파장 테이블)

RCWA(simulator)는 이 IR 만 소비한다 — 위저드 스키마, DTI/CF/ML 같은 구조
개념을 전혀 모른다. 따라서 새로운 구조 생성기(빌더)를 붙이거나 물질을
바꿔도 RCWA/결과 블록 코드는 그대로다.

두 모드
-------
  mode="region" : region_map 이 uint8 공간 번호 -> 파장별 n,k 로 eps 계산 (표준)
  mode="eps"    : region_map 이 complex128 eps 스냅샷 (λ 고정, npy 직접 입력용)
"""
import json
import numpy as np
from dataclasses import dataclass, field


# --------------------------------------------------------------------------
@dataclass
class Detector:
    """QE 집계 규약 — '스택 하단 어느 밴드가 검출기이고 어떻게 픽셀로 나누나'.

    band_um     : 검출 밴드(예: Si) 두께. 스택 마지막 n_layers 층이 이 밴드.
    n_layers    : 검출 밴드에 속하는 층 수 (스택 끝에서 셈)
    pixel_map   : [ny,nx] int — 셀 전체를 픽셀 인덱스로 분할 (없으면 단일 픽셀)
    pixel_labels: 픽셀 인덱스 -> 라벨('R'/'G'/'B'/...). 라벨별 QE 는 같은 라벨 평균.
    exclude_mask: [ny,nx] bool — 픽셀 창에서 제외할 영역 (예: DTI 트렌치 내부)
    deep_is_detector: 밴드 아래 반무한 기판도 같은 검출 물질 (심부 흡수를 QE 에 포함)
    """
    band_um: float = 0.0
    n_layers: int = 0
    pixel_map: np.ndarray = None
    pixel_labels: list = field(default_factory=list)
    exclude_mask: np.ndarray = None
    deep_is_detector: bool = True

    @property
    def n_pixels(self):
        return len(self.pixel_labels) if self.pixel_labels else \
            (int(self.pixel_map.max()) + 1 if self.pixel_map is not None else 1)


# --------------------------------------------------------------------------
@dataclass
class StructureIR:
    span_x: float
    span_y: float
    layers: list                                   # [(map[ny,nx], th_um)] 위->아래
    region_materials: dict                         # {int: str}
    ambient: str = "air"
    substrate: str = "si"
    mode: str = "region"                           # "region" | "eps"
    eps_lambda_um: float = 0.0                     # eps 모드: 스냅샷 파장
    ambient_eps: complex = 1.0 + 0j                # eps 모드 전용
    substrate_eps: complex = None                  # eps 모드 전용
    detector: Detector = None
    materials: dict = field(default_factory=dict)  # {name: {n,k[,src]}}
    dispersion: dict = field(default_factory=dict) # {name: [[lam,n,k],..]}
    layer_tags: list = None                        # 층별 출신 블록 이름 (위->아래, 옵션)

    # ---------------------------------------------------------------- 기본
    @property
    def grid_shape(self):
        return self.layers[0][0].shape if self.layers else (0, 0)

    def material_names(self):
        """구조에 등장하는 전 물질 (ambient/substrate 포함, 순서 보존)."""
        seen = []
        for n in list(self.region_materials.values()) + [self.ambient, self.substrate]:
            if n not in seen:
                seen.append(n)
        return seen

    # ---------------------------------------------------------------- 검증
    def validate(self):
        """모든 빌더 출력이 지켜야 하는 계약 — 위반 시 즉시 명확한 에러."""
        assert self.layers, "layers 비었음"
        ny, nx = self.grid_shape
        used = set()
        for i, (m, th) in enumerate(self.layers):
            assert m.shape == (ny, nx), f"layer{i} 격자 {m.shape} != {(ny, nx)}"
            assert th > 0, f"layer{i} 두께 {th} <= 0"
            if self.mode == "region":
                assert not np.iscomplexobj(m), f"layer{i}: region 모드에 복소 맵"
                used.update(np.unique(m).tolist())
            else:
                assert np.iscomplexobj(m), f"layer{i}: eps 모드에 실수 맵"
        if self.mode == "region":
            missing = [r for r in used if int(r) not in self.region_materials]
            assert not missing, f"물질 매핑 없는 공간 번호: {missing}"
        else:
            assert self.substrate_eps is not None, "eps 모드: substrate_eps 필요"
        if self.layer_tags is not None:
            assert len(self.layer_tags) == len(self.layers), "layer_tags 길이 != 층 수"
        d = self.detector
        if d is not None:
            assert d.n_layers <= len(self.layers), "detector.n_layers > 층 수"
            if d.pixel_map is not None:
                assert d.pixel_map.shape == (ny, nx), "pixel_map 격자 불일치"
                assert int(d.pixel_map.max()) + 1 <= len(d.pixel_labels), \
                    "pixel_labels 개수 < 픽셀 인덱스"
            if d.exclude_mask is not None:
                assert d.exclude_mask.shape == (ny, nx), "exclude_mask 격자 불일치"
        return self

    # ---------------------------------------------------------------- 물질 교체
    def remap_material(self, old, new):
        """공간 기하는 그대로 두고 물질만 교체 (구조 재생성 불필요)."""
        hit = [r for r, n in self.region_materials.items() if n == old]
        for r in hit:
            self.region_materials[r] = new
        for attr in ("ambient", "substrate"):
            if getattr(self, attr) == old:
                setattr(self, attr, new)
        return len(hit)

    # ---------------------------------------------------------------- 저장/로드
    def save_npz(self, path):
        maps = [m for m, _ in self.layers]
        ths = np.array([th for _, th in self.layers], dtype=np.float64)
        d = self.detector
        meta = {
            "span_x": self.span_x, "span_y": self.span_y,
            "region_materials": {str(k): v for k, v in self.region_materials.items()},
            "ambient": self.ambient, "substrate": self.substrate,
            "mode": self.mode, "eps_lambda_um": self.eps_lambda_um,
            "ambient_eps": [self.ambient_eps.real, self.ambient_eps.imag],
            "substrate_eps": ([self.substrate_eps.real, self.substrate_eps.imag]
                              if self.substrate_eps is not None else None),
            "materials": self.materials, "dispersion": self.dispersion,
            "layer_tags": self.layer_tags,
            "detector": ({"band_um": d.band_um, "n_layers": d.n_layers,
                          "pixel_labels": d.pixel_labels,
                          "deep_is_detector": d.deep_is_detector,
                          "has_pixel_map": d.pixel_map is not None,
                          "has_exclude": d.exclude_mask is not None}
                         if d is not None else None),
        }
        arrs = {"thicknesses": ths, "meta_json": np.frombuffer(
            json.dumps(meta, ensure_ascii=False).encode("utf-8"), dtype=np.uint8)}
        for i, m in enumerate(maps):
            arrs[f"layer_{i:04d}"] = m
        if d is not None and d.pixel_map is not None:
            arrs["pixel_map"] = d.pixel_map
        if d is not None and d.exclude_mask is not None:
            arrs["exclude_mask"] = d.exclude_mask
        np.savez_compressed(path, **arrs)

    @classmethod
    def load_npz(cls, path):
        z = np.load(path, allow_pickle=False)
        meta = json.loads(bytes(z["meta_json"]).decode("utf-8"))
        ths = z["thicknesses"]
        layers = [(z[f"layer_{i:04d}"], float(ths[i])) for i in range(len(ths))]
        det = None
        dm = meta.get("detector")
        if dm:
            det = Detector(band_um=dm["band_um"], n_layers=dm["n_layers"],
                           pixel_labels=dm["pixel_labels"],
                           deep_is_detector=dm.get("deep_is_detector", True),
                           pixel_map=z["pixel_map"] if dm["has_pixel_map"] else None,
                           exclude_mask=z["exclude_mask"] if dm["has_exclude"] else None)
        se = meta.get("substrate_eps")
        return cls(span_x=meta["span_x"], span_y=meta["span_y"], layers=layers,
                   region_materials={int(k): v for k, v in meta["region_materials"].items()},
                   ambient=meta["ambient"], substrate=meta["substrate"],
                   mode=meta["mode"], eps_lambda_um=meta["eps_lambda_um"],
                   ambient_eps=complex(*meta["ambient_eps"]),
                   substrate_eps=complex(*se) if se else None,
                   detector=det, materials=meta.get("materials", {}),
                   dispersion=meta.get("dispersion", {}),
                   layer_tags=meta.get("layer_tags")).validate()


# --------------------------------------------------------------------------
def ir_from_voxels(matid3d, dz_um, dxy_um, id2name, substrate="si", ambient="air",
                   detector=None, materials=None, dispersion=None):
    """3D voxel 물질 배열 -> IR. 동일한 연속 z-slice 는 한 층으로 병합.

    matid3d: [nz,ny,nx] (z=0 이 아래/투과쪽). 반환 layers 는 위->아래.
    """
    layers = []
    prev, count = None, 0
    for z in range(matid3d.shape[0]):
        sl = matid3d[z]
        if prev is not None and np.array_equal(sl, prev):
            count += 1
        else:
            if prev is not None:
                layers.append((prev, count * dz_um))
            prev, count = sl, 1
    layers.append((prev, count * dz_um))
    layers = list(reversed(layers))                 # 위->아래
    ny, nx = matid3d.shape[1], matid3d.shape[2]
    return StructureIR(span_x=nx * dxy_um, span_y=ny * dxy_um, layers=layers,
                       region_materials={int(k): v for k, v in id2name.items()},
                       ambient=ambient, substrate=substrate, detector=detector,
                       materials=materials or {}, dispersion=dispersion or {}).validate()


def ir_from_eps_npy(eps_path, downsample=1):
    """위저드 eps 스냅샷(<name>_eps.npy + <name>_meta.json) -> eps 모드 IR.

    meta 에 si_thickness/pixel_pitch/bayer 가 있으면 detector 까지 구성.
    옆에 <name>.yaml 이 있으면 정확한 기하(트렌치 마스크)로 detector 를 대체.
    """
    import os
    arr = np.load(eps_path)                          # complex64 [nz,ny,nx]
    mpath = eps_path.replace("_eps.npy", "_meta.json")
    with open(mpath, encoding="utf-8") as f:
        meta = json.load(f)
    dz = float(meta["voxel_um"]["dz"])
    dxy = float(meta["voxel_um"]["dx"])
    ds = max(1, int(downsample))
    a = arr[:, ::ds, ::ds].astype(np.complex128)
    ny, nx = a.shape[1], a.shape[2]

    layers = []
    prev, count = None, 0
    for z in range(a.shape[0]):
        sl = a[z]
        if prev is not None and np.array_equal(sl, prev):
            count += 1
        else:
            if prev is not None:
                layers.append((prev, count * dz))
            prev, count = sl, 1
    layers.append((prev, count * dz))
    layers = list(reversed(layers))

    mats = {m["name"]: m for m in meta.get("materials", [])}
    sub = meta.get("substrate_material", "si")
    ms = mats.get(sub, {"n": 4.08, "k": 0.028})
    sub_eps = complex(float(ms["n"]), abs(float(ms.get("k", 0)))) ** 2
    amb_nk = meta.get("ambient_nk") or {"n": 1.0}
    amb_eps = complex(float(amb_nk.get("n", 1.0)), 0) ** 2

    # ---- detector ----
    det = None
    band = float(meta.get("si_thickness_um", 0) or 0)
    if band > 0:
        n_lay, acc = 0, 0.0
        for _, th in reversed(layers):
            if acc + 1e-9 >= band:
                break
            acc += th
            n_lay += 1
        pixel_map = None; labels = []; excl = None
        ypath = eps_path.replace("_eps.npy", ".yaml")
        if os.path.exists(ypath):                    # 정확 기하 (권장)
            from ..config.loader import load_config
            from .wizard_builder import WizardBuilder
            b = WizardBuilder(load_config(ypath))
            pixidx, labels = b.pixel_maps()
            pixel_map = pixidx[::ds, ::ds]
            excl = b.dti_trench_mask()[::ds, ::ds]
        elif meta.get("bayer"):                      # meta 근사
            npx = int(meta["n_pixels"]); p = float(meta["pixel_pitch_um"])
            yy, xx = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
            pr = np.clip(((yy * ds + 0.5) * dxy / p).astype(int), 0, npx - 1)
            pc = np.clip(((xx * ds + 0.5) * dxy / p).astype(int), 0, npx - 1)
            pixel_map = (pr * npx + pc).astype(np.int32)
            bay = meta["bayer"]
            labels = [bay[r][c] for r in range(npx) for c in range(npx)]
            excl = np.abs(layers[-1][0] - sub_eps) > 1e-6   # 라이너 근사 검출
        det = Detector(band_um=band, n_layers=n_lay, pixel_map=pixel_map,
                       pixel_labels=labels, exclude_mask=excl)

    return StructureIR(span_x=nx * dxy * ds, span_y=ny * dxy * ds, layers=layers,
                       region_materials={}, ambient="air", substrate=sub,
                       mode="eps", eps_lambda_um=float(meta.get("eps_lambda_um", 0) or 0),
                       ambient_eps=amb_eps, substrate_eps=sub_eps,
                       detector=det, materials={n: m for n, m in mats.items()}).validate()
