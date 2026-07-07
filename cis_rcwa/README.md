# CIS qcell → npy 구조 빌더

CIS(CMOS Image Sensor)의 **qcell**(= (2×2 pixel) quad ×4 = 4×4 픽셀 단위)을
YAML 로 정의하고 3D voxel 구조(`.npy`)로 뽑아 확인하는 도구.
이후 RCWA 광학 해석의 입력으로 사용.

## 층 구조 (아래 → 위)

| # | layer | 설명 |
|---|-------|------|
| 1 | `substrate_si` | Photodiode (Si) |
| 2 | `metal_grid` | DTI / backside metal grid — 격벽=metal, 내부=oxide |
| 3 | `arc` | planarization / anti-reflection (균일) |
| 4 | `color_filter` | R/G/B, 사이에 metal grid 격벽, 그 사이를 CF가 채움 |
| 5 | `ml_spacer` | ML 동일물질 평탄화층 (focus 거리 확보) |
| 6 | `microlens` | ML dome (spacer 위) |

## Microlens 레이아웃

qcell = 2×2 quad. **각 quad 마다** 렌즈 형태를 선택:

- `1x1` : 픽셀마다 원형 렌즈 4개
- `2x1` : 2픽셀 덮는 타원 렌즈 2개 (`orient: h` 가로 / `v` 세로)
- `2x2` : quad 중앙 초점 원형 렌즈 1개

`qcell_config.yaml` 의 `microlens_layout.quads[qy][qx]` 로 지정.
(주의: 배열 origin 은 아래(y=0)가 `qy=0`. top-view 이미지 기준 아래줄이 `quads[0]`)

## Microlens 변형 (thermal reflow)

이상적 원/타원 base 에서 **응집(reflow)** 으로 변형된 형상을 렌즈별로 지정 가능.

- 표현: 정규화 경계반경 **B(θ)** (이상형=1.0). `θ` 는 렌즈 local 정규화 좌표
  `u=(x-cx)/ax, v=(y-cy)/ay` 기준 `atan2(v,u)`.
- Dome: `sag = h·√(1-(ρ/B(θ))²)`, `ρ=√(u²+v²)`.
- **Volume 보존**: 이상 반타원체 부피 `V0 = h0·(2/3)π·ax·ay` 를 유지하도록 `h` 자동 재계산.
  → footprint 가 넓어지면 높이가 낮아짐 (재료량 일정).

### 렌즈 id 규칙
`q{qy}{qx}_{shape}_{k}` — 예: `q00_2x2_0`, `q11_1x1_0..3` (`k=dyp*2+dxp`), `q10_2x1_0..1`

### 지정 방법 (둘 다 지원)
1. **YAML inline** — `microlens_layout.deformations`:
   ```yaml
   deformations:
     q00_2x2_0:
       ctrl: [[0,1.18],[90,0.86],[180,1.18],[270,0.86]]   # [angle_deg, radius_mult]
     q11_1x1_0:
       b_theta: [ ...128 values... ]                       # 직접 B(θ) 배열
   ```
2. **HTML 에디터 export** — `microlens_layout.deform_file: "ml_shapes.json"`
   (파일에 `quads` 포함 시 그 레이아웃이 우선 적용)

## 위저드 스키마 v3 → RCWA (Python)

위저드가 저장한 `<product>.yaml` 을 **그대로 RCWA 파이프라인에 사용** 가능:
```bash
# 구조 npy (위저드와 동일 지오메트리, src/structure/wizard_builder.py)
PYTHONPATH=. python3 -m src.structure.wizard_builder -c conf/wizard_config.yaml -o out
# QE sweep — simulator 가 스키마 자동 인식 (wizard v3 / 구버전 qcell)
python3 run_qe.py -c conf/wizard_config.yaml --nG 101
```
구조 모델 (경계/표면 RCWA 정합):
- **DTI**: Si 식각(1×1 분리 | 2×2 center-open **클로버**) → 판 표면(사이드월+팔 끝벽)
  oxide 라이너 → 채움. 클로버는 4픽셀 Si 가 중앙에서 연결.
- **BARL**: 4×4 전면 blanket 다층.
- **Grid**: 울타리(fence) 1×1/2×2, 다층 stack(동일물질 연결) → 병합 표면(옆+위) oxide 코팅 옵션.
  **taper**: 하부 width 기준, 상부 `top_ratio`(0.8~1)로 선형 축소.
- **CF**: 울타리 셀 채움, 컬러별 두께 + **상부 곡률(meniscus, +볼록/−오목)** — RCWA 반영.
- **ML 풍선(balloon) 모델**: 렌즈 = 원/타원 footprint(중심·반경)만 정의 →
  높이 = `hr`×min(반경), 곡률 자동. 겹치면 max(sag) → 교선이 풍선 '찌부' 접촉선.
  step7 인라인 에디터에서 드래그(이동/크기). 파라미터 (cx,cy,ax,ay,hr)뿐이라 일반화/최적화 용이.
- **ARL**: ML 표면 conformal(ALD).

## 구조 위저드 — `editors/structure_wizard.html`

브라우저에서 **step-by-step**(1~8)으로 구조 설정 → 3D/단면 확인 → **npy 생성**:
1. pixel pitch·n×n  2. Si 물질(폴더 n,k)·두께  3. DTI(width·oxide 라이너, depth=Si두께)
4. BARL 다층  5. Grid(1×1/2×2·width·물질 stack·표면 코팅)  6. CF 컬러별 물질·두께
7. ML 물질·평탄층·형상(ml_shapes.json import)  8. ARL top

- **📁 Materials**: 폴더 n,k txt 로드 → 물질별 파장의존 n,k
- **⤒ Import / ⤓ YAML**: 구조를 `<product>.yaml` 로 저장/불러오기
- **⤓ npy + yaml**: 브라우저에서 `<product>_matid.npy` + `_meta.json` + `.yaml` 동시 생성
  (numpy 로 바로 로드 가능 — Python 불필요. 4GB 초과 해상도는 Chrome/Edge 에서
  디스크 스트리밍 저장. 브라우저 npy == Python 빌더 npy, voxel 100% 동일 검증)
- 뷰: XZ 단면 / 3D isometric / top-view

### 뽑은 npy 확인 — `view_npy.py`
```bash
python3 view_npy.py -i <product>_matid.npy        # 옆의 _meta.json 자동 사용
# -> <product>_view.png : XZ/YZ 단면(물질 다양한 슬라이스 자동 선택) +
#    z-점유율 스택 + 대표 z 3곳 XY 평면 + 물질 범례 / z-밴드 요약 출력
```
npy 규약: `uint8` 물질 id, shape `[nz, ny, nx]` — **z=0 이 바닥(Si)**, 빛은 +z 위에서 입사.
`a[z]` 가 높이 z 의 XY 평면. 단면을 정확히 pixel 경계(y=1.0µm 등)에서 자르면
grid 벽/DTI 를 따라 잘려 벽 물질만 보이니 셀 중앙(y=0.5µm 등)으로 자를 것.

## UI 에디터 — `ml_shape_editor.html`

브라우저에서 열어 **드래그로 base 2D 형상을 변형**하고 dome/height 실시간 확인:

- 4개 quad 렌즈 형태(1×1/2×1/2×2) 선택
- 렌즈별 control point 드래그 또는 preset(가로/세로 응집, 각짐, 3-lobe, pear) + strength
- volume 보존 height, dome 단면 실시간 표시
- **Export ml_shapes.json** (→ `deform_file` 로 사용) / **Copy YAML** (→ `microlens_layout` 붙여넣기)

## 사용법

```bash
pip install numpy pyyaml matplotlib

# 1) 구조 생성 -> out/qcell_matid.npy, qcell_index.npy, qcell_meta.json
python3 build_qcell.py -c qcell_config.yaml -o out

# 2) 구조 확인 -> out/qcell_view.png (단면 + ML top-view + CF 배치)
python3 view_qcell.py -o out
```

## 출력물

- `qcell_matid.npy` — `uint8` 물질 id 배열, shape `[nz, ny, nx]`
  (z=0 이 바닥 Si, 빛은 +z 위에서 입사)
- `qcell_index.npy` — `complex64` 복소굴절률 `n+ik` 배열 (동일 shape, RCWA 입력)
- `qcell_meta.json` — voxel 크기, 층 z-경계, 물질 범례, 축 규약

## RCWA 해석 (PyTorch, GPU 지원)

grcwa 기반 FMM + 강화 S-matrix 를 PyTorch 로 포팅. `device='cuda'` 자동 감지.

```
config(YAML) → build_qcell(구조) → RCWAPlaneWaveSimulator → RCWASolver
             → R / QE(Si) / A_stack → runner(파장 sweep) → QE 스펙트럼
```

### 모듈 (`rcwa/`)
| 파일 | 역할 |
|------|------|
| `rcwa.py` | `RCWASolver` — 역격자·입사 K → 층 eps FFT 컨볼루션 → 고유모드(eig) → S-matrix → R/T/QE |
| `fft_funs.py` | eps(x,y) 격자 → Fourier 컨볼루션 행렬 (FFT) |
| `kbloch.py` | 역격자 / G-order truncation(원형·사각형) / 입사파 K |
| `torch_eig.py` | 복소 고유값 분해 + 분기(branch) 함수 |

특징: 원형/사각형 truncation · 복소굴절률(흡수) · 비수직 입사(θ,φ) · GPU 가속.

### 폴더 구조 (src/ 블록)
```
cis_rcwa/
  conf/            설정 YAML (qcell_config, structure_config)
  data/materials/  물질별 파장 n,k txt
  editors/         structure_editor.html, ml_shape_editor.html
  src/
    config/        loader(YAML) + schema(dataclass)
    materials/     library.py (MaterialLibrary)
    structure/     builder.py(RCWATensorStack) + color_filter/si_dti/shrink[stub]
    rcwa/          rcwa/kbloch/fft_funs/torch_eig + tests/validate + eig/[stub]
    sim/           simulator, runner, qe_calc/cone/models/option[stub]
    viz/ eval/ opt/ utils/ api/ cache/ core/   [블록]
  validate.py / build_structure.py / run_qe.py   (최상위 실행 진입점)
```

### 실행
```bash
pip install torch pyyaml numpy matplotlib

# 검증 (Fresnel / thin-film TMM / 에너지 보존 — 해석해 일치)
python3 validate.py

# 구조 생성(npy)
python3 build_structure.py -c conf/qcell_config.yaml -o out

# QE 파장 sweep  (금속 grid 고대비 -> nG>=101 권장)
python3 run_qe.py -c conf/qcell_config.yaml --lam0 0.45 --lam1 0.65 --n 5 --nG 101 --downsample 2
# 출력: out/qe_spectrum.csv, out/qe_spectrum.png
```

### QE 정의
투과매질 = Si 반무한 → **T = Si 로 결합되는 광량 = 광학 QE** (검증됨).
반사 R + 스택 흡수 A_stack(metal/CF) + QE = 1.

### 검증 (`rcwa/tests/validate.py`)
- 단일 계면 Fresnel (수직/사입사) = 해석해 정확 일치
- 단일 박막 = 1D transfer-matrix 정확 일치
- 무손실 패턴층 에너지 보존 R+T=1
- 흡수층 물리적 R/T/A

### 파장 의존 n,k — `materials/` 폴더 (`materials.py`)
물질별 txt 파일을 자동 로드해 **임의 파장 step/range 를 보간**으로 커버.
```
materials/<name>.txt     # 열: wavelength  n  k  (nm/um 자동감지, # 주석 허용)
```
```python
from materials import MaterialLibrary
lib = MaterialLibrary("materials")
n, k = lib.nk("si", 0.55)      # 임의 파장 보간
```
우선순위: `materials/` 폴더 → config `dispersion:` → 물질 상수 n,k.
구조 에디터에서 **📁 Load materials folder** 로 불러오면 물질별 λ-dep n,k 가
물리고(λ✓), 편집상태는 자동 저장(localStorage).

### 진행중(WIP)
- 픽셀별 QE / 3D |E|² 볼륨(Si 내부 필드): `RCWASolver` 필드복원 메서드
  (`absorption_profile`, `layer_internal_fields`) 확장 — flux 매핑 보정 필요.
