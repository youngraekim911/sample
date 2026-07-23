# CIS qcell → npy 구조 빌더

CIS(CMOS Image Sensor)의 **qcell**(= (2×2 pixel) quad ×4 = 4×4 픽셀 단위)을
YAML 로 정의하고 3D voxel 구조(`.npy`)로 뽑아 확인하는 도구.
이후 RCWA 광학 해석의 입력으로 사용.

## 빠른 시작 (처음 받은 분)

**윈도우**: `run_windows.bat` 더블클릭 (Python 3.9+ 필요 — python.org 설치 시 "Add to PATH" 체크).
**mac/linux**: `./run.sh`
→ 최초 1회 가상환경+패키지 자동 설치 후 http://127.0.0.1:8787 이 열립니다.

수동 설치 시:
```
python -m pip install -r requirements.txt
python app.py
```
`ModuleNotFoundError: No module named 'yaml'` 등이 나오면 위 설치가 안 된 것 —
app.py 가 시작 시 누락 패키지를 알려주고 자동 설치도 제안합니다.

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

## 블록 아키텍처 — StructureIR (구조/RCWA 분리 계약)

```
[구조 블록]                        [계약]                [RCWA 블록]        [결과 블록]
WizardBuilder.to_ir() ─┐                                                  QE 스펙트럼
eps npy(ir_from_eps_npy)├──►  StructureIR  ──────►  RCWAPlaneWaveSimulator ─ 픽셀/라벨별 QE
voxel(ir_from_voxels) ─┘      · layers[(공간맵,두께)]                        워터폴 진단
임의의 새 빌더 ────────┘      · region_materials{공간→물질}                  구조 이미지(viz)
                              · detector(밴드+픽셀분할+제외마스크)
```

구조란 본질적으로 **공간 분할(boundary) + 각 공간을 채우는 물질** 이다.
`src/structure/ir.py` 의 `StructureIR` 이 그 두 가지만 담는 계약이고,
RCWA(`src/sim/simulator.py`)는 IR 만 소비한다 — DTI/CF/ML 같은 구조 개념을 모른다.
따라서 **구조·물질이 바뀌어도 RCWA/결과 코드는 재작성이 필요 없다**:

```python
from src.sim.simulator import RCWAPlaneWaveSimulator
sim = RCWAPlaneWaveSimulator("conf/wizard_config.yaml")   # yaml | *_eps.npy | ir.npz | StructureIR
sim.remap_material("cf_green", "my_new_cf")   # 기하 그대로 물질만 교체
sim.ir.save_npz("structure.npz")              # IR 저장/재사용
o = sim.run(0.55)                             # {R, QE, QE_rgb(라벨별), ...}

from src.viz.structure_view import render_ir  # IR -> 구조 이미지 (빌더 무관)
render_ir(sim.ir, "structure.png")
```

### 구조 블록 (src/structure/blocks.py)

구조 블록도 부품화되어 있다 — 아래(Si)→위(공기) 순으로 조립:

| 블록 | 역할 | 주요 파라미터 |
|---|---|---|
| `SiDtiBlock` | Si + DTI (none/1x1/2x2_open 클로버) | thickness, width, **liners 스택**(벽면 겹겹이), fill(poly), center gap |
| `BarlBlock` | blanket 다층 AR — 층 수 임의 | [{material, thickness}, ...] (gradual n) |
| `GridCfBlock` | grid 울타리(stack+표면코팅) 안 CF 채움 | grid{pitch,width,taper,stack,coat}, cf{색별 두께+**reflow 응집 곡률**} |
| `PlanarBlock` | ML 평탄층 — CF/grid 위 전부 채움 | material, thickness |
| `MlBlock` | ML 렌즈 (1x1/1x2/2x1/2x2 × quad 4) | height(곡률 자동), scale, quads/lenses |
| `ConformalCoatBlock` | ML 표면 conformal 코팅(ARL) | material, thickness — **여러 겹 가능** |

```python
from src.structure.blocks import *
ctx = BlockContext(pitch_um=1.0, n_pixels=2, lateral_n=200, bayer=[["R","G"],["G","B"]])
ir = BlockStack([
    SiDtiBlock("si", 4.0, dti={"mode":"2x2_open","width_um":0.10,
        "liners":[{"material":"oxide","thickness_um":0.03}],  # 벽면 30nm 좌우 -> 남는 40nm
        "fill":"poly","center_gap_x_um":0.2,"center_gap_y_um":0.2}),
    BarlBlock([...7층이든 몇 층이든...]),
    GridCfBlock(grid, cf, bg_material="ml"),
    PlanarBlock("ml", 0.10),
    MlBlock("ml", height_um=0.35, quads=[[{"shape":"1x1","scale":1}]*2]*2),
    ConformalCoatBlock("ml_arl", 0.12),          # 겹겹이 코팅 가능
], materials={...}).to_ir(ctx)                   # -> 그대로 RCWA 에
```

위저드 yaml 은 이 블록 조립의 한 사례일 뿐이다(`blocks_from_wizard_cfg`) —
블록 경로와 기존 위저드 경로가 **맵/두께/detector 완전 일치**함을
`tests/regress_ir.py` [6] 이 보증한다. 블록 간 결합은 `BlockContext` 공표
필드(트렌치 마스크, CF 상면, 돔 sag)로만 이루어지므로 블록 추가/교체가 자유롭다.

### RCWA 실행 블록 (src/sim/blocks.py)

RCWA 쪽도 블록이다 — 광원/메쉬/프로브를 조합해 실행:

```python
from src.sim.blocks import RCWAEngine, LightSource, MeshPolicy, QEProbe, BoundaryProbe
from src.viz.analysis_view import plot_boundary_T, plot_qe

eng = RCWAEngine(ir, mesh=MeshPolicy(nG=101, downsample=2))   # ① IR 수신 ② mesh 정책
res = eng.run(LightSource(lam=(0.40, 0.70, 13), theta=0, pol="avg"),  # ③ 빛 옵션 블록
              probes=[QEProbe(), BoundaryProbe()])            # ④ 원하는 분석 프로브

res["qe"]        # 픽셀별 QE — [{pixel,row,col,x_um,y_um,cf(어느 CF 아래인지),qe[λ]}]
res["qe_by_cf"]  # CF 라벨별 평균 QE 스펙트럼
res["boundary"]  # 블록 경계별 T(λ) — [{tag:"ml+ml_arl"|"cf_grid"|..., T:{R,G,B}}]
plot_boundary_T(res, "boundary_T.png")   # 경계 Transmittance 그래프 (컬러 분기)
plot_qe(res, "pixel_qe.png")             # 픽셀/CF별 QE 그래프
```

- **MeshPolicy**: 공간 크기별 mesh 자동 최적 — z 는 해석적 층 경계(Å층 정확,
  복셀화 없음), 연속 곡면만 슬라이스, 균일층은 고유분해 생략 → λ당 수십 초.
- **BoundaryProbe** 는 IR 의 `layer_tags`(블록 provenance)로 층을 묶어, 빛
  시작(1-R)부터 Si 유입까지 각 블록 경계의 T(λ) 를 컬러 분기로 준다. 흡수맵은
  QE 계산과 같은 solve 를 재사용하므로 추가 비용이 거의 없다.
- **QEProbe** 는 모든 픽셀의 QE 를 위치(µm)·bayer (row,col)·상부 CF 라벨과
  함께 준다 — 어떤 QE 가 어떤 CF 에 의한 것인지 명확.

- 물질 해석은 `src/materials/resolver.py` 단일 창구: ① yaml dispersion(브라우저
  테이블) → ② materials/ 폴더 → ③ 상수. k 는 전 경로 |k|.
- QE 집계는 `IR.detector` 규약만 따른다: 스택 하단 검출 밴드(n_layers) 3D 흡수
  + 심부(반무한) 흡수를 `pixel_map`/`exclude_mask` 로 픽셀 귀속, 라벨별 평균.
- 새 구조 생성기는 `StructureIR` 만 만들면 끝 (`validate()` 가 계약 위반을 즉시 검출).
- 회귀: `python3 tests/regress_ir.py` (yaml 회귀 / npz 라운드트립 / remap / eps 모드 / 렌더).

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

## 한 줄 실행 — structure → RCWA → QE (프론트 flow)

```bash
pip install torch numpy pyyaml
python3 app.py            # http://127.0.0.1:8787 자동 오픈 (GPU 있으면 자동 사용)
```
브라우저에서 step1~8 로 구조 설정 → 상단 **[▶ QE 해석]** → 파장범위·nG 설정 → Run
→ **QE / R / A 스펙트럼** 차트·표·CSV. 구조는 현재 위저드 상태가 그대로 RCWA 에 들어간다.

블록 구조 (개발자가 아니어도 이 경계만 알면 됨):
```
[프론트]  structure_wizard.html  ── structYAML() ──►  POST /api/qe
[백엔드]  src/api/server.py  ─►  WizardBuilder(구조) ─►  RCWAPlaneWaveSimulator
          ─►  RCWASolver(FMM+S-matrix, torch) ─►  QE=Si 결합 T  ─►  GET /api/qe/status
```
- nG: Fourier 차수. metal grid 수렴에 **101 이상 권장** (기본값). 빠른 미리보기는 41~61.
- downsample: 구조 격자 축소(속도↑). 최종 결과는 1~2 권장.
- 소요: CPU 기준 nG=101 에서 ~수 초/파장(TE+TM). CUDA 자동 감지.

### 적응 mesh (기본값 mesh="auto")
RCWA 는 z 방향 mesh 가 필요 없다 — 층 두께를 해석적으로 정확히 지정:
- **z**: 복셀화 없이 yaml 기하에서 층 경계 직접 생성. BARL Å 단위·20nm 코팅 cap·
  grid stack 경계·Si 밴드 = **정확한 두께**. 연속 곡면만 계단화(ML 돔+ARL 48,
  CF 응집면 8, grid taper 8 슬라이스 — `rcwa_layers()` 인자).
- **가로**: npy dxy 와 독립적인 미세 래스터 (기본 5nm×downsample; ds=2 -> 10nm).
- npy 복셀 해상도(step1 dxy/dz)는 이제 **뷰어/npy 출력 전용** — RCWA 정확도와 무관.
- 검증: 37Å BARL 층이 voxel mesh(dz=0.02)에선 소실, auto 에선 정확 반영.
- 구버전 비교용: `RCWAPlaneWaveSimulator(..., mesh="voxel")`.

### QE 물리 모델 (검증 요약)
- **파동광학 엄밀해**: 굴절·회절·간섭·다중반사·흡수가 한 풀이에 전부 포함
  (ML 집광, BARL/ARL 간섭, grid metal R/T/A — A/B 실험으로 개별 확인).
- **무한 반복 배열**: RCWA 는 주기 경계조건 — unit(4×4) 이 상하좌우·대각으로
  무한 반복된 배열의 해 (구조 평행이동 불변으로 검증).
- **DTI 포함**: Si 밴드(트렌치+라이너)가 패턴층으로 스택에 들어가 DTI 벽의
  반사/도파(픽셀 격리)까지 반영. 투과 매질은 트렌치 바닥 아래 벌크 Si.
- **픽셀별 QE = 픽셀 Si 볼륨의 3D 흡수**: Im(ε)|E|²(Ez 포함) 을 픽셀 1×1 의
  순수 Si 창(DTI 제외)으로 z-적분 + 밴드 아래 심부 유입분. 절대 스케일은
  에너지 보존(Σ흡수=1−R−T)으로 고정. 컬러 QE = 같은 색 픽셀 평균.

### complex64 eps 텐서 직접 입력
위저드 npy 저장 시 `<product>_eps.npy` (complex64, `(n+ik)²` @ 저장 시점 λ)가
함께 생성되며, **RCWA 에 그대로 입력** 가능:
```python
sim = RCWAPlaneWaveSimulator("foo_eps.npy", nG=101)   # meta.json 자동 사용
sim.run(0.55)      # yaml 경로와 R/QE/픽셀QE 1e-8 일치 검증됨
```
옆에 `<product>.yaml` 이 있으면 픽셀 마스크(DTI 제외)를 정확 기하로 구성.
주의: eps 는 λ 고정 스냅샷 — 파장 sweep 의 물질 분산은 yaml 경로 사용.

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

### surrogate DB (위치 지정 · 동기화)

DOE surrogate 는 상태해시 기반 영속 캐시(DB)에 저장되어 같은 구조·조건이면 재계산을
생략한다. DB 폴더는 위저드 `🧪 DOE` 카드의 **🗂️ surrogate DB 위치·동기화** 에서 지정.

- **위치 지정**: 공유 드라이브/팀 폴더를 지정하면 여러 사람이 같은 DB 를 공유. 우선순위는
  환경변수 `CIS_SURROGATE_DB` > 포인터파일(`surrogate_db_path.txt`) > 기본(`out/surrogate_cache`).
- **동기화(Sync)**: 로컬 DB ↔ 공유 폴더를 양방향 병합(서로 없는 `<key>.json` 만 복사).
  같은 key 는 상태해시가 동일 = 내용 동일이라 충돌이 없다.
- 목록은 파일 스캔으로 **자가치유** — 공유 폴더에 누가 `<key>.json` 을 복사만 해도 목록에 뜬다.
- API: `GET/POST /api/doe/cache/config`, `POST /api/doe/cache/sync`, `GET /api/doe/cache[/file]`.

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

### 물질 n,k 우선순위 (물질 변경 -> QE 반영 경로, 검증됨)
1. **yaml `dispersion:`** — 위저드가 브라우저에서 import 한 λ-테이블을 yaml 에
   자동 동봉 (화면에서 편집한 값 그대로 RCWA 에 들어감. 서버 폴더보다 우선)
2. **materials/ 폴더** (`src` 지정 파일 -> 동명 파일)
3. **yaml `materials:` 상수** (테이블 없는 커스텀 물질)
위저드에서 물질 import 시 새 물질이 전 step 선택 목록에 자동 등록되고,
dispersion 동봉 yaml 은 Import 시 테이블까지 복원된다 (라운드트립).

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
