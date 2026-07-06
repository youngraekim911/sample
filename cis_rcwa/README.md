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

### 실행
```bash
pip install torch pyyaml numpy matplotlib

# 검증 (Fresnel / thin-film TMM / 에너지 보존 — 해석해 일치)
python3 rcwa/tests/validate.py

# QE 파장 sweep  (금속 grid 고대비 -> nG>=101 권장)
python3 runner.py -c qcell_config.yaml --lam0 0.45 --lam1 0.65 --n 5 --nG 101 --downsample 2
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

### 파장 의존 n,k
`qcell_config.yaml` 의 `dispersion:` 에 `[lambda_um, n, k]` 테이블 지정
(Si·CF 예시 포함 — 실측 값으로 교체 권장).

### 진행중(WIP)
- 픽셀별 QE / 3D |E|² 볼륨(Si 내부 필드): `RCWASolver` 필드복원 메서드
  (`absorption_profile`, `layer_internal_fields`) 확장 — flux 매핑 보정 필요.
