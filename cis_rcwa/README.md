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

## RCWA 로 잇기 (다음 단계)

RCWA 는 층별 2D 주기 패턴 + 파장 의존 n,k 를 요구.
- `qcell_index.npy` 를 z 방향으로 층 슬랩(slab)화 하거나, `meta` 의 `layer_bounds`
  구간별로 대표 2D 패턴을 뽑아 solver 에 전달.
- 엔진 후보: `S4`, `grcwa`, `torcwa`, `RETICOLO`.
- 파장별 QE / crosstalk / angular response 계산으로 확장.
