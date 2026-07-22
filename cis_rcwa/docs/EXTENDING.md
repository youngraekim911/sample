# 구조 확장 가이드 — 구조를 바꿔도 RCWA 는 안 건드린다

이 시스템은 **구조(블록) → IR(중간표현) → RCWA** 로 완전히 분리돼 있습니다.
RCWA 는 `StructureIR` 계약만 알기 때문에, **구조를 어떻게 바꾸든 RCWA/QE 코드는
수정할 필요가 없습니다.** 잘못된 구조는 린터(`src/structure/lint.py`)와
`StructureIR.validate()` 가 실행 전에 사람이 읽을 수 있는 메시지로 잡아줍니다.

## 1) yaml 만 바꾸는 경우 (대부분)
`stack:` 아래 값(두께·물질·grid·cf·ml…)을 바꾸면 됩니다. 위저드 UI 또는 yaml
직접 편집 → Run. 치명 실수(두께 0, 물질 누락, bayer 크기 불일치 등)는
`POST /api/lint` 및 run 직전 점검이 "어디를 고쳐라"로 알려줍니다.

물질 n,k 는 코드가 아니라 **`data/materials/<이름>.txt`** (3열: 파장 n k)가
단일 진실원입니다. 새 물질은 파일만 추가하면 됩니다.

## 2) 새로운 층/구조 블록을 추가하는 경우
블록은 딱 하나의 계약만 지키면 됩니다:

```python
class MyBlock:
    def build(self, ctx):
        # ctx.X, ctx.Y : (N,N) 가로 좌표 격자 (µm)
        # ctx.mat_id("이름") : 물질 이름 -> 정수 공간번호
        # 반환: [(map2d[int], 두께_um), ...]  (아래->위 순서)
        m = ctx.zeros("oxide")          # 전부 oxide 로 채운 한 층
        # ... m[...] = ctx.mat_id("metal")  식으로 패턴 그리기 ...
        return [(m, 0.05)]
```

그리고 조립 목록에 끼워 넣으면 끝:

```python
from src.structure.blocks import BlockStack, BlockContext, SiDtiBlock, ...
ctx = BlockContext(pitch_um=0.7, n_pixels=4, lateral_n=256, bayer=[[...]])
ir  = BlockStack([SiDtiBlock(...), MyBlock(), GridCfBlock(...), ...],
                 materials={...}).to_ir(ctx)
# 이후는 기존과 동일:
RCWAPlaneWaveSimulator(ir, nG=151).run(0.55)
```

- 블록 간 결합은 **ctx 의 공표 필드**로만 이뤄집니다
  (트렌치 마스크 `ctx.trench`, CF 상면 `ctx.cf_top_min`, 돔 sag `ctx.dome_sag` 등).
  이 필드만 규약대로 채우면 위/아래 블록과 자동으로 이어집니다.
- 검출기(어느 층이 광다이오드인지)는 `ctx.det_band_um`, `ctx.det_n_layers`,
  `ctx.trench`(제외영역) 로 공표 → `to_ir` 이 `Detector` 로 묶습니다.
- RCWA 는 `ir.layers`(맵·두께)와 `ir.detector` 계약만 보므로 **블록을 아무리
  늘려도 손대지 않습니다.**

## 3) 안전장치 (누가 바꿔도 깨지지 않게)
- `src/structure/lint.py` : 조립 전 흔한 실수 차단 (errors=실행중단 / warnings=안내).
  모든 경로(위저드·CLI·DOE)가 통과하는 단일 관문 `ir_from_wizard_cfg` 에 걸려 있음.
- `StructureIR.validate()` : 격자 크기·두께>0·물질매핑·검출기 인덱싱 계약 강제.
- 회귀 `tests/regress_ir.py` [6] : "블록 조립 == 레거시 rcwa_layers" 를 매번 검증
  → 블록 경로가 기존 결과와 동일함을 보장.

## 4) 위저드에 UI 노브를 노출하려면
`editors/structure_wizard.html` 의 `STEPS` 배열에 패널 하나 추가하고,
`structYAML()` 이 S(상태)를 그대로 내보내므로 값은 자동으로 yaml 에 실립니다.
(모델링 옵션 dti.optical/collection 등은 UI 없이 엔진이 자동 판단합니다.)
