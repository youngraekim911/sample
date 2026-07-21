# CIS RCWA QE 시뮬레이터 — 인수인계 (다른 로컬 Claude용)

## 목적/구조
CIS(CMOS 이미지센서) 구조를 위저드로 만들고 RCWA(FMM)로 파장별 QE(R/G/B) 계산.
- repo: `youngraekim911/sample`, 작업브랜치 `claude/cis-rcwa-analysis-ol0h4u`, 루트 `cis_rcwa/`
- 실행: `python app.py` → http://127.0.0.1:8787 (torch/numpy/pyyaml, GPU 자동감지)
- 회귀: `python tests/regress_ir.py` (8개, 반드시 ALL PASS 유지)
- 규약: 변경 후 commit+push, 결과물은 .tar.gz(gz)로 전달, 회귀 통과 확인

## 핵심 파일
- `src/rcwa/rcwa.py` RCWA 솔버(S-matrix, 층 eig/solve) ← **속도 병목**
- `src/rcwa/kbloch.py` G-order truncation(대칭 shell)
- `src/structure/blocks.py` 6블록(SiDti/Barl/GridCf/Planar/Ml/Coat)→IR, `ir_from_wizard_cfg`, 인접동일층 머지
- `src/structure/ir.py` StructureIR + Detector(n_layers/n_below_band)
- `src/sim/simulator.py` `RCWAPlaneWaveSimulator.run/_detector_qe`(밴드흡수=QE)
- `src/sim/converge.py` nG 수렴(구조맞춤 nG추천+다중nG평균→참값±밴드)
- `src/materials/{library,resolver}.py` n,k 해석 (폴더 우선)
- `src/api/server.py` /api/qe(+converged), /api/materials
- `editors/structure_wizard.html` 위저드(페이지 라우터 UX)
- `conf/wizard_config.yaml` 예시 소자, `data/materials/*.txt` n,k(플레이스홀더)

## 아키텍처 원칙
- **물질 n,k는 폴더 txt가 유일 진실원**. 코드에 하드코딩 상수 없음. 폴더에 없는 물질→계산 거부.
  우선순위: 폴더 > yaml embedded dispersion > yaml 상수. (저장 yaml 재실행 시 폴더 최신값 반영됨)
- 구조=블록 조립→IR→RCWA. 구조 바뀌어도 RCWA 안 건드림(IR 계약).
- QE = Si 밴드(광다이오드 두께) 3D 흡수. `collect_deep_substrate=False`(기본)=밴드만.
- 재현성: RCWA 난수 없음. 같은 (구조,n,k,nG) → 동일 QE. **단 nG 고정 필요**.

## 물리 노브(무엇이 뭘 바꾸나)
- **cf_*.txt k 곡선**: off-peak 크로스톡/피크. 그린 청색차단 약하면 blue가 green CF 뚫고 누설.
- **back_reflector**(conf `stack.si.back_reflector:{enabled,material:cu,coverage}`): Si하부 Cu가 심투과 red 되돌려 2차흡수. coverage 0~1로 red@700 조절. solid=과다.
- **ML 초점**: 2x2렌즈는 매질(n≈1.58) 때문에 초점이 Si표면보다 ~1µm 깊음(그린 흡수구간과 일치→OK). 곡률=height/footprint.
- **ARL top 두께**: 그린 λ/4n AR. 얇으면 그린↑ 적색↓(공유층 트레이드오프).
- **nG(푸리에차수)**: 클수록 정확·느림(≈nG³). 이 큰 셀(2.56µm)은 nG≤250에서 ±3% 진동(미수렴). 참값은 converged 모드(다중 nG 평균) 또는 nG↑.

## 지금까지 개선(무엇으로 뭘)
1. 물질 폴더화: /api/materials + 매 run 재fetch, localStorage 캐시 제거 → '옛 물질 걸림' 제거
2. CF k 실제화 → 그린 청색누설 39~54%→6~12%, blue 피크 470
3. back_reflector(Cu, 부분커버리지) → red 2차흡수, Detector.n_below_band로 밴드 인덱싱
4. converge.py + 'converged' 품질 → 타겟튜닝 대신 수렴으로 일반화(참값±불확도)
5. 절대수치 제거 → 제네릭 플레이스홀더(순수 프로그램)
6. 코드 4각도 검증+버그6수정(폴더우선/파라미터가드/n_below검증/Wood재귀/CSV헤더/voxel가드)
7. UX 페이지 라우터: 팝업→페이지(시작→구조→시뮬→설정→실행→결과), export/import탭 제거, run시 재바인드, 💾구조저장
8. **속도**(핵심): patterned 층 60→14
   - ML 돔 슬라이스 48→8 (4에서 blue −2%p 붕괴, 8이 안전마진)
   - 인접 동일맵 층 머지(to_ir): flat CF meniscus redundant 제거
   - `_layer_smatrix` inv→solve(LU재사용)+W0=I2 활용(matmul2회↓, uniform inv생략)
   - **GPU 배치 eig**: patterned 층 eig 독립→`_all_layer_modes`가 OM2 stack해 `torch.linalg.eig` 1회(GPU 층 병렬)
   - kbloch 대칭 shell(±G 대칭, jitter 일부↓)
   전부 QE 무손실·에너지보존 1.0·회귀 8/8 검증.

## 남은 병목/다음 개선
- **eig가 patterned 층 최대비용(nG³)**. 14층까지 줄임. nG=500 수렴엔 GPU 필수(코드 device=cuda 자동).
- cf_grid 5층=grid TiN/TEOS(2,물리필수)+CF본체(1)+CF높이step(2). step 2개는 R/G/B CF 두께 상이 때문→두께통일시 3층(구조결정).
- nG 진동은 본질적 미수렴. 근본해결=nG↑(GPU) 또는 converged 평균. 대칭 truncation만으론 부족.
- 저순위(미착수): 에너지보존 독립검증(현재 A_stack=1-R-QE 항등식), trench흡수 vs 픽셀QE 정합, JOBS dict evict, recommend_center_nG 대형셀 cap(701) 스케일.
- 추가 속도 아이디어: OM2 빌드(conv/inv(ER))도 배치화, downsample↑, 대칭구조면 order 축소.

## 검증 방법(변경 후)
`python tests/regress_ir.py` ALL PASS + 에너지보존(R+QE+A=1) 확인. QE 바뀌면 EXPECT_G(현 0.6491, ml_slices=8·플레이스홀더 기준) 갱신.
