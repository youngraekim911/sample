"""pixem — CIS qcell 구조/광학(RCWA) 시뮬레이션 패키지.

블록:
  config     설정 로드/파싱 · 데이터클래스 스키마
  materials  물질 파장별 n,k 라이브러리 (data/materials/*.txt)
  structure  qcell 구조 생성 (builder=RCWATensorStack, color_filter, si_dti, shrink)
  rcwa       RCWA 솔버 (FMM + S-matrix, torch/GPU)
  sim        시뮬레이터/러너/QE·신호/원뿔입사/데이터클래스
  eval       비교 평가
  opt        대리모델/최적화
  viz        시각화
  utils/api/cache/core  유틸·인터페이스·캐시·코어
"""
