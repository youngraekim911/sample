# -*- coding: utf-8 -*-
"""안료 산란 바닥 — 소프트웨어(코드) 소관의 측정↔소자 괴리 보정.

왜 코드에 있나
    material n,k 파일과 구조 yaml 은 '순수한 입력'(실측 그대로)으로 유지한다.
    평막 투과 n,k 측정은 안료 입자 산란광이 검출기로 들어가 '투과'로 집계되기
    때문에 산란 소광을 원리적으로 못 잡는다. 소자(서브µm 픽셀)에서는 그 산란이
    옆 픽셀 손실이 된다. 이 "측정 방식이 못 보는 몫"은 입력 데이터의 문제가
    아니라 측정↔소자 사이의 괴리이므로, 소프트웨어가 로딩 시 보정한다:

        k_eff(λ) = max(k_측정(λ), 바닥)

검증 (docs/최종구성_v134.md, docs/Green원인분석_쿠폰측정.md)
    - 4개 파장(520/540/550/560) 독립 역산 필요 k 가 0.0100~0.0112 한 점 수렴(±7%)
    - 쿠폰 투과 peak ~90%(실측 확인)와 원본 n,k 는 일치 — 데이터는 옳고
      빠진 것은 산란뿐
    - green 만 기본 등재: blue 통과대역 k(0.014)는 바닥보다 높아 무영향,
      red 는 장파장에서 시뮬<실측이라 바닥 근거 없음 (안료가 다르면 산란도 다름)

조정
    UI(위저드 우측 상단 ⚙ 설정) 또는 out/scatter_settings.json 으로 재정의.
    값 0 은 '끔'. 적분구(haze) 실측이 오면 그 값으로 교체할 것 (색마다 1회).
"""
import json
import os

# 기본값 (코드 소관) — 물질이름 -> k 바닥
DEFAULTS = {"cf_green": 0.0105}

_SETTINGS_PATH = None      # server 가 APP_DIR 기준으로 지정. 미지정이면 cwd/out


def settings_path():
    if _SETTINGS_PATH:
        return _SETTINGS_PATH
    return os.path.join(os.getcwd(), "out", "scatter_settings.json")


def set_settings_path(path):
    global _SETTINGS_PATH
    _SETTINGS_PATH = path


def load_overrides():
    """사용자 재정의 {물질: 바닥}. 파일 없음/깨짐 -> {}."""
    try:
        with open(settings_path(), encoding="utf-8") as f:
            d = json.load(f)
        return {str(k): float(v) for k, v in (d.get("k_floor") or {}).items()}
    except (OSError, ValueError, TypeError):
        return {}


def save_overrides(floors):
    """재정의 저장. 값은 float, 0 = 해당 물질 바닥 끔."""
    clean = {}
    for k, v in (floors or {}).items():
        try:
            clean[str(k)] = abs(float(v))
        except (ValueError, TypeError):
            continue
    p = settings_path()
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump({"k_floor": clean}, f, ensure_ascii=False, indent=1)
    return clean


def get_floors():
    """실효 바닥 = DEFAULTS 에 사용자 재정의를 덮은 것. 값 0/음수는 제외(끔)."""
    eff = dict(DEFAULTS)
    eff.update(load_overrides())
    return {k: v for k, v in eff.items() if v and v > 0}


def get_state():
    """UI 용 상태: 기본값 / 재정의 / 실효값."""
    return {"defaults": dict(DEFAULTS), "overrides": load_overrides(),
            "effective": get_floors(), "path": settings_path()}
