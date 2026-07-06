# -*- coding: utf-8 -*-
"""설정 데이터클래스 (ConfigData, MicroLensData 등). [스텁]"""
from dataclasses import dataclass, field

@dataclass
class MicroLensData:
    sag_height_um: float = 0.55
    material: str = "ml"
    quads: list = field(default_factory=list)

@dataclass
class ConfigData:
    """구조/시뮬 전체 설정 컨테이너 (추후 확장)."""
    raw: dict = field(default_factory=dict)
