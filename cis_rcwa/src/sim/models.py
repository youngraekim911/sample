# -*- coding: utf-8 -*-
"""SimCase 데이터클래스 (파장 range/step, 편광, 입사각 등). [스텁]"""
from dataclasses import dataclass, field
@dataclass
class SimCase:
    lam0_um: float = 0.40
    lam1_um: float = 0.70
    n_lambda: int = 13
    nG: int = 101
    theta_deg: float = 0.0
    phi_deg: float = 0.0
