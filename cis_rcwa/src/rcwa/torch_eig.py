# -*- coding: utf-8 -*-
"""pytorch 고유값 솔버 래퍼.

torch.linalg.eig 는 복소 일반행렬 고유분해 + autograd 를 지원한다.
여기서는 RCWA 층 고유모드용 얇은 래퍼와, 분기(branch) 안전한 sqrt 를 제공.
GPU tensor 그대로 처리 (device 보존).
"""
import torch


def eig(A):
    """일반(비대칭) 복소행렬 고유분해.

    Returns
    -------
    w : (..., n)      고유값
    V : (..., n, n)   열이 고유벡터
    """
    return torch.linalg.eig(A)


def sqrt_decaying(x):
    """lam = sqrt(x) 를 취하되, 매질 내에서 +z 로 '감쇠'하는 분기를 고름.

    RCWA S-matrix 는 X = exp(-lam*k0*L) 로 전파를 다루므로 Re(lam) >= 0 이어야
    수치적으로 안정 (evanescent 모드가 감쇠). 실수부가 음이면 부호 반전.
    """
    r = torch.sqrt(x + 0j)
    r = torch.where(r.real < 0, -r, r)
    # 실수부가 (거의) 0 인 순수 evanescent 는 허수부로 분기 결정
    zero_re = r.real.abs() < 1e-12
    r = torch.where(zero_re & (r.imag < 0), -r, r)
    return r


def sqrt_outgoing(x):
    """반/투과 반무한 매질의 kz = sqrt(eps - kt^2). 바깥으로 나가는(전파) 분기.

    전파모드는 Re(kz) > 0, evanescent 는 Im(kz) > 0 (바깥으로 감쇠).
    """
    r = torch.sqrt(x + 0j)
    r = torch.where(r.imag < 0, -r, r)
    small_im = r.imag.abs() < 1e-12
    r = torch.where(small_im & (r.real < 0), -r, r)
    return r
