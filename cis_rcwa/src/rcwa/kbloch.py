# -*- coding: utf-8 -*-
"""Bloch 격자: 역격자 벡터, G-order truncation, 입사파 K 설정.

규약
----
정규화 파수: kx, ky 는 k0 = 2*pi/lambda 로 나눈 무차원 값.
G-order 는 정수쌍 (m, n) 으로 표현, 물리 역격자벡터 = m*g1 + n*g2.
truncation: 'circular' (|G| 작은 순 nG 개) 또는 'rectangular'.
"""
import math
import torch


def reciprocal(L1, L2):
    """실공간 격자벡터 L1,L2 (2,) -> 역격자 g1,g2 (2,), g_i . L_j = 2*pi δ_ij."""
    A = torch.tensor([[L1[0], L1[1]], [L2[0], L2[1]]], dtype=torch.float64)
    G = 2 * math.pi * torch.linalg.inv(A).T   # 행이 g1, g2
    return G[0], G[1]


def get_G(nG_target, g1, g2, trunc="circular"):
    """G-order 집합 선택.

    Returns
    -------
    m, n   : (N,) long        정수 order
    Gx, Gy : (N,) float64     물리 역격자 성분
    """
    gmin = min(float(torch.linalg.norm(g1)), float(torch.linalg.norm(g2)))
    M = int(math.sqrt(nG_target)) + 3
    ms, ns, mag = [], [], []
    for m in range(-M, M + 1):
        for n in range(-M, M + 1):
            gx = m * g1[0] + n * g2[0]
            gy = m * g1[1] + n * g2[1]
            ms.append(m); ns.append(n)
            mag.append(float(gx * gx + gy * gy))
    idx = sorted(range(len(mag)), key=lambda i: mag[i])
    if trunc == "circular":
        keep = idx[:nG_target]
    else:  # rectangular: |m|,|n| <= M0
        M0 = int(round(math.sqrt(nG_target)) // 2)
        keep = [i for i in idx if abs(ms[i]) <= M0 and abs(ns[i]) <= M0]
    m = torch.tensor([ms[i] for i in keep], dtype=torch.long)
    n = torch.tensor([ns[i] for i in keep], dtype=torch.long)
    Gx = m.to(torch.float64) * g1[0] + n.to(torch.float64) * g2[0]
    Gy = m.to(torch.float64) * g1[1] + n.to(torch.float64) * g2[1]
    return m, n, Gx, Gy


def set_incidence(k0, n_inc, theta, phi, Gx, Gy):
    """입사(theta, phi) + G-order -> 정규화 kx, ky (N,).

    theta, phi : radian.  theta=0 수직입사.
    """
    kx0 = n_inc * math.sin(theta) * math.cos(phi)
    ky0 = n_inc * math.sin(theta) * math.sin(phi)
    kx = kx0 + Gx / k0
    ky = ky0 + Gy / k0
    return kx, ky
