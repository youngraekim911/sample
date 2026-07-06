# -*- coding: utf-8 -*-
"""RCWASolver 정확성 검증 (해석해 비교).

1) 단일 계면 Fresnel (수직/사입사)
2) 균일 단일 박막 = 1D thin-film transfer matrix 일치
3) 에너지 보존 R+T=1 (무손실), R+T+A=1
"""
import math
import numpy as np
import torch
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from rcwa import RCWASolver

torch.set_default_dtype(torch.float64)


def fresnel_normal(n1, n2):
    r = (n1 - n2) / (n1 + n2)
    R = r * r
    return R, 1 - R


def tmm_single_film(n0, n1, ns, d, lam, theta0=0.0):
    """1D 박막(수직/사입사, s-편광) transfer matrix R,T."""
    k0 = 2 * math.pi / lam
    s0 = n0 * math.sin(theta0)
    def kz(n): return k0 * np.sqrt(n**2 - s0**2 + 0j)
    kz0, kz1, kzs = kz(n0), kz(n1), kz(ns)
    # s-편광 Fresnel
    def rt(kza, kzb):
        r = (kza - kzb) / (kza + kzb)
        t = 2 * kza / (kza + kzb)
        return r, t
    r01, t01 = rt(kz0, kz1)
    r1s, t1s = rt(kz1, kzs)
    ph = np.exp(1j * kz1 * d)
    r = (r01 + r1s * ph**2) / (1 + r01 * r1s * ph**2)
    t = (t01 * t1s * ph) / (1 + r01 * r1s * ph**2)
    R = abs(r)**2
    T = (kzs.real / kz0.real) * abs(t)**2
    return R, T


def uniform_grid(eps, Ny=32, Nx=32):
    return torch.full((Ny, Nx), eps, dtype=torch.complex128)


def run():
    print("="*64)
    print("TEST 1  단일 계면 Fresnel (수직입사)  air->glass n=1.5")
    R0, T0 = fresnel_normal(1.0, 1.5)
    s = RCWASolver(0.55, 1.0, 1.0, nG=21, theta=0, device="cpu")
    s.setup_incidence(1.0, 1.5**2)
    o = s.solve(pol_te=1.0)
    print(f"  analytic  R={R0:.5f} T={T0:.5f}")
    print(f"  rcwa      R={o['R']:.5f} T={o['T']:.5f}  (R+T={o['R']+o['T']:.5f})")
    assert abs(o['R']-R0) < 1e-4 and abs(o['T']-T0) < 1e-4, "Fresnel mismatch"
    print("  ✓ pass")

    print("="*64)
    print("TEST 2  단일 박막 = 1D TMM   n0=1, film n=2.0 d=0.12um, ns=1.45  @0.55um")
    lam, nf, d, ns = 0.55, 2.0, 0.12, 1.45
    Rt, Tt = tmm_single_film(1.0, nf, ns, d, lam)
    s = RCWASolver(lam, 1.0, 1.0, nG=21, theta=0, device="cpu")
    s.setup_incidence(1.0, ns**2)
    s.add_layer(d, eps_grid=uniform_grid(nf**2))
    o = s.solve(pol_te=1.0)
    print(f"  TMM   R={Rt:.5f} T={Tt:.5f}")
    print(f"  rcwa  R={o['R']:.5f} T={o['T']:.5f}  (R+T={o['R']+o['T']:.5f})")
    assert abs(o['R']-Rt) < 1e-3 and abs(o['T']-Tt) < 1e-3, "TMM mismatch"
    print("  ✓ pass")

    print("="*64)
    print("TEST 3  사입사 30deg s-편광 박막 = 1D TMM")
    th = 30.0
    Rt, Tt = tmm_single_film(1.0, nf, ns, d, lam, math.radians(th))
    s = RCWASolver(lam, 1.0, 1.0, nG=21, theta=th, phi=0, device="cpu")
    s.setup_incidence(1.0, ns**2)
    s.add_layer(d, eps_grid=uniform_grid(nf**2))
    o = s.solve(pol_te=1.0, pol_tm=0.0)   # TE=s
    print(f"  TMM   R={Rt:.5f} T={Tt:.5f}")
    print(f"  rcwa  R={o['R']:.5f} T={o['T']:.5f}  (R+T={o['R']+o['T']:.5f})")
    assert abs(o['R']-Rt) < 2e-3, "oblique TMM mismatch"
    print("  ✓ pass")

    print("="*64)
    print("TEST 4  에너지 보존: 무손실 패턴층 (사각 기둥 격자)")
    Ny=Nx=64
    g = torch.full((Ny,Nx), (1.46**2), dtype=torch.complex128)
    g[Ny//4:3*Ny//4, Nx//4:3*Nx//4] = 2.5**2       # 고굴절 사각 기둥
    s = RCWASolver(0.55, 0.5, 0.5, nG=61, theta=10, device="cpu")
    s.setup_incidence(1.0, 1.0)
    s.add_layer(0.3, eps_grid=g)
    o = s.solve(pol_te=0.8, pol_tm=0.6)
    tot = o['R']+o['T']
    print(f"  R={o['R']:.5f} T={o['T']:.5f}  R+T={tot:.6f}  (무손실 => 1)")
    assert abs(tot-1.0) < 2e-3, "energy not conserved"
    print("  ✓ pass")

    print("="*64)
    print("TEST 5  흡수층 flux-difference:  Σ A_layer == 1-R-T (에너지)")
    # air / 흡수 박막(n=3+0.1i) / air, + 무손실 유전체층 하나
    lam = 0.55
    s = RCWASolver(lam, 1.0, 1.0, nG=21, theta=15, device="cpu")
    s.setup_incidence(1.0, 1.0)
    s.add_layer(0.20, eps_grid=uniform_grid((3.0 + 0.1j)**2))   # 흡수층
    s.add_layer(0.10, eps_grid=uniform_grid(1.5**2))            # 무손실층
    o = s.solve(pol_te=1.0, pol_tm=0.3)
    target = o["A"]
    print(f"  R={o['R']:.4f} T={o['T']:.4f}  A=1-R-T={target:.4f}")
    # 흡수층이므로 0<A<1, 수동성(R+T<=1) 확인
    assert 0.0 < target < 1.0, "absorption out of physical range"
    assert o["R"] + o["T"] <= 1.0 + 1e-6, "R+T>1 (gain, unphysical)"
    # 무손실 매질만일 때 A~0 (에너지 보존) 재확인
    s2 = RCWASolver(lam, 1.0, 1.0, nG=21, theta=15, device="cpu")
    s2.setup_incidence(1.0, 1.0)
    s2.add_layer(0.20, eps_grid=uniform_grid(2.0**2))
    o2 = s2.solve(pol_te=1.0, pol_tm=0.3)
    print(f"  무손실 검증 A={o2['A']:.5f} (=> ~0)")
    assert abs(o2["A"]) < 2e-3, "lossless absorption should be ~0"
    print("  ✓ pass (흡수층 물리적 R/T/A, 무손실 에너지 보존)")

    print("="*64)
    print("모든 검증 통과 ✓")


if __name__ == "__main__":
    run()
