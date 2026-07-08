# -*- coding: utf-8 -*-
"""FFT 기반 Fourier 변환: eps(x,y) 격자 -> Fourier 컨볼루션 행렬.

FMM 규약
--------
eps(x,y) = Σ_{m,n} c_{m,n} exp(+i (m g1 + n g2)·r)
torch.fft.fft2(eps)[a,b] = Σ eps[y,x] exp(-2πi (a·y/Ny + b·x/Nx))
따라서 c_{m,n} = fft2(eps)[n mod Ny, m mod Nx] / (Nx*Ny)   (m: x-order, n: y-order)

컨볼루션 행렬 C[i,j] = c_{ (m_i-m_j), (n_i-n_j) }
"""
import torch


def conv_matrix(eps_grid, m, n):
    """eps 2D 격자 -> (N,N) Fourier 컨볼루션 행렬.

    Parameters
    ----------
    eps_grid : (Ny, Nx) complex tensor
    m, n     : (N,) long   G-order 정수 (x=m, y=n)
    """
    Ny, Nx = eps_grid.shape
    F = torch.fft.fft2(eps_grid) / (Nx * Ny)     # c_{n,m}
    dm = (m[:, None] - m[None, :]) % Nx          # (N,N)
    dn = (n[:, None] - n[None, :]) % Ny
    return F[dn, dm]


def conv_matrix_inv_rule(eps_grid, m, n):
    """1/eps 의 컨볼루션 (Li inverse rule 의 1D 근사용). 반환: <<1/eps>> 의 역.

    고대비(금속) 층 수렴 개선용 옵션. 기본 솔버는 Laurent rule(conv_matrix) 사용.
    """
    inv = conv_matrix(1.0 / eps_grid, m, n)
    return torch.linalg.inv(inv)


def conv_matrix_fff(eps_grid, m, n):
    """Li 인수분해(FFF, crossed grating) 컨볼루션 행렬 3종: (E, Ex_inv, Ey_inv).

    E      : Laurent rule (기존)
    Ex_inv : x 방향 inverse rule + y 방향 Laurent  — x 경계에서 불연속인 Ex 항용
    Ey_inv : y 방향 inverse rule + x 방향 Laurent  — y 경계용
    고대비(금속) 격자의 TM 수렴을 크게 개선 (Li 1997, Lalanne).
    구현: 행/열별 1D Toeplitz(1/eps) 역행렬 -> 직교방향 FFT -> G-order 인덱싱.
    """
    mx = int(m.abs().max()); my = int(n.abs().max())
    # 격자가 차수 범위(2·max+1)보다 작으면 최근접 반복 업샘플 (계단형 격자에 정확)
    ry, rx = 2 * my + 2, 2 * mx + 2
    if eps_grid.shape[0] < ry:
        eps_grid = eps_grid.repeat_interleave(-(-ry // eps_grid.shape[0]), dim=0)
    if eps_grid.shape[1] < rx:
        eps_grid = eps_grid.repeat_interleave(-(-rx // eps_grid.shape[1]), dim=1)
    Ny, Nx = eps_grid.shape
    E = conv_matrix(eps_grid, m, n)
    dev = eps_grid.device
    inv_eps = 1.0 / eps_grid

    def _half(dim_len, half, F1, along):
        # F1: 1D FFT(1/eps) along 'along' 축 / half: 그 축 최대 차수
        rng = torch.arange(-half, half + 1, device=dev)
        idx = (rng[:, None] - rng[None, :]) % dim_len
        if along == "x":                      # (Ny, r, r) 행별 Toeplitz
            T = F1[:, idx]
        else:                                 # (Nx, r, r) 열별 Toeplitz
            T = F1.transpose(0, 1)[:, idx]
        return torch.linalg.inv(T)            # 행/열별 <<1/eps>>^-1

    # Ex_inv: x inverse rule -> y Laurent
    F1x = torch.fft.fft(inv_eps, dim=1) / Nx
    Einvx = _half(Nx, mx, F1x, "x")           # (Ny, rx, rx)
    Gx = torch.fft.fft(Einvx, dim=0) / Ny     # y 방향 Laurent 계수 (Ny, rx, rx)
    mi = (m + mx).long()
    dn = (n[:, None] - n[None, :]) % Ny
    Ex_inv = Gx[dn, mi[:, None].expand_as(dn), mi[None, :].expand_as(dn)]

    # Ey_inv: y inverse rule -> x Laurent
    F1y = torch.fft.fft(inv_eps, dim=0) / Ny
    Einvy = _half(Ny, my, F1y, "y")           # (Nx, ry, ry)
    Gy = torch.fft.fft(Einvy, dim=0) / Nx     # x 방향 Laurent (Nx, ry, ry)
    ni = (n + my).long()
    dm = (m[:, None] - m[None, :]) % Nx
    Ey_inv = Gy[dm, ni[:, None].expand_as(dm), ni[None, :].expand_as(dm)]
    return E, Ex_inv, Ey_inv


def field_ifft(coeffs, m, n, Ny, Nx):
    """Fourier order 계수 (N,) -> 실공간 (Ny,Nx) 필드 (검증/시각화용)."""
    F = torch.zeros((Ny, Nx), dtype=coeffs.dtype, device=coeffs.device)
    F[n % Ny, m % Nx] = coeffs
    return torch.fft.ifft2(F) * (Nx * Ny)
