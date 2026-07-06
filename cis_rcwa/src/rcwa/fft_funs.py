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


def field_ifft(coeffs, m, n, Ny, Nx):
    """Fourier order 계수 (N,) -> 실공간 (Ny,Nx) 필드 (검증/시각화용)."""
    F = torch.zeros((Ny, Nx), dtype=coeffs.dtype, device=coeffs.device)
    F[n % Ny, m % Nx] = coeffs
    return torch.fft.ifft2(F) * (Nx * Ny)
