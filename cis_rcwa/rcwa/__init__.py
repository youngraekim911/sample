"""PyTorch RCWA (FMM + S-matrix) for CIS qcell — GPU 가속 지원.

modules:
    kbloch    : 역격자 / G-truncation / 입사파 K 설정
    fft_funs  : eps 격자 -> Fourier 컨볼루션 행렬 (FFT)
    torch_eig : pytorch 고유값 솔버 래퍼
    rcwa      : RCWASolver (FFT -> eig -> S-matrix -> R/T/QE)
"""
from .rcwa import RCWASolver
from . import kbloch, fft_funs, torch_eig

__all__ = ["RCWASolver", "kbloch", "fft_funs", "torch_eig"]
