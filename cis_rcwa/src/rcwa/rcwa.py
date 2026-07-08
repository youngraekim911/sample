# -*- coding: utf-8 -*-
"""RCWASolver — Fourier Modal Method + 강화 S-matrix (Redheffer) 알고리즘.

파이프라인:  역격자/K 설정 -> 층별 eps FFT 컨볼루션 -> 층 고유모드(eig)
             -> 층 S-matrix -> Redheffer star 로 전체 결합 -> R/T -> Si QE

공식: EMPossible 정규화 PQ 형식 (k0 = 2π/λ 로 정규화).
      P Q 고유분해로 층 모드 (W, V, λ) 를 얻고, gap(자유공간) 기준으로
      각 층 S-matrix 를 조립.  비자성(μ=1) 가정.

특징
----
- 원형/사각형 order truncation (kbloch.get_G)
- 복소굴절률(흡수) 처리
- 비수직 입사 (theta, phi)
- GPU 가속: 모든 텐서가 지정 device/dtype 로 동작 (cuda 자동)
"""
import math
import torch

from . import kbloch, fft_funs
from .torch_eig import eig, sqrt_decaying, sqrt_outgoing


class RCWASolver:
    def __init__(self, wavelength, Lx, Ly, nG=101, theta=0.0, phi=0.0,
                 trunc="circular", device=None, dtype=torch.complex128):
        """
        wavelength : 진공 파장 (um, 격자 Lx,Ly 와 같은 단위)
        Lx, Ly     : 단위셀 주기 (um)
        nG         : Fourier order 개수(목표). 원형 truncation 시 근사 개수.
        theta,phi  : 입사각 (deg)
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.cdt = dtype
        self.rdt = torch.float64 if dtype == torch.complex128 else torch.float32
        self.lam = float(wavelength)
        self.k0 = 2 * math.pi / self.lam
        self.Lx, self.Ly = float(Lx), float(Ly)
        self.theta = math.radians(theta)
        self.phi = math.radians(phi)

        # 역격자 + G-order
        g1, g2 = kbloch.reciprocal([self.Lx, 0.0], [0.0, self.Ly])
        self.m, self.n, Gx, Gy = kbloch.get_G(nG, g1, g2, trunc)
        self.nG = self.m.numel()

        # 매질/입사는 solve 시점에 지정. K 는 입사매질 n_inc 필요 -> setup_incidence.
        self._Gx, self._Gy = Gx, Gy
        self.layers = []          # [(ER, ERinv, thickness)] patterned; None ER => uniform(eps scalar)
        self._built = False

    # -----------------------------------------------------------------
    def _C(self, x):
        return torch.as_tensor(x, dtype=self.cdt, device=self.device)

    def _eye(self, k):
        return torch.eye(k, dtype=self.cdt, device=self.device)

    def setup_incidence(self, eps_inc, eps_trn):
        """입사(위)/투과(아래) 반무한 매질 유전율 지정 후 K 행렬 구성."""
        self.eps_inc = complex(eps_inc)
        self.eps_trn = complex(eps_trn)
        n_inc = (self.eps_inc ** 0.5).real
        kx, ky = kbloch.set_incidence(self.k0, n_inc, self.theta, self.phi,
                                      self._Gx.to(self.device), self._Gy.to(self.device))
        self.kx = kx.to(self.rdt)
        self.ky = ky.to(self.rdt)
        self.Kx = torch.diag(self._C(self.kx))
        self.Ky = torch.diag(self._C(self.ky))
        N = self.nG
        self.I = self._eye(N)
        self.I2 = self._eye(2 * N)

        # gap(자유공간) 기준 매질
        self.W0 = self.I2
        Kz0 = sqrt_outgoing(1.0 - self._C(self.kx) ** 2 - self._C(self.ky) ** 2)
        self.V0 = self._homogeneous_V(1.0, Kz0)
        return self

    # -----------------------------------------------------------------
    def _homogeneous_Q(self, er):
        """균일매질 Q block (2N,2N)."""
        Kx, Ky, I = self.Kx, self.Ky, self.I
        erI = er * I
        return torch.cat([
            torch.cat([Kx @ Ky,        erI - Kx @ Kx], dim=1),
            torch.cat([Ky @ Ky - erI, -Ky @ Kx],       dim=1)], dim=0)

    def _homogeneous_V(self, er, Kz):
        """반무한/gap 기준 매질 V = Q @ inv(Lam), Lam = 1j*[Kz;Kz] (outgoing 분기)."""
        Q = self._homogeneous_Q(er)
        lam = torch.cat([1j * Kz, 1j * Kz])
        return Q @ torch.diag(1.0 / lam)

    # -----------------------------------------------------------------
    def add_layer(self, thickness, eps_grid=None, eps_scalar=None):
        """층 추가.  eps_grid: (Ny,Nx) 패턴 / eps_scalar: 균일층."""
        if eps_scalar is not None:
            self.layers.append(("uniform", complex(eps_scalar), float(thickness)))
        else:
            g = torch.as_tensor(eps_grid, dtype=self.cdt, device=self.device)
            self.layers.append(("patterned", g, float(thickness)))
        return self

    def clear_layers(self):
        self.layers = []

    # -----------------------------------------------------------------
    def _uniform_grid(self, er):
        """균일 유전율 -> conv_matrix 가 정확한(앨리어싱 없는) 최소 상수 격자."""
        gx = int(self.m.abs().max()) * 2 + 2
        gy = int(self.n.abs().max()) * 2 + 2
        return torch.full((gy, gx), complex(er), dtype=self.cdt, device=self.device)

    def _layer_modes(self, kind, data):
        """층 고유모드 (W, V, lam).  lam = sqrt(eig(P@Q)), 감쇠분기."""
        N = self.nG
        Kx, Ky, I = self.Kx, self.Ky, self.I
        if kind == "uniform":
            # 균일층도 patterned 와 동일 경로(eig) 사용 — 해석식 분기(analytic V)가
            # gap 규약과 propagating/evanescent 에서 어긋나 A 가 특이해지는 버그 회피
            data = self._uniform_grid(data)

        ER = fft_funs.conv_matrix(data, self.m, self.n)     # (N,N)
        ERinv = torch.linalg.inv(ER)
        P = torch.cat([
            torch.cat([Kx @ ERinv @ Ky,     I - Kx @ ERinv @ Kx], dim=1),
            torch.cat([Ky @ ERinv @ Ky - I, -Ky @ ERinv @ Kx],    dim=1)], dim=0)
        Q = torch.cat([
            torch.cat([Kx @ Ky,      ER - Kx @ Kx], dim=1),
            torch.cat([Ky @ Ky - ER, -Ky @ Kx],     dim=1)], dim=0)
        OM2 = P @ Q
        lam2, W = eig(OM2)
        lam = sqrt_decaying(lam2)               # Re(lam)>=0, X=exp(-lam k0 L) 안정
        # V 부호: gap/영역(lam=1j*Kz)과 분기 반대 -> 정합 위해 음부호
        V = -(Q @ W @ torch.diag(1.0 / lam))
        return W, V, lam

    def _layer_smatrix(self, W, V, lam, thickness):
        """gap 기준 층 S-matrix (2N block). A = Wᵢ⁻¹W₀ + Vᵢ⁻¹V₀."""
        Winv = torch.linalg.inv(W)
        Vinv = torch.linalg.inv(V)
        A = Winv @ self.W0 + Vinv @ self.V0
        B = Winv @ self.W0 - Vinv @ self.V0
        X = torch.diag(torch.exp(-lam * self.k0 * thickness))
        Ai = torch.linalg.inv(A)
        XB = X @ B
        XA = X @ A
        D = A - XB @ Ai @ XB
        Dinv = torch.linalg.inv(D)
        S11 = Dinv @ (XB @ Ai @ XA - B)
        S12 = Dinv @ X @ (A - B @ Ai @ B)
        return {"11": S11, "12": S12, "21": S12, "22": S11}

    @staticmethod
    def _star(SA, SB):
        """Redheffer star product."""
        I = torch.eye(SA["11"].shape[0], dtype=SA["11"].dtype, device=SA["11"].device)
        D = SA["12"] @ torch.linalg.inv(I - SB["11"] @ SA["22"])
        F = SB["21"] @ torch.linalg.inv(I - SA["22"] @ SB["11"])
        return {
            "11": SA["11"] + D @ SB["11"] @ SA["21"],
            "12": D @ SB["12"],
            "21": F @ SA["21"],
            "22": SB["22"] + F @ SA["22"] @ SB["12"],
        }

    # -----------------------------------------------------------------
    def _region_smatrices(self):
        """반사(위)/투과(아래) 반무한 영역 S-matrix."""
        N = self.nG
        # reflection region (eps_inc)
        Kzr = sqrt_outgoing(self.eps_inc - self._C(self.kx) ** 2 - self._C(self.ky) ** 2)
        Wref = self.I2
        Vref = self._homogeneous_V(self.eps_inc, Kzr)
        V0i = torch.linalg.inv(self.V0)
        Ar = self.I2 + V0i @ Vref                     # W0inv Wref = I
        Br = self.I2 - V0i @ Vref
        Ari = torch.linalg.inv(Ar)
        Sref = {"11": -Ari @ Br, "12": 2 * Ari,
                "21": 0.5 * (Ar - Br @ Ari @ Br), "22": Br @ Ari}

        # transmission region (eps_trn)
        Kzt = sqrt_outgoing(self.eps_trn - self._C(self.kx) ** 2 - self._C(self.ky) ** 2)
        Wtrn = self.I2
        Vtrn = self._homogeneous_V(self.eps_trn, Kzt)
        At = self.I2 + V0i @ Vtrn
        Bt = self.I2 - V0i @ Vtrn
        Ati = torch.linalg.inv(At)
        Strn = {"11": Bt @ Ati, "12": 0.5 * (At - Bt @ Ati @ Bt),
                "21": 2 * Ati, "22": -Ati @ Bt}
        return Sref, Strn, Kzr, Kzt

    # -----------------------------------------------------------------
    def solve(self, pol_te=1.0, pol_tm=0.0):
        """전체 S-matrix 조립 후 R, T, 흡수 계산.

        pol_te, pol_tm : 입사 편광 진폭 (TE=s, TM=p).
        Returns dict: R, T, A, R_orders, T_orders, Kz.
        """
        Sref, Strn, Kzr, Kzt = self._region_smatrices()
        self._modes = []
        self._layerS = []
        self.n_regularized = 0
        S = Sref
        for kind, data, th in self.layers:
            W, V, lam = self._layer_modes(kind, data)
            SL = self._layer_smatrix(W, V, lam, th)
            # 준-균일(quasi-uniform) 패턴층 가드: 소수 픽셀만 다른 층은 OM2 고유값이
            # near-degenerate -> 고유벡터 행렬 병적 조건수 -> S 폭발.
            # 이때 층을 평균 유전율(균일)로 재계산 — 물리 오차는 미소(패턴이 거의 없음).
            bad = (not torch.isfinite(SL["11"]).all()) or (not torch.isfinite(SL["21"]).all()) \
                  or float(SL["11"].abs().amax()) > 1e4 or float(SL["21"].abs().amax()) > 1e4
            if bad and kind == "patterned":
                mean_eps = complex(data.mean())
                W, V, lam = self._layer_modes("uniform", mean_eps)
                SL = self._layer_smatrix(W, V, lam, th)
                data = self._uniform_grid(mean_eps)
                self.n_regularized += 1
            self._modes.append((W, V, lam, th, kind, data))
            self._layerS.append(SL)
            S = self._star(S, SL)
        self._S_pre_trn = S                       # 투과 계면 필드 복원용 (픽셀별 QE)
        S = self._star(S, Strn)
        self.S_global = S
        self._Sref, self._Strn, self._Kzr, self._Kzt = Sref, Strn, Kzr, Kzt

        N = self.nG
        # 입사 편광 벡터 (order 0)
        i0 = self._order0_index()
        kx0 = float(self.kx[i0]); ky0 = float(self.ky[i0])
        kz0 = math.sqrt(max(self.eps_inc.real - kx0**2 - ky0**2, 1e-12))
        # TE: 면(z)과 수직인 s-편광, TM: p-편광
        theta = self.theta
        if abs(theta) < 1e-9:
            ate = torch.tensor([0.0, 1.0, 0.0], dtype=self.rdt)   # y
            atm = torch.tensor([1.0, 0.0, 0.0], dtype=self.rdt)   # x
        else:
            khat = torch.tensor([kx0, ky0, kz0], dtype=self.rdt)
            khat = khat / torch.linalg.norm(khat)
            zhat = torch.tensor([0.0, 0.0, 1.0], dtype=self.rdt)
            ate = torch.linalg.cross(zhat, khat); ate = ate / torch.linalg.norm(ate)
            atm = torch.linalg.cross(ate, khat); atm = atm / torch.linalg.norm(atm)
        P = pol_te * ate + pol_tm * atm
        P = P / torch.linalg.norm(P).clamp_min(1e-12)      # 단위 입사파워 |P|=1

        esrc = torch.zeros(2 * N, dtype=self.cdt, device=self.device)
        esrc[i0] = self._C(P[0])
        esrc[N + i0] = self._C(P[1])
        cinc = esrc                                # Wref = I
        self._cinc = cinc

        rmode = S["11"] @ cinc
        tmode = S["21"] @ cinc
        rx, ry = rmode[:N], rmode[N:]
        tx, ty = tmode[:N], tmode[N:]
        rz = -(self._C(self.kx) * rx + self._C(self.ky) * ry) / Kzr
        tz = -(self._C(self.kx) * tx + self._C(self.ky) * ty) / Kzt

        r2 = rx.abs()**2 + ry.abs()**2 + rz.abs()**2
        t2 = tx.abs()**2 + ty.abs()**2 + tz.abs()**2
        R_ord = (r2 * (Kzr.real / kz0)).real
        T_ord = (t2 * (Kzt.real / kz0)).real
        R = float(R_ord.sum()); T = float(T_ord.sum())
        self._R, self._T, self._kz0 = R, T, kz0
        return {"R": R, "T": T, "A": 1.0 - R - T,
                "R_orders": R_ord.detach().cpu(), "T_orders": T_ord.detach().cpu(),
                "i0": i0, "nG": N}

    def _order0_index(self):
        z = (self.m == 0) & (self.n == 0)
        return int(torch.nonzero(z, as_tuple=False)[0, 0])

    # =================================================================
    # 필드 복원 / 층·픽셀 흡수 (QE)
    # =================================================================
    def _cumulative_S(self):
        """elems = [Sref, L1..LM, Strn] 의 좌/우 누적 S-matrix."""
        elems = [self._Sref] + self._layerS + [self._Strn]
        K = len(elems)
        cumL = [None] * K
        cumR = [None] * K
        cumL[0] = elems[0]
        for k in range(1, K):
            cumL[k] = self._star(cumL[k - 1], elems[k])
        cumR[K - 1] = elems[K - 1]
        for k in range(K - 2, -1, -1):
            cumR[k] = self._star(elems[k], cumR[k + 1])
        return elems, cumL, cumR

    def _node_amps(self, cumL_k, cumR_k1):
        """node(=elems[k]와 elems[k+1] 사이) gap-basis 전/후진 진폭 a,b."""
        a = torch.linalg.solve(self.I2 - cumL_k["22"] @ cumR_k1["11"],
                               cumL_k["21"] @ self._cinc)
        b = cumR_k1["11"] @ a
        return a, b

    def _node_tangential(self, a, b):
        """node 접선 필드 Fourier: E_tan=[sx;sy]=W0(a+b), H_tan=[hx;hy]=V0(a-b)."""
        E = self.W0 @ (a + b)           # W0 = I2
        H = self.V0 @ (a - b)
        return E, H

    @staticmethod
    def _flux_from_tangential(E, H, N):
        """접선 Fourier 로부터 하향 Poynting flux (order 합) = 0.5 Σ Re(Ex Hy* - Ey Hx*).

        V 규약상 H_tan = [Hx; Hy] (아래 calibration 로 부호 확인됨).
        """
        sx, sy = E[:N], E[N:]
        hx, hy = H[:N], H[N:]
        return 0.5 * torch.sum((sx * hy.conj() - sy * hx.conj()).real)

    def absorption_profile(self):
        """층별 흡수 A_i = flux(top_i) - flux(bottom_i).  gap-basis node flux 사용.

        Returns
        -------
        dict: A_layers (list), flux_nodes (list), calib (scale), check (Σ vs 1-R-T)
        """
        N = self.nG
        elems, cumL, cumR = self._cumulative_S()
        K = len(elems)                       # nodes: 0 .. K-2
        node_ab = []
        for k in range(K - 1):
            a, b = self._node_amps(cumL[k], cumR[k + 1])
            node_ab.append((a, b))
        # 각 node flux
        flux = []
        for (a, b) in node_ab:
            E, H = self._node_tangential(a, b)
            flux.append(float(self._flux_from_tangential(E, H, N)))
        # calibration: 최상단 node flux 는 (1-R), 최하단은 T 여야 함.
        # 스케일/부호 보정 (V 정규화 상수 흡수).
        top, bot = flux[0], flux[-1]
        calib = 1.0
        if abs(top) > 1e-12:
            calib = (1.0 - self._R) / top
        flux = [f * calib for f in flux]
        # 층 i: elems index i+1, top node = i, bottom node = i+1
        M = len(self.layers)
        A_layers = [flux[i] - flux[i + 1] for i in range(M)]
        self._node_ab = node_ab
        self._node_flux = flux
        self._flux_calib = calib
        return {"A_layers": A_layers, "flux_nodes": flux, "calib": calib,
                "sum_A": sum(A_layers), "check_1_R_T": 1.0 - self._R - self._T}

    def transmitted_flux_map(self, Ny, Nx):
        """투과(기판) 계면 바로 위 node 의 하향 Poynting flux Sz(x,y) — (Ny,Nx).

        평균값이 정확히 T 가 되도록 calibration (Parseval: 셀 평균 = order 합).
        픽셀별 QE = 픽셀 마스크 영역 flux 적분 / 픽셀 면적 비율.
        """
        N = self.nG
        Spre, Strn = self._S_pre_trn, self._Strn
        a = torch.linalg.solve(self.I2 - Spre["22"] @ Strn["11"],
                               Spre["21"] @ self._cinc)
        b = Strn["11"] @ a
        E = self.W0 @ (a + b)
        H = self.V0 @ (a - b)
        f = float(self._flux_from_tangential(E, H, N))
        calib = (self._T / f) if abs(f) > 1e-300 else 0.0
        Ex = fft_funs.field_ifft(E[:N], self.m, self.n, Ny, Nx)
        Ey = fft_funs.field_ifft(E[N:], self.m, self.n, Ny, Nx)
        Hx = fft_funs.field_ifft(H[:N], self.m, self.n, Ny, Nx)
        Hy = fft_funs.field_ifft(H[N:], self.m, self.n, Ny, Nx)
        Sz = 0.5 * (Ex * Hy.conj() - Ey * Hx.conj()).real * calib
        return Sz

    def node_tangential_realspace(self, node_k, Ny, Nx):
        """node_k 접선 필드를 실공간 (Ny,Nx) 로 ifft.  반환 Ex,Ey,Hx,Hy, Sz(x,y)."""
        a, b = self._node_ab[node_k]
        E, H = self._node_tangential(a, b)
        N = self.nG
        Ex = fft_funs.field_ifft(E[:N], self.m, self.n, Ny, Nx)
        Ey = fft_funs.field_ifft(E[N:], self.m, self.n, Ny, Nx)
        Hx = fft_funs.field_ifft(H[:N] * self._flux_calib, self.m, self.n, Ny, Nx)
        Hy = fft_funs.field_ifft(H[N:] * self._flux_calib, self.m, self.n, Ny, Nx)
        Sz = 0.5 * (Ex * Hy.conj() - Ey * Hx.conj()).real
        return {"Ex": Ex, "Ey": Ey, "Hx": Hx, "Hy": Hy, "Sz": Sz}

    def layer_internal_fields(self, i, zfracs, Ny, Nx):
        """층 i 내부 depth(zfracs∈[0,1]) 에서 E 접선필드 실공간 재구성.

        3D |E|^2 볼륨(Si)용.  안정 분해: cp(top), cm(bottom).
        반환: list of dict(Ex,Ey,|Et|^2)  (zfracs 순)
        """
        W, V, lam, th, kind, data = self._modes[i]
        N = self.nG
        # top(node i) / bottom(node i+1) 접선 필드
        aE_t, aH_t = self._node_tangential(*self._node_ab[i])
        aE_b, aH_b = self._node_tangential(*self._node_ab[i + 1])
        Winv = torch.linalg.inv(W)
        Vinv = torch.linalg.inv(V)
        cp = 0.5 * (Winv @ aE_t + Vinv @ (aH_t * self._flux_calib))
        cm = 0.5 * (Winv @ aE_b - Vinv @ (aH_b * self._flux_calib))
        out = []
        for zf in zfracs:
            zt = zf * th
            ep = torch.exp(-lam * self.k0 * zt)
            em = torch.exp(-lam * self.k0 * (th - zt))
            Et = W @ (ep * cp + em * cm)
            Ex = fft_funs.field_ifft(Et[:N], self.m, self.n, Ny, Nx)
            Ey = fft_funs.field_ifft(Et[N:], self.m, self.n, Ny, Nx)
            out.append({"Ex": Ex, "Ey": Ey, "E2": (Ex.abs()**2 + Ey.abs()**2)})
        return out
