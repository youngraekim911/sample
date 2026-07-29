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
                 trunc="circular", device=None, dtype=torch.complex128, fff=False):
        """
        wavelength : 진공 파장 (um, 격자 Lx,Ly 와 같은 단위)
        Lx, Ly     : 단위셀 주기 (um)
        nG         : Fourier order 개수(목표). 원형 truncation 시 근사 개수.
        theta,phi  : 입사각 (deg)
        fff        : Li 인수분해(inverse rule) — 고대비(금속) 층 수렴 가속
        """
        self.fff = bool(fff)
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
        """반무한/gap 기준 매질 V = Q @ inv(Lam), Lam = 1j*[Kz;Kz] (outgoing 분기).

        Wood anomaly(회절 차수가 정확히 kz=0) 가드: |lam| 하한 클램프로 0-나눗셈 방지
        (해당 λ 는 simulator 가 미세 이동 재계산으로 처리 — 이건 NaN 전파 방지용)."""
        Q = self._homogeneous_Q(er)
        lam = torch.cat([1j * Kz, 1j * Kz])
        small = lam.abs() < 1e-10
        lam = torch.where(small, lam + 1e-10, lam)
        return Q @ torch.diag(1.0 / lam)

    # -----------------------------------------------------------------
    def add_layer(self, thickness, eps_grid=None, eps_scalar=None):
        """층 추가.  eps_grid: (Ny,Nx) 패턴 / eps_scalar: 균일층."""
        if eps_scalar is not None:
            self.layers.append(("uniform", complex(eps_scalar), float(thickness)))
        else:
            g = torch.as_tensor(eps_grid, dtype=self.cdt, device=self.device)
            self.layers.append(("patterned", g, float(thickness)))
        self._assembled = False
        return self

    def clear_layers(self):
        self.layers = []
        self._assembled = False

    # -----------------------------------------------------------------
    def _uniform_grid(self, er):
        """균일 유전율 -> conv_matrix 가 정확한(앨리어싱 없는) 최소 상수 격자."""
        gx = int(self.m.abs().max()) * 2 + 2
        gy = int(self.n.abs().max()) * 2 + 2
        return torch.full((gy, gx), complex(er), dtype=self.cdt, device=self.device)

    def _layer_modes(self, kind, data):
        """층 고유모드 (W, V, lam).  patterned: lam=sqrt(eig(P@Q)) 감쇠분기 /
        uniform: 해석식 q (eig 생략 — 수 배 빠름).

        해석식 균일층: lam = sqrt_decaying(kt²−er), W=I, V=−Q·diag(1/lam).
        sqrt_decaying 의 전파모드 tie-break(Im<0) 수정 후 gap V0(outgoing)와
        전 모드에서 forward 짝이 일치 -> 과거의 A 특이(층≈gap 크래시) 문제 없음.
        (검증: patterned 경로와 R/T 일치, validate.py 5종 통과)"""
        N = self.nG
        Kx, Ky, I = self.Kx, self.Ky, self.I
        if kind == "uniform":
            er = complex(data)
            lam1 = sqrt_decaying(self._C(self.kx) ** 2 + self._C(self.ky) ** 2 - er)
            lam = torch.cat([lam1, lam1])
            W = self.I2
            V = -(self._homogeneous_Q(er) @ torch.diag(1.0 / lam))
            return W, V, lam

        OM2, Q = self._patterned_OM2_Q(data)
        lam2, W = eig(OM2)
        lam = sqrt_decaying(lam2)               # Re(lam)>=0, X=exp(-lam k0 L) 안정
        # V 부호: gap/영역(lam=1j*Kz)과 분기 반대 -> 정합 위해 음부호
        V = -(Q @ W @ torch.diag(1.0 / lam))
        return W, V, lam

    def _patterned_OM2_Q(self, data):
        """패턴층의 OM2=P@Q 와 Q 를 만든다 (eig 입력). eig 는 층 독립이라 solve()
        에서 여러 층 OM2 를 배치로 묶어 한 번에 분해(GPU 병렬)할 때 재사용."""
        Kx, Ky, I = self.Kx, self.Ky, self.I
        if getattr(self, "fff", False):
            # Li 인수분해(FFF): Q 의 eps·Ey 항엔 y-inverse, eps·Ex 항엔 x-inverse
            ER, EX, EY = fft_funs.conv_matrix_fff(data, self.m, self.n)
            ERinv = torch.linalg.inv(ER)
            E_ey, E_ex = EY, EX
        else:
            ER = fft_funs.conv_matrix(data, self.m, self.n)  # (N,N) Laurent
            ERinv = torch.linalg.inv(ER)
            E_ey = E_ex = ER
        P = torch.cat([
            torch.cat([Kx @ ERinv @ Ky,     I - Kx @ ERinv @ Kx], dim=1),
            torch.cat([Ky @ ERinv @ Ky - I, -Ky @ ERinv @ Kx],    dim=1)], dim=0)
        Q = torch.cat([
            torch.cat([Kx @ Ky,        E_ey - Kx @ Kx], dim=1),
            torch.cat([Ky @ Ky - E_ex, -Ky @ Kx],       dim=1)], dim=0)
        return P @ Q, Q

    def _all_layer_modes(self):
        """전 층의 (W,V,lam) 를 계산. **patterned 층 eig 는 배치로 묶어 한 번에**
        분해 -> GPU 가 층들을 병렬 처리(순차 eig N회 -> 배치 eig 1회). uniform 층은
        해석식(즉시). S-matrix 사슬(순차)과 분리해 독립적인 eig 만 병렬화한다."""
        modes = [None] * len(self.layers)
        om2_list, q_list, idx_list = [], [], []
        for i, (kind, data, th) in enumerate(self.layers):
            if kind == "uniform":
                modes[i] = self._layer_modes("uniform", data)
            else:
                om2, Q = self._patterned_OM2_Q(data)
                om2_list.append(om2); q_list.append(Q); idx_list.append(i)
        if om2_list:
            lam2b, Wb = eig(torch.stack(om2_list))       # 배치 eig [P,2N,2N]
            for j, i in enumerate(idx_list):
                lam = sqrt_decaying(lam2b[j])
                V = -(q_list[j] @ Wb[j] @ torch.diag(1.0 / lam))
                modes[i] = (Wb[j], V, lam)
        return modes

    def _layer_smatrix(self, W, V, lam, thickness):
        """gap 기준 층 S-matrix (2N block). A = Wᵢ⁻¹W₀ + Vᵢ⁻¹V₀.

        최적화: W0=I2 이므로 Winv@W0=Winv (matmul 생략), uniform 층(W=I2)은 inv(W)
        생략. 명시적 역행렬 대신 LU 재사용 solve — A⁻¹·[XB|XA|B], D⁻¹·[·|·] 를
        각각 한 번의 인수분해로 처리 (역행렬 형성 회피 → 더 빠르고 안정)."""
        Winv = self.I2 if W is self.I2 else torch.linalg.inv(W)   # W0=I2 → Winv@W0=Winv
        ViV0 = torch.linalg.solve(V, self.V0)                     # V⁻¹V0 (inv 안 만듦)
        A = Winv + ViV0
        B = Winv - ViV0
        X = torch.diag(torch.exp(-lam * self.k0 * thickness))
        XB = X @ B
        XA = X @ A
        n2 = A.shape[0]
        # A 인수분해 1회로 A⁻¹·XB, A⁻¹·XA, A⁻¹·B 동시 (LU 재사용)
        AiN = torch.linalg.solve(A, torch.cat([XB, XA, B], dim=1))
        Ai_XB, Ai_XA, Ai_B = AiN[:, :n2], AiN[:, n2:2 * n2], AiN[:, 2 * n2:]
        D = A - XB @ Ai_XB
        rhs = torch.cat([XB @ Ai_XA - B, X @ (A - B @ Ai_B)], dim=1)
        DiN = torch.linalg.solve(D, rhs)                          # D⁻¹·[S11rhs|S12rhs]
        S11, S12 = DiN[:, :n2], DiN[:, n2:]
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
    def _assemble(self):
        """편광 무관 부분 전체 — 층 eig + S-matrix 사슬 조립 (한 번만).

        TE/TM 은 입사 진폭만 다르고 S-matrix 는 동일하므로, 여기서 만든 결과를
        solve() 가 편광마다 재사용한다 (비편광 QE 런에서 조립 비용 절반).
        """
        if getattr(self, "_assembled", False):
            return
        Sref, Strn, Kzr, Kzt = self._region_smatrices()
        self._modes = []
        self._layerS = []
        self.n_regularized = 0
        S = Sref
        all_modes = self._all_layer_modes()         # patterned eig 배치(GPU 병렬) 후 순차 조립
        for (kind, data, th), (W, V, lam) in zip(self.layers, all_modes):
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
        self.S_global = self._star(S, Strn)
        self._Sref, self._Strn, self._Kzr, self._Kzt = Sref, Strn, Kzr, Kzt
        self._cumS = None                         # 누적 S 캐시 (lazy, 편광 무관)
        self._ERconv = {}                         # 층별 (ER, ERinv) 캐시 (흡수 재구성용)
        self._winspec = {}                        # 창 스펙트럼 캐시 (편광 무관)
        self._assembled = True

    def solve(self, pol_te=1.0, pol_tm=0.0):
        """전체 S-matrix 조립(캐시) 후 R, T, 흡수 계산.

        pol_te, pol_tm : 입사 편광 진폭 (TE=s, TM=p).
        Returns dict: R, T, A, R_orders, T_orders, Kz.
        """
        self._assemble()
        S = self.S_global
        Kzr, Kzt = self._Kzr, self._Kzt

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
        """elems = [Sref, L1..LM, Strn] 의 좌/우 누적 S-matrix (편광 무관 -> 캐시)."""
        if getattr(self, "_cumS", None) is not None:
            return self._cumS
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
        self._cumS = (elems, cumL, cumR)
        return self._cumS

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

    def node_flux_map_raw(self, k, Ny, Nx):
        """node k 의 하향 net Poynting Sz(x,y) — '비교정' raw (진단용).

        절대 스케일은 신뢰 불가(gap-basis) — 호출측에서 에너지 보존으로 얻은
        그 경계의 총 투과 T_after 에 평균을 맞춰 재스케일해 사용.
        (absorption_maps 호출 후 사용 가능)"""
        N = self.nG
        a, b = self._diag_node_ab[k]
        E = a + b
        H = self.V0 @ (a - b)
        Ex = fft_funs.field_ifft(E[:N], self.m, self.n, Ny, Nx)
        Ey = fft_funs.field_ifft(E[N:], self.m, self.n, Ny, Nx)
        Hx = fft_funs.field_ifft(H[:N], self.m, self.n, Ny, Nx)
        Hy = fft_funs.field_ifft(H[N:], self.m, self.n, Ny, Nx)
        return 0.5 * (Ex * Hy.conj() - Ey * Hx.conj()).real

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

    def absorption_maps(self, Ny, Nx, nz_per_um=48, min_nz=4):
        """모든 '손실 층'의 z-적분 흡수밀도 raw 맵 + 전역 에너지 정합 상수 C.

        층 i 흡수 A_i = C · raw_i.sum(),  Σ_i A_i = 1−R−T (정확한 에너지 보존으로
        절대 스케일 고정 — gap-basis node flux 공식(WIP)에 의존하지 않음).

        층 내부 E 복원은 위/아래 node 의 '접선 E 만' 사용:
            cp_j = (ut_j − X_j·ub_j)/(1−X_j²),  cm_j = (ub_j − X_j·ut_j)/(1−X_j²)
        (X=exp(−lam·k0·th) 감쇠 대각 -> 두꺼운 층도 안정, H/flux 규약 비의존)
        Ez 는 Ampere 법칙 ez = ERinv(Kx·hy − Ky·hx) 로 포함 (크기만 사용).

        Returns: (maps: {layer_i: (Ny,Nx) float64 raw}, C: float)
        """
        elems, cumL, cumR = self._cumulative_S()
        node_ab = [self._node_amps(cumL[k], cumR[k + 1])
                   for k in range(len(elems) - 1)]
        self._diag_node_ab = node_ab                  # 경계 flux 맵(진단)용 캐시
        aE = [a + b for a, b in node_ab]              # W0 = I2 -> node 접선 E
        maps = {}
        total = 0.0
        for i in range(len(self._modes)):
            r = self._layer_abs_slices(i, aE, Ny, Nx, nz_per_um, min_nz)
            if r is None:
                continue                              # 무손실층: 흡수 없음
            _zc, stack = r
            dens = stack.sum(0)
            maps[i] = dens
            total += float(dens.sum())
        C = max(0.0, 1.0 - self._R - self._T) / total if total > 1e-300 else 0.0   # 음수 방지(유니터리티 잔차)
        return maps, C

    def _layer_abs_slices(self, i, aE, Ny, Nx, nz_per_um=48, min_nz=4):
        """층 i 를 nz 개 z-슬라이스로 나눈 흡수밀도 raw 맵 반환.
        (모드/필드는 이미 있으므로 추가 eig 없음 — 깊이분해가 사실상 무료)
        무손실층은 None. 반환: (z_centers[층상단 기준 µm 리스트], dens(nz,Ny,Nx)).

        z-슬라이스는 배치 GEMM + 배치 ifft2 로 한꺼번에 (루프 대비 수 배 빠름,
        수치 동일). 메모리는 z-청크로 제한.
        """
        N = self.nG
        W, V, lam, th, kind, data = self._modes[i]
        if kind == "uniform" or (hasattr(data, "dim") and data.dim() == 0):
            data = self._uniform_grid(data)
        if float(data.imag.abs().max()) < 1e-12:
            return None
        cache = getattr(self, "_ERconv", None)
        if cache is not None and i in cache:               # 편광 간 재사용
            ERinv = cache[i]
        else:
            ERinv = torch.linalg.inv(fft_funs.conv_matrix(data, self.m, self.n))
            if cache is not None:
                cache[i] = ERinv
        eps_xy = data
        if eps_xy.shape != (Ny, Nx):
            ii = (torch.arange(Ny, device=data.device) * data.shape[0] // Ny)
            jj = (torch.arange(Nx, device=data.device) * data.shape[1] // Nx)
            eps_xy = data[ii][:, jj]
        imeps = eps_xy.imag.to(torch.float64)
        Winv = torch.linalg.inv(W)
        ut = Winv @ aE[i]
        ub = Winv @ aE[i + 1]
        X = torch.exp(-lam * self.k0 * th)
        den = 1.0 - X * X
        den = torch.where(den.abs() < 1e-12, den + 1e-12, den)   # FP 공진 가드
        cp = (ut - X * ub) / den
        cm = (ub - X * ut) / den
        nz = max(min_nz, int(round(th * nz_per_um)))
        zf = (torch.arange(nz, dtype=self.rdt, device=W.device) + 0.5) / nz
        dens = torch.empty((nz, Ny, Nx), dtype=torch.float64, device=W.device)
        chunk = max(1, int(4e6 // (Ny * Nx)))              # 복소 중간체 메모리 상한
        for s in range(0, nz, chunk):
            zc_f = zf[s:s + chunk]                         # (B,)
            ep = torch.exp(-lam[None, :] * (self.k0 * th) * zc_f[:, None])        # (B,2N)
            em = torch.exp(-lam[None, :] * (self.k0 * th) * (1 - zc_f)[:, None])
            Et = (ep * cp[None, :] + em * cm[None, :]) @ W.T                      # (B,2N)
            Ht = (ep * cp[None, :] - em * cm[None, :]) @ V.T
            ez = (Ht[:, N:] @ self.Kx.T - Ht[:, :N] @ self.Ky.T) @ ERinv.T        # (B,N)
            Ex = fft_funs.field_ifft_batch(Et[:, :N], self.m, self.n, Ny, Nx)
            Ey = fft_funs.field_ifft_batch(Et[:, N:], self.m, self.n, Ny, Nx)
            Ez = fft_funs.field_ifft_batch(ez, self.m, self.n, Ny, Nx)
            dens[s:s + chunk] = (Ex.abs()**2 + Ey.abs()**2 + Ez.abs()**2) \
                * imeps[None, :, :] * (th / nz)
        return [float(z) * th for z in zf], dens

    # ---------------- 검출 QE 고속 경로: order-공간 창 적분 ----------------
    def _dg_index(self):
        """order 차분 (mᵢ−mⱼ, nᵢ−nⱼ) 의 평면 인덱스 (N²,) 와 dg 격자 크기.

        |E|²(x)=Σᵢⱼ cᵢcⱼ* e^{i(gᵢ−gⱼ)x} 는 차분격자(dg) 위의 삼각다항식 —
        임의 창 w 에 대한 Σₓ w·|E|² 를 dg 계수 × 창 스펙트럼 내적으로 정확 계산.
        """
        if getattr(self, "_dgidx", None) is None:
            mx = int(self.m.abs().max()); my = int(self.n.abs().max())
            Sx, Sy = 4 * mx + 1, 4 * my + 1
            DM = (self.m[:, None] - self.m[None, :]) + 2 * mx
            DN = (self.n[:, None] - self.n[None, :]) + 2 * my
            self._dgidx = ((DN * Sx + DM).reshape(-1).to(torch.long),
                           (Sy, Sx), (mx, my))
        return self._dgidx

    def _window_spectra(self, wmaps, imeps_xy):
        """창들 × Imε(x,y) 의 fft2 를 dg 격자점에서 샘플 — (nw, ndg) complex.

        Σₓ w(x)Imε(x)|E(x)|² = Σ_dg A[dg]·conj(FFT2[w·Imε][dg])  (실수부).
        |E|² 대역폭(2·max차수) ≪ 격자수라 앨리어싱 없음 = 실공간 합과 동일값.
        """
        _, (Sy, Sx), (mx, my) = self._dg_index()
        Ny, Nx = imeps_xy.shape
        dn = (torch.arange(-2 * my, 2 * my + 1, device=imeps_xy.device) % Ny)
        dm = (torch.arange(-2 * mx, 2 * mx + 1, device=imeps_xy.device) % Nx)
        out = []
        for w in wmaps:
            F = torch.fft.fft2((w * imeps_xy).to(self.cdt))
            out.append(F[dn[:, None], dm[None, :]].reshape(-1))
        return torch.stack(out)                              # (nw, ndg)

    def _layer_abs_zcoeffs(self, i, aE, nz_per_um=48, min_nz=4):
        """층 i z-슬라이스별 |E|²(성분합·Ez 포함) 의 dg 푸리에 계수 (nz, ndg).

        실공간 iFFT 없이 c·c* 자기상관만 — _layer_abs_slices 와 수치 동일한
        적분을 훨씬 싸게 제공(픽셀 창 적분 전용). 무손실층은 None.
        """
        N = self.nG
        W, V, lam, th, kind, data = self._modes[i]
        if kind == "uniform" or (hasattr(data, "dim") and data.dim() == 0):
            data = self._uniform_grid(data)
        if float(data.imag.abs().max()) < 1e-12:
            return None
        cache = getattr(self, "_ERconv", None)
        if cache is not None and i in cache:
            ERinv = cache[i]
        else:
            ERinv = torch.linalg.inv(fft_funs.conv_matrix(data, self.m, self.n))
            if cache is not None:
                cache[i] = ERinv
        Winv = torch.linalg.inv(W)
        ut = Winv @ aE[i]
        ub = Winv @ aE[i + 1]
        X = torch.exp(-lam * self.k0 * th)
        den = 1.0 - X * X
        den = torch.where(den.abs() < 1e-12, den + 1e-12, den)
        cp = (ut - X * ub) / den
        cm = (ub - X * ut) / den
        nz = max(min_nz, int(round(th * nz_per_um)))
        zf = (torch.arange(nz, dtype=self.rdt, device=W.device) + 0.5) / nz
        ep = torch.exp(-lam[None, :] * (self.k0 * th) * zf[:, None])         # (nz,2N)
        em = torch.exp(-lam[None, :] * (self.k0 * th) * (1 - zf)[:, None])
        Et = (ep * cp[None, :] + em * cm[None, :]) @ W.T                     # (nz,2N)
        Ht = (ep * cp[None, :] - em * cm[None, :]) @ V.T
        ez = (Ht[:, N:] @ self.Kx.T - Ht[:, :N] @ self.Ky.T) @ ERinv.T       # (nz,N)
        idxflat, (Sy, Sx), _ = self._dg_index()
        idxflat = idxflat.to(W.device)
        A = torch.zeros((nz, Sy * Sx), dtype=self.cdt, device=W.device)
        for c in (Et[:, :N], Et[:, N:], ez):
            O = torch.einsum("zi,zj->zij", c, c.conj()).reshape(nz, -1)
            A.index_add_(1, idxflat, O)
        return [float(z) * th for z in zf], A, th, nz, data

    def band_window_absorption(self, windows, band_layers, nz_per_um=48, min_nz=4):
        """검출 QE 전용 고속 창 적분 — 실공간 재구성 없이 정확 계산.

        windows : list[(Ny,Nx) float64 tensor]  (예: [전체=1, 트렌치, 픽셀창들])
        반환: ({li: (z_centers, WA (nz,nw) — Σₓ wᵢ·Imε·|E|²·th/nz)}, C)
        C 는 전 손실층 합 = 1−R−T 로 맞추는 전역 정합 상수 (기존과 동일 정의).
        """
        elems, cumL, cumR = self._cumulative_S()
        node_ab = [self._node_amps(cumL[k], cumR[k + 1])
                   for k in range(len(elems) - 1)]
        self._diag_node_ab = node_ab
        aE = [a + b for a, b in node_ab]
        band = set(band_layers)
        wcache = getattr(self, "_winspec", None)
        if wcache is None:
            wcache = self._winspec = {}                  # 편광 간 재사용 (λ 종속 aE 무관)
        out = {}
        total = 0.0
        for i in range(len(self._modes)):
            r = self._layer_abs_zcoeffs(i, aE, nz_per_um, min_nz)
            if r is None:
                continue
            zc, A, th, nz, data = r
            nw = len(windows) if i in band else 1        # 비밴드 손실층: 전체합만
            key = (i, nw)
            if key not in wcache:
                Ny, Nx = windows[0].shape
                eps_xy = data
                if eps_xy.shape != (Ny, Nx):
                    ii = (torch.arange(Ny, device=data.device) * data.shape[0] // Ny)
                    jj = (torch.arange(Nx, device=data.device) * data.shape[1] // Nx)
                    eps_xy = data[ii][:, jj]
                imeps = eps_xy.imag.to(torch.float64)
                wcache[key] = self._window_spectra(windows[:nw], imeps)
            WA = (A @ wcache[key].conj().T).real * (th / nz)   # (nz, nw)
            total += float(WA[:, 0].sum())
            if i in band:
                out[i] = (zc, WA)
        C = max(0.0, 1.0 - self._R - self._T) / total if total > 1e-300 else 0.0
        return out, C

    def absorption_maps_zresolved(self, Ny, Nx, band_layers, nz_per_um=48, min_nz=4):
        """band_layers(층 인덱스 집합)만 z-분해 흡수를 반환. C 는 전역 정합 상수.
        Returns: (zres: {i: (z_centers_um[list], dens(nz,Ny,Nx))}, C)."""
        elems, cumL, cumR = self._cumulative_S()
        node_ab = [self._node_amps(cumL[k], cumR[k + 1])
                   for k in range(len(elems) - 1)]
        self._diag_node_ab = node_ab
        aE = [a + b for a, b in node_ab]
        band = set(band_layers)
        zres = {}
        total = 0.0
        for i in range(len(self._modes)):
            r = self._layer_abs_slices(i, aE, Ny, Nx, nz_per_um, min_nz)
            if r is None:
                continue
            zc, stack = r
            total += float(stack.sum())
            if i in band:
                zres[i] = (zc, stack)
        C = max(0.0, 1.0 - self._R - self._T) / total if total > 1e-300 else 0.0   # 음수 방지(유니터리티 잔차)
        return zres, C

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
