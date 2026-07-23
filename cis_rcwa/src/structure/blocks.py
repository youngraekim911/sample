# -*- coding: utf-8 -*-
"""blocks — 구조의 블록화/일반화. 각 물리 블록이 독립 부품이고, 조합해서 IR 을 만든다.

블록 (아래->위 조립 순서):
    SiDtiBlock         Si 기판 + DTI (none/1x1/2x2_open 클로버, liner '스택', fill)
    BarlBlock          blanket 다층 AR — 층 수 임의 (gradual n 탐색용)
    GridCfBlock        grid 울타리(stack+표면코팅) 안에 CF 채움 (+reflow 응집 곡면)
    PlanarBlock        ML 평탄층 — CF/grid 위를 지정 두께로 전부 채움
    MlBlock            ML 렌즈 (1x1/1x2/2x1/2x2 × quad 4개, 두께/배율 -> 곡률 자동)
    ConformalCoatBlock ML 표면을 덮는 conformal 코팅(ARL) — 여러 겹 가능

사용:
    ctx = BlockContext(pitch_um=1.0, n_pixels=2, lateral_n=200,
                       bayer=[["R","G"],["G","B"]])
    ir  = BlockStack([SiDtiBlock(...), BarlBlock(...), GridCfBlock(...),
                      PlanarBlock(...), MlBlock(...), ConformalCoatBlock(...)],
                     ambient="air", materials={...}).to_ir(ctx)

각 블록은 build(ctx)->[(map2d,두께)](아래->위) 만 구현하면 되고, 블록 간 결합은
ctx 의 공표 필드(트렌치 마스크, CF 상면, 돔 sag 등)로만 이뤄진다.
구조가 바뀌면 블록 목록만 바꾸면 된다 — RCWA/QE 쪽은 IR 계약이라 그대로.
"""
import numpy as np

from .ir import StructureIR, Detector


# ==========================================================================
class BlockContext:
    """가로 격자 + 물질 id 할당 + 블록 간 공유 상태."""

    def __init__(self, pitch_um, n_pixels, lateral_n, bayer=None, cf_array=None):
        self.p = float(pitch_um)
        self.npx = int(n_pixels)
        self.span = self.p * self.npx
        n = int(lateral_n)
        self.n = n
        xc = (np.arange(n) + 0.5) * (self.span / n)
        self.X, self.Y = np.meshgrid(xc, xc)
        self.bayer = bayer
        self.cf_array = cf_array
        self.names = ["air"]
        self._idx = {"air": 0}
        # ---- 블록 간 공표 필드 ----
        self.trench = np.zeros_like(self.X, dtype=bool)   # SiDti -> detector 제외
        self.det_band_um = 0.0                            # SiDti -> detector 밴드
        self.det_n_layers = 0
        self.det_below = 0                                # 밴드 아래 비검출 층 수(후면 반사경)
        self.cf_top_min = 0.0                             # GridCf -> Planar
        self.band_top = 0.0                               # GridCf -> Planar
        self.dome_sag = None                              # Ml -> Coat/flush
        self.dome_mat = None
        self.coats = []                                   # ConformalCoat 누적

    def mat_id(self, name):
        if name not in self._idx:
            self._idx[name] = len(self.names)
            self.names.append(name)
        return self._idx[name]

    def zeros(self, name="air"):
        return np.full(self.X.shape, self.mat_id(name), dtype=np.uint8)


# ==========================================================================
class SiDtiBlock:
    """Si 기판 + DTI.

    dti=None            : DTI 없음 (통 Si)
    mode "1x1"          : 픽셀마다 격벽 -> 2x2 가 1x1 로 다 나뉨
    mode "2x2_open"     : intra-quad 팔이 center gap 에서 끊김 -> 네잎 클로버 Si
    liners              : [{material, thickness_um}, ...] 식각벽에서 안쪽으로
                          겹겹이 (격벽 좌우 각각). 남는 가운데는 fill 로 채움.
    """

    def __init__(self, material, thickness_um, dti=None, back_reflector=None):
        self.mat = material
        self.th = float(thickness_um)
        self.dti = dti
        self.back_reflector = back_reflector          # Si 하부 metal routing 반사경

    def _reflector_layer(self, ctx):
        """후면 반사경 층 (Si 밴드 아래). Cu 가 면적비율 coverage 만 덮고 나머지는
        filler(=Si)라 심부로 투과(손실). coverage=1 이면 solid 거울. 반환 (map, th)."""
        br = self.back_reflector or {}
        cov = min(max(float(br.get("coverage", 1.0)), 0.0), 1.0)
        th = float(br.get("thickness_um", 0.15))       # Cu 두께 (>~0.1µm 이면 불투명)
        metal = br.get("material", "cu")
        rp = float(br.get("routing_pitch_um", 0) or ctx.p)   # 라우팅 피치(기본 픽셀피치)
        out = ctx.zeros(self.mat)                      # 갭 = Si (심부로 투과)
        if cov >= 0.999:
            out[:] = ctx.mat_id(metal)
        elif cov > 0:
            side = float(np.sqrt(cov))                 # 정사각 Cu 패치 변비율 (면적=cov)
            fx = (ctx.X / rp) % 1.0
            fy = (ctx.Y / rp) % 1.0
            out[(fx < side) & (fy < side)] = ctx.mat_id(metal)
        return (out, th)

    def _liners(self):
        d = self.dti or {}
        if d.get("liners"):
            return [(l["material"], float(l["thickness_um"])) for l in d["liners"]]
        ol = float(d.get("oxide_liner_um", 0) or 0)       # 구 스키마 호환 (단일 liner)
        return [(d.get("liner", "oxide"), ol)] if ol > 0 else []

    def build(self, ctx):
        si_id = ctx.mat_id(self.mat)
        ctx.det_band_um = self.th
        ctx.det_n_layers = 1
        br = self.back_reflector or {}
        refl = ([self._reflector_layer(ctx)]                  # 밴드 아래(=맨 아래) 반사경
                if br.get("enabled") and float(br.get("coverage", 1.0)) > 0 else [])
        if refl:
            ctx.det_below = len(refl)
        d = self.dti
        # optical=False -> DTI 를 광학 스택에 넣지 않음(전기적 격리만). 밴드는 균일 Si,
        # 트렌치 흡수/산란/제외 없음 -> 매끈한 Si 광학모델(상용 CIS QE 툴 관행).
        if not d or (d.get("mode") or "").lower() in ("", "none") \
                or d.get("optical") is False:
            ctx.trench = np.zeros_like(ctx.X, dtype=bool)
            return refl + [(ctx.zeros(self.mat), self.th)]

        p = ctx.p
        W = float(d["width_um"])
        liners = self._liners()
        cum = np.cumsum([t for _, t in liners]) if liners else np.array([])
        tot_liner = min(float(cum[-1]) if len(cum) else 0.0, W / 2)
        mode = d.get("mode") or ("2x2_open" if d.get("center_open") else "1x1")
        per = 2 * p if mode == "2x2" else p
        dx = np.abs(ctx.X - np.round(ctx.X / per) * per)
        dy = np.abs(ctx.Y - np.round(ctx.Y / per) * per)
        vAct = dx < W / 2
        hAct = dy < W / 2
        dyc = dxc = None
        oddX = oddY = None
        gx = gy = 0.0
        if mode == "2x2_open":
            gx = float(d.get("center_gap_x_um", d.get("center_gap_um", 0)) or 0) / 2
            gy = float(d.get("center_gap_y_um", d.get("center_gap_um", 0)) or 0) / 2
            bi = np.round(ctx.X / p)
            bj = np.round(ctx.Y / p)
            odd = lambda V: (lambda i: np.where(i % 2 != 0, i,
                             np.where(V > i * p, i + 1, i - 1)))(np.round(V / p)) * p
            dyc = np.abs(ctx.Y - odd(ctx.Y))
            dxc = np.abs(ctx.X - odd(ctx.X))
            oddX = (bi % 2 != 0)                          # intra-quad 수직 경계만 끊김/끝벽
            oddY = (bj % 2 != 0)
            vAct = vAct & ~(oddX & (dyc < gy))            # 팔 끊김 -> Si
            hAct = hAct & ~(oddY & (dxc < gx))
        any_t = vAct | hAct
        out = np.full(ctx.X.shape, si_id, dtype=np.uint8)
        fill_id = ctx.mat_id(d.get("fill", self.mat))
        core = (vAct & (dx <= W / 2 - tot_liner)) | (hAct & (dy <= W / 2 - tot_liner))
        # 우선순위(교차점 규약): 팔 끝벽 liner > core fill > 벽 liner > fill 기본
        out[any_t] = fill_id
        lo = 0.0
        for (lname, lth), hi in zip(liners, cum):         # 벽 liner 스택 (벽->안쪽)
            lid = ctx.mat_id(lname)
            band_v = vAct & (dx > W / 2 - hi) & (dx <= W / 2 - lo)
            band_h = hAct & (dy > W / 2 - hi) & (dy <= W / 2 - lo)
            out[band_v | band_h] = lid
            lo = float(hi)
        out[core] = fill_id                               # 교차점: core 가 벽 band 를 덮음
        if mode == "2x2_open":                            # 팔 끝벽 스택 (최우선)
            lo = 0.0
            for (lname, lth), hi in zip(liners, cum):
                lid = ctx.mat_id(lname)
                endv = vAct & oddX & (dyc >= gy + lo) & (dyc < gy + hi)
                endh = hAct & oddY & (dxc >= gx + lo) & (dxc < gx + hi)
                out[endv | endh] = lid
                lo = float(hi)
        ctx.trench = any_t
        return refl + [(out, self.th)]


# ==========================================================================
class BarlBlock:
    """blanket 다층 — [{material, thickness_um}, ...] 아래->위. 층 수 임의."""

    def __init__(self, layers):
        self.layers = layers

    def build(self, ctx):
        out = []
        for l in self.layers:
            t = float(l["thickness_um"])
            if t > 0:
                out.append((ctx.zeros(l["material"]), t))
        return out


# ==========================================================================
class GridCfBlock:
    """grid 울타리 + CF 채움.

    grid: {pitch:1|2, width_um, top_ratio(taper), stack:[{material,height_um}..],
           coat_material, coat_um, coat_on}   — stack 쌓고 표면(옆+위) 코팅 옵션
    cf  : {R:{material,thickness_um,curvature_um}, G:{...}, B:{...}}
          curvature = liquid reflow 응집(부피 유지) 상면 곡률 (+볼록/-오목)
    bg  : 이 밴드에서 CF/grid 밖(위)을 채우는 물질 = ML 평탄층 물질
    """

    def __init__(self, grid, cf, bg_material, men_slices=8, taper_slices=8):
        self.g = grid
        self.cf = cf
        self.bg = bg_material
        self.men = men_slices
        self.tap = taper_slices

    def build(self, ctx):
        g, cf, p = self.g, self.cf, ctx.p
        gridH = sum(float(l["height_um"]) for l in g["stack"])
        cw = float(g.get("coat_um", 0)) if g.get("coat_on", True) else 0.0
        th = np.array([float(cf[c]["thickness_um"]) for c in "RGB"])
        cv = np.array([float(cf[c].get("curvature_um", 0) or 0) for c in "RGB"])
        cfTopMax = float(np.max(th + np.maximum(2 * cv / 3, -4 * cv / 3)))
        cfTopMin = float(max(0.002, np.min(th + np.minimum(2 * cv / 3, -4 * cv / 3))))
        bandTop = max(gridH + cw, cfTopMax)
        ctx.cf_top_min = cfTopMin
        ctx.band_top = bandTop

        per = (g.get("pitch", 1) or 1) * p
        dgx = np.abs(ctx.X - np.round(ctx.X / per) * per)   # 세로벽까지 거리
        dgy = np.abs(ctx.Y - np.round(ctx.Y / per) * per)   # 가로벽까지 거리
        dg = np.minimum(dgx, dgy)
        # 교차점 deadzone(DZ): 울타리 교차점(코너)에서 금속이 대각선으로 자라 CF 를
        # 8각형으로 컷. dz_um = 코너에서 벽 안쪽으로 파고드는 대각 reach (≤0.25µm).
        dz = float(g.get("deadzone_um", g.get("dz_um", 0)) or 0)
        dz = max(0.0, min(dz, 0.25))
        W = float(g["width_um"])
        ratio = min(1.0, max(0.0, float(g.get("top_ratio", 1) or 0)))
        coat_id = ctx.mat_id(g.get("coat_material", "oxide")) if cw > 0 else 0
        gbounds = []
        acc = 0.0
        for l in g["stack"]:
            acc += float(l["height_um"])
            gbounds.append((acc, ctx.mat_id(l["material"])))

        # CF 색 배치 (bayer / tetra)
        pr = np.clip((ctx.Y / p).astype(int), 0, ctx.npx - 1)
        pc = np.clip((ctx.X / p).astype(int), 0, ctx.npx - 1)
        colmap = {"R": 0, "G": 1, "B": 2}
        colcode = np.zeros_like(pr)
        for r in range(ctx.npx):
            for c in range(ctx.npx):
                colcode[(pr == r) & (pc == c)] = colmap[ctx.bayer[r][c]]
        cf_per = (2 if (ctx.cf_array == "tetra") else 1) * p
        u = (ctx.X - (np.floor(ctx.X / cf_per) + 0.5) * cf_per) / (cf_per / 2)
        v = (ctx.Y - (np.floor(ctx.Y / cf_per) + 0.5) * cf_per) / (cf_per / 2)
        pin = (2.0 / 3.0) - (u * u + v * v)
        zTop = np.maximum(0.002, th[colcode] + cv[colcode] * pin)
        # CF 상부 모서리 chamfer: grid 벽 근처(안쪽 reach 이내) 상면을 angle 로 각지게
        # 컷 -> 가장자리 빛을 픽셀 중앙으로 funnel(크로스톡↓). chamfer_um=수평 reach,
        # chamfer_angle=경사(도). 0 이면 없음.
        ch_um = np.array([float(cf[c].get("chamfer_um", 0) or 0) for c in "RGB"])
        if np.any(ch_um > 0):
            ch_ang = np.array([float(cf[c].get("chamfer_angle", 45) or 45) for c in "RGB"])
            cr = ch_um[colcode]
            tang = np.tan(np.deg2rad(np.clip(ch_ang[colcode], 1.0, 89.0)))
            edge = dg - W / 2.0                       # grid 벽에서 CF 안쪽으로의 거리
            cut = np.maximum(0.0, cr - np.maximum(edge, 0.0)) * tang
            zTop = np.maximum(0.002, zTop - cut)
            cfTopMin = float(max(0.002, np.min(zTop)))   # 슬라이싱 범위를 chamfer 까지 확장
            cfTopMax = float(np.max(zTop))
            ctx.cf_top_min = cfTopMin
        cf_ids = np.array([ctx.mat_id(cf[c]["material"]) for c in "RGB"], dtype=np.uint8)
        cf_map = cf_ids[colcode]
        bg_id = ctx.mat_id(self.bg)
        # CF 풋프린트 scale (ML scale 과 동일 개념): 색별 가로/세로 배율.
        # u,v 는 CF 셀 중심 기준 정규화(-1..1) -> |u|<=sx & |v|<=sy 안쪽만 CF,
        # 바깥은 bg(ML 평탄층 물질). 기본 1.0 = 기존과 동일(셀 가득).
        sx = np.array([float(cf[c].get("scale_x", 1) or 1) for c in "RGB"])[colcode]
        sy = np.array([float(cf[c].get("scale_y", 1) or 1) for c in "RGB"])[colcode]
        cf_in = (np.abs(u) <= sx) & (np.abs(v) <= sy)

        # ---- z 경계점: 정확 경계 + 연속 구간(taper/meniscus)만 세분 ----
        bps = {0.0, bandTop}
        for b_, _ in gbounds:
            bps.add(min(b_, bandTop))
        if cw > 0:
            bps.add(min(gridH + cw, bandTop))
        if ratio < 1.0 and gridH > 0:
            # taper 슬라이스 자동: 벽 이동량(한쪽 W(1-ratio)/2)이 슬라이스당
            # 래스터 반 셀 이하가 되도록 — 계단을 래스터가 표현 가능한 최소로
            cell = ctx.span / ctx.n
            dw_half = W * (1.0 - ratio) / 2.0
            n_tap = min(48, max(self.tap, int(np.ceil(dw_half / (cell / 2)))))
            for k in range(1, n_tap):
                bps.add(gridH * k / n_tap)
        if cfTopMax - cfTopMin > 1e-6:
            for k in range(self.men + 1):
                z = cfTopMin + (cfTopMax - cfTopMin) * k / self.men
                if 0 < z < bandTop:
                    bps.add(z)
        elif 0 < cfTopMax < bandTop:
            bps.add(cfTopMax)
        bps = sorted(b for b in bps if -1e-12 <= b <= bandTop + 1e-12)

        layers = []
        for z0, z1 in zip(bps[:-1], bps[1:]):
            if z1 - z0 < 1e-9:
                continue
            zc = 0.5 * (z0 + z1)
            m = np.full(ctx.X.shape, bg_id, dtype=np.uint8)
            sel = (zc <= zTop) & cf_in
            m[sel] = cf_map[sel]
            wz = W * (1 - (1 - ratio) * min(zc / gridH, 1.0)) if gridH > 0 else W
            # 옆면 코팅이 래스터 셀보다 얇으면 샘플을 빠져나감 -> 최소 1셀 폭 보장
            cws = max(cw, ctx.span / ctx.n) if cw > 0 else 0.0
            if zc <= gridH:
                gid = gbounds[-1][1]
                for b_, i_ in gbounds:
                    if zc <= b_:
                        gid = i_
                        break
                m[dg < wz / 2] = gid
                if cw > 0:
                    m[(dg >= wz / 2) & (dg < wz / 2 + cws)] = coat_id
                if dz > 0:                                 # 교차점 deadzone: 코너 45° 컷 → 8각 CF
                    exx = dgx - wz / 2
                    eyy = dgy - wz / 2
                    corner = (exx > 0) & (eyy > 0) & (exx + eyy < dz)
                    m[corner] = gid                        # 코너 삼각형을 grid 금속으로
            elif cw > 0 and zc <= gridH + cw:
                m[dg < wz / 2 + cws] = coat_id
            layers.append((m, z1 - z0))
        return layers


# ==========================================================================
class PlanarBlock:
    """ML 평탄층 — CF 최저 상면 위로 지정 두께까지 전부 채움 (grid/CF 꼭대기 이상)."""

    def __init__(self, material, thickness_um):
        self.mat = material
        self.th = float(thickness_um)

    def build(self, ctx):
        top = max(ctx.cf_top_min + self.th, ctx.band_top)
        ctx.planar_top = top
        t = top - ctx.band_top
        return [(ctx.zeros(self.mat), t)] if t > 1e-9 else []


# ==========================================================================
class MlBlock:
    """ML 렌즈 (풍선 모델) — 층을 직접 내지 않고 sag 표면을 공표.

    quads : 2x2 quad 각각 {shape:"1x1"|"2x1"|"1x2"|"2x2", scale, orient}
            (4x4 픽셀 = 2x2 quad 4개. 1x1=픽셀당 1개, 2x2=quad 당 1개, ...)
    lenses: [{cx,cy,ax,ay}] 명시 배치 (quads 대신)
    height_um: 돔 두께 (곡률은 footprint 와 두께에서 자동, ROC=(r²+h²)/2h)
    겹치면 max(sag) -> 풍선 '찌부' 접촉선.
    """

    def __init__(self, material, height_um=0.0, hr=0.55, quads=None, lenses=None,
                 power=2.0):
        self.mat = material
        self.h = float(height_um or 0)
        self.hr = float(hr)
        self.quads = quads
        self.lenses = lenses
        # superellipse(Lamé) 지수: 2=원/타원(기존), >2=squircle(둥근 사각, gapless
        # 무간극 ML), →∞=사각, <2=오목. footprint 경계 |u|^n+|v|^n<1.
        self.power = float(power or 2.0)

    def _lens_list(self, ctx):
        if self.lenses:
            return [dict(L) for L in self.lenses]
        p = ctx.p
        out = []
        for qy in range(2):
            for qx in range(2):
                sp = self.quads[qy][qx]
                sc = float(sp.get("scale", 1) or 1)
                sh = sp["shape"]
                orient = sp.get("orient", "h")
                qh = float(sp.get("height_um", 0) or 0)   # quad별 돔 두께 (0=전역값)
                qn = float(sp.get("power", 0) or 0)        # quad별 superellipse 지수 (0=전역)
                x0, y0 = qx * 2 * p, qy * 2 * p
                cx, cy = x0 + p, y0 + p
                add = lambda ccx, ccy, ax, ay: out.append(
                    {"cx": ccx, "cy": ccy, "ax": ax * sc, "ay": ay * sc,
                     **({"h": qh} if qh > 0 else {}),
                     **({"power": qn} if qn > 0 else {})})
                if sh == "2x2":
                    add(cx, cy, p, p)
                elif sh == "1x1":
                    for a in range(2):
                        for b in range(2):
                            add(x0 + (b + .5) * p, y0 + (a + .5) * p, p / 2, p / 2)
                elif sh in ("2x1", "1x2"):
                    horiz = (sh == "2x1" and orient == "h")   # 1x2 = 세로 2픽셀
                    if horiz:
                        for a in range(2):
                            add(cx, y0 + (a + .5) * p, p, p / 2)
                    else:
                        for b in range(2):
                            add(x0 + (b + .5) * p, cy, p / 2, p)
        return out

    def _height(self, L):
        # 우선순위: 렌즈/quad 개별 h(height_um) > 전역 height_um > hr×min(반경)
        lh = float(L.get("h", L.get("height_um", 0)) or 0)
        if lh > 0:
            return lh
        if self.h > 0:
            return self.h
        return self.hr * min(float(L["ax"]), float(L["ay"]))

    def _power(self, L):
        n = float(L.get("power", 0) or 0)
        return n if n > 0 else self.power

    def build(self, ctx):
        sag = np.zeros_like(ctx.X)
        for L in self._lens_list(ctx):
            u = (ctx.X - float(L["cx"])) / float(L["ax"])
            v = (ctx.Y - float(L["cy"])) / float(L["ay"])
            n = self._power(L)
            # superellipse(Lamé): |u|^n+|v|^n. n=2 면 u²+v²(원) 과 동일(하위호환).
            re = (u * u + v * v) if abs(n - 2.0) < 1e-9 \
                else (np.abs(u) ** n + np.abs(v) ** n)
            s = np.where(re < 1, self._height(L) * np.sqrt(np.clip(1 - re, 0, 1)), 0.0)
            np.maximum(sag, s, out=sag)
        ctx.dome_sag = sag
        ctx.dome_mat = self.mat
        ctx.dome_h = max((self._height(L) for L in self._lens_list(ctx)), default=0.0)
        return []                                         # 슬라이스는 flush 에서 (코팅과 함께)


# ==========================================================================
class ConformalCoatBlock:
    """직전 곡면(ML 돔)을 덮는 conformal 코팅 — ARL. 여러 개 쌓으면 다층 conformal."""

    def __init__(self, material, thickness_um):
        self.mat = material
        self.th = float(thickness_um)

    def build(self, ctx):
        ctx.coats.append((self.mat, self.th))
        return []


# ==========================================================================
class BlockStack:
    """블록 조립 -> StructureIR. blocks 는 아래(Si)->위(공기) 순."""

    def __init__(self, blocks, ambient="air", materials=None, dispersion=None,
                 ml_slices=8, collect_deep=False, collect_r0=0.0, collect_ld_um=0.0):
        self.blocks = blocks
        self.ambient = ambient
        self.materials = materials or {}
        self.dispersion = dispersion or {}
        # collect_deep: 광다이오드 밴드 아래 반무한 기판 흡수를 QE 로 셀지.
        #   False(기본)=밴드만(유한 광다이오드 — 적색이 기판 뚫으면 손실, 물리적 rolloff)
        #   True=밴드+심부 전체 Si 흡수 (반무한 수집 가정)
        self.collect_deep = collect_deep
        # 캐리어 수집효율 η(z)=1-r0·exp(-z/Ld) (광학 QE -> 소자 QE). r0=0 이면 순수광학.
        self.collect_r0 = float(collect_r0 or 0.0)
        self.collect_ld_um = float(collect_ld_um or 0.0)
        self.ml_slices = ml_slices

    def _flush_dome(self, ctx):
        """ML sag + conformal 코팅들을 균등 슬라이스로 층화."""
        if ctx.dome_sag is None:
            return []
        sag = ctx.dome_sag
        coat_tot = sum(t for _, t in ctx.coats)
        top = ctx.dome_h + coat_tot
        if top <= 1e-9:
            return []
        ml_id = ctx.mat_id(ctx.dome_mat)
        air_id = ctx.mat_id("air")
        cums = np.cumsum([t for _, t in ctx.coats]) if ctx.coats else []
        layers = []
        nsl = max(4, int(self.ml_slices))
        for k in range(nsl):
            z0 = top * k / nsl
            z1 = top * (k + 1) / nsl
            zc = 0.5 * (z0 + z1)
            m = np.full(ctx.X.shape, air_id, dtype=np.uint8)
            for (cname, _), cum in reversed(list(zip(ctx.coats, cums))):
                m[sag + cum > zc] = ctx.mat_id(cname)     # 바깥 코팅부터, 안쪽이 덮어씀
            m[sag > zc] = ml_id
            if not (m == air_id).all():
                layers.append((m, z1 - z0))
        return layers

    @staticmethod
    def _tag_of(b):
        """블록 -> 경계 프로브용 태그 이름 (name 지정 우선)."""
        if getattr(b, "name", None):
            return b.name
        if isinstance(b, ConformalCoatBlock):
            return b.mat                                  # 코팅은 물질명 (예: ml_arl)
        return {"SiDtiBlock": "si", "BarlBlock": "barl", "GridCfBlock": "cf_grid",
                "PlanarBlock": "planar", "MlBlock": "ml"}.get(
                    type(b).__name__, type(b).__name__)

    def to_ir(self, ctx, substrate=None):
        layers = []                                       # 아래->위
        tags = []
        for b in self.blocks:
            new = b.build(ctx)
            layers += new
            tags += [self._tag_of(b)] * len(new)
        dome = self._flush_dome(ctx)
        layers += dome
        # 돔 구간은 ML+conformal 코팅이 같은 z 를 공유 -> 하나의 태그로 묶음
        dome_tag = "+".join([self._tag_of(b) for b in self.blocks
                             if isinstance(b, (MlBlock, ConformalCoatBlock))]) or "ml"
        tags += [dome_tag] * len(dome)
        assert layers, "블록이 층을 하나도 만들지 않음"
        # 인접 동일 맵 층 머지 — flat CF meniscus의 redundant 슬라이스나 동일 blanket 을
        # 하나의 두꺼운 층으로 합쳐 patterned 층 수(=eig/solve 횟수)를 줄인다. 물리 동일
        # (같은 lateral 맵 = 같은 S-matrix). 밴드/반사경 층은 맵이 distinct 라 안 합쳐짐.
        mlayers, mtags = [], []
        for (m, th), tg in zip(layers, tags):
            if mlayers and np.array_equal(mlayers[-1][0], m):
                mlayers[-1] = (mlayers[-1][0], mlayers[-1][1] + float(th))
            else:
                mlayers.append((m, float(th))); mtags.append(tg)
        layers, tags = mlayers, mtags
        # detector: 픽셀 분할(bayer) + SiDti 공표값
        det = None
        if ctx.det_band_um > 0 and ctx.bayer:
            pr = np.clip((ctx.Y / ctx.p).astype(int), 0, ctx.npx - 1)
            pc = np.clip((ctx.X / ctx.p).astype(int), 0, ctx.npx - 1)
            pixidx = (pr * ctx.npx + pc).astype(np.int32)
            labels = [ctx.bayer[r][c] for r in range(ctx.npx) for c in range(ctx.npx)]
            det = Detector(band_um=ctx.det_band_um, n_layers=ctx.det_n_layers,
                           pixel_map=pixidx, pixel_labels=labels,
                           exclude_mask=ctx.trench,
                           deep_is_detector=self.collect_deep,
                           n_below_band=ctx.det_below,
                           collect_r0=self.collect_r0,
                           collect_ld_um=self.collect_ld_um)
        sub = substrate or next((b.mat for b in self.blocks
                                 if isinstance(b, SiDtiBlock)), "si")
        ir = StructureIR(
            span_x=ctx.span, span_y=ctx.span,
            layers=[(m.astype(np.uint8), th) for m, th in reversed(layers)],
            region_materials={int(i): n for n, i in ctx._idx.items()},
            ambient=self.ambient, substrate=sub, detector=det,
            materials=self.materials, dispersion=self.dispersion,
            layer_tags=list(reversed(tags)))
        if self.ambient != "air":
            ir.remap_material("air", self.ambient)
        return ir.validate()


# ==========================================================================
def blocks_from_wizard_cfg(cfg, men_slices=8, taper_slices=8):
    """위저드 yaml v3 -> 블록 목록 (아래->위). 위저드는 블록 조립의 한 사례일 뿐."""
    s = cfg["stack"]
    ml = s["ml"]
    blocks = [
        SiDtiBlock(s["si"]["material"], s["si"]["thickness_um"], dti=s.get("dti"),
                   back_reflector=s["si"].get("back_reflector")),
        BarlBlock(s.get("barl") or []),
        GridCfBlock(s["grid"], s["cf"], bg_material=ml["material"],
                    men_slices=men_slices, taper_slices=taper_slices),
        PlanarBlock(ml["material"], ml.get("planar_um", 0)),
        MlBlock(ml["material"], height_um=ml.get("height_um", 0),
                hr=ml.get("hr", 0.55), quads=ml.get("quads"),
                lenses=ml.get("lenses"), power=ml.get("power", 2.0)),
    ]
    arl = s.get("arl_top")
    if arl and float(arl.get("thickness_um", 0)) > 0:
        blocks.append(ConformalCoatBlock(arl["material"], arl["thickness_um"]))
    return blocks


def ir_from_wizard_cfg(cfg, lateral_n, ml_slices=8, men_slices=8, taper_slices=8):
    """yaml v3 -> (블록 조립) -> IR."""
    # 구조 린트 — 누가 바꿔도 치명 실수는 여기서 명확한 메시지로 차단(모든 경로 공통 관문).
    from .lint import assert_wizard_cfg
    assert_wizard_cfg(cfg)
    g = cfg["grid"]
    ctx = BlockContext(pitch_um=g["pixel_pitch_um"], n_pixels=g["n_pixels"],
                       lateral_n=lateral_n, bayer=cfg.get("bayer"),
                       cf_array=cfg.get("cf_array"))
    stack = BlockStack(blocks_from_wizard_cfg(cfg, men_slices, taper_slices),
                       ambient=cfg.get("ambient", "air") or "air",
                       materials=dict(cfg.get("materials", {}) or {}),
                       dispersion=dict(cfg.get("dispersion", {}) or {}),
                       ml_slices=ml_slices,
                       collect_deep=bool(cfg.get("collect_deep_substrate", False)),
                       collect_r0=float((cfg.get("collection") or {}).get("r0", 0.0)),
                       collect_ld_um=float((cfg.get("collection") or {}).get("ld_um", 0.0)))
    # 후면 반사경은 SiDtiBlock 이 '밴드 아래 패턴 층'으로 삽입 (부분 커버리지 지원).
    # substrate 는 Si 유지 -> Cu 갭 사이로 투과된 빛은 심부 Si 흡수(손실).
    return stack.to_ir(ctx)


# ==========================================================================
def apply_cra_shift(ir, shift_ml_um=(0.0, 0.0), shift_cfgrid_um=(0.0, 0.0)):
    """CRA 렌즈 shift(shrink): ML 그룹과 CF+grid 그룹 층을 횡방향으로 순환이동.

    빗각(CRA) 입사 시 초점이 틀어지는 걸 상부 구조를 빛 오는 쪽으로 밀어 Si 중심에
    다시 모으는 lens-shift 보정. Si/DTI/BARL/검출기(pixel_map)는 고정.

    RCWA 는 supercell 을 상하좌우·대각 무한반복 -> shift 는 np.roll(주기 wrap):
    unit 밖으로 나간 부분이 반대편서 들어옴 = '옆 unit 침범'을 물리적으로 정확히 표현.
    Si/BARL 은 안 밀어 원래 주기 유지 -> 주기성 자동 정합. region 인덱스 맵이라
    정수픽셀 이동(보간 없음; N 세밀하면 오차 <픽셀).

    shift_*_um = (dx, dy) µm. dx>0 = +x 방향. ML 이 CF+grid 보다 크게(더 위라).
    반환: 층 맵만 교체한 새 IR (원본 불변).
    """
    tags = getattr(ir, "layer_tags", None)
    if not tags:
        raise ValueError("layer_tags 없음 — 블록 조립 IR(ir_from_wizard_cfg) 필요")
    ny, nx = ir.grid_shape
    dxp, dyp = ir.span_x / nx, ir.span_y / ny

    def rollpx(m, sh):
        sx, sy = int(round(sh[0] / dxp)), int(round(sh[1] / dyp))
        return m if (sx == 0 and sy == 0) else np.roll(m, (sy, sx), axis=(0, 1))

    new_layers = []
    for (m, th), tg in zip(ir.layers, tags):
        if "cf_grid" in tg:
            m = rollpx(m, shift_cfgrid_um)
        elif "ml" in tg or tg == "planar":               # ML 돔+상부 코팅+planar
            m = rollpx(m, shift_ml_um)
        new_layers.append((m, th))
    import copy
    ir2 = copy.copy(ir)
    ir2.layers = new_layers
    return ir2
