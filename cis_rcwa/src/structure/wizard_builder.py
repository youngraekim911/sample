# -*- coding: utf-8 -*-
"""WizardBuilder — 구조 위저드(<product>.yaml, schema v3) -> 3D voxel (RCWA 입력).

editors/structure_wizard.html 의 matAt() 과 동일한 지오메트리 (경계/표면 일치):
  Si substrate (DTI: 식각[1x1 분리 | 2x2 center-open 클로버] -> 표면 oxide liner -> 채움)
  -> BARL blanket 다층 (4x4 전면)
  -> Grid 울타리(fence, 1x1/2x2 pitch, 다층 stack, 표면 oxide 코팅 옵션)
  -> CF (울타리 셀 채움, 컬러별 두께 + 상부 곡률 meniscus)
  -> ML 평탄층(최저 CF top 기준) -> ML dome (quads + b_theta 변형)
  -> ARL top (ML 표면 conformal) -> air
"""
import math
import numpy as np


class WizardBuilder:
    def __init__(self, cfg, base_dir="."):
        self.cfg = cfg
        self.base_dir = base_dir
        g = cfg["grid"]
        self.p = float(g["pixel_pitch_um"])
        self.npx = int(g["n_pixels"])
        self.dxy = float(g["dxy_um"])
        self.dz = float(g["dz_um"])
        self.air = float(g.get("air_margin_um", 0.2))
        self.span = self.p * self.npx
        self.nx = self.ny = int(round(self.span / self.dxy))
        self.s = cfg["stack"]
        xc = (np.arange(self.nx) + 0.5) * self.dxy
        self.X, self.Y = np.meshgrid(xc, xc)          # [ny,nx], X=x, Y=y (row j = y)
        self.names = ["air"]
        self._idx = {"air": 0}

    def _id(self, name):
        if name not in self._idx:
            self._idx[name] = len(self.names)
            self.names.append(name)
        return self._idx[name]

    # ------------------------------------------------------------- DTI
    def _si_map(self):
        """Si 밴드 2D 물질 id 맵: 식각(1x1|2x2|클로버) -> 식각면 라이너 -> 채움.

        mode "2x2_open": intra-quad 트렌치 팔이 중앙 못미쳐 끊김(가로/세로 open
        길이 별도) -> 4픽셀 Si 직각 클로버 연결. 라이너=사이드월+팔 끝벽.
        """
        st = self.s["dti"]
        p = self.p
        W = float(st["width_um"])
        ol = min(float(st["oxide_liner_um"]), W / 2)
        mode = st.get("mode") or ("2x2_open" if st.get("center_open") else "1x1")
        per = 2 * p if mode == "2x2" else p
        dx = np.abs(self.X - np.round(self.X / per) * per)
        dy = np.abs(self.Y - np.round(self.Y / per) * per)
        vAct = dx < W / 2                      # 수직 트렌치
        hAct = dy < W / 2                      # 수평 트렌치
        vEnd = np.zeros_like(vAct)
        hEnd = np.zeros_like(hAct)
        if mode == "2x2_open":
            gx = float(st.get("center_gap_x_um", st.get("center_gap_um", 0)) or 0) / 2
            gy = float(st.get("center_gap_y_um", st.get("center_gap_um", 0)) or 0) / 2
            bi = np.round(self.X / p)
            bj = np.round(self.Y / p)
            odd = lambda V: (lambda i: np.where(i % 2 != 0, i,
                             np.where(V > i * p, i + 1, i - 1)))(np.round(V / p)) * p
            dyc = np.abs(self.Y - odd(self.Y))
            dxc = np.abs(self.X - odd(self.X))
            oddX = (bi % 2 != 0)               # intra-quad 수직 경계
            oddY = (bj % 2 != 0)
            gapV = vAct & oddX & (dyc < gy)    # 팔 끊김(안 파임) -> Si
            vEnd = vAct & oddX & (dyc >= gy) & (dyc < gy + ol)   # 팔 끝벽 라이너
            gapH = hAct & oddY & (dxc < gx)
            hEnd = hAct & oddY & (dxc >= gx) & (dxc < gx + ol)
            vAct = vAct & ~gapV
            hAct = hAct & ~gapH
        si_id = self._id(self.s["si"]["material"])
        li_id = self._id(st.get("liner", "oxide"))
        fi_id = self._id(st.get("fill", "si"))
        endwall = (vEnd & vAct) | (hEnd & hAct)
        core = (vAct & (dx <= W / 2 - ol)) | (hAct & (dy <= W / 2 - ol))
        any_t = vAct | hAct
        self._trench = any_t                   # DTI 트렌치(라이너+채움) 마스크 캐시
        out = np.where(endwall, li_id,
              np.where(core, fi_id,
              np.where(any_t, li_id, si_id)))
        return out.astype(np.uint8)

    # ----------------------------------------------------- 픽셀 QE 마스크
    def dti_trench_mask(self):
        """(ny,nx) bool — DTI 트렌치 내부(라이너+채움). 픽셀 Si 창에서 제외용."""
        if not hasattr(self, "_trench"):
            self._si_map()
        return self._trench

    def pixel_maps(self):
        """픽셀 QE 용 맵: (pixidx[ny,nx] int, colors[list len npx²] 'R'/'G'/'B').

        pixidx = pr*npx+pc (pr: y-행, pc: x-열). colors[k] 는 bayer 색.
        """
        p = self.p
        pr = np.clip((self.Y / p).astype(int), 0, self.npx - 1)
        pc = np.clip((self.X / p).astype(int), 0, self.npx - 1)
        pixidx = (pr * self.npx + pc).astype(np.int32)
        bay = self.cfg["bayer"]
        colors = [bay[r][c] for r in range(self.npx) for c in range(self.npx)]
        return pixidx, colors

    # ------------------------------------------------------------- ML (풍선 모델)
    def _ml_lenses(self):
        """렌즈 footprint 목록 [{cx,cy,ax,ay},...] — 명시 lenses 우선, 없으면 quad 템플릿."""
        ml = self.s["ml"]
        if ml.get("lenses"):
            return [dict(L) for L in ml["lenses"]]
        p = self.p
        out = []
        for qy in range(2):
            for qx in range(2):
                sp = ml["quads"][qy][qx]
                sc = float(sp.get("scale", 1) or 1)
                sh = sp["shape"]
                orient = sp.get("orient", "h")
                x0, y0 = qx * 2 * p, qy * 2 * p
                cx, cy = x0 + p, y0 + p
                add = lambda ccx, ccy, ax, ay: out.append(
                    {"cx": ccx, "cy": ccy, "ax": ax * sc, "ay": ay * sc})
                if sh == "2x2":
                    add(cx, cy, p, p)
                elif sh == "1x1":
                    for a in range(2):
                        for b in range(2):
                            add(x0 + (b + .5) * p, y0 + (a + .5) * p, p / 2, p / 2)
                elif sh == "2x1":
                    if orient == "h":
                        for a in range(2):
                            add(cx, y0 + (a + .5) * p, p, p / 2)
                    else:
                        for b in range(2):
                            add(x0 + (b + .5) * p, cy, p / 2, p)
        return out

    def _ml_hr(self):
        ml = self.s["ml"]
        if ml.get("hr") is not None:
            return float(ml["hr"])
        return float(ml.get("sag_height_um", 0.55)) / (self.p / 2)

    def _ml_height(self, L):
        """렌즈 높이: height_um>0 이면 고정(µm) -> footprint 따라 곡률 변화, 아니면 hr*min(반경)."""
        hu = float(self.s["ml"].get("height_um", 0) or 0)
        if hu > 0:
            return hu
        return self._ml_hr() * min(float(L["ax"]), float(L["ay"]))

    def _ml_sag(self):
        """풍선 모델: 각 렌즈 = 타원 캡. 겹침은 max -> 접촉 교선(찌부)."""
        sag = np.zeros_like(self.X)
        for L in self._ml_lenses():
            u = (self.X - float(L["cx"])) / float(L["ax"])
            v = (self.Y - float(L["cy"])) / float(L["ay"])
            r2 = u * u + v * v
            hL = self._ml_height(L)
            s = np.where(r2 < 1, hL * np.sqrt(np.clip(1 - r2, 0, 1)), 0.0)
            np.maximum(sag, s, out=sag)
        return sag

    def _ml_maxh(self):
        return max((self._ml_height(L) for L in self._ml_lenses()), default=0.0)

    # ------------------------------------------------------------- build
    def build(self):
        s = self.s
        dz = self.dz
        zSi = float(s["si"]["thickness_um"])
        barl = s.get("barl") or []
        barlH = sum(float(l["thickness_um"]) for l in barl)
        g = s["grid"]
        gridH = sum(float(l["height_um"]) for l in g["stack"])
        cw = float(g.get("coat_um", 0)) if g.get("coat_on", True) else 0.0
        gridTot = gridH + cw
        cf = s["cf"]
        th = np.array([float(cf[c]["thickness_um"]) for c in "RGB"])
        cv = np.array([float(cf[c].get("curvature_um", 0) or 0) for c in "RGB"])
        cfTopMax = float(np.max(th + np.maximum(2 * cv / 3, -4 * cv / 3)))
        cfTopMin = float(max(0.002, np.min(th + np.minimum(2 * cv / 3, -4 * cv / 3))))
        planar = max(cfTopMin + float(s["ml"]["planar_um"]), cfTopMax, gridTot)
        sag = self._ml_sag()
        sagH = self._ml_maxh()                 # 풍선 모델: hr*min(반경) 최대
        arlT = float(s["arl_top"]["thickness_um"])
        Ht = zSi + barlH + planar + sagH + arlT + self.air
        nz = int(round(Ht / dz))

        # 2D 맵 사전계산
        si_map = self._si_map()
        p = self.p
        per = (g.get("pitch", 1) or 1) * p
        dg = np.minimum(np.abs(self.X - np.round(self.X / per) * per),
                        np.abs(self.Y - np.round(self.Y / per) * per))
        W = float(g["width_um"])
        ratio = min(1.0, max(0.0, float(g.get("top_ratio", 1) or 0)))   # 상부/하부 폭 비율 (0~1)
        coat_id = self._id(g.get("coat_material", "oxide")) if cw > 0 else 0
        gbounds = []
        acc = 0.0
        for l in g["stack"]:
            acc += float(l["height_um"])
            gbounds.append((acc, self._id(l["material"])))
        # CF: 색 코드 + meniscus zTop
        pr = np.clip((self.Y / p).astype(int), 0, self.npx - 1)
        pc = np.clip((self.X / p).astype(int), 0, self.npx - 1)
        colmap = {"R": 0, "G": 1, "B": 2}
        colcode = np.zeros_like(pr)
        bay = self.cfg["bayer"]
        for r in range(self.npx):
            for c in range(self.npx):
                colcode[(pr == r) & (pc == c)] = colmap[bay[r][c]]
        cf_per = (2 if (self.cfg.get("cf_array") == "tetra") else 1) * p   # CF array 단위 셀
        u = (self.X - (np.floor(self.X / cf_per) + 0.5) * cf_per) / (cf_per / 2)
        v = (self.Y - (np.floor(self.Y / cf_per) + 0.5) * cf_per) / (cf_per / 2)
        # 응집(cohesion, 볼륨 보존): 셀 평균=thickness. k>0 중앙 응집 / k<0 벽 젖음
        pin = (2.0 / 3.0) - (u * u + v * v)
        zTop = np.maximum(0.002, th[colcode] + cv[colcode] * pin)
        cf_ids = np.array([self._id(cf[c]["material"]) for c in "RGB"], dtype=np.uint8)
        cf_map = cf_ids[colcode]
        ml_id = self._id(s["ml"]["material"])
        arl_id = self._id(s["arl_top"]["material"])
        bbounds = []
        acc = 0.0
        for l in barl:
            acc += float(l["thickness_um"])
            bbounds.append((acc, self._id(l["material"])))

        barlTop = zSi + barlH
        planarTop = barlTop + planar
        surf = planarTop + sag

        A = np.zeros((nz, self.ny, self.nx), dtype=np.uint8)
        for zi in range(nz):
            zc = (zi + 0.5) * dz
            if zc < zSi:
                A[zi] = si_map
            elif zc < barlTop:
                zz = zc - zSi
                mid = bbounds[-1][1]
                for b_, i_ in bbounds:
                    if zz <= b_:
                        mid = i_
                        break
                A[zi] = mid
            elif zc < planarTop:
                zabs = zc - barlTop
                sl = np.full((self.ny, self.nx), ml_id, dtype=np.uint8)   # 평탄화 fill
                m = zabs <= zTop
                sl[m] = cf_map[m]                                          # CF (곡면 top)
                # taper: 아래 W -> 위 W*ratio 선형
                wz = W * (1 - (1 - ratio) * min(zabs / gridH, 1.0)) if gridH > 0 else W
                fence = dg < wz / 2
                side = dg < wz / 2 + cw
                if zabs <= gridH:                                          # fence stack
                    gid = gbounds[-1][1]
                    for b_, i_ in gbounds:
                        if zabs <= b_:
                            gid = i_
                            break
                    sl[fence] = gid
                    if cw > 0:
                        sl[side & ~fence] = coat_id                        # 옆면 코팅
                elif cw > 0 and zabs <= gridTot:
                    sl[side] = coat_id                                     # 윗면 코팅
                A[zi] = sl
            else:
                sl = np.zeros((self.ny, self.nx), dtype=np.uint8)
                mlm = zc <= surf
                sl[mlm] = ml_id                                            # ML dome
                am = (~mlm) & (zc <= surf + arlT)
                sl[am] = arl_id                                            # ARL conformal
                A[zi] = sl

        self.matid = A
        self._bounds_um = [("substrate_si", 0.0, zSi), ("barl", zSi, barlTop),
                           ("cf_grid_planar", barlTop, planarTop),
                           ("microlens", planarTop, planarTop + sagH + arlT),
                           ("air_margin", planarTop + sagH + arlT, Ht)]
        self.nz = nz
        return A

    def meta(self):
        mats = {}
        cfg_m = self.cfg.get("materials", {}) or {}
        for name in self.names:
            m = cfg_m.get(name, {}) or {}
            mats[name] = {"id": self._idx[name],
                          "n": float(m.get("n", 1.0)), "k": float(m.get("k", 0.0))}
        vz = lambda um: int(round(um / self.dz))
        return {
            "product": self.cfg.get("product", ""),
            "shape_zyx": [int(self.nz), int(self.ny), int(self.nx)],
            "voxel_um": {"dz": self.dz, "dy": self.dxy, "dx": self.dxy},
            "lateral_span_um": self.span,
            "pixel_pitch_um": self.p,
            "n_pixels": self.npx,
            "materials": mats,
            "substrate_material": self.s["si"]["material"],
            "layer_bounds_vox": [
                {"name": n, "z0": vz(a), "z1": vz(b),
                 "z0_um": round(a, 4), "z1_um": round(b, 4)}
                for (n, a, b) in self._bounds_um],
            "axis_note": "matid[z,y,x]; z=0 bottom(Si), light incident from top(+z)",
        }


def main():
    import os, json, argparse
    import yaml
    ap = argparse.ArgumentParser(description="wizard yaml -> npy")
    ap.add_argument("-c", "--config", required=True)
    ap.add_argument("-o", "--outdir", default="out")
    args = ap.parse_args()
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    b = WizardBuilder(cfg, base_dir=os.path.dirname(os.path.abspath(args.config)))
    A = b.build()
    os.makedirs(args.outdir, exist_ok=True)
    nm = (cfg.get("product") or "structure").replace(" ", "_")
    np.save(os.path.join(args.outdir, f"{nm}_matid.npy"), A)
    with open(os.path.join(args.outdir, f"{nm}_meta.json"), "w", encoding="utf-8") as f:
        json.dump(b.meta(), f, indent=2, ensure_ascii=False)
    print(f"[saved] {args.outdir}/{nm}_matid.npy shape={A.shape}")
    print(f"[saved] {args.outdir}/{nm}_meta.json  materials={b.names}")


if __name__ == "__main__":
    main()
