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
        """Si 밴드 2D 물질 id 맵: 식각(모양) -> 표면 라이너 -> 채움."""
        st = self.s["dti"]
        p = self.p
        W = float(st["width_um"])
        ol = min(float(st["oxide_liner_um"]), W / 2)
        d = np.minimum(np.abs(self.X - np.round(self.X / p) * p),
                       np.abs(self.Y - np.round(self.Y / p) * p))
        trench = d < W / 2
        liner = trench & (d > W / 2 - ol)
        si_id = self._id(self.s["si"]["material"])
        li_id = self._id(st.get("liner", "oxide"))
        fi_id = self._id(st.get("fill", "si"))
        if st.get("center_open"):
            gh = float(st.get("center_gap_um", 0)) / 2
            i = np.round(self.X / p)
            ix = np.where(i % 2 != 0, i, np.where(self.X > i * p, i + 1, i - 1)) * p
            j = np.round(self.Y / p)
            jy = np.where(j % 2 != 0, j, np.where(self.Y > j * p, j + 1, j - 1)) * p
            dxc, dyc = np.abs(self.X - ix), np.abs(self.Y - jy)
            inbox = (dxc < gh) & (dyc < gh)                 # center box: 식각 안 됨 -> 순수 Si
            near_box = (dxc < gh + ol) & (dyc < gh + ol) & ~inbox   # 팔 '끝벽'(식각면) 라이너
            liner = liner | (trench & near_box)
            trench = trench & ~inbox
            liner = liner & ~inbox
        out = np.where(liner, li_id, np.where(trench, fi_id, si_id))
        return out.astype(np.uint8)

    # ------------------------------------------------------------- ML sag
    def _ml_sag(self):
        ml = self.s["ml"]
        p = self.p
        h = float(ml["sag_height_um"])
        quads = ml["quads"]
        defs = ml.get("deformations") or {}
        sag = np.zeros_like(self.X)

        def btheta(lid, theta):
            spec = defs.get(lid)
            if not spec or "b_theta" not in spec:
                return 1.0
            b = np.asarray(spec["b_theta"], float)
            n = len(b)
            ang = np.mod(theta, 2 * math.pi)
            f = ang / (2 * math.pi) * n
            i0 = np.floor(f).astype(int) % n
            t = f - np.floor(f)
            return b[i0] * (1 - t) + b[(i0 + 1) % n] * t

        def dome(lid, cx, cy, ax, ay):
            u = (self.X - cx) / ax
            v = (self.Y - cy) / ay
            rho = np.hypot(u, v)
            B = btheta(lid, np.arctan2(v, u))
            r = rho / np.maximum(B, 1e-9)
            s = np.where(r <= 1, h * np.sqrt(np.clip(1 - r * r, 0, 1)), 0.0)
            np.maximum(sag, s, out=sag)

        for qy in range(2):
            for qx in range(2):
                spec = quads[qy][qx]
                sh = spec["shape"]
                orient = spec.get("orient", "h")
                x0, y0 = qx * 2 * p, qy * 2 * p
                cx0, cy0 = x0 + p, y0 + p
                if sh == "2x2":
                    dome(f"q{qy}{qx}_2x2_0", cx0, cy0, p, p)
                elif sh == "1x1":
                    for a in range(2):
                        for b in range(2):
                            dome(f"q{qy}{qx}_1x1_{a*2+b}",
                                 x0 + (b + .5) * p, y0 + (a + .5) * p, p / 2, p / 2)
                elif sh == "2x1":
                    if orient == "h":
                        for a in range(2):
                            dome(f"q{qy}{qx}_2x1_{a}", cx0, y0 + (a + .5) * p, p, p / 2)
                    else:
                        for b in range(2):
                            dome(f"q{qy}{qx}_2x1_{b}", x0 + (b + .5) * p, cy0, p / 2, p)
        return sag

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
        cfTopMax = float(np.max(th + np.maximum(0, cv)))
        cfTopMin = float(np.min(th + np.minimum(0, cv)))
        planar = max(cfTopMin + float(s["ml"]["planar_um"]), cfTopMax, gridTot)
        sag = self._ml_sag()
        sagH = float(s["ml"]["sag_height_um"])
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
        fence = dg < W / 2
        side = dg < W / 2 + cw
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
        u = (self.X - (np.floor(self.X / per) + 0.5) * per) / (per / 2)
        v = (self.Y - (np.floor(self.Y / per) + 0.5) * per) / (per / 2)
        pin = 1 - np.maximum(u * u, v * v)              # 벽=0, 중앙=1
        zTop = th[colcode] + cv[colcode] * pin
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
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    b = WizardBuilder(cfg, base_dir=os.path.dirname(os.path.abspath(args.config)))
    A = b.build()
    os.makedirs(args.outdir, exist_ok=True)
    nm = (cfg.get("product") or "structure").replace(" ", "_")
    np.save(os.path.join(args.outdir, f"{nm}_matid.npy"), A)
    with open(os.path.join(args.outdir, f"{nm}_meta.json"), "w") as f:
        json.dump(b.meta(), f, indent=2, ensure_ascii=False)
    print(f"[saved] {args.outdir}/{nm}_matid.npy shape={A.shape}")
    print(f"[saved] {args.outdir}/{nm}_meta.json  materials={b.names}")


if __name__ == "__main__":
    main()
