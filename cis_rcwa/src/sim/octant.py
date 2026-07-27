# -*- coding: utf-8 -*-
"""옥탄트(8-fold) 대칭 확장 + 센서 이미지 조립.

센서가 D4 대칭(좌우/상하/대각 반전 + 회전)이라 가정하면, 이미지의 1/8(옥탄트)만
RCWA 로 계산하고 나머지 7개는 그 결과를 기하 변환해 채운다 → 계산 1/8.

CFA 채널 재배치의 일반화: 기하 변환 T 를 하면 unit 의 라벨 격자도 T(labels) 로
움직인다. 실제 센서의 CFA 방향은 어디서나 동일(labels)하므로, 변환된 QE 를
'라벨이 맞는 위치'로 되돌리는 순열을 라벨 매칭(같은 라벨끼리 최근접)으로 구한다.
— 2×2 RGGB 든 4×4 tetra 든 하드코딩 없이 동작 (tetra 의 Gr/Gb 스왑이 자동 재현).
"""
import math

import numpy as np


# ------------------------------------------------------------- D4 기하 변환
def flip_array(a, angle):
    """반사: 0=상하, 90=좌우, 45=주대각(↘) 기준, -45=부대각(↙) 기준."""
    a = np.asarray(a)
    if angle == 0:
        return a[::-1, :]
    if angle == 90:
        return a[:, ::-1]
    if angle == 45:
        return a.T[::-1, ::-1]
    if angle == -45:
        return a.T
    raise ValueError(f"flip angle {angle}")


def rotate_array(a, angle):
    """회전(반시계): 90/-90/180 — np.rot90 래퍼."""
    a = np.asarray(a)
    if angle == 90:
        return np.rot90(a, 1)
    if angle == -90:
        return np.rot90(a, -1)
    if angle == 180:
        return np.rot90(a, 2)
    raise ValueError(f"rotate angle {angle}")


def transform_octant(a, octant_index):
    """옥탄트 1~8 에 대응하는 D4 변환 (1=항등)."""
    if octant_index == 1:
        return np.asarray(a)
    return {2: lambda x: flip_array(x, 45), 3: lambda x: rotate_array(x, 90),
            4: lambda x: flip_array(x, 90), 5: lambda x: rotate_array(x, 180),
            6: lambda x: flip_array(x, -45), 7: lambda x: rotate_array(x, -90),
            8: lambda x: flip_array(x, 0)}[octant_index](a)


def transform_unit(unit, labels, octant_index):
    """unit(QE 격자, npx×npx)을 옥탄트 변환하되 CFA 라벨 배치를 보존.

    원리: 주기 CFA(bayer/tetra)는 D4 변환을 해도 '주기 평행이동' 한 번이면 원래
    패턴과 정확히 겹친다. RCWA 슈퍼셀은 주기 구조라 roll(원점 이동)은 물리적으로
    정확한 재라벨링 — 변환된 라벨 격자를 원래 라벨과 일치시키는 (dr,dc)를 찾아
    QE 격자도 같이 roll 한다. (tetra 의 Gr/Gb 대각 스왑 규칙이 자동 재현되며,
    4×4 하드코딩 없이 임의 npx 주기 CFA 에 동작.)
    """
    U = transform_octant(np.asarray(unit, float), octant_index)
    L0 = np.asarray(labels, dtype=object)
    LT = transform_octant(L0, octant_index)
    if np.array_equal(LT, L0):
        return U
    n, m = L0.shape
    for dr in range(n):
        for dc in range(m):
            if np.array_equal(np.roll(LT, (dr, dc), axis=(0, 1)), L0):
                return np.roll(U, (dr, dc), axis=(0, 1))
    raise ValueError("CFA 라벨이 이 대칭 변환과 주기이동으로 겹치지 않습니다 — "
                     "octant 대칭을 끄고 region='full' 로 계산하세요")


# ------------------------------------------------------------- 동컬러 diff
def unit_color_diff(arr, labels):
    """unit QE 격자(npx×npx) → 같은 라벨 픽셀끼리 (max−min)/mean 편차.

    사광 핵심 지표: shrink 후에도 남는 초점 틀어짐이 같은 색 픽셀들(tetra 2×2
    quad, bayer Gr/Gb)에 서로 다른 QE 를 주는 정도. 스펙: diff_pct ≤ 30 권장.
    반환: {label: {"min","max","mean","n","diff_pct"}}
    """
    A = np.asarray(arr, float)
    L = np.asarray(labels, dtype=object)
    out = {}
    for c in sorted({str(v) for v in L.flat}):
        v = A[L == c]
        mn, mx, mu = float(v.min()), float(v.max()), float(v.mean())
        dp = (mx - mn) / mu * 100.0 if mu > 1e-12 else 0.0
        out[c] = {"min": round(mn, 5), "max": round(mx, 5),
                  "mean": round(mu, 5), "n": int(v.size),
                  "diff_pct": round(dp, 2)}
    return out


# ------------------------------------------------------------- 필드 격자
def make_xy_fields_set(field_step=0.1, region="octant", x_max=0.8, y_max=0.6):
    """정규화 필드 좌표 (x,y) 목록. octant=1옥탄트(x≥y≥0)만, full=전체.

    x∈[−x_max,x_max], y∈[−y_max,y_max] (4:3 센서 → 0.8/0.6, 코너 r=1).
    """
    eps = 1e-9
    ys = np.arange(-y_max if region == "full" else 0.0, y_max + eps, field_step)
    xs = np.arange(-x_max if region == "full" else 0.0, x_max + eps, field_step)
    out = []
    for y in ys:
        for x in xs:
            if region == "octant" and x < y - eps:
                continue
            out.append((round(float(x), 6), round(float(y), 6)))
    return out


# ------------------------------------------------------------- 이미지 조립
def assemble_image(units, field_step, labels, x_max=0.8, y_max=0.6):
    """옥탄트 unit 들 → 센서 전면 2D 이미지.

    units: {(x,y): npx×npx array}  (1옥탄트 필드만 있으면 됨; full 좌표가 있으면
           그대로 우선 사용). 계산 안 된 타일은 NaN.
    반환: (image[H,W], meta) — H=npx·unit_rows, W=npx·unit_cols.
    """
    L0 = np.asarray(labels, dtype=object)
    npx = L0.shape[0]
    unit_cols = int(round(2 * x_max / field_step)) + 1
    unit_rows = int(round(2 * y_max / field_step)) + 1
    H, W = npx * unit_rows, npx * unit_cols
    img = np.full((H, W), np.nan)

    def put(xf, yf, u):
        if not (-x_max - 1e-3 <= xf <= x_max + 1e-3
                and -y_max - 1e-3 <= yf <= y_max + 1e-3):
            return
        xs = npx * int(round((x_max + xf) / field_step))
        ys = npx * int(round((y_max - yf) / field_step))   # y↑ = 이미지 위쪽
        img[ys:ys + npx, xs:xs + npx] = u

    for (x, y), u in units.items():
        u = np.asarray(u, float)
        put(x, y, u)                                        # 1: 원본
        if x < y - 1e-9 or x < -1e-9 or y < -1e-9:
            continue                                        # full 좌표는 확장 없이
        put(y, x, transform_unit(u, L0, 2))
        put(-y, x, transform_unit(u, L0, 3))
        put(-x, y, transform_unit(u, L0, 4))
        put(-x, -y, transform_unit(u, L0, 5))
        put(-y, -x, transform_unit(u, L0, 6))
        put(y, -x, transform_unit(u, L0, 7))
        put(x, -y, transform_unit(u, L0, 8))
    meta = {"unit_rows": unit_rows, "unit_cols": unit_cols, "npx": npx,
            "x_max": x_max, "y_max": y_max, "field_step": field_step}
    return img, meta
