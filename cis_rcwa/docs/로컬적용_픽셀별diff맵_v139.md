# [로컬 클로드 작업지시] 사광 ch.diff — 픽셀 16개 개별 field-맵 + 4×4 모자이크

이 문서를 로컬 클로드에게 그대로 붙여넣으면 됩니다. 수정 3개 파일 + 신설 1개
파일이고, 모든 코드가 아래 전문으로 들어 있습니다.

## 0. 배경 — 무엇을 왜 만드나

기존 사광(oblique) diff 는 채널 통계(R/Gr/Gb/B 별 (max−min)/mean)만 보여줬다.
요구사항: **단위셀(4×4 tetra = 16픽셀)의 픽셀 하나하나에 대해** "센서
전체(field)에서 그 픽셀의 동컬러 대비 편차 맵"을 만들고, 그 16장을 단위셀
배열 그대로 4×4 모자이크로 보여줄 것.

물리 (사용자 확인 완료):
- 센서 가장자리 field 에서는 주광선이 렌즈 중심 쪽으로 기울어 들어와, 초점
  스팟이 동컬러 2×2 그룹의 중심이 아니라 **센서중심 쪽 코너로 치우쳐** 맺힘
- 예: 센서 우측상단 field → 그룹 내 신호 세기 **좌하 > 우하 = 좌상 > 우상**
- 따라서 각 픽셀의 diff 맵은 "그룹 내 코너 위치 × field 방향"에 지배되는
  경사판 + 구조 기인 비대칭이 얹힌 모양이 됨
- 이건 공식으로 넣는 게 아니라 **CRA 기반 빗각 RCWA 정계산에서 저절로 나옴**
  (shrink 로 보정하고 남는 잔차를 보는 것이 이 지표의 목적)

정의:
```
diff%(r,c; field) = ( QE[r,c] − μ_ch ) / μ_ch × 100
μ_ch = 그 field 에서 픽셀 (r,c)와 같은 채널(R/Gr/Gb/B — refine_channels 기준,
       tetra 는 동컬러 2×2 quad)에 속한 픽셀들의 QE 평균
```

데이터 흐름 (기존 인프라 재사용 — 새 RCWA 계산 없음):
```
사광 이미지 잡(_run_image_job)이 이미 field 마다 unit 픽셀 QE 격자(npx×npx)를
계산하고 옥탄트 8배 대칭으로 센서 전면 이미지(assemble_image)를 만든다.
조립된 이미지에서 픽셀 (r,c)의 field-맵은 스트라이드 한 줄:  img[r::npx, c::npx]
→ 후처리 함수 하나 + 결과 JSON 필드 + UI 뷰 + CLI 툴만 추가하면 됨.
```

---

## 1. 수정 — `src/sim/octant.py`

`unit_color_diff()` 함수 바로 **아래**에 다음 함수를 추가:

```python
def per_pixel_diff_image(img, labels):
    """조립된 센서 QE 이미지 → 단위셀 픽셀 하나하나의 field-diff 맵.

    사용자 요구: 4×4 단위셀이면 픽셀 16개 각각에 대해 "센서 전체(field)에서
    그 픽셀의 동컬러 대비 편차 맵"이 나와야 한다 (Gr/Gb 통계 하나로 뭉개지
    말 것). 물리: field 위치의 주광선이 렌즈 중심 쪽으로 기울어 초점 스팟이
    동컬러 2×2 그룹의 중심이 아니라 센서중심 쪽 코너로 치우침 → 그룹 내
    코너별 신호 차이 (예: 우상단 field 에서 좌하 > 우하=좌상 > 우상).

    입력  img   : assemble_image() 결과 (H,W)=(npx·rows, npx·cols).
                  타일(I,J)=field, 타일 안 (r,c)=단위셀 픽셀 — 그래서
                  픽셀 (r,c) 의 field 맵은 단순 스트라이드 img[r::npx, c::npx].
          labels: npx×npx CFA 라벨 (assemble 과 동일한 것)
    diff% = (QE_rc − μ_ch) / μ_ch × 100,
            μ_ch = 그 field 타일에서 같은 채널(refine_channels: R/Gr/Gb/B —
            tetra 는 동컬러 2×2 quad) 픽셀들의 평균. NaN 타일은 NaN 유지.
    반환: (diff[npx,npx,rows,cols], 채널격자 npx×npx list)
    """
    A = np.asarray(img, float)
    L = refine_channels(labels)
    npx = L.shape[0]
    rows, cols = A.shape[0] // npx, A.shape[1] // npx
    sub = np.empty((npx, npx, rows, cols))
    for r in range(npx):
        for c in range(npx):
            sub[r, c] = A[r::npx, c::npx]
    out = np.full_like(sub, np.nan)
    for ch in {str(v) for v in L.flat}:
        idx = [(int(r), int(c)) for r, c in np.argwhere(L == ch)]
        mu = np.nanmean(np.stack([sub[r, c] for r, c in idx]), axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            for r, c in idx:
                out[r, c] = np.where(np.abs(mu) > 1e-12,
                                     (sub[r, c] - mu) / mu * 100.0, np.nan)
    return out, [[str(v) for v in row] for row in L]
```

## 2. 수정 — `src/api/server.py` (`_run_image_job` 내부)

앵커: `diff = _same_color_diff(units, labels, waves, ...)` 줄 **바로 아래**에 추가:

```python
        # ── 픽셀별 diff 맵 — 단위셀 픽셀(npx×npx) 각각의 field-diff 맵.
        #    diff% = (픽셀QE − 동채널(quad) 평균)/평균 ×100, 파장별 + 평균.
        from ..sim.octant import per_pixel_diff_image
        pix_maps = {}
        ch_grid = None
        for w in list(waves) + ["mean"]:
            src = mean if w == "mean" else images[int(w)]
            pd, ch_grid = per_pixel_diff_image(src, labels)
            npx = len(ch_grid)
            pix_maps[str(w) if w == "mean" else str(int(w))] = \
                [[[[None if not np.isfinite(v) else round(float(v), 3)
                    for v in rw] for rw in pd[r][c]]
                  for c in range(npx)] for r in range(npx)]
        pixel_diff = {"maps": pix_maps, "channels": ch_grid,
                      "note": "diff% = (픽셀QE − 동채널 quad 평균)/평균 ×100 · "
                              "maps[λ][r][c] = 픽셀(r,c)의 field 맵(rows×cols)"}
```

그리고 결과 dict(res) 저장부의

```python
               "images": {str(int(w)): j2(images[int(w)]) for w in waves},
               "mean": j2(mean), "diff": diff}
```

를 아래로 교체 (pixel_diff 추가):

```python
               "images": {str(int(w)): j2(images[int(w)]) for w in waves},
               "mean": j2(mean), "diff": diff, "pixel_diff": pixel_diff}
```

## 3. 수정 — `editors/structure_wizard.html` (사광·이미지 페이지)

### 3-1. 보기 셀렉터에 옵션 추가

앵커(기존):
```html
<label class="hint">보기 <select id="imgViewSel"><option value="qe" selected>QE 맵</option><option value="diff">동컬러 diff 맵</option></select></label>
```
교체:
```html
<label class="hint">보기 <select id="imgViewSel"><option value="qe" selected>QE 맵</option><option value="diff">동컬러 diff 맵</option><option value="pix">픽셀별 diff 모자이크</option></select></label>
```

### 3-2. `function imgDraw(){` 정의 **바로 위**에 렌더러 추가 + imgDraw 분기

```js
// ── 픽셀별 diff 모자이크 — 단위셀 픽셀(4×4=16개) 각각의 field-diff 맵을
//    단위셀 배열 그대로 4×4 로 배열. 소맵 하나 = 센서 전체(field) 좌표,
//    색 = 그 픽셀의 동채널(quad) 평균 대비 편차%. 물리: 주광선이 렌즈중심
//    쪽으로 기울어 스팟이 그룹의 센서중심쪽 코너로 치우침 → 코너별 차이.
function imgPixColor(t){ // t∈[-1,1] → 파랑-흰-빨강 (diverging)
  const L=(a,b,f)=>[a[0]+(b[0]-a[0])*f,a[1]+(b[1]-a[1])*f,a[2]+(b[2]-a[2])*f];
  const b=[33,102,172], w=[247,247,247], r=[178,24,43];
  return t<0? L(w,b,Math.min(1,-t)) : L(w,r,Math.min(1,t)); }
function imgDrawPixMosaic(){
  const r=IMG.res, P=r&&r.pixel_diff; if(!P||!P.maps) return;
  const sel=el("imgWlSel").value, M=P.maps[sel]||P.maps["mean"]; if(!M) return;
  const npx=M.length, rows=M[0][0].length, cols=M[0][0][0].length;
  let amax=0;                                            // 대칭 스케일
  for(let a=0;a<npx;a++)for(let b=0;b<npx;b++)for(const rw of M[a][b])
    for(const v of rw) if(v!=null&&Math.abs(v)>amax) amax=Math.abs(v);
  if(!(amax>0)) amax=1e-9;
  const cv=el("imgCv"), ctx=cv.getContext("2d");
  ctx.clearRect(0,0,cv.width,cv.height);
  const CC={R:"#d62728",Gr:"#2ca02c",Gb:"#1f9e63",G:"#2ca02c",B:"#1f77b4"};
  const gap=8, head=16;
  const tw=(cv.width-gap*(npx+1))/npx, th=(cv.height-gap*(npx+1)-head*npx)/npx;
  const sc=Math.min(tw/cols, th/rows);
  for(let a=0;a<npx;a++)for(let b=0;b<npx;b++){
    const X0=gap+b*(tw+gap), Y0=gap+a*(th+head+gap)+head;
    const dw=cols*sc, dh=rows*sc, ox=X0+(tw-dw)/2, oy=Y0+(th-dh)/2;
    for(let i=0;i<rows;i++)for(let j=0;j<cols;j++){
      const v=M[a][b][i][j];
      if(v==null){ ctx.fillStyle="rgba(128,128,128,.25)"; }
      else{ const c=imgPixColor(v/amax);
        ctx.fillStyle=`rgb(${c[0]|0},${c[1]|0},${c[2]|0})`; }
      ctx.fillRect(ox+j*sc, oy+i*sc, Math.ceil(sc), Math.ceil(sc)); }
    const ch=(P.channels&&P.channels[a]&&P.channels[a][b])||"";
    ctx.strokeStyle=CC[ch]||"#888"; ctx.lineWidth=2.5;
    ctx.strokeRect(ox-1.5, oy-1.5, dw+3, dh+3);
    ctx.fillStyle=CC[ch]||"#888"; ctx.font="600 11px var(--mono,monospace)";
    ctx.textAlign="center"; ctx.textBaseline="alphabetic";
    ctx.fillText(`(${a},${b}) ${ch}`, X0+tw/2, Y0-4); }
  el("imgScale").textContent=`픽셀별 diff 모자이크 · 소맵=센서 전체 field(${cols}×${rows})`
    +` · 파랑 −${amax.toFixed(1)}% ~ 빨강 +${amax.toFixed(1)}% (동채널 quad 평균 대비)`
    +` · ${sel==="mean"?"파장 평균":"λ "+sel+"nm"}`;
}
```

`function imgDraw(){` 첫 부분(기존):
```js
function imgDraw(){
  const r=IMG.res; if(!r) return;
  if(el("imgViewSel").value==="diff"){ imgDrawDiff(); return; }
```
교체:
```js
function imgDraw(){
  const r=IMG.res; if(!r) return;
  if(el("imgViewSel").value==="pix"){ imgDrawPixMosaic(); return; }
  if(el("imgViewSel").value==="diff"){ imgDrawDiff(); return; }
```

## 4. 신설 — `tools/pixel_diff_mosaic.py` (전문)

```python
# -*- coding: utf-8 -*-
"""픽셀별 diff 모자이크 PNG — 사광 이미지 잡 결과(_image.json)에서 생성.

    PYTHONPATH=. python3 tools/pixel_diff_mosaic.py out/jobs/<jid>_image.json [λ|mean] [out.png]

단위셀 픽셀(npx×npx) 각각의 "센서 전체(field) diff 맵"을 단위셀 배열 그대로
npx×npx 모자이크로 그린다. 소맵 색 = 그 픽셀의 동채널(quad) 평균 대비 편차%.
결과 json 에 pixel_diff 가 없으면(구버전 잡) images/labels 로 즉석 재계산한다.
"""
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np                                             # noqa: E402


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    path = sys.argv[1]
    sel = sys.argv[2] if len(sys.argv) > 2 else "mean"
    out = sys.argv[3] if len(sys.argv) > 3 else \
        os.path.splitext(path)[0] + f"_pixdiff_{sel}.png"
    res = json.load(open(path, encoding="utf-8"))

    if res.get("pixel_diff", {}).get("maps"):
        maps = res["pixel_diff"]["maps"]
        key = sel if sel in maps else "mean"
        M = np.array([[np.array(m, float) for m in row] for row in maps[key]],
                     dtype=float)                      # (npx,npx,rows,cols) — None→nan
        M = np.where(np.isfinite(M), M, np.nan)
        ch = res["pixel_diff"]["channels"]
    else:                                              # 구버전 잡 — 즉석 계산
        from src.sim.octant import per_pixel_diff_image
        img = res["mean"] if sel == "mean" else res["images"][sel]
        A = np.array([[np.nan if v is None else v for v in row] for row in img])
        M, ch = per_pixel_diff_image(A, res["labels"])

    npx = len(ch)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    CC = {"R": "#d62728", "Gr": "#2ca02c", "Gb": "#1f9e63",
          "G": "#2ca02c", "B": "#1f77b4"}
    amax = float(np.nanmax(np.abs(M))) or 1e-9
    fig, axes = plt.subplots(npx, npx, figsize=(2.1 * npx, 1.9 * npx))
    im = None
    for r in range(npx):
        for c in range(npx):
            ax = axes[r][c]
            im = ax.imshow(M[r][c], cmap="RdBu_r", vmin=-amax, vmax=amax,
                           interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            col = CC.get(str(ch[r][c]), "#888")
            for sp in ax.spines.values():
                sp.set_edgecolor(col); sp.set_linewidth(2.2)
            ax.set_title(f"({r},{c}) {ch[r][c]}", fontsize=8, color=col, pad=2)
    fig.suptitle(f"per-pixel ch.diff mosaic  [{sel}]  "
                 f"(map = full sensor field, +-{amax:.1f}%)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 0.92, 0.96))
    cax = fig.add_axes([0.94, 0.15, 0.02, 0.65])
    fig.colorbar(im, cax=cax, label="diff (%)")
    fig.savefig(out, dpi=140)
    print("saved:", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

## 5. 검증 절차

1) 위저드 → 🌈 사광·이미지 시뮬 → 실행 (제품 CRA yaml 필요: `data/cra/<product>.yaml`)
2) 완료 후 보기 셀렉터에서 **"픽셀별 diff 모자이크"** 선택 → 16장 소맵이
   단위셀 배열대로 나오는지, 파장 셀렉터(λ/평균) 전환이 되는지 확인
3) CLI 로도 동일 결과:
   ```bash
   PYTHONPATH=. python3 tools/pixel_diff_mosaic.py out/jobs/<jid>_image.json mean
   ```
4) 빠른 기능 데모(코드 검증용 거친 설정 — 값 자체는 판정용 아님):
   field_step 0.2, nG 41, downsample 4, theta/phi_samples 1, 파장 3개

## 6. 해석 가이드 + 주의사항

- **정상 물리 패턴**: 그룹내 "좌하" 픽셀 소맵은 센서 우상단에서 +(빨강),
  좌하단에서 −(파랑). 대각 코너 픽셀끼리는 부호 반전. 중심 field ≈ 0,
  가장자리로 갈수록 진해짐(CRA∝상고). shrink 보정이 완벽하면 전부 0.
- **field_step 은 x_max(0.8)·y_max(0.6)를 나눠떨어지게** (0.1/0.2 권장).
  0.4 처럼 안 나눠떨어지면 가장자리 행이 NaN(회색)으로 남는다.
- 거친 설정(nG 41, 주광선만)에서는 구조 기인 노이즈가 코너 규칙 위에 얹혀
  얼룩덜룩하게 보임 — **판정용 실행은 nG 101+, theta/phi_samples 3×3(F# 원뿔
  적분), field_step 0.1** 로. 계산량 = 필드수 × 파장 × 원뿔점 × 2편광.
- diff 의 기준은 refine_channels 채널(R/Gr/Gb/B) = tetra 의 동컬러 2×2 quad.
  bayer 모자이크(RGGB)에서도 같은 코드가 동작 (Gr/Gb 픽셀 단위 분리).
- 기존 "동컬러 diff 맵"(채널 통계)은 그대로 유지 — 이 기능은 추가 뷰.
- 이미 계산해둔 구버전 잡(_image.json 에 pixel_diff 없음)도 CLI 툴이
  images/labels 에서 즉석 재계산해 그려준다 (재실행 불필요).

## 7. 수정 파일 요약

| 파일 | 변경 |
|---|---|
| `src/sim/octant.py` | `per_pixel_diff_image()` 추가 |
| `src/api/server.py` | `_run_image_job` 에 pixel_diff 계산 + res 에 포함 |
| `editors/structure_wizard.html` | 보기 옵션 "픽셀별 diff 모자이크" + 렌더러 |
| `tools/pixel_diff_mosaic.py` | 신설 (CLI PNG 생성) |
| 엔진(RCWA)/기존 diff 로직 | 무변경 |
