# -*- coding: utf-8 -*-
"""surrogate 영속 캐시 (DB화) — 구조·조건이 동일하면 재계산 없이 즉시 재사용.

핵심 아이디어: DOE surrogate 는 (구조 + DOE 조건 + 실제 사용 물질 n,k) 로
'완전히 결정'된다. 이 셋을 정규화해 sha256 해시(=상태키)를 만들고, 결과를
    <cache_dir>/<key>.json   (surrogate + csv + meta)
로 저장한다. 같은 키가 이미 있으면 RCWA 를 아예 돌리지 않는다 →
같은 구조를 반복해서 돌리는 불필요한 리소스 낭비 제거.

키에 물질 지문(n,k 내용 해시)까지 넣으므로, 물질을 수정하면 키가 달라져
자동으로 재계산된다(잘못된 캐시 히트 방지). ENGINE 을 올리면 전체 무효화.
"""
import hashlib
import json
import os
import time

ENGINE = "sur-cache/v1"          # 계산 로직 버전 — 엔진 규약 바뀌면 올려서 캐시 무효화


def _canon(obj):
    return json.dumps(obj, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"), default=str)


def _sha(s):
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _collect_strings(obj, out):
    """cfg 안의 모든 문자열 토큰 수집 (물질 이름 참조 탐지용)."""
    if isinstance(obj, str):
        out.add(obj)
    elif isinstance(obj, dict):
        for v in obj.values():
            _collect_strings(v, out)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _collect_strings(v, out)


def referenced_materials(cfg, materials_dir):
    """이 구조가 실제로 참조하는 물질 파일만 골라낸다.

    ① cfg['materials'][role].src (없으면 role 이름) ② ambient
    ③ cfg 어디든 등장하는 문자열 중 폴더에 같은 이름 .txt 가 있는 것
    셋의 합집합 ∩ 실제 존재 파일.
    """
    try:
        avail = {fn[:-4] for fn in os.listdir(materials_dir)
                 if fn.lower().endswith(".txt")}
    except OSError:
        avail = set()
    used = set()
    mats = (cfg.get("materials") or {})
    for role, spec in mats.items():
        src = (spec or {}).get("src") if isinstance(spec, dict) else None
        used.add(src or role)
    if cfg.get("ambient"):
        used.add(cfg["ambient"])
    toks = set()
    _collect_strings(cfg, toks)
    used |= (toks & avail)
    return sorted(n for n in used if n in avail)


def materials_fingerprint(cfg, materials_dir):
    """참조 물질들의 (이름, 내용 sha8) 목록 — n,k 가 바뀌면 지문도 바뀐다."""
    fp = []
    for name in referenced_materials(cfg, materials_dir):
        path = os.path.join(materials_dir, name + ".txt")
        try:
            with open(path, "rb") as f:
                fp.append([name, _sha(f.read().decode("utf-8", "replace"))[:12]])
        except OSError:
            fp.append([name, "missing"])
    return fp


def compute_key(cfg, conds, axes, materials_dir):
    """상태키 = sha256(정규화 구조 + 조건 + 축정의 + 물질지문 + 엔진버전).

    conds: {waves:[nm...], mode, nG, downsample, lateral_n}
    axes:  DOE 축 정의 (이름·스텝·단위) — 축이 바뀌면 surrogate 의미가 달라짐.
    """
    payload = {
        "engine": ENGINE,
        "structure": cfg,                          # _auto_model_defaults 반영 후 dict
        "axes": [[a[0], round(float(a[2]), 6), a[3]] for a in axes],
        "waves": [int(w) for w in conds["waves"]],
        "mode": conds["mode"],
        "nG": int(conds["nG"]),
        "downsample": int(conds["downsample"]),
        "lateral_n": int(conds.get("lateral_n", 256)),
        "materials": materials_fingerprint(cfg, materials_dir),
    }
    return _sha(_canon(payload))


def _path(cache_dir, key):
    return os.path.join(cache_dir, key + ".json")


def load(cache_dir, key):
    """캐시 히트면 {surrogate, csv, meta, ...} dict, 아니면 None."""
    p = _path(cache_dir, key)
    if not os.path.isfile(p):
        return None
    try:
        with open(p, encoding="utf-8") as f:
            rec = json.load(f)
        if rec.get("engine") != ENGINE:      # 엔진 규약 불일치 → 무효
            return None
        return rec
    except (OSError, ValueError):
        return None


def save(cache_dir, key, surrogate, meta, csv_text="", created=None):
    """surrogate + csv + meta 를 캐시에 저장하고 index 갱신. created=시각(주입식)."""
    os.makedirs(cache_dir, exist_ok=True)
    rec = {"engine": ENGINE, "key": key,
           "created": created if created is not None else _now(),
           "meta": meta, "surrogate": surrogate, "csv": csv_text}
    tmp = _path(cache_dir, key) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(rec, f, ensure_ascii=False)
    os.replace(tmp, _path(cache_dir, key))
    _update_index(cache_dir, key, meta, rec["created"])
    return rec


def _now():
    try:
        return int(time.time())
    except Exception:
        return 0


def _index_path(cache_dir):
    return os.path.join(cache_dir, "index.json")


def _update_index(cache_dir, key, meta, created):
    idx = _read_index(cache_dir)
    idx = [e for e in idx if e.get("key") != key]
    entry = {"key": key, "created": created}
    entry.update(meta or {})
    idx.append(entry)
    idx.sort(key=lambda e: -e.get("created", 0))
    try:
        with open(_index_path(cache_dir), "w", encoding="utf-8") as f:
            json.dump(idx, f, ensure_ascii=False)
    except OSError:
        pass


def _read_index(cache_dir):
    try:
        with open(_index_path(cache_dir), encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return []


def index(cache_dir):
    """저장된 surrogate 목록 (최신순). 파일이 실제 존재하는 항목만."""
    out = []
    for e in _read_index(cache_dir):
        if os.path.isfile(_path(cache_dir, e.get("key", ""))):
            out.append(e)
    return out
