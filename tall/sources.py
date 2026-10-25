"""Candidate sources: ArcGIS REST layers, CSV exports, and the snapshot cache.

Every source is normalised to one record format:
    api_raw, api10, name, well_no, operator, lat, lon, status, spud_year, td, lease, layer
`api10` is None when the source gives no usable 10-digit identifier.
"""
from __future__ import annotations

import gzip
import json
import os
import re
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import requests
from scipy.spatial import cKDTree

from .parsing import api10_from_raw, norm

HEADERS = {"User-Agent": "tall/1.0 (+https://github.com/)"}
ROLES = ("api", "name", "well_no", "operator", "lat", "lon", "status", "spud_year", "td")


def _year(v):
    try:
        y = int(str(v)[:4])
        return y if 1900 <= y <= 2100 else None
    except (TypeError, ValueError):
        try:
            y = pd.to_datetime(float(v), unit="ms").year
            return y if 1900 <= y <= 2100 else None
        except Exception:
            return None


def _depth(v):
    try:
        x = float(v)
        return x if 0 < x < 99999 else None
    except (TypeError, ValueError):
        return None


def _lease_from_name(name, well_no, unit_words):
    """Lease name from a 'LEASE NUMBER' well name; unitised names carry no lease."""
    n, k = str(name or "").upper().strip(), str(well_no or "").upper().strip()
    if k and n.endswith(" " + k):
        n = n[: -len(k) - 1]
    n = norm(re.sub(r"\bETAL\b", "ET AL", n))
    if not n or re.search(r"\b(" + "|".join(unit_words) + r")\b", n):
        return None
    return n


def to_record(attrs: dict, roles: dict, state_code: str, layer: str, unit_words) -> dict:
    g = lambda role: attrs.get(roles.get(role)) if roles.get(role) else None
    name, well_no = g("name"), g("well_no")
    if well_no in (None, "") and name:
        m = re.search(r"#\s*([0-9A-Z\-]+)\s*$", str(name).upper())
        well_no = m.group(1) if m else None
    lat, lon = g("lat"), g("lon")
    try:
        lat, lon = float(lat), float(lon)
    except (TypeError, ValueError):
        lat = lon = None
    return {"api_raw": None if g("api") is None else str(g("api")),
            "api10": api10_from_raw(g("api"), state_code),
            "name": None if name in (None, "") else str(name),
            "well_no": None if well_no in (None, "") else str(well_no),
            "operator": g("operator"), "lat": lat, "lon": lon, "status": g("status"),
            "spud_year": _year(g("spud_year")), "td": _depth(g("td")),
            "lease": _lease_from_name(name, well_no, unit_words), "layer": layer}


# --- ArcGIS REST ---------------------------------------------------------------------
def _get(url, params, timeout=40):
    try:
        return requests.get(url, params=params, headers=HEADERS, timeout=timeout).json()
    except Exception:
        return None


def resolve_layer(url: str, role_fields: dict):
    """Layer metadata with field names resolved to roles, or None if unusable."""
    meta = _get(url, {"f": "json"}, timeout=25)
    if not meta or not meta.get("fields"):
        return None
    have = {f["name"].lower(): f["name"] for f in meta["fields"]}
    roles = {r: next((have[c.lower()] for c in role_fields.get(r, []) if c.lower() in have), None)
             for r in ROLES}
    if not roles["api"]:
        return None
    return {"url": url, "name": meta.get("name", url), "roles": roles,
            "max_rec": int(meta.get("maxRecordCount") or 1000)}


def expand_service(source: str):
    if re.search(r"/(MapServer|FeatureServer)/\d+$", source):
        return [source]
    root = _get(source, {"f": "json"}, timeout=25) or {}
    return [f"{source}/{l['id']}" for l in root.get("layers", [])]


def query_box(layer, xmin, ymin, xmax, ymax, depth=0):
    base = {"geometry": f"{xmin},{ymin},{xmax},{ymax}", "geometryType": "esriGeometryEnvelope",
            "inSR": 4326, "outSR": 4326, "spatialRel": "esriSpatialRelIntersects",
            "outFields": ",".join(v for v in layer["roles"].values() if v),
            "returnGeometry": "true", "f": "json"}
    out, offset = [], 0
    while True:
        j = _get(layer["url"] + "/query", {**base, "resultOffset": offset,
                                           "resultRecordCount": layer["max_rec"]})
        if j is None:
            raise RuntimeError("request failed")
        if "error" in j:                                   # no paging: split the box
            j = _get(layer["url"] + "/query", base)
            if j is None or "error" in j:
                raise RuntimeError("request failed")
            feats = j.get("features", [])
            if j.get("exceededTransferLimit") and depth < 4:
                xm, ym = (xmin + xmax) / 2, (ymin + ymax) / 2
                return (query_box(layer, xmin, ymin, xm, ym, depth + 1)
                        + query_box(layer, xm, ymin, xmax, ym, depth + 1)
                        + query_box(layer, xmin, ym, xm, ymax, depth + 1)
                        + query_box(layer, xm, ym, xmax, ymax, depth + 1))
            return feats
        feats = j.get("features", [])
        out += feats
        if not j.get("exceededTransferLimit") or not feats:
            return out
        offset += len(feats)


def fetch_arcgis(points: pd.DataFrame, layers: list, state_code: str, radius_m: float,
                 unit_words, workers: int = 8) -> dict:
    """Records within `radius_m` of every point (columns key, lat, lon), keyed by `key`."""
    cache = {}

    def one(key, lat, lon):
        dlat = radius_m / 111_320.0
        dlon = dlat / max(np.cos(np.radians(lat)), 0.2)
        recs, seen = [], set()
        for L in layers:
            for f in query_box(L, lon - dlon, lat - dlat, lon + dlon, lat + dlat):
                a = f.get("attributes", {}); g = f.get("geometry") or {}
                if a.get(L["roles"]["lat"]) is None and g:
                    a[L["roles"]["lat"] or "__lat"] = g.get("y"); a[L["roles"]["lon"] or "__lon"] = g.get("x")
                roles = dict(L["roles"]); roles.setdefault("lat", "__lat"); roles.setdefault("lon", "__lon")
                r = to_record(a, roles, state_code, L["name"], unit_words)
                sig = (r["api_raw"], round(r["lat"] or 0, 5), round(r["lon"] or 0, 5))
                if sig not in seen:
                    seen.add(sig); recs.append(r)
        return key, recs

    with ThreadPoolExecutor(workers) as ex:
        futs = [ex.submit(one, k, la, lo) for k, la, lo in zip(points.key, points.lat, points.lon)]
        for fut in as_completed(futs):
            key, recs = fut.result()
            cache[key] = recs
    return cache


# --- CSV exports (manual downloads from any provider) ----------------------------------
def load_csv_source(path_or_glob: str, columns: dict, state_code: str, unit_words,
                    layer: str = "csv", one_per_api: bool = True) -> list:
    """Records from one or more CSV files. `columns` maps roles to column names.
    With `one_per_api`, rows that repeat an identifier (one per completion or sidetrack)
    collapse to the first row, matching the wellbore level of regulatory layers."""
    import glob
    files = sorted(glob.glob(path_or_glob)) or [path_or_glob]
    frames = [pd.read_csv(f, dtype=str) for f in files]
    df = pd.concat(frames, ignore_index=True).replace({"-": None, "": None})
    if one_per_api and columns.get("api") in df:
        df = df.drop_duplicates(columns["api"])
    recs = []
    for row in df.to_dict("records"):
        attrs = {v: row.get(v) for v in columns.values() if v}
        r = to_record(attrs, columns, state_code, layer, unit_words)
        if r["lat"] is not None:
            recs.append(r)
    return recs


def cache_from_records(points: pd.DataFrame, records: list, radius_m: float) -> dict:
    """Candidate cache (key -> records within radius) built from a flat record list."""
    lat0 = float(np.mean([r["lat"] for r in records]))
    xy = np.array([[r["lon"] * np.cos(np.radians(lat0)), r["lat"]] for r in records]) * 111_320.0
    tree = cKDTree(xy)
    cache = {}
    for k, la, lo in zip(points.key, points.lat, points.lon):
        idx = tree.query_ball_point([lo * np.cos(np.radians(lat0)) * 111_320.0, la * 111_320.0], radius_m)
        cache[k] = [records[i] for i in idx]
    return cache


# --- snapshot cache ----------------------------------------------------------------------
def save_cache(path: str, cache: dict, meta: dict):
    """Atomic write of a compressed JSON snapshot."""
    d = os.path.dirname(os.path.abspath(path)) or "."
    fd, tmp = tempfile.mkstemp(dir=d, suffix=".tmp")
    with os.fdopen(fd, "wb") as fh, gzip.GzipFile(fileobj=fh, mode="wb") as gz:
        gz.write(json.dumps({"__meta__": meta, **cache}).encode("utf-8"))
    os.replace(tmp, path)


def load_cache(path: str):
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        data = json.load(fh)
    meta = data.pop("__meta__", {})
    return data, meta
