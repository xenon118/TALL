"""Command line: fetch -> split -> fit -> evaluate -> apply -> reconcile, driven by a config file."""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import yaml

from . import linker, sources
from .parsing import split_name


def load_config(path):
    with open(path) as fh:
        return yaml.safe_load(fh)


def load_legacy(cfg) -> pd.DataFrame:
    """Legacy table in the common form: key, state, county, well_name, lease, wn, lat, lon, year, td, label."""
    c = cfg["legacy"]["columns"]
    df = pd.read_csv(cfg["legacy"]["path"], dtype=str)
    out = pd.DataFrame({
        "key": df[c["id"]].astype(str),
        "state": df[c["state"]].astype(str).str.upper().str.strip(),
        "county": df[c["county"]].astype(str).str.upper().str.strip(),
        "well_name": df[c["well_name"]].astype(str),
        "lat": pd.to_numeric(df[c["latitude"]], errors="coerce"),
        "lon": pd.to_numeric(df[c["longitude"]], errors="coerce"),
        "year": pd.to_numeric(df[c["completion_year"]], errors="coerce") if c.get("completion_year") else np.nan,
        "td": pd.to_numeric(df[c["total_depth"]], errors="coerce") if c.get("total_depth") else np.nan,
    })
    sp = out.well_name.map(split_name)
    out["lease"], out["wn"] = sp.str[0], sp.str[1]
    if c.get("api"):
        out["label"] = df[c["api"]].map(lambda x: _api10(x))
    else:
        out["label"] = None
    out = out[out.lat.notna() & out.lon.notna()].reset_index(drop=True)
    out["year"] = out.year.where((out.year >= 1800) & (out.year <= 2100))
    return out


def _api10(x):
    import re
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    try:
        d = str(int(float(x)))
    except (TypeError, ValueError):
        d = re.sub(r"\D", "", str(x))
    return d[:10] if len(d) >= 10 else None


def settings_for(cfg, state) -> linker.Settings:
    st = cfg["states"][state]
    s = linker.Settings(**cfg.get("thresholds", {}))
    s.vintage_bands = st.get("vintage_bands", [])
    s.dummy_from = st.get("dummy_from")
    s.county_codes = dict(st.get("county_codes", {}))
    return s


def county_codes_from_labels(legacy: pd.DataFrame, cfg, split: pd.Series):
    """County code table from labelled tuning records only (no test leakage)."""
    for state, st in cfg["states"].items():
        lab = legacy[(legacy.state == state) & legacy.label.notna() & (split.reindex(legacy.key).values == "tune")]
        codes = lab.groupby("county").label.agg(lambda s: s.str[2:5].mode().iloc[0]).to_dict()
        st.setdefault("county_codes", {}).update({k: v for k, v in codes.items() if k not in st["county_codes"]})


def cache_path(cfg, state, source):
    return os.path.join(cfg["workdir"], f"cache_{state}_{source}.json.gz")


def cmd_fetch(cfg, args):
    legacy = load_legacy(cfg)
    os.makedirs(cfg["workdir"], exist_ok=True)
    for state, st in cfg["states"].items():
        pts = legacy[legacy.state == state][["key", "lat", "lon"]]
        for name, src in st["sources"].items():
            path = cache_path(cfg, state, name)
            if os.path.exists(path) and not args.refresh:
                print(f"[{state}/{name}] snapshot exists: {path}"); continue
            unit_words = cfg.get("unitised_words", ["UNIT", "SWD", "DISPOSAL"])
            if src["type"] == "arcgis":
                layers = [L for u in src["urls"] for lu in sources.expand_service(u)
                          if (L := sources.resolve_layer(lu, src["fields"]))]
                if not layers:
                    sys.exit(f"[{state}/{name}] no usable layer")
                cache = sources.fetch_arcgis(pts, layers, st["code"], cfg["radius_m"], unit_words)
                meta = {"type": "arcgis", "layers": [L["url"] for L in layers]}
            elif src["type"] == "csv":
                recs = sources.load_csv_source(src["path"], src["columns"], st["code"], unit_words, layer=name,
                                               one_per_api=src.get("one_per_api", True))
                cache = sources.cache_from_records(pts, recs, cfg["radius_m"])
                meta = {"type": "csv", "path": src["path"], "records": len(recs)}
            else:
                sys.exit(f"unknown source type {src['type']}")
            sources.save_cache(path, cache, {**meta, "radius_m": cfg["radius_m"]})
            print(f"[{state}/{name}] {len(cache):,} points cached -> {path}")


def cmd_split(cfg, args):
    legacy = load_legacy(cfg)
    lab = legacy[legacy.label.notna()].copy()
    rng = np.random.default_rng(cfg.get("seed", 0))
    lab["stratum"] = lab.state + "_" + (lab.year // 10 * 10).fillna(-1).astype(int).astype(str)
    lab["set"] = "tune"
    for _, g in lab.groupby("stratum"):
        idx = rng.permutation(g.index)
        lab.loc[idx[: len(idx) // 2], "set"] = "test"
    path = os.path.join(cfg["workdir"], "split.csv")
    lab[["key", "set"]].to_csv(path, index=False)
    print(f"split written -> {path}: {lab.set.value_counts().to_dict()}")


def _load_split(cfg):
    return pd.read_csv(os.path.join(cfg["workdir"], "split.csv"), dtype=str).set_index("key")["set"]


def cmd_fit(cfg, args):
    legacy = load_legacy(cfg); split = _load_split(cfg)
    county_codes_from_labels(legacy, cfg, split)
    for state, st in cfg["states"].items():
        for name in st["sources"]:
            if args.source and name != args.source:
                continue
            cache, _ = sources.load_cache(cache_path(cfg, state, name))
            tune = legacy[(legacy.state == state) & legacy.label.notna()
                          & (split.reindex(legacy.key).values == "tune")]
            model = linker.fit(tune, cache, settings_for(cfg, state))
            model["county_codes"] = settings_for(cfg, state).county_codes
            path = os.path.join(cfg["workdir"], f"model_{state}_{name}.json")
            json.dump(model, open(path, "w"), indent=1)
            print(f"[{state}/{name}] fitted on {model['wells']} records ({model['pairs']} pairs) -> {path}")
            for f, lv in model["weights"].items():
                print(f"  {f:<20}" + "  ".join(f"{k} {v:+.1f}" for k, v in lv.items() if k != "missing"))


def _model(cfg, state, name, use):
    """Model for a source; `use` names another source whose weights are transferred unchanged."""
    path = os.path.join(cfg["workdir"], f"model_{state}_{use or name}.json")
    m = json.load(open(path))
    return m


def cmd_evaluate(cfg, args):
    legacy = load_legacy(cfg); split = _load_split(cfg)
    for state, st in cfg["states"].items():
        for name in st["sources"]:
            if args.source and name != args.source:
                continue
            cache, _ = sources.load_cache(cache_path(cfg, state, name))
            model = _model(cfg, state, name, args.weights_from)
            S = settings_for(cfg, state); S.county_codes = model.get("county_codes", S.county_codes)
            test = legacy[(legacy.state == state) & legacy.label.notna()
                          & (split.reindex(legacy.key).values == "test")]
            res = linker.run(test, cache, model, S)
            print(f"\n== {state} / {name}" + (f" (weights from {args.weights_from})" if args.weights_from else "") +
                  f": {len(test)} test records ==")
            print(linker.evaluate(res))
            res.drop(columns=["cand_apis"]).to_csv(os.path.join(cfg["workdir"], f"eval_{state}_{name}.csv"))


def cmd_apply(cfg, args):
    legacy = load_legacy(cfg)
    existing = {a: k for k, a in zip(legacy.key, legacy.label) if a}
    for state, st in cfg["states"].items():
        for name in st["sources"]:
            if args.source and name != args.source:
                continue
            cache, _ = sources.load_cache(cache_path(cfg, state, name))
            model = _model(cfg, state, name, args.weights_from)
            S = settings_for(cfg, state); S.county_codes = model.get("county_codes", S.county_codes)
            target = legacy[(legacy.state == state) & legacy.label.isna()]
            res = linker.safeguards(linker.run(target, cache, model, S), legacy, existing)
            path = os.path.join(cfg["workdir"], f"links_{state}_{name}.csv")
            res.drop(columns=["cand_apis"]).to_csv(path)
            lk = res.status != "unresolved"
            print(f"[{state}/{name}] {int(lk.sum())} of {len(res)} records linked "
                  f"(assigned {int((res.tier=='HIGH').sum())}, provisional {int((res.tier=='MEDIUM').sum())}); "
                  f"expected correct ~{res.p[lk].sum():.0f} -> {path}")


def cmd_reconcile(cfg, args):
    legacy = load_legacy(cfg)
    existing = {a: k for k, a in zip(legacy.key, legacy.label) if a}
    frames = []
    for state, st in cfg["states"].items():
        results = {}
        for name in st["sources"]:
            path = os.path.join(cfg["workdir"], f"links_{state}_{name}.csv")
            if os.path.exists(path):
                results[name] = pd.read_csv(path, dtype={"api": str}).set_index("key")
        if results:
            frames.append(linker.safeguards(linker.reconcile(results), legacy, existing))
    final = pd.concat(frames)
    path = os.path.join(cfg["workdir"], "links_final.csv")
    final.to_csv(path)
    lk = final.status != "unresolved"
    print(f"{int(lk.sum())} of {len(final)} records linked; by source: {final.source[lk].value_counts().to_dict()} -> {path}")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="tall", description="Tiered API Linking for Legacy wells")
    ap.add_argument("-c", "--config", default="config.yaml")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("fetch", help="snapshot candidate records around every legacy record"); p.add_argument("--refresh", action="store_true")
    sub.add_parser("split", help="tuning / test split of the labelled records")
    for name in ("fit", "evaluate", "apply"):
        p = sub.add_parser(name); p.add_argument("--source"); p.add_argument("--weights-from")
    sub.add_parser("reconcile", help="one status per record across sources")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    {"fetch": cmd_fetch, "split": cmd_split, "fit": cmd_fit, "evaluate": cmd_evaluate,
     "apply": cmd_apply, "reconcile": cmd_reconcile}[args.cmd](cfg, args)


if __name__ == "__main__":
    main()
