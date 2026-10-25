"""The linking method: candidates, evidence weights, calibration, status, reconciliation."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from .parsing import WN_TYPE, dist_m, lease_score, split_name, vintage_ok, wilson, wn_match

FIELDS = ["distance", "well number", "lease name", "spud year", "depth",
          "same-number rivals", "unnumbered closer", "dummy number", "twin within 30 m"]
D_BANDS = [(0, 20), (20, 50), (50, 100), (100, 200), (200, 300), (300, 500), (500, 1001)]
SMOOTH = 0.5


class Settings:
    """Thresholds; every value can be overridden from the config file."""
    max_assign_m = 1000
    unnumbered_m = 300
    twin_m = 30
    lease_match = 85
    lease_contra = 50
    p_high = 0.95
    p_med = 0.85
    permit_words = ("PERMIT", "CANCEL", "NEW", "NEVER DRILLED")
    dummy_from = 80000
    vintage_bands = []
    county_codes = {}

    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


def lvl_dist(d):
    for a, b in D_BANDS:
        if a <= d < b:
            return f"{a}-{b if b < 1001 else 1000} m"
    return "1000 m+"


def lvl_wn(leg, st, t):
    if not leg or not st:
        return "missing"
    if t is None:
        return "contradicts"
    if t == 1:
        return "exact, No. 1" if (leg.strip().lstrip("0") or "0") == "1" else "exact, other number"
    return WN_TYPE[t]


def lvl_lease(ls):
    if ls is None:
        return "missing"
    return ">=85" if ls >= 85 else "70-85" if ls >= 70 else "50-70" if ls >= 50 else "<50"


def lvl_year(dy):
    return "missing" if dy is None else ("<=1 yr" if dy <= 1 else ">1 yr")


def lvl_td(rel):
    return "missing" if rel is None else ("<=5%" if rel <= 0.05 else "5-15%" if rel <= 0.15 else ">15%")


def candidates(w, recs, S: Settings):
    """Plausible candidates for legacy record `w` (a Series) with their comparison levels."""
    cc = S.county_codes.get(w.county)
    out, unnum = [], []
    for r in recs or []:
        if r.get("lat") is None:
            continue
        d = dist_m(w.lat, w.lon, r["lat"], r["lon"])
        a = r.get("api10")
        if not a:
            unnum.append(d); continue
        if d > S.max_assign_m:
            continue
        if cc and a[2:5] != cc:
            continue
        if not vintage_ok(w.year, int(a[5:10]), S.vintage_bands, S.dummy_from):
            continue
        if any(x in str(r.get("status")).upper() for x in S.permit_words):
            continue
        t = wn_match(w.wn, r["well_no"]) if (w.wn and r.get("well_no")) else None
        clease = r.get("lease") or split_name(r.get("name"))[0]
        ls = lease_score(w.lease, clease) if (w.lease and clease) else None
        dy = abs(r["spud_year"] - w.year) if (r.get("spud_year") and not pd.isna(w.year)) else None
        rel = (abs(r["td"] - w.td) / w.td if (r.get("td") and not pd.isna(w.td) and w.td > 0) else None)
        out.append({"api10": a, "d": d, "well_no": r.get("well_no") or "", "name": r.get("name") or "",
                    "lat": r["lat"], "lon": r["lon"], "t": t,
                    "distance": lvl_dist(d), "well number": lvl_wn(w.wn, r.get("well_no"), t),
                    "lease name": lvl_lease(ls), "spud year": lvl_year(dy), "depth": lvl_td(rel),
                    "dummy number": "yes" if int(a[5:10]) >= (S.dummy_from or 10**9) else "no"})
    for c in out:
        n = sum(1 for o in out if o is not c and o["t"] is not None and o["well number"] != "missing")
        c["same-number rivals"] = "0" if n == 0 else ("1" if n == 1 else "2+")
        c["unnumbered closer"] = "yes" if any(u < c["d"] and u <= S.unnumbered_m for u in unnum) else "no"
        c["twin within 30 m"] = "yes" if any(
            o is not c and o["well number"] != "contradicts"
            and dist_m(c["lat"], c["lon"], o["lat"], o["lon"]) <= S.twin_m for o in out) else "no"
    return out, any(u <= S.unnumbered_m for u in unnum)


def fit_weights(pairs: pd.DataFrame) -> dict:
    """log2(m/u) per field level from labelled pairs (tuning set only)."""
    Wt = {}
    T, F = pairs[pairs.match], pairs[~pairs.match]
    for f in FIELDS:
        levels = sorted(pairs[f].unique())
        m = T[f].value_counts().reindex(levels, fill_value=0) + SMOOTH
        u = F[f].value_counts().reindex(levels, fill_value=0) + SMOOTH
        Wt[f] = np.log2((m / m.sum()) / (u / u.sum())).to_dict()
        Wt[f]["missing"] = 0.0
    return Wt


def _points(c, Wt):
    return sum(Wt[f].get(c[f], 0.0) for f in FIELDS)


def best_two(cands, Wt):
    for c in cands:
        c["score"] = _points(c, Wt)
    cands = sorted(cands, key=lambda c: -c["score"])
    s1 = cands[0]["score"] if cands else np.nan
    s2 = cands[1]["score"] if len(cands) > 1 else s1 - 20
    return cands, s1, s1 - s2


def fit(legacy: pd.DataFrame, cache: dict, S: Settings):
    """Weights and calibration from legacy records that carry a known API (`label`)."""
    rows, X, y = [], [], []
    for _, w in legacy.iterrows():
        cands, _ = candidates(w, cache.get(w.key), S)
        for c in cands:
            rows.append({"match": c["api10"] == w.label, **{f: c[f] for f in FIELDS}})
    pairs = pd.DataFrame(rows)
    Wt = fit_weights(pairs)
    for _, w in legacy.iterrows():
        cands, _ = candidates(w, cache.get(w.key), S)
        if not cands:
            continue
        cands, s1, gap = best_two(cands, Wt)
        X.append([s1, min(gap, 20)]); y.append(cands[0]["api10"] == w.label)
    y = np.array(y, dtype=bool)
    if y.all() or not y.any():                      # one class only: calibration cannot be estimated
        raise ValueError("every best candidate on the tuning set is " + ("correct" if y.all() else "wrong")
                         + "; the tuning set is too easy or too small to calibrate")
    cal = LogisticRegression(C=1e6, max_iter=1000).fit(np.array(X), y)
    return {"weights": Wt, "calibration": {"coef": cal.coef_[0].tolist(), "intercept": float(cal.intercept_[0])},
            "pairs": int(len(pairs)), "wells": int(len(legacy))}


def _prob(model, s1, gap):
    z = model["calibration"]["intercept"] + model["calibration"]["coef"][0] * s1 \
        + model["calibration"]["coef"][1] * min(gap, 20)
    return float(1 / (1 + np.exp(-np.clip(z, -40, 40))))


def score(w, recs, model, S: Settings) -> dict:
    """One legacy record -> best candidate, probability, status, reason. Never reads a label."""
    cands, near_unnum = candidates(w, recs, S)
    out = {"api": None, "p": np.nan, "tier": "", "status": "unresolved", "reason": "",
           "d": np.nan, "well_no_check": "", "points": "", "n_cand": len(cands), "cand_apis": []}
    if not cands:
        out["reason"] = ("coverage: unnumbered record at location" if near_unnum
                         else "coverage: no plausible record within assignment limit")
        return out
    Wt = model["weights"]
    cands, s1, gap = best_two(cands, Wt)
    b = cands[0]
    p = _prob(model, s1, gap)
    t_txt = WN_TYPE.get(b["t"], "contradicts" if b["t"] is None else "?")
    out.update(api=b["api10"], p=p, d=b["d"], cand_apis=[c["api10"] for c in cands],
               well_no_check=f"{w.wn} -> {b['well_no']} ({t_txt})" if (w.wn and b["well_no"]) else "",
               points="; ".join(f"{f} {b[f]} {Wt[f].get(b[f], 0):+.1f}" for f in FIELDS if b[f] != "missing"))
    contra = b["well number"] == "contradicts" or b["lease name"] == "<50"
    if p >= S.p_high and not contra:
        out.update(tier="HIGH", status="assigned")
    elif p >= S.p_med:
        out.update(tier="MEDIUM", status="provisional")
    elif b["unnumbered closer"] == "yes":
        out["reason"] = "coverage: unnumbered record closer than the best candidate"
    elif contra:
        out["reason"] = "ambiguity: best candidate contradicts the record"
    elif gap < 2:
        out["reason"] = "ambiguity: several candidates score alike"
    else:
        out["reason"] = "ambiguity: evidence too weak"
    return out


def run(legacy: pd.DataFrame, cache: dict, model, S: Settings) -> pd.DataFrame:
    rows = []
    for _, w in legacy.iterrows():
        o = score(w, cache.get(w.key), model, S)
        o["key"] = w.key
        if "label" in legacy.columns and pd.notna(w.get("label")):
            o["label"] = w.label
            o["correct"] = (o["api"] == w.label) if o["status"] != "unresolved" else np.nan
            o["label_among_candidates"] = w.label in o["cand_apis"]
        rows.append(o)
    return pd.DataFrame(rows).set_index("key")


def evaluate(res: pd.DataFrame) -> str:
    """Accuracy by status, calibration bands and reasons, for records with a label."""
    lines = []
    for tier, status in (("HIGH", "assigned"), ("MEDIUM", "provisional")):
        g = res[res.tier == tier]; k = int(g.correct.sum()) if len(g) else 0
        lo, hi = wilson(k, len(g))
        lines.append(f"{tier:<7}{status:<12}{len(g):>6} links  " + (f"{k/len(g)*100:5.1f}%  [{lo:.1f}, {hi:.1f}]" if len(g) else ""))
    L = res[res.status != "unresolved"]; k = int(L.correct.sum()) if len(L) else 0
    lo, hi = wilson(k, len(L))
    lines.append(f"LINKED {len(L)} of {len(res)}  coverage {len(L)/max(len(res),1)*100:.1f}%  "
                 f"accuracy {k/max(len(L),1)*100:.1f}%  [{lo:.1f}, {hi:.1f}]")
    U = res[res.status == "unresolved"]
    if len(U):
        lines.append("unresolved: " + ", ".join(f"{r} {c}" for r, c in U.reason.str.split(":").str[0].value_counts().items()))
    cb = res[res.p.notna()].copy()
    if len(cb):
        cb["right"] = cb.api == cb.label
        cb["band"] = pd.cut(cb.p, [0, .5, .7, .85, .95, 1.0001], right=False,
                            labels=["<50%", "50-70%", "70-85%", "85-95%", ">=95%"])
        tab = cb.groupby("band", observed=True).agg(wells=("p", "size"), predicted=("p", "mean"), observed=("right", "mean"))
        lines.append("calibration (predicted vs observed, %):")
        lines.append((tab * [1, 100, 100]).round(1).to_string())
    if "label_among_candidates" in res and len(U):
        lines.append("unresolved records whose label is among the candidates:")
        lines.append(pd.crosstab(U.reason, U.label_among_candidates).to_string())
    return "\n".join(lines)


def safeguards(res: pd.DataFrame, legacy: pd.DataFrame, existing_apis: dict) -> pd.DataFrame:
    """One-to-one assignment: refuse an API already held by another record, or picked twice."""
    res = res.copy()
    name_of = legacy.set_index("key").well_name if "well_name" in legacy else None

    def same_well(i, j):
        return name_of is not None and split_name(name_of[i]) == split_name(name_of[j])

    def refuse(ix, why):
        res.loc[ix, ["tier", "status", "reason"]] = ["", "unresolved", why]
        res.loc[ix, ["api", "p", "d"]] = [None, np.nan, np.nan]

    for i, r in res[res.api.notna()].iterrows():
        j = existing_apis.get(r.api)
        if j is not None and j != i:
            refuse(i, f"safeguard: {'duplicate row' if same_well(i, j) else 'conflict'}: API already held by {j}")
    for a, g in res[res.api.notna()].groupby("api"):
        if len(g) > 1:
            if all(same_well(g.index[0], k) for k in g.index[1:]):
                for k in g.index[1:]:
                    refuse(k, f"safeguard: duplicate row: same well as {g.index[0]}")
            else:
                refuse(g.index, "safeguard: conflict: same API picked for " + ", ".join(map(str, g.index)))
    return res


def reconcile(results: dict) -> pd.DataFrame:
    """One status per record from several sources. Sources that agree keep the best status
    and the highest probability; sources that disagree leave the record unresolved."""
    RANK = {"HIGH": 2, "MEDIUM": 1}
    keys = sorted(set().union(*[set(r.index) for r in results.values()]))
    rows = []
    for k in keys:
        links = [(src, r.at[k, "api"], r.at[k, "tier"], r.at[k, "p"], r.at[k, "d"])
                 for src, r in results.items() if k in r.index and r.at[k, "status"] != "unresolved"]
        apis = {a for _, a, *_ in links}
        base = next(r.loc[k] for r in results.values() if k in r.index)
        if not links:
            rows.append({"key": k, "api": None, "tier": "", "status": "unresolved", "p": np.nan,
                         "source": "", "reason": base.reason})
        elif len(apis) > 1:
            rows.append({"key": k, "api": None, "tier": "", "status": "unresolved", "p": np.nan,
                         "source": "", "reason": "ambiguity: sources disagree (" + "; ".join(f"{s} {a}" for s, a, *_ in links) + ")"})
        else:
            best = max(links, key=lambda x: (RANK[x[2]], x[3]))
            rows.append({"key": k, "api": best[1], "tier": best[2],
                         "status": "assigned" if best[2] == "HIGH" else "provisional",
                         "p": max(x[3] for x in links), "source": " + ".join(s for s, *_ in links), "reason": ""})
    return pd.DataFrame(rows).set_index("key")
