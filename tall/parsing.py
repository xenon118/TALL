"""Name and number parsing, comparison helpers, geometry and statistics."""
from __future__ import annotations

import re
import numpy as np
try:
    from rapidfuzz import fuzz
except ImportError:                                   # slower fallback with the same scores (0-100)
    from difflib import SequenceMatcher

    class fuzz:  # noqa: N801
        @staticmethod
        def token_sort_ratio(a, b):
            a, b = " ".join(sorted(a.split())), " ".join(sorted(b.split()))
            return 100 * SequenceMatcher(None, a, b).ratio()

        @staticmethod
        def token_set_ratio(a, b):
            ta, tb = set(a.split()), set(b.split())
            inter = " ".join(sorted(ta & tb))
            s1, s2 = (inter + " " + " ".join(sorted(ta - tb))).strip(), (inter + " " + " ".join(sorted(tb - ta))).strip()
            r = lambda x, y: 100 * SequenceMatcher(None, x, y).ratio()
            return max(r(inter, s1), r(inter, s2), r(s1, s2)) if inter else r(s1, s2)

M_PER_DEG = 111_320.0

# Words that carry no identifying information inside a lease name.
GENERIC = {"COM", "UNIT", "FEDERAL", "FED", "STATE", "ST", "ET", "AL",
           "ESTATE", "EST", "TRACT", "NCT", "CO", "INC"}

_WN = re.compile(r"(?:\bN[O0]L?\b\.?|\bNA\b\.?|#)\s*([0-9A-Z][0-9A-Z\-\(\)]*)", re.I)
_FOLD = str.maketrans({"O": "0", "I": "1", "L": "1", "S": "5", "Z": "2", "B": "8"})

WN_TYPE = {1: "exact", 2: "letters dropped", 3: "hyphen part", 4: "folded",
           5: "unit renumbered", 6: "letters differ"}


def norm(s) -> str:
    """Upper-case, punctuation removed, generic tokens kept (lease_score drops them)."""
    s = "" if s is None else str(s).upper()
    s = re.sub(r"\b(NO|NUM|NUMBER)\b\.?", " ", s)
    s = re.sub(r"[^A-Z0-9 ]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def split_name(s, placeholders=("ONGARD",)):
    """'Lea-State A No. 4' -> ('LEA STATE A', '4'). Placeholder names give no lease."""
    s = str(s or "")
    m = _WN.search(s)
    lease = norm(s[: m.start()] if m else s)
    if not lease or any(p in lease for p in placeholders):
        lease = None
    return lease, (m.group(1).upper() if m else None)


def _wn_forms(s, state_side=False):
    s = re.sub(r"[^0-9A-Z\-]", "", str(s or "").upper())
    out = {}

    def add(f, t):
        f = f.lstrip("0") or "0"
        if f not in out or t < out[f]:
            out[f] = t

    if not s or not re.search(r"\d", s):
        return out
    flat = s.replace("-", "")
    add(flat, 1)
    parts = re.findall(r"\d+", s)
    if len(parts) == 1:
        add(parts[0], 2)
    elif "-" in s:
        add(parts[0], 3); add(parts[-1], 3)
        add("".join(parts), 3)
        add(parts[0] + "".join(p.zfill(2) for p in parts[1:]), 3)
    folded = re.sub(r"(?<=\d)[OILSZB](?=\d)", lambda m: m.group(0).translate(_FOLD), flat)
    if folded != flat:
        add(folded, 4)
    if state_side and len(parts) == 1 and len(parts[0]) >= 3 and parts[0][-2:] != "00":
        add(parts[0][-2:], 5)
    return out


def wn_match(legacy, state):
    """Best match type (1-6) between two well numbers; None = contradiction."""
    fa, fb = _wn_forms(legacy), _wn_forms(state, state_side=True)
    common = set(fa) & set(fb)
    if not common:
        return None
    t = min(max(fa[f], fb[f]) for f in common)
    la = set(re.sub(r"[^A-Z]", "", str(legacy).upper()))
    lb = set(re.sub(r"[^A-Z]", "", str(state).upper()))
    if t >= 2 and la and lb and la != lb:
        return 6
    return t


def lease_score(a, b) -> float:
    """0-100 token similarity; generic words ignored; differing designators (A vs B) -> 60."""
    ca, cb = set(a.split()) - GENERIC, set(b.split()) - GENERIC
    if not ca or not cb:
        return fuzz.token_sort_ratio(a, b)
    desig = lambda s: {t for t in s if len(t) <= 2 or t.isdigit()}
    if desig(ca - cb) and desig(cb - ca):
        return 60.0
    return fuzz.token_set_ratio(" ".join(sorted(ca)), " ".join(sorted(cb)))


def dist_m(lat1, lon1, lat2, lon2) -> float:
    """Local equirectangular distance in metres (adequate within a few km)."""
    return float(np.hypot(lat2 - lat1, (lon2 - lon1) * np.cos(np.radians(lat1)))) * M_PER_DEG


def wilson(k, n, z=1.96):
    """Wilson score interval for k successes in n, as percentages."""
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0, c - h) * 100, min(1, c + h) * 100)


def api10_from_raw(raw, state_code: str):
    """10-digit API from the digits a source provides. 8 digits -> state code added.
    County-only fragments and all-zero sequences give None; nothing is padded."""
    d = re.sub(r"\D", "", str(raw)) if raw is not None else ""
    if len(d) >= 10:
        a = d[:10]
    elif len(d) == 8:
        a = state_code + d
    else:
        return None
    if a[:2] != state_code or a[5:10] == "00000":
        return None
    return a


def vintage_ok(year, seq: int, bands, exempt_from: int | None) -> bool:
    """Era filter. `bands` is a list of [max_year, max_sequence]; a candidate whose
    sequence exceeds the limit for the legacy record's completion year is excluded.
    Sequences at or above `exempt_from` (e.g. dummy numbers) carry no era information."""
    if year is None or (isinstance(year, float) and np.isnan(year)):
        return True
    if exempt_from is not None and seq >= exempt_from:
        return True
    for max_year, max_seq in bands:
        if year <= max_year:
            return seq < max_seq
    return True
