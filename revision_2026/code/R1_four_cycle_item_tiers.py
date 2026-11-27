"""Revision November 2026, check R1. Four-cycle series (2015-2025) by comparability level of the items.

The paper's four-cycle harmonized HOMEPOS (step 07) uses the thirteen items asked in every cycle.
They are not equally comparable. This check classifies them in three levels and rebuilds the
four-cycle HOMEPOS and ESCS with the same procedure as step 07 on each nested set:

  Level 1, identical      : same question and same response categories in every cycle
                            room, cars, bathrooms, musical instruments                       (4 items)
  Level 2, exact nesting  : same question; 2022/25 categories split the 2015/18 ones and are
                            collapsed back without loss
                            books, televisions, tablets, e-book readers                     (+4 = 8)
  Level 3, wording/format : the question itself changed
                            computer for schoolwork (+ tablets), educational software (+ apps),
                            internet (excl. smartphones), works of art (yes/no -> count),
                            number of computers (one count -> desktops + laptops)            (+5 = 13)

Sets: "L123" (the paper's thirteen, used to validate against Table A.1), "L12" and "L1".
Minimum number of valid items to be scored: ceil(0.6 x items) (8 of 13 as in step 07; 5 of 8; 3 of 4).
Everything else is step 07 unchanged: 2PL/GPCM, one calibration for the four pooled cycles with
every country-by-cycle cell weighted equally, WLE scores, OECD composite rule, 34 systems,
equal-weight quartiles recomputed in each of the 80 BRR replicates, ten plausible values.

Run from the repository root with PISA_REPO pointing to a folder whose data/extract holds the extract:
    PISA_REPO=<scratch folder> python revision_2026/code/R1_four_cycle_item_tiers.py
Output: revision_2026/output/tables/R1_four_cycle_item_tiers.csv
"""
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / "code" / "python" / "scripts"))      # the paper's library, unchanged

import numpy as np
import pandas as pd

from lib import brr
from lib import config as C
from lib import irt
from lib.escs_rule import escs
from lib.io_utils import load_table
from lib.stats import cell_weights, zscore

OUT = HERE.parent / "output" / "tables"
OUT.mkdir(parents=True, exist_ok=True)
YEARS = (2015, 2018, 2022, 2025)

# identical to ITEMS13 of step 07
ITEMS13 = {"room": ("st011q02ta", "st250q01ja", "bin"), "computer": ("st011q04ta", "st250q02ja", "bin"),
           "software": ("st011q05ta", "st250q03ja", "bin"), "internet": ("st011q06ta", "st250q05ja", "bin"),
           "art": ("st011q09ta", "st251q07ja", "bin_from_count"),
           "instruments": ("st012q09na", "st251q06ja", "count4"), "cars": ("st012q02ta", "st251q01ja", "count4"),
           "bathrooms": ("st012q03ta", "st251q03ja", "count4"),
           "televisions": ("st012q01ta", "st254q01ja", "band3"), "tablets": ("st012q07na", "st254q04ja", "band3"),
           "ereaders": ("st012q08na", "st254q05ja", "band3"),
           "computers": ("st012q06na", ("st254q02ja", "st254q03ja"), "any_computer"),
           "books": ("books6", "st255q01ja", "books6")}
LEVEL = {"room": 1, "cars": 1, "bathrooms": 1, "instruments": 1,
         "books": 2, "televisions": 2, "tablets": 2, "ereaders": 2,
         "computer": 3, "software": 3, "internet": 3, "art": 3, "computers": 3}
SETS = {"L123": [k for k in ITEMS13], "L12": [k for k in ITEMS13 if LEVEL[k] <= 2],
        "L1": [k for k in ITEMS13 if LEVEL[k] == 1]}


def old_band(x):
    return x.where(x.between(1, 4)).map({1: 0.0, 2: 1.0, 3: 1.0, 4: 2.0})


def new_band(x):
    return x.where(x.between(1, 4)).map({1: 0.0, 2: 1.0, 3: 2.0, 4: 2.0})


def recode(g, year):          # step 07, unchanged
    out = pd.DataFrame(index=g.index)
    for name, (old, new, kind) in ITEMS13.items():
        if year <= 2018:
            x = g[old]
            if kind in ("bin", "bin_from_count"):
                out[name] = x.where(x.isin([1, 2])).map({1: 1.0, 2: 0.0})
            elif kind == "count4":
                out[name] = x.where(x.between(1, 4)) - 1
            elif kind == "band3":
                out[name] = old_band(x)
            elif kind == "any_computer":
                out[name] = (x.where(x.between(1, 4)) >= 2).astype(float).where(x.notna())
            else:
                out[name] = x.where(x.between(1, 6)) - 1
        else:
            if kind == "bin":
                out[name] = g[new].where(g[new].isin([1, 2])).map({1: 1.0, 2: 0.0})
            elif kind == "bin_from_count":
                out[name] = (g[new].where(g[new].between(1, 4)) > 1).astype(float).where(g[new].between(1, 4))
            elif kind == "count4":
                out[name] = g[new].where(g[new].between(1, 4)) - 1
            elif kind == "band3":
                out[name] = new_band(g[new])
            elif kind == "any_computer":
                a, b = (g[v].where(g[v].between(1, 4)) for v in new)
                out[name] = ((a >= 2) | (b >= 2)).astype(float).where(g[new[0]].notna() | g[new[1]].notna())
            else:
                out[name] = (g[new].where(g[new].between(1, 7)) - 1).clip(lower=1) - 1
    return out


PV = [f"pv{i}{dom}" for dom in C.DOMAINS for i in range(1, 11)]
common = set.intersection(*[set(load_table(f"stu_{y}", ["cnt"]).cnt) for y in YEARS]) - set(C.EXCLUDE)
parts, rows = [], []
for y in YEARS:
    cols = ["wave", "cnt", "cntstuid", "w_fstuwt", "escs", "homepos", "hisei", "paredint"] + PV
    if y <= 2018:
        cols.append("hisced")
    src = sorted({v for old, new, _ in ITEMS13.values() for v in ((old,) if y <= 2018 else (new if isinstance(new, tuple) else (new,)))})
    g = load_table(f"stu_{y}", cols + src)
    keep = g.cnt.isin(common).to_numpy()
    g = g[keep].reset_index(drop=True)
    parts.append(pd.concat([g[cols], recode(g, y)], axis=1))
    rows.append(np.flatnonzero(keep))
d = pd.concat(parts, ignore_index=True)
del parts, g
d["src_row"] = np.concatenate(rows)
WTS = {y: np.load(C.EXTRACT / f"wts_{y}.npy", mmap_mode="r") for y in YEARS}
print(f"{len(common)} OECD systems present in {YEARS}; N = {len(d):,}", flush=True)

m18 = d[(d.wave == 2018) & d.hisced.notna() & d.paredint.notna()]
mapping = m18.groupby("hisced").paredint.agg(lambda s: s.mode().iloc[0]).to_dict()
is15 = d.wave == 2015
d.loc[is15, "paredint"] = d.loc[is15, "hisced"].map(mapping)
d["pared_c"] = d.paredint.replace({3.0: 6.0, 14.5: 14.0})
w = cell_weights(d)
group = d.groupby(["cnt", "wave"], sort=False).ngroup().to_numpy()

IDX = {"escs": "ESCS published", "homepos": "HOMEPOS published", "hisei": "HISEI", "pared_c": "PARED (common scale)"}
for tag, names in SETS.items():
    X = np.full((len(d), len(names)), -1, dtype=np.int8)
    for j, c in enumerate(names):
        v = d[c].to_numpy("float64")
        X[np.isfinite(v), j] = v[np.isfinite(v)].astype(np.int8)
        k = math.ceil(0.6 * len(names))
    theta, A, B = irt.fit_and_score(X, w / w.sum() * len(d), min_items=k, verbose=False)
    d[f"homepos_{tag}"] = zscore(theta, w)
    RAW = np.column_stack([d.hisei.to_numpy("float64"), d.pared_c.to_numpy("float64"), theta])
    d[f"escs_{tag}"], _, _ = escs(RAW, group, w)
    IDX[f"homepos_{tag}"] = f"HOMEPOS harmonized, {tag} ({len(names)} items, min {k})"
    IDX[f"escs_{tag}"] = f"ESCS harmonized, {tag} ({len(names)} items)"
    r = [d[d.wave == y][["homepos", f"homepos_{tag}"]].corr().iloc[0, 1] for y in YEARS]
    print(f"{tag}: {len(names)} items, scored {np.isfinite(theta).mean():.1%}; r with published HOMEPOS "
          + ", ".join(f"{y} {x:.3f}" for y, x in zip(YEARS, r)), flush=True)


def weights(g):
    out = np.empty((len(g), 1 + C.N_REP))
    out[:, 0] = g.w_fstuwt.to_numpy("float64")
    out[:, 1:] = WTS[int(g.wave.iloc[0])][g.src_row.to_numpy()]
    return out


res = []
for dom in C.DOMAINS:
    PVc = [f"pv{i}{dom}" for i in range(1, 11)]
    TH = {}
    for (cnt, wave), g in d.groupby(["cnt", "wave"], sort=True):
        P, W = g[PVc].to_numpy("float64"), weights(g)
        ok = np.isfinite(P).all(axis=1)
        for idx in IDX:
            m = ok & g[idx].notna().to_numpy()
            if m.sum() >= 200:
                TH[(idx, wave, cnt)] = brr.theta_quartiles(g[idx].to_numpy("float64")[m], W[m], P[m])
    for idx, label in IDX.items():
        cnts = sorted(set.intersection(*[{k[2] for k in TH if k[0] == idx and k[1] == y} for y in YEARS]))
        avg = {y: np.mean([TH[(idx, y, c)] for c in cnts], axis=0) for y in YEARS}
        r = {"domain": dom, "index": idx, "label": label, "n_systems": len(cnts)}
        for y in YEARS:
            r[f"gap{y}"], r[f"se{y}"] = brr.contrast(avg[y], [-1, 0, 0, 1])
        for a, b in [(2022, 2025), (2015, 2025), (2018, 2022), (2015, 2018)]:
            r[f"d{a}_{b}"] = r[f"gap{b}"] - r[f"gap{a}"]
            r[f"se_d{a}_{b}"] = float(np.hypot(r[f"se{a}"], r[f"se{b}"]))
        res.append(r)
    t = pd.DataFrame(res)
    print(f"\n=== {dom} ===\n" + t[t.domain == dom][["label", "n_systems", "gap2015", "gap2018", "gap2022", "gap2025",
                                                    "d2022_2025", "d2015_2025"]].round(1).to_string(index=False), flush=True)
pd.DataFrame(res).to_csv(OUT / "R1_four_cycle_item_tiers.csv", index=False)
print("saved", OUT / "R1_four_cycle_item_tiers.csv")
