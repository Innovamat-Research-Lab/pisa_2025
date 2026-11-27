"""Revision November 2026, check R2. The 2022 filter on the device counts (footnote).

In 2022, students who answered "none" to the number of screen devices (ST253) skipped the five
device counts (ST254Q01-Q05); in 2025 ST253 was not asked and every student answered ST254. The
sixteen common items have identical wording and categories in both cycles, but the device counts
are therefore missing by design for those 2022 students. This check reports the size of that group
and rebuilds the 2022-2025 harmonized HOMEPOS and ESCS in two ways:

  "no_devices" : the sixteen common items minus the five device counts (eleven items, min 7 valid);
  "filled"     : the sixteen items, with the device counts of 2022 students filtered out by ST253 set to
                 "none" (what the filter implies), min 10 valid as in step 03.

Everything else is steps 03-05 unchanged. Needs steps 02-04 run first under the same PISA_REPO
(data/interim/python/base.parquet, base_weights.npy, indices_h.parquet).
    PISA_REPO=<scratch folder> python revision_2026/code/R2_device_filter_2022.py
Output: revision_2026/output/tables/R2_device_filter_2022.csv
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / "code" / "python" / "scripts"))

import numpy as np
import pandas as pd

from lib import brr
from lib import config as C
from lib import irt
from lib.escs_rule import escs
from lib.io_utils import load_table
from lib.stats import cell_weights, zscore

OUT = HERE.parent / "output" / "tables"
ITEMS = [c.lower() for c in C.COMMON]
DEV = [c.lower() for c in C.DEV]
BINS = [b.lower() for b in C.BIN]

d = load_table("base", ["wave", "cnt", "cntstuid", "w_fstuwt", "in_sample", "escs", "homepos", "hisei", "paredint",
                        "st253q01ja"] + ITEMS + [f"pv{i}{dm}" for dm in C.DOMAINS for i in range(1, 11)])
h = load_table("indices_h")
s22 = (d.wave == 2022) & (d.in_sample == 1)
none = s22 & (d.st253q01ja == 1)
dev_missing = d[DEV].isna().all(axis=1)
share = [np.average(g.st253q01ja.eq(1), weights=g.w_fstuwt) for _, g in d[s22 & d.st253q01ja.notna()].groupby("cnt")]
print(f"2022: ST253 = none for {np.mean(share):.2%} of students with a valid ST253 (simple average of systems); "
      f"device counts all missing for {dev_missing[none].mean():.1%} of them")


def matrix(frame, items):
    X = np.full((len(frame), len(items)), -1, dtype=np.int8)
    for j, c in enumerate(items):
        v = frame[c].to_numpy("float64")
        k = v if c in BINS else v - 1
        m = np.isfinite(k)
        X[m, j] = k[m].astype(np.int8)
    return X


w = cell_weights(d)
variants = {}
X = matrix(d, [c for c in ITEMS if c not in DEV])
variants["no_devices"] = irt.fit_and_score(X, w / w.sum() * len(d), min_items=7, verbose=False)[0]
f = d.copy()
for c in DEV:
    f.loc[none & f[c].isna(), c] = 1.0          # category 1 = "none"
X = matrix(f, ITEMS)
variants["filled"] = irt.fit_and_score(X, w / w.sum() * len(d), min_items=10, verbose=False)[0]
del f

# as in step 04: the composite and the z-scores use the analysis sample only (its rows come first)
n = int((d.in_sample == 1).sum())
ds = d.iloc[:n]
ws = cell_weights(ds)
group = ds.groupby(["cnt", "wave"], sort=False).ngroup().to_numpy()
pared_c = ds.paredint.replace({3.0: 6.0, 14.5: 14.0}).to_numpy("float64")
IDX = {"escs": ds.escs.to_numpy("float64"), "escs_h": h.escs_h.to_numpy("float64")[:n],
       "homepos_h": h.homepos_h.to_numpy("float64")[:n]}
for tag, theta in variants.items():
    th = theta[:n]
    RAW = np.column_stack([ds.hisei.to_numpy("float64"), pared_c, th])
    IDX[f"escs_{tag}"], _, _ = escs(RAW, group, ws)
    IDX[f"homepos_{tag}"] = zscore(th, ws)
    ok = np.isfinite(th) & np.isfinite(IDX["homepos_h"])
    print(f"{tag}: r with the paper's harmonized HOMEPOS {np.corrcoef(th[ok], IDX['homepos_h'][ok])[0, 1]:.3f}")

W_ALL = np.load(C.INTERIM / "base_weights.npy", mmap_mode="r")
rows = []
for dom in C.DOMAINS:
    PVc = [f"pv{i}{dom}" for i in range(1, 11)]
    TH = {}
    for (cnt, wave), g in ds.groupby(["cnt", "wave"], sort=True):
        pos = g.index.to_numpy()
        W, P = W_ALL[pos], g[PVc].to_numpy("float64")
        ok = np.isfinite(P).all(axis=1)
        for idx, x in IDX.items():
            m = ok & np.isfinite(x[pos])
            if m.sum() >= 200:
                TH[(cnt, wave, idx)] = brr.theta_quartiles(x[pos][m], W[m], P[m])
    for idx in IDX:
        cnts = sorted({k[0] for k in TH if k[2] == idx and (k[0], 2022, idx) in TH and (k[0], 2025, idx) in TH})
        a = np.mean([TH[(p, 2022, idx)] for p in cnts], axis=0)
        b = np.mean([TH[(p, 2025, idx)] for p in cnts], axis=0)
        (g22, t22), (g25, t25) = brr.contrast(a, [-1, 0, 0, 1]), brr.contrast(b, [-1, 0, 0, 1])
        rows.append(dict(domain=dom, index=idx, n_systems=len(cnts), gap2022=g22, gap2025=g25,
                         change=g25 - g22, se=float(np.hypot(t22, t25))))
t = pd.DataFrame(rows)
t.to_csv(OUT / "R2_device_filter_2022.csv", index=False)
print(t.round(2).to_string(index=False))
