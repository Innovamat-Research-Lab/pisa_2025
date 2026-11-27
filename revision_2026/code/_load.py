"""Reads the OECD public-use student files (SPSS) for the revision scripts.

`cargar(ola, ocde=True, vars=[...])` returns one row per student of the requested cycle with the
identifiers, weights, published indices and plausible values used by 100_items_base.py, plus the
extra variables requested in `vars`. Column names are lower case; user-missing codes are read as
missing. The files are looked up in PISA_RAW_DIR (default: <repository>/data/raw), with the same
file names as code/python/scripts/lib/config.py.
"""
import os
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = Path(os.environ.get("PISA_RAW_DIR") or ROOT / "data" / "raw")
SAV = {2015: "CY6_MS_CMB_STU_QQQ.sav", 2018: "CY07_MSU_STU_QQQ.sav",
       2022: "CY08MSP_STU_QQQ.SAV", 2025: "CY09_MS_STU_PUF.sav"}
CORE = (["CNT", "REGION", "CNTSCHID", "CNTSTUID", "W_FSTUWT", "SENWT", "ESCS", "HISEI", "PAREDINT", "HOMEPOS"]
        + [f"PV{i}{d}" for d in ("MATH", "READ", "SCIE") for i in range(1, 11)])


def _find(ola):
    want = SAV[ola].lower()
    for p in RAW.iterdir() if RAW.exists() else []:
        if p.name.lower() == want:
            return p
    raise SystemExit(f"{SAV[ola]} not found in {RAW}. Set PISA_RAW_DIR to the folder with the OECD files.")


def cargar(ola, ocde=True, vars=()):
    import pyreadstat
    path = _find(ola)
    _, meta = pyreadstat.read_sav(str(path), metadataonly=True)
    actual = {n.upper(): n for n in meta.column_names}
    want = list(dict.fromkeys(CORE + [v.upper() for v in vars] + ["OECD"]))
    absent = [c for c in want if c not in actual]
    if absent:
        print(f"  {ola}: not in the file, left missing: {absent}")
    d, _ = pyreadstat.read_sav(str(path), usecols=[actual[c] for c in want if c in actual])
    d.columns = [c.lower() for c in d.columns]
    for c in absent:
        d[c.lower()] = pd.NA
    if ocde is not None:
        d = d[d["oecd"] == (1 if ocde else 0)]
    d = d.drop(columns="oecd").reset_index(drop=True)
    d.insert(0, "wave", ola)
    d["cntstuid"] = pd.to_numeric(d["cntstuid"]).astype("int64")
    for c in ("region", "cntschid"):                  # stored as text in some releases
        d[c] = pd.to_numeric(d[c], errors="coerce")
    if "langtest_qqq" in d:                           # 997-999 are missing codes not always flagged as such
        d["langtest_qqq"] = pd.to_numeric(d["langtest_qqq"], errors="coerce").where(lambda x: x < 997)
    return d
