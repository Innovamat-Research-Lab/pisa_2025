"""Huecos Q4-Q1 por cuartiles de cada variante de HOMEPOS y del ESCS que la lleva dentro.

    python code/103_huecos.py --olas 2022 2025 --dom math --tags <tag> <tag> ...

Indices de referencia que se anaden solos:
  pub        HOMEPOS y ESCS publicados (cada ola en su escala)
  grm        el nuestro: GRM concurrente de items comunes (16 en 22-25 -> homepos_h; 13 en 4 olas -> homepos_h4)
  tend_ocde  tendencia oficial de la OCDE 2022 (solo olas <= 2022): HOMEPOS/ESCS de 2015-18 reescalados
             a 2022 con parametros fijos; 2022 = publicado
El ESCS de cada variante se construye con la regla de la OCDE (_escs.py) sobre HISEI y PARED comunes,
estandarizando en el pool de las olas y paises de la ventana. Cuartiles de peso igual dentro de pais x
ola, recalculados en cada replica; 80 BRR x 10 PV; media simple de paises.
"""
import os, sys, argparse, numpy as np, pandas as pd
sys.path.insert(0, "code"); import _brr as B, _escs as E

ap = argparse.ArgumentParser()
ap.add_argument("--olas", type=int, nargs="+", default=[2022, 2025])
ap.add_argument("--dom", default="math")
ap.add_argument("--tags", nargs="*", default=[])
ap.add_argument("--sin_escs", action="store_true")
ap.add_argument("--salida", default="")
A = ap.parse_args()
OLAS = sorted(A.olas)
cuatro = OLAS != [2022, 2025]

it = pd.read_parquet("data/metodos/items.parquet",
                     columns=["wave", "cnt", "cntstuid", "w_fstuwt", "homepos", "escs", "homepos_trend", "escs_trend",
                              "hisei", "paredint"] + [f"pv{i}{A.dom}" for i in range(1, 11)])
it = it[it.wave.isin(OLAS)]
# Reference index "grm": the common-item HOMEPOS of the main pipeline (code/python/scripts/03_irt_homepos.py),
# read from its interim output. It also fixes the student sample of the 2022-2025 window.
INTERIM = os.environ.get("PISA_INTERIM", "../data/interim/python")
if cuatro:
    raise SystemExit("Four-cycle windows: see code/python/scripts/07_four_cycle_series.py and code/R1_four_cycle_item_tiers.py")
else:
    ref = pd.read_parquet(f"{INTERIM}/homepos_irt.parquet").rename(columns={"homepos_irt": "grm"})
ref["cntstuid"] = ref.cntstuid.astype("int64")
d = it.merge(ref, on=["wave", "cnt", "cntstuid"], how="inner", validate="1:1")
if not cuatro:
    d["pared_c"] = d.paredint.replace({3.0: 6.0, 14.5: 14.0})
d = d[d.cnt != "CRI"]
for t in A.tags:
    h = pd.read_parquet(f"data/metodos/hp_{t}.parquet")[["wave", "cnt", "cntstuid", "wle"]]
    h["cntstuid"] = h.cntstuid.astype("int64")
    d = d.merge(h.rename(columns={"wle": t}), on=["wave", "cnt", "cntstuid"], how="left", validate="1:1")
paises = sorted(set.intersection(*[set(d[d.wave == o].cnt) for o in OLAS]))
d = d[d.cnt.isin(paises)].reset_index(drop=True)
print(f"ventana {OLAS}, {A.dom}: {len(paises)} paises, {len(d):,} alumnos")

hp = ["pub", "grm"] + list(A.tags)
d["pub"] = d.homepos
if max(OLAS) <= 2022:
    hp.insert(2, "tend_ocde"); d["tend_ocde"] = d.homepos_trend
# ESCS de cada variante (regla OCDE, pool de la ventana)
w = E.senate(d); grupo = d.groupby(["cnt", "wave"], sort=False).ngroup().to_numpy()
indices = {}
for h in hp:
    indices[f"HOMEPOS|{h}"] = h
    if A.sin_escs:
        continue
    if h == "pub":
        d["ESCS|pub"] = d.escs
    elif h == "tend_ocde":
        d["ESCS|tend_ocde"] = d.escs_trend
    else:
        X = np.column_stack([d.hisei.to_numpy("float64"), d.pared_c.to_numpy("float64"), d[h].to_numpy("float64")])
        d[f"ESCS|{h}"] = E.escs(X, grupo, w)[0]
    indices[f"ESCS|{h}"] = f"ESCS|{h}"
if not A.sin_escs:
    for comp in ("hisei", "pared_c"):
        indices[f"{comp.upper()}|-"] = comp

PVc = [f"pv{i}{A.dom}" for i in range(1, 11)]
W_ALL = B.matriz_pesos(d)
TH = {}
for (cnt, ola), g in d.groupby(["cnt", "wave"], sort=True):
    pos = g.index.to_numpy()
    W, PV = W_ALL[pos], g[PVc].to_numpy("float64")
    fin = np.isfinite(PV).all(axis=1)
    for nom, col in indices.items():
        x = g[col].to_numpy("float64")
        m = fin & np.isfinite(x)
        if m.sum() < 200:
            continue
        if col == "pared_c":       # PARED tiene masa en el maximo: cuartiles con desempate aleatorio
            pass
        TH[(nom, cnt, ola)] = B.theta_cuartiles(x[m], W[m], PV[m])

c = np.array([-1, 0, 0, 1.0])
fil, pais = [], []
for nom in indices:
    ps = [p for p in paises if all((nom, p, o) in TH for o in OLAS)]
    if not ps:
        continue
    gaps = {o: np.mean([np.einsum("k,krm->rm", c, TH[(nom, p, o)]) for p in ps], axis=0)[None] for o in OLAS}
    q = {o: np.mean([TH[(nom, p, o)] for p in ps], axis=0) for o in OLAS}
    r = dict(indice=nom.split("|")[0], variante=nom.split("|")[1], dom=A.dom, n_paises=len(ps))
    for o in OLAS:
        e, s, _, _ = B.resume(gaps[o]); r[f"gap_{o}"], r[f"ee_{o}"] = e[0], s[0]
        eq, sq, _, _ = B.resume(q[o])
        for k in range(4):
            r[f"Q{k+1}_{o}"] = eq[k]
    pares = list(zip(OLAS[:-1], OLAS[1:])) + ([(OLAS[0], OLAS[-1])] if len(OLAS) > 2 else [])
    for a, b in pares:
        r[f"d_{a}_{b}"] = r[f"gap_{b}"] - r[f"gap_{a}"]
        r[f"ee_d_{a}_{b}"] = float(np.hypot(r[f"ee_{a}"], r[f"ee_{b}"]))
        for k in range(4):
            r[f"dQ{k+1}_{a}_{b}"] = r[f"Q{k+1}_{b}"] - r[f"Q{k+1}_{a}"]
        # paises con cambio significativo del hueco
        sig_n = sig_p = 0
        for p in ps:
            ga = B.resume(np.einsum("k,krm->rm", c, TH[(nom, p, a)])[None])
            gb = B.resume(np.einsum("k,krm->rm", c, TH[(nom, p, b)])[None])
            dd, ee = gb[0][0] - ga[0][0], np.hypot(ga[1][0], gb[1][0])
            sig_n += dd / ee < -1.96; sig_p += dd / ee > 1.96
            pais.append(dict(indice=r["indice"], variante=r["variante"], dom=A.dom, cnt=p, a=a, b=b,
                             gap_a=ga[0][0], gap_b=gb[0][0], d=dd, ee=ee))
        r[f"sig_neg_{a}_{b}"], r[f"sig_pos_{a}_{b}"] = sig_n, sig_p
    fil.append(r)

t = pd.DataFrame(fil)
suf = A.salida or f"{'-'.join(str(o)[2:] for o in OLAS)}_{A.dom}"
t.to_csv(f"output/tables/metodos/huecos_{suf}.csv", index=False)
pd.DataFrame(pais).to_csv(f"output/tables/metodos/huecos_pais_{suf}.csv", index=False)
cols = ["indice", "variante"] + [f"gap_{o}" for o in OLAS] + [c for c in t.columns if c.startswith("d_")] + \
       [c for c in t.columns if c.startswith("ee_d_")][:1] + [c for c in t.columns if c.startswith("sig_neg")][:1]
with pd.option_context("display.width", 250, "display.max_columns", 30):
    print(t[cols].round(2).to_string(index=False))
