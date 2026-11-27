"""Propiedades psicometricas de cada variante de HOMEPOS, por pais x ola (media simple de paises).

- fiabilidad WLE: rho = 1 - media(SE^2) / var(WLE)   (ponderado con w_fstuwt)
- r(indice, nota) media de los 10 PV; r con HISEI y PARED
- r desatenuada = r / sqrt(rho): la correlacion del CONSTRUCTO con la nota
- descomposicion del cambio: ln r = ln r_constructo + 0,5 ln rho
  -> d ln r = d ln r_constructo (contenido) + 0,5 d ln rho (fiabilidad)
- hueco predicho con normalidad bivariante: 2,54 * sd(Y) * r  (y su cambio), frente al observado
- reclasificacion: % de alumnos cuyo cuartil difiere del de nuestro indice (GRM comunes)
"""
import sys, glob, warnings, numpy as np, pandas as pd
warnings.filterwarnings('ignore')
sys.path.insert(0, "code")

OLAS = [int(x) for x in sys.argv[1].split(",")] if len(sys.argv) > 1 else [2022, 2025]
pref = "-".join(str(o)[2:] for o in OLAS)
DOM = sys.argv[2] if len(sys.argv) > 2 else "math"
tags = sorted(f.split("hp_")[1][:-8] for f in glob.glob(f"data/metodos/hp_{pref}_*.parquet"))
print(len(tags), "variantes")

it = pd.read_parquet("data/metodos/items.parquet", columns=["wave", "cnt", "cntstuid", "w_fstuwt", "homepos", "hisei",
                                                            "paredint"] + [f"pv{i}{DOM}" for i in range(1, 11)])
it = it[it.wave.isin(OLAS) & (it.cnt != "CRI")]
it["cntstuid"] = it.cntstuid.astype("int64")
it["pared_c"] = it.paredint.replace({3.0: 6.0, 14.5: 14.0})
import os
INTERIM = os.environ.get("PISA_INTERIM", "../data/interim/python")   # main pipeline, step 03
grm = pd.read_parquet(f"{INTERIM}/homepos_irt.parquet").rename(columns={"homepos_irt": "grm"})[["wave", "cnt", "cntstuid", "grm"]]
grm["cntstuid"] = grm.cntstuid.astype("int64")
grm["cntstuid"] = grm.cntstuid.astype("int64")
it = it.merge(grm, on=["wave", "cnt", "cntstuid"], how="left")
PV = [f"pv{i}{DOM}" for i in range(1, 11)]


def wcorr(x, y, w):
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 100:
        return np.nan
    x, y, w = x[m], y[m], w[m]
    mx, my = np.average(x, weights=w), np.average(y, weights=w)
    return np.average((x - mx) * (y - my), weights=w) / np.sqrt(np.average((x - mx) ** 2, weights=w) * np.average((y - my) ** 2, weights=w))


def cuartil(x, w):
    o = np.argsort(x); c = np.cumsum(w[o]) / w.sum()
    q = np.empty(len(x), int); q[o] = np.minimum((c * 4 - 1e-12).astype(int), 3)
    return q


fil = []
for tag in tags + ["pub", "grm"]:
    if tag in ("pub", "grm"):
        d = it.assign(x=it.homepos if tag == "pub" else it.grm, se=np.nan)
    else:
        h = pd.read_parquet(f"data/metodos/hp_{tag}.parquet")
        h["cntstuid"] = h.cntstuid.astype("int64")
        d = it.merge(h[["wave", "cnt", "cntstuid", "wle", "se"]].rename(columns={"wle": "x"}),
                     on=["wave", "cnt", "cntstuid"], how="inner")
    for (cnt, ola), g in d.groupby(["cnt", "wave"]):
        g = g[np.isfinite(g.x)]
        if len(g) < 300:
            continue
        w = g.w_fstuwt.to_numpy(); x = g.x.to_numpy()
        vx = np.average((x - np.average(x, weights=w)) ** 2, weights=w)
        se = g.se.to_numpy()
        rho = 1 - np.average(se ** 2, weights=w) / vx if np.isfinite(se).all() else np.nan
        r_y = np.mean([wcorr(x, g[p].to_numpy(), w) for p in PV])
        sdy = np.mean([np.sqrt(np.average((g[p] - np.average(g[p], weights=w)) ** 2, weights=w)) for p in PV])
        m = np.isfinite(g.grm.to_numpy())
        rec = np.average(cuartil(x[m], w[m]) != cuartil(g.grm.to_numpy()[m], w[m]), weights=w[m]) if m.sum() > 300 else np.nan
        fil.append(dict(variante=tag, cnt=cnt, wave=ola, n=len(g), rho=rho, r_nota=r_y, sd_nota=sdy,
                        r_hisei=wcorr(x, g.hisei.to_numpy(), w), r_pared=wcorr(x, g.pared_c.to_numpy(), w),
                        r_grm=wcorr(x, g.grm.to_numpy(), w), reclasif_vs_grm=rec,
                        n_items_medio=np.nan))
t = pd.DataFrame(fil)
t["r_constructo"] = t.r_nota / np.sqrt(t.rho)
t["hueco_predicho"] = 2.542 * t.sd_nota * t.r_nota
t.to_csv(f"output/tables/metodos/propiedades_pais_{pref}_{DOM}.csv", index=False)

agg = t.groupby(["variante", "wave"])[["rho", "r_nota", "r_constructo", "r_hisei", "r_pared", "r_grm",
                                        "reclasif_vs_grm", "hueco_predicho"]].mean().unstack("wave")
res = pd.DataFrame(index=agg.index)
for c in ["rho", "r_nota", "r_constructo", "r_hisei", "r_pared", "reclasif_vs_grm", "hueco_predicho"]:
    for o in OLAS:
        res[f"{c}_{o}"] = agg[(c, o)]
a, b = OLAS[0], OLAS[-1]
res["d_rho"] = res[f"rho_{b}"] - res[f"rho_{a}"]
res["d_r_nota"] = res[f"r_nota_{b}"] - res[f"r_nota_{a}"]
# descomposicion de d ln r (media de paises de las diferencias de logaritmos)
p = t.pivot_table(index=["variante", "cnt"], columns="wave", values=["r_nota", "rho"])
dl_r = np.log(p[("r_nota", b)]) - np.log(p[("r_nota", a)])
dl_rho = 0.5 * (np.log(p[("rho", b)]) - np.log(p[("rho", a)]))
dec = pd.DataFrame(dict(dln_r=dl_r, parte_fiabilidad=dl_rho, parte_constructo=dl_r - dl_rho)).groupby("variante").mean()
res = res.join(dec)
res["d_hueco_predicho"] = res[f"hueco_predicho_{b}"] - res[f"hueco_predicho_{a}"]
res.to_csv(f"output/tables/metodos/propiedades_{pref}_{DOM}.csv")
with pd.option_context("display.width", 250, "display.max_columns", 40):
    print(res[[f"rho_{a}", f"rho_{b}", "d_rho", f"r_nota_{a}", f"r_nota_{b}", f"r_constructo_{a}", f"r_constructo_{b}",
               f"r_hisei_{a}", f"r_hisei_{b}", "parte_fiabilidad", "parte_constructo", "d_hueco_predicho",
               f"reclasif_vs_grm_{a}", f"reclasif_vs_grm_{b}"]].round(3).to_string())
