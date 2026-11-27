"""H5: ¿derivan los items comunes? DIF temporal sin modelo.

Para cada item comun y ola, dentro de cada pais (media simple de paises, OCDE-34 en las cuatro olas):
  r_resto  correlacion del item con la suma estandarizada del resto de items comunes (discriminacion)
  r_hisei  correlacion del item con HISEI (criterio externo, metrica estable)
  r_nota   correlacion del item con matematicas (media de los 10 PV)
  p        proporcion en la categoria mas alta (techo)
Un cambio grande de r_resto o r_hisei entre olas es DIF no uniforme: el item pesa distinto en el
constructo o se relaciona distinto con la riqueza. Ademas, para 2022-25, las discriminaciones de la
calibracion SEPARADA de cada ola (solo comunes) frente a las de la concurrente.
"""
import sys, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, "code"); import _items_homepos as I

d = pd.read_parquet("data/metodos/items.parquet")
p34 = open("data/paises34.txt").read().strip().split(",")
PV = [f"pv{i}math" for i in range(1, 11)]
d["nota"] = d[PV].mean(axis=1)


def wcorr(x, y, w):
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 100:
        return np.nan
    x, y, w = x[m], y[m], w[m]
    mx, my = np.average(x, weights=w), np.average(y, weights=w)
    vx, vy = np.average((x - mx) ** 2, weights=w), np.average((y - my) ** 2, weights=w)
    return np.average((x - mx) * (y - my), weights=w) / np.sqrt(vx * vy) if vx > 0 and vy > 0 else np.nan


fil = []
for olas, paises, nombre in [((2015, 2018, 2022, 2025), p34, "4 olas (13 comunes)"),
                             ((2022, 2025), sorted(set(d[d.wave == 2022].cnt) & set(d[d.wave == 2025].cnt) - {"CRI"}),
                              "2022-25 (16 comunes)")]:
    com = I.comunes(olas)
    for ola in olas:
        g0 = d[(d.wave == ola) & d.cnt.isin(paises)]
        for cnt, g in g0.groupby("cnt"):
            Z = g[com].to_numpy("float64")
            Zs = (Z - np.nanmean(Z, axis=0)) / np.nanstd(Z, axis=0)
            w = g.w_fstuwt.to_numpy()
            for j, it in enumerate(com):
                resto = np.nanmean(np.delete(Zs, j, axis=1), axis=1)
                x = Z[:, j]
                if np.isfinite(x).sum() < 100:
                    continue
                fil.append(dict(ventana=nombre, ola=ola, cnt=cnt, item=it,
                                r_resto=wcorr(x, resto, w), r_hisei=wcorr(x, g.hisei.to_numpy("float64"), w),
                                r_nota=wcorr(x, g.nota.to_numpy("float64"), w),
                                p_techo=np.average(x[np.isfinite(x)] == np.nanmax(x), weights=w[np.isfinite(x)])))
t = pd.DataFrame(fil)
t.to_csv("output/tables/metodos/dif_comunes_pais.csv", index=False)
a = t.groupby(["ventana", "item", "ola"])[["r_resto", "r_hisei", "r_nota", "p_techo"]].mean().unstack("ola")
a.to_csv("output/tables/metodos/dif_comunes.csv")
for v in a.index.get_level_values(0).unique():
    x = a.loc[v]
    print(f"\n=== {v}")
    with pd.option_context("display.width", 250):
        print(x.round(3).to_string())
    olas = sorted(x.columns.get_level_values(1).unique())
    for c in ["r_resto", "r_hisei"]:
        dd = (x[(c, olas[-1])] - x[(c, olas[0])])
        print(f"  cambio {c} {olas[0]}->{olas[-1]}: medio {dd.mean():+.3f}; mayores: " +
              ", ".join(f"{k} {v:+.3f}" for k, v in dd.sort_values().iloc[[0, 1, -2, -1]].items()))

# discriminaciones: calibracion separada (solo comunes) 2022 vs 2025
import glob, pickle
f = "data/metodos/modelo_22-25_comunes_gpcm_pi0_separada.pkl"
try:
    m = pickle.load(open(f, "rb"))
    A = {o: {m["items"][st["j"]]: st["a"] for st in mm.sets} for o, mm in m["mods"].items()}
    r = pd.DataFrame(A)
    r["ratio"] = r[2025] / r[2022]
    r["ratio_rel"] = r.ratio / np.exp(np.log(r.ratio).mean())       # sin el cambio comun de escala
    print("\n=== discriminaciones, calibracion separada de cada ola (16 comunes)")
    print(r.round(3).sort_values("ratio_rel").to_string())
    r.to_csv("output/tables/metodos/dif_comunes_discriminaciones.csv")
except FileNotFoundError:
    print("(falta la calibracion separada de comunes)")

# ---- que miden los items ESPECIFICOS de cada ola (2022-25): mismas correlaciones, dentro de pais
paises = sorted(set(d[d.wave == 2022].cnt) & set(d[d.wave == 2025].cnt) - {"CRI"})
com = I.comunes((2022, 2025))
fil = []
for ola in (2022, 2025):
    esp = [s for s in I.slots_de(ola) if s not in com]
    g0 = d[(d.wave == ola) & d.cnt.isin(paises)]
    for cnt, g in g0.groupby("cnt"):
        Zc = g[com].to_numpy("float64")
        nucleo = np.nanmean((Zc - np.nanmean(Zc, axis=0)) / np.nanstd(Zc, axis=0), axis=1)
        w = g.w_fstuwt.to_numpy()
        for it in esp + com:
            x = g[it].to_numpy("float64")
            if np.isfinite(x).sum() < 200 or np.nanstd(x) == 0:
                continue
            fil.append(dict(ola=ola, cnt=cnt, item=it, comun=it in com, r_nucleo=wcorr(x, nucleo, w),
                            r_hisei=wcorr(x, g.hisei.to_numpy("float64"), w), r_nota=wcorr(x, g.nota.to_numpy("float64"), w),
                            n=int(np.isfinite(x).sum())))
e = pd.DataFrame(fil)
s = e.groupby(["ola", "comun", "item"]).agg(paises=("cnt", "size"), r_nucleo=("r_nucleo", "mean"),
                                            r_hisei=("r_hisei", "mean"), r_nota=("r_nota", "mean")).reset_index()
s.to_csv("output/tables/metodos/items_especificos.csv", index=False)
print("\n=== items especificos de cada ola vs comunes (media de paises)")
print(s.groupby(["ola", "comun"])[["r_nucleo", "r_hisei", "r_nota"]].mean().round(3).to_string())
with pd.option_context("display.width", 200):
    print(s[~s.comun].sort_values(["ola", "r_nota"], ascending=[True, False]).round(3).to_string(index=False))
