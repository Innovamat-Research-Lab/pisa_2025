"""Puerta 0: con los parametros PUBLICADOS de la OCDE (TR 2025, hoja STQ del anexo del cap. 22:
internacionales, liberados por pais x lengua y propios de pais) y nuestra puntuacion WLE, ¿se reproduce
el HOMEPOS publicado de 2025? Valida a la vez las recodificaciones, la parametrizacion, la asignacion
de grupos y el WLE, antes de calibrar nada."""
import sys, numpy as np, pandas as pd, openpyxl
sys.path.insert(0, "code"); import _gpcm_mg as G, _items_homepos as I

REF = "../reference/"            # technical2025_ch22_tables.xlsx and iso_language_codes.csv, in the repository
wb = openpyxl.load_workbook(REF + "technical2025_ch22_tables.xlsx", read_only=True)
rows = [r for r in wb["STQ"].iter_rows(values_only=True) if r[1] == "HOMEPOS"]
k = [r for r in wb["T22.A.2"].iter_rows(values_only=True) if r and r[0] == "HOMEPOS"][0]
MEAN, SD = float(k[1]), float(k[2])
iso = pd.read_csv(REF + "iso_language_codes.csv").set_index("alpha3").alpha2.to_dict()
iso.update(dict(esp="es", slo="sk", val="ca"))

VAR2SLOT = {v: s for s, d in I.SLOTS.items() for o, (v, f) in d.items() if o == 2025}
par = {}        # (slot, clave) -> (a, b) ; clave = 'INT' | ('REL', cnt, lang2) | ('NAT', pais)
for r in rows:
    pid = r[2]; item = pid[:10].lower()
    a, b = G.desde_ocde(r[3], r[4], list(r[5:10]))
    if pid.endswith("_R"):
        par[(VAR2SLOT[item], "INT")] = (a, b)
    elif "_R_" in pid:
        pares = [(r[i], r[i + 1]) for i in range(10, len(r) - 1, 2) if r[i]]
        for c, l in pares:
            par[(VAR2SLOT[item], ("REL", c, l.split("-")[0]))] = (a, b)
    else:
        par[(item, ("NAT", pid.split("_R")[-1]))] = (a, b)

d = pd.read_parquet("data/metodos/items.parquet")
d = d[d.wave == 2025].reset_index(drop=True)
d["lang2"] = d.lang.map(iso)
slots = I.slots_de(2025)
nac = {"st251q08ja": "st251d08ja", "st251q09ja": "st251d09ja"}


def pais_nat(cnt, region):
    if cnt == "BEL":
        return {5601: "QBL", 5602: "QBR", 5603: "QBR"}.get(int(region), cnt)
    if cnt == "GBR":
        return "QSC" if int(region) == 82604 else "GBR"
    return cnt


out = np.full(len(d), np.nan)
diag = []
for (cnt, lang2, reg), g in d.groupby(["cnt", "lang2", d.region.fillna(0)], dropna=False):
    A, B, cols = [], [], []
    for s in slots:
        p = par.get((s, ("REL", cnt, lang2)), par[(s, "INT")])
        A.append(p[0]); B.append(p[1]); cols.append(g[s])
    pn = pais_nat(cnt, reg)
    for item, v in nac.items():
        if (item, ("NAT", pn)) in par:
            p = par[(item, ("NAT", pn))]
            A.append(p[0]); B.append(p[1]); cols.append(g[v])
    X = np.column_stack([c.fillna(-1).to_numpy() for c in cols]).astype(int)
    th = G.wle_grupo(X, A, B, min_items=3)
    out[g.index] = (th - MEAN) / SD
d["rec"] = out

ok = d.homepos.notna() & d.rec.notna()
dif = (d.rec - d.homepos)[ok]
print(f"2025, {d.cnt.nunique()} paises OCDE: r(reconstruido, publicado) = {d.loc[ok, ['rec','homepos']].corr().iloc[0,1]:.6f}")
print(f"  |dif| media {dif.abs().mean():.5f}  p99 {dif.abs().quantile(.99):.4f}  max {dif.abs().max():.3f}")
print(f"  cobertura: publicado {d.homepos.notna().mean()*100:.2f}%  reconstruido {d.rec.notna().mean()*100:.2f}%")
t = d[ok].groupby("cnt").apply(lambda x: pd.Series(dict(
        r=x[["rec", "homepos"]].corr().iloc[0, 1], mad=(x.rec - x.homepos).abs().mean(),
        max=(x.rec - x.homepos).abs().max(), lang_na=(x.lang == "NA").mean())), include_groups=False)
print(t.sort_values("r").head(8).round(4).to_string())
t.to_csv("output/tables/metodos/101_valida_oficial_2025.csv")
