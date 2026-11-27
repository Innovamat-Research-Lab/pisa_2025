"""Validacion cruzada del motor GPCM/PCM multigrupo contra mirt (R), en la misma submuestra.

Modelo: 16 items comunes 2022-25, dos grupos (olas), parametros de item invariantes, media y varianza
libres en 2025 (2022 ~ N(0,1) en mirt). Se compara: discriminaciones y umbrales (tras llevar nuestra
escala a la de mirt), diferencia de medias latentes en DT de 2022, y WLE alumno a alumno. Lo mismo con
PCM (a = 1). Y una segunda prueba con items propios de cada ola (missing by design), que es la
estructura del metodo Avvisati sin invarianza parcial.
"""
import sys, subprocess, numpy as np, pandas as pd
sys.path.insert(0, "code"); import _gpcm_mg as G, _items_homepos as I

d = pd.read_parquet("data/metodos/items.parquet")
PAISES = ["AUS", "DEU", "ESP", "FRA", "JPN", "MEX", "POL", "USA"]
rng = np.random.default_rng(1)
d = d[d.wave.isin([2022, 2025]) & d.cnt.isin(PAISES)]
d = d.groupby(["cnt", "wave"]).sample(n=1500, random_state=1)
d = d.reset_index(drop=True)
com = I.comunes([2022, 2025])
uni = ["classic_lit", "dictionary", "art_books", "mopeds"] + ["pool", "dishwasher", "alarm", "guest_room"]
out = {}
for prueba, items in [("comunes", com), ("con_unicos", com + uni)]:
    Xdf = d[items].copy()
    Xdf = Xdf.loc[:, Xdf.notna().sum() > 200]
    items = list(Xdf.columns)
    Xdf.assign(wave=d.wave.values).to_csv(f"data/metodos/mirt_{prueba}.csv", index=False, na_rep="NA")
    X = Xdf.fillna(-1).to_numpy().astype(int)
    K = [int(Xdf[c].max()) + 1 for c in items]
    gi = (d.wave == 2025).astype(int).to_numpy()
    for modelo in ("gpcm", "pcm"):
        mod = G.Modelo(items, K, 2, pcm=(modelo == "pcm"))
        for j in range(len(items)):
            s = mod.nuevo_set(j, etiqueta=items[j])
            mod.M[j, [g for g in (0, 1) if (X[gi == g, j] >= 0).any()]] = s
        dat = G.Datos(X, gi, np.ones(len(X)))
        G.em(mod, dat, log=None, tol=1e-5, max_iter=2000)
        # a la escala de mirt: grupo 2022 con media 0; en GPCM ademas DT 1 (en Rasch la escala la fija a = 1)
        m0, s0 = mod.mu[0], (mod.sd[0] if modelo == "gpcm" else 1.0)
        A = np.array([st["a"] * s0 for st in mod.sets])
        Bs = [(st["b"] - m0) / s0 for st in mod.sets]
        wle = G.puntua(mod, dat, "wle")
        out[(prueba, modelo)] = dict(items=items, a=A, b=Bs, dmu=(mod.mu[1] - m0) / s0, sdr=mod.sd[1] / s0, sd2_abs=mod.sd[1],
                                    wle=(wle - m0) / s0)

R = r'''
suppressMessages(library(mirt))
res <- list()
for (prueba in c("comunes","con_unicos")) {
  x <- read.csv(sprintf("data/metodos/mirt_%s.csv", prueba))
  g <- factor(x$wave); x$wave <- NULL
  for (modelo in c("gpcm","pcm")) {
    it <- if (modelo == "gpcm") "gpcm" else "Rasch"
    m <- multipleGroup(x, 1, group = g, itemtype = it, SE = FALSE, verbose = FALSE,
                       invariance = c(colnames(x), "free_means", "free_var"),
                       technical = list(NCYCLES = 3000), TOL = 1e-6)
    cf <- coef(m, simplify = TRUE)
    ip <- cf[[1]]$items
    gp <- cf[[2]]$means; gv <- cf[[2]]$cov
    th <- fscores(m, method = "WLE", full.scores = TRUE, max_theta = 10)
    write.csv(ip, sprintf("data/metodos/mirt_par_%s_%s.csv", prueba, modelo))
    write.csv(data.frame(mu2 = as.numeric(gp), var2 = as.numeric(gv)), sprintf("data/metodos/mirt_grp_%s_%s.csv", prueba, modelo), row.names = FALSE)
    write.csv(data.frame(wle = th[, 1]), sprintf("data/metodos/mirt_wle_%s_%s.csv", prueba, modelo), row.names = FALSE)
  }
}
cat("R_OK\n")
'''
open("data/metodos/mirt_check.R", "w").write(R)
import os
if not os.path.exists("data/metodos/mirt_wle_con_unicos_pcm.csv") or "--rehacer" in sys.argv:
    r = subprocess.run(["Rscript", "data/metodos/mirt_check.R"], capture_output=True, text=True)
    print(r.stdout[-2000:], r.stderr[-3000:])

fil = []
for (prueba, modelo), o in out.items():
    ip = pd.read_csv(f"data/metodos/mirt_par_{prueba}_{modelo}.csv", index_col=0)
    gp = pd.read_csv(f"data/metodos/mirt_grp_{prueba}_{modelo}.csv")
    wm = pd.read_csv(f"data/metodos/mirt_wle_{prueba}_{modelo}.csv").wle.to_numpy()
    a_m = ip["a1"].to_numpy() if "a1" in ip else np.ones(len(ip))
    dcols = [c for c in ip.columns if c.startswith("d")]
    b_m, b_p = [], []
    for j, it in enumerate(o["items"]):
        dd = ip.loc[it, dcols].to_numpy(float)
        dd = dd[np.isfinite(dd)]
        # mirt gpcm: P_k ∝ exp(a k theta + d_k), d_0 = 0  ->  b_k = -(d_k - d_{k-1}) / a
        if modelo == "pcm":
            dd = np.concatenate([[0.0], dd]) if len(dd) == len(o["b"][j]) else dd
            bm = -np.diff(dd)
        else:
            bm = -np.diff(dd) / a_m[j]
        b_m += list(bm); b_p += list(o["b"][j])
    ok = np.isfinite(wm) & np.isfinite(o["wle"]) & (np.abs(wm) < 9) & (np.abs(o["wle"]) < 9)
    fil.append(dict(prueba=prueba, modelo=modelo, n_items=len(o["items"]),
                    r_a=np.corrcoef(a_m, o["a"])[0, 1] if modelo == "gpcm" else np.nan,
                    max_dif_a=np.abs(a_m - o["a"]).max() if modelo == "gpcm" else np.nan,
                    r_b=np.corrcoef(b_m, b_p)[0, 1], max_dif_b=np.abs(np.array(b_m) - np.array(b_p)).max(),
                    dmu_mirt=float(gp.mu2[0]) / np.sqrt(1), dmu_nuestro=o["dmu"],
                    sd2_mirt=float(np.sqrt(gp.var2[0])), sd2_nuestro=o["sdr"] if modelo == "gpcm" else o["sd2_abs"],
                    r_wle=np.corrcoef(wm[ok], o["wle"][ok])[0, 1], mad_wle=np.abs(wm[ok] - o["wle"][ok]).mean()))
t = pd.DataFrame(fil)
print(t.round(4).to_string(index=False))
t.to_csv("output/tables/metodos/107_mirt_check.csv", index=False)
