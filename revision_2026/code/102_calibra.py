"""Calibra y puntua una variante de HOMEPOS en una ventana de olas.

    python code/102_calibra.py --olas 2022 2025 --items todos --modelo gpcm --pi 1 --calib concurrente

--items    comunes | todos | sin:<bloque de _items_homepos.BLOQUES> | anclas:<slot,slot,...>
--modelo   gpcm | pcm
--pi       1 = invarianza parcial (liberacion por RMSD > umbral, grupos pais x lengua x ola)
--calib    concurrente   una sola calibracion del pool de olas; los slots comunes comparten parametros
           separada      cada ola se calibra sola con sus items (lo que hizo la OCDE en 2025)
           fpl<ola>      enlace de parametros fijos a <ola> (TR 2022 cap. 19): se calibra <ola> sola;
                         en las demas, los slots compartidos se fijan a sus parametros (los del mismo
                         pais x lengua si estaban liberados), los nuevos se estiman y los que no ajustan se
                         liberan
           anclas        dos pasos (Avvisati y Wuyts 2024, p. 48): anclas concurrentes e invariantes;
                         con ellas fijas, el resto de items se calibra ola a ola
Salida: data/metodos/hp_<tag>.parquet (wle, eap) + output/tables/metodos/params_<tag>.csv y rmsd_<tag>.csv
"""
import sys, argparse, time, json, numpy as np, pandas as pd
sys.path.insert(0, "code"); import _gpcm_mg as G, _items_homepos as I

ap = argparse.ArgumentParser()
ap.add_argument("--olas", type=int, nargs="+", default=[2022, 2025])
ap.add_argument("--items", default="todos")
ap.add_argument("--modelo", default="gpcm")
ap.add_argument("--pi", type=int, default=1)
ap.add_argument("--calib", default="concurrente")
ap.add_argument("--umbral", type=float, default=0.25)
ap.add_argument("--nmin", type=int, default=150)
ap.add_argument("--paises", default="")
ap.add_argument("--tag", default="")
ap.add_argument("--max_iter", type=int, default=300)
ap.add_argument("--amplia22", type=int, default=0, help="1: ST256 y smartphones de 2022 con todas sus categorias")
A = ap.parse_args()
OLAS = sorted(A.olas)
tag = A.tag or f"{'-'.join(str(o)[2:] for o in OLAS)}_{A.items.replace(':','-').replace(',','+')}_{A.modelo}_pi{A.pi}_{A.calib}" + ("_amplia22" if A.amplia22 else "")
t0 = time.time()
def log(*x):
    print(f"[{time.time()-t0:7.1f}s]", *x, flush=True)

# ------------------------------------------------------------------ datos
cols = None
d = pd.read_parquet("data/metodos/items.parquet")
d = d[d.wave.isin(OLAS)].reset_index(drop=True)
if A.amplia22:
    for c in [c for c in d.columns if c.startswith("x4_")]:
        m = d.wave == 2022
        d.loc[m, c[3:]] = d.loc[m, c]
# paises: OCDE en todas las olas de la ventana y con HOMEPOS publicado en todas
ok = d.groupby(["cnt", "wave"]).homepos.apply(lambda x: x.notna().mean() > .5).unstack()
paises = sorted(ok.index[ok.reindex(columns=OLAS).fillna(False).all(axis=1)])
if A.paises:
    paises = sorted(set(paises) & set(A.paises.split(",")))
d = d[d.cnt.isin(paises)].reset_index(drop=True)
log(f"ventana {OLAS}: {len(paises)} paises, {len(d):,} alumnos")

# ------------------------------------------------------------------ items de la variante
por_ola = {o: set(I.slots_de(o)) for o in OLAS}
if A.items.startswith("lista:"):                       # lista cerrada de slots (nucleos estrictos)
    slots, con_propios = A.items[6:].split(","), False
elif A.items == "comunes" or A.items.startswith("anclas"):
    slots = I.comunes(OLAS) if A.items == "comunes" else sorted(set().union(*por_ola.values()))
    con_propios = A.items.startswith("anclas")
else:
    slots = sorted(set().union(*por_ola.values()))
    con_propios = True
    if A.items.startswith("sin:"):
        quita = sum((I.BLOQUES[b] for b in A.items[4:].split("+")), [])
        if "__propios__" in quita:
            con_propios = False
        slots = [s for s in slots if s not in quita]
anclas = A.items.split(":")[1].split(",") if A.items.startswith("anclas") else []
# slots sin ninguna respuesta en la ventana (p. ej. items nacionales que no eligio ningun pais OCDE)
slots = [s for s in slots if d[s].notna().sum() > 0] if slots else slots
log(f"{len(slots)} slots: {slots}")

# columnas de items propios de pais: una por (ola, variable, prefijo)
prop_cols = []
nuevas = {}
if con_propios:
    for o in OLAS:
        for v, K in I.PROPIOS[o]:
            m = d.wave == o
            for pref in sorted(d.loc[m, v + "_pref"].dropna().unique()):
                nm = f"{v}@{o}@{pref}"
                nuevas[nm] = d[v].where(m & (d[v + "_pref"] == pref))
                if nuevas[nm].notna().sum() >= 100:
                    prop_cols.append(nm)
                else:
                    del nuevas[nm]
d = pd.concat([d, pd.DataFrame(nuevas, index=d.index)], axis=1)
log(f"{len(prop_cols)} columnas de items propios de pais")

items = slots + prop_cols
Xdf = d[items]
K = [int(np.nanmax(Xdf[c].to_numpy())) + 1 for c in items]
X = Xdf.fillna(-1).to_numpy().astype(np.int16)
del Xdf
d = d[["wave", "cnt", "cntstuid", "lang", "w_fstuwt", "homepos"]].copy()     # libera memoria
import gc; gc.collect()

# grupos: pais x lengua x ola
d["grupo"] = d.wave.astype(str) + "-" + d.cnt + "-" + d.lang.astype(str)
gcod, gnom = pd.factorize(d.grupo, sort=True)
gnom = np.asarray(gnom)
g_ola = np.array([int(x.split("-")[0]) for x in gnom])
g_cnt = np.array([x.split("-")[1] for x in gnom])
# peso de calibracion: senado por pais x ola (cada celda pesa igual), media 1
w = d.w_fstuwt.to_numpy("float64")
cel = d.groupby(["cnt", "wave"]).w_fstuwt.transform("sum").to_numpy()
w = w / cel
w = w * len(w) / w.sum()
Gn = len(gnom)
ADM = np.zeros((Gn, len(items)), bool)                 # item administrado (alguna respuesta) en el grupo
for j in range(len(items)):
    ADM[:, j] = np.bincount(gcod[X[:, j] >= 0], minlength=Gn) > 0
log(f"{Gn} grupos pais x lengua x ola; X {X.shape}")


def modelo_base(olas_mod, pcm):
    """Modelo con un conjunto internacional por slot (compartido entre olas) y uno por item propio."""
    mod = G.Modelo(items, K, Gn, pcm=pcm)
    for j, it in enumerate(items):
        if "@" in it:
            v, o, pref = it.split("@")
            gs = np.where(ADM[:, j])[0]
            s = mod.nuevo_set(j, etiqueta=f"{it}")
            mod.M[j, gs] = s
        else:
            gs = [g for g in range(Gn) if g_ola[g] in olas_mod and g_ola[g] in [o for o in OLAS if it in por_ola[o]]]
            gs = [g for g in gs if ADM[g, j]]
            if gs:
                s = mod.nuevo_set(j, etiqueta=it)
                mod.M[j, gs] = s
    # arranque: logits de las proporciones acumuladas
    for st in mod.sets:
        j = st["j"]; v = X[X[:, j] >= 0, j]
        cum = np.cumsum(np.bincount(v, minlength=K[j])[::-1])[::-1][1:] / max(len(v), 1)
        st["b"] = np.clip(-np.log(np.clip(cum, 1e-3, 1 - 1e-3) / (1 - np.clip(cum, 1e-3, 1 - 1e-3))), -4, 4)
        if np.any(np.diff(st["b"]) == 0):
            st["b"] = st["b"] + np.linspace(0, 1e-3, len(st["b"]))
    return mod


def subdatos(filas):
    return G.Datos(X[filas], gcod[filas], w[filas])


pcm = A.modelo == "pcm"
umbral = A.umbral if A.pi else 99.0
historial = []

if A.calib == "concurrente":
    mod = modelo_base(OLAS, pcm)
    dat = subdatos(np.arange(len(d)))
    mod, historial = G.calibra_pi(mod, dat, umbral=umbral, n_min=A.nmin, log=log, max_iter=A.max_iter)
    theta = G.puntua(mod, dat, "wle"); eap = G.puntua(mod, dat, "eap")
    se = G.se_wle(mod, dat, theta)
    mods = {"pool": mod}

elif A.calib == "separada":
    theta = np.full(len(d), np.nan); eap = theta.copy(); se = theta.copy(); mods = {}
    for o in OLAS:
        f = np.where(d.wave == o)[0]
        mod = modelo_base([o], pcm)
        dat = G.Datos(X[f], gcod[f], w[f])
        # identificacion dentro de la ola: los grupos de otras olas no tienen filas
        mod, h = G.calibra_pi(mod, dat, umbral=umbral, n_min=A.nmin, log=log, max_iter=A.max_iter)
        historial += [dict(ola=o, **x) for x in h]
        theta[f] = G.puntua(mod, dat, "wle"); eap[f] = G.puntua(mod, dat, "eap")
        se[f] = G.se_wle(mod, dat, theta[f])
        mods[o] = mod

elif A.calib.startswith("fpl"):
    ref = int(A.calib[3:])
    otras = [o for o in OLAS if o != ref]
    f = np.where(d.wave == ref)[0]
    mref = modelo_base([ref], pcm)
    mref, h = G.calibra_pi(mref, G.Datos(X[f], gcod[f], w[f]), umbral=umbral, n_min=A.nmin, log=log, max_iter=A.max_iter)
    historial += [dict(ola=ref, **x) for x in h]
    mod = modelo_base(OLAS, pcm)
    # copiar los conjuntos de la referencia; para cada grupo de otra ola, el del mismo pais x lengua
    mod.sets = [dict(st) for st in mref.sets]
    mod.M = mref.M.copy()
    for st in mod.sets:
        st["fijo"] = True
    grupo_ref = {(g_cnt[g], gnom[g].split("-", 2)[2]): g for g in range(Gn) if g_ola[g] == ref}
    for j, it in enumerate(items):
        for g in range(Gn):
            if g_ola[g] == ref or not ADM[g, j]:
                continue
            if "@" in it:            # propios de otra ola: siempre libres, uno por prefijo
                if mod.M[j, g] < 0:
                    ss = [s for s, st in enumerate(mod.sets) if st["j"] == j and not st["fijo"]]
                    mod.M[j, g] = ss[0] if ss else mod.nuevo_set(j, etiqueta=it)
                continue
            gr = grupo_ref.get((g_cnt[g], gnom[g].split("-", 2)[2]))
            s_ref = mod.M[j, gr] if gr is not None else -1
            if s_ref < 0:            # el pais x lengua no existe en la referencia: el internacional
                cand = [s for s, st in enumerate(mod.sets) if st["j"] == j and st["etiqueta"] == it and st["fijo"]]
                s_ref = cand[0] if cand else -1
            if s_ref >= 0:
                mod.M[j, g] = s_ref
            else:                    # item nuevo en esta ola: parametros internacionales libres
                ss = [s for s, st in enumerate(mod.sets) if st["j"] == j and not st["fijo"]]
                mod.M[j, g] = ss[0] if ss else mod.nuevo_set(j, etiqueta=it + "_new")
    # arranque de los conjuntos nuevos
    for st in mod.sets:
        if not st["fijo"]:
            j = st["j"]; v = X[X[:, j] >= 0, j]
            cum = np.cumsum(np.bincount(v, minlength=K[j])[::-1])[::-1][1:] / max(len(v), 1)
            cum = np.clip(cum, 1e-3, 1 - 1e-3); st["b"] = -np.log(cum / (1 - cum))
    mod.mu[g_ola == ref], mod.sd[g_ola == ref] = mref.mu[g_ola == ref], mref.sd[g_ola == ref]
    f2 = np.where(d.wave.isin(otras))[0]
    dat2 = G.Datos(X[f2], gcod[f2], w[f2])
    mod, h = G.calibra_pi(mod, dat2, umbral=umbral, n_min=A.nmin, log=log, max_iter=A.max_iter)
    historial += [dict(ola=otras[0], **x) for x in h]
    theta = np.full(len(d), np.nan); eap = theta.copy()
    theta[f] = G.puntua(mref, G.Datos(X[f], gcod[f], w[f]), "wle")
    eap[f] = G.puntua(mref, G.Datos(X[f], gcod[f], w[f]), "eap")
    theta[f2] = G.puntua(mod, dat2, "wle"); eap[f2] = G.puntua(mod, dat2, "eap")
    se = np.full(len(d), np.nan)
    se[f] = G.se_wle(mref, G.Datos(X[f], gcod[f], w[f]), theta[f]); se[f2] = G.se_wle(mod, dat2, theta[f2])
    mods = {"ref": mref, "enlazado": mod}

elif A.calib == "anclas":
    # paso 1: anclas concurrentes e invariantes (sin liberar)
    ja = [items.index(s) for s in anclas]
    mod = modelo_base(OLAS, pcm)
    dat = subdatos(np.arange(len(d)))
    m1 = G.Modelo([items[j] for j in ja], [K[j] for j in ja], Gn, pcm=pcm)
    for i, j in enumerate(ja):
        s = m1.nuevo_set(i, etiqueta=items[j]); m1.M[i] = np.where(mod.M[j] >= 0, s, -1)
        m1.sets[s]["b"] = mod.sets[mod.M[j][mod.M[j] >= 0][0]]["b"].copy()
    d1 = G.Datos(X[:, ja], gcod, w)
    G.em(m1, d1, log=log, max_iter=A.max_iter)
    # paso 2: anclas fijas; el resto con parametros propios de cada ola (un conjunto por ola)
    for i, j in enumerate(ja):
        st = m1.sets[m1.M[i][m1.M[i] >= 0][0]]
        s = mod.M[j][mod.M[j] >= 0][0]
        mod.sets[s].update(a=st["a"], b=st["b"].copy(), fijo=True)
    for j, it in enumerate(items):
        if j in ja or "@" in it:
            continue
        s0 = mod.M[j][mod.M[j] >= 0]
        if not len(s0):
            continue
        s0 = s0[0]
        for o in OLAS[1:]:
            gs = [g for g in range(Gn) if g_ola[g] == o and mod.M[j, g] == s0]
            if gs:
                s = mod.nuevo_set(j, mod.sets[s0]["a"], mod.sets[s0]["b"].copy(), etiqueta=f"{it}_{o}")
                mod.M[j, gs] = s
    mod.mu, mod.sd = m1.mu.copy(), m1.sd.copy()
    mod, historial = G.calibra_pi(mod, dat, umbral=umbral, n_min=A.nmin, log=log, max_iter=A.max_iter,
                                  no_liberables=ja)
    theta = G.puntua(mod, dat, "wle"); eap = G.puntua(mod, dat, "eap")
    se = G.se_wle(mod, dat, theta)
    mods = {"pool": mod}
else:
    raise SystemExit(A.calib)

out = d[["wave", "cnt", "cntstuid"]].copy()
out["wle"], out["eap"], out["se"] = theta, eap, se
out["n_items"] = (X >= 0).sum(axis=1)
out.to_parquet(f"data/metodos/hp_{tag}.parquet", index=False)
pd.concat([G.resumen_sets(m, gnom).assign(modelo=k) for k, m in mods.items()]).to_csv(
    f"output/tables/metodos/params_{tag}.csv", index=False)
pd.DataFrame(historial).to_csv(f"output/tables/metodos/rmsd_{tag}.csv", index=False)
# distribuciones de grupo y liberaciones por grupo (para comparar con T22.A.12 de la OCDE)
mm = list(mods.values())[-1]
lib = []
for g in range(Gn):
    js = np.where(mm.M[:, g] >= 0)[0]
    n_int = sum(1 for j in js if "@" not in items[j] and "_" not in mm.sets[mm.M[j, g]]["etiqueta"].replace(items[j], "", 1))
    lib.append(dict(grupo=gnom[g], mu=mm.mu[g], sd=mm.sd[g], n_items=len(js), n_internacionales=n_int))
pd.DataFrame(lib).to_csv(f"output/tables/metodos/grupos_{tag}.csv", index=False)
# informacion del test por grupo (con el modelo que puntua a cada grupo)
INF = np.full((Gn, G.Q), np.nan)
for k_, m_ in mods.items():
    for g in range(Gn):
        if (k_ == "ref" and g_ola[g] != int(A.calib[3:])) or (k_ == "enlazado" and g_ola[g] == int(A.calib[3:])):
            continue
        if isinstance(k_, int) and g_ola[g] != k_:
            continue
        if (m_.M[:, g] >= 0).any():
            INF[g] = G.informacion(m_, g)
np.savez(f"data/metodos/info_{tag}.npz", info=INF, grupos=gnom, mu=mm.mu, sd=mm.sd)
# guardar el modelo completo para reutilizarlo (simulacion, mirt)
import pickle
pickle.dump(dict(mods=mods, items=items, K=K, gnom=gnom), open(f"data/metodos/modelo_{tag}.pkl", "wb"))
m = out.wle.notna()
log(f"guardado hp_{tag}: cobertura {m.mean()*100:.1f}%  r(wle, homepos publicado) = "
    f"{np.corrcoef(out.wle[m & d.homepos.notna()], d.homepos[m & d.homepos.notna()])[0,1]:.4f}")
for o in OLAS:
    mo = m & (d.wave == o) & d.homepos.notna()
    r_pais = pd.DataFrame(dict(a=out.wle[mo], b=d.homepos[mo], c=d.cnt[mo])).groupby("c").apply(
        lambda x: x.a.corr(x.b), include_groups=False)
    log(f"  {o}: r intra-pais con el publicado, mediana {r_pais.median():.4f}  min {r_pais.min():.4f} ({r_pais.idxmin()})")
json.dump(dict(tag=tag, olas=OLAS, items=A.items, modelo=A.modelo, pi=A.pi, calib=A.calib, umbral=A.umbral,
               paises=paises, n_slots=len(slots), n_propios=len(prop_cols), historial=historial,
               segundos=time.time() - t0), open(f"output/tables/metodos/meta_{tag}.json", "w"), indent=1)
