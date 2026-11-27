"""Monte Carlo: ¿que metodo recupera la tendencia del hueco Q4-Q1 cuando se conoce la verdad?

Poblacion (por pais c y ola t):
  theta  ~ N(mu_c, 1)                      riqueza (el constructo objetivo de HOMEPOS)
  u      ~ N(0, 1) independiente           parte del capital cultural no explicada por la riqueza
  kappa  = rho*theta + sqrt(1-rho^2)*u     capital cultural (rho = 0,5)
  z      ~ N(0, 1) independiente           factor local sin relacion con la nota (clima, ruralidad...)
  Y      = 100*(g_th*theta + g_u*u + e)    nota; g_th = 0,30 y g_u = 0,25 dan corr(Y,theta) = 0,30 y
                                           corr(Y,kappa) = 0,37, los ordenes de magnitud observados
Items comunes: los 16 de 2022-25 con los parametros internacionales estimados (C-GPCM), generados desde theta.

Escenarios (verdad: el hueco por cuartiles de theta NO cambia, salvo S4):
  S0  solo items comunes en las dos olas
  S1  fiabilidad: la ola 2 anade 12 items que miden theta (a = 1)            -> cambia solo la fiabilidad
  S2  constructo: la ola 1 anade 8 items de capital cultural (driven by kappa, a = 1,6);
      la ola 2 anade 6 items de "condiciones locales" (0,3*theta + 0,95*z, a = 0,9, faciles/dificiles)
                                                                              -> el patron 2022/2025
  S3  DIF temporal no uniforme en comunes: en la ola 2 'books', 'instruments' y 'art' pierden la mitad de
      su discriminacion (y ganan carga de kappa)                              -> el riesgo de NUESTRO metodo
  S4  S2 + cambio real: g_th baja de 0,30 a 0,25 en la ola 2 (el hueco verdadero cae)
Metodos: C-GPCM (comunes, concurrente, sin PI), A-GPCM-PI (todos, concurrente, RMSD 0,25 por pais x ola),
A-PCM-PI (idem con a = 1), y el oraculo (cuartiles de theta verdadera).
Metricas: sesgo, DT de Monte Carlo y RECM del cambio del hueco (puntos), frente al cambio verdadero en la
misma muestra (cuartiles de theta).
"""
import sys, pickle, time, numpy as np, pandas as pd
sys.path.insert(0, "code"); import _gpcm_mg as G

ESC = sys.argv[1] if len(sys.argv) > 1 else "S2"
R = int(sys.argv[2]) if len(sys.argv) > 2 else 40
import os
_prev = f"output/tables/metodos/sim_{ESC}.csv"
res = pd.read_csv(_prev).to_dict("records") if os.path.exists(_prev) else []   # reanuda
NP, N = 12, 2000                       # paises y alumnos por pais x ola
RHO, G_TH, G_U = 0.5, 0.30, 0.25

m = pickle.load(open("data/metodos/modelo_22-25_comunes_gpcm_pi0_concurrente.pkl", "rb"))
mod0 = m["mods"]["pool"]
COM = [(st["a"], st["b"]) for st in mod0.sets[:len(m["items"])]]
NOM_COM = m["items"]


def gen_item(lat, a, b, rng):
    P = G.probs(a, b, lat)                        # (K, n)
    c = P.cumsum(axis=0)
    u = rng.random(len(lat))
    return (u[None, :] > c).sum(axis=0).astype(np.int16)


def simula(rng, esc):
    filas = []
    Xs = []
    for c in range(NP):
        mu = rng.normal(0, 0.4)
        for t in (0, 1):
            th = rng.normal(mu, 1, N); u = rng.normal(0, 1, N); z = rng.normal(0, 1, N)
            ka = RHO * th + np.sqrt(1 - RHO ** 2) * u
            gth = 0.25 if (esc == "S4" and t == 1) else G_TH
            Y = 100 * (gth * th + G_U * u + rng.normal(0, np.sqrt(1 - gth ** 2 - G_U ** 2), N))
            cols = {}
            for j, (a, b) in enumerate(COM):
                nm = NOM_COM[j]
                if esc == "S3" and t == 1 and nm in ("books", "instruments", "art"):
                    lat = 0.6 * th + 0.8 * u          # pierde relacion con la riqueza
                    cols[nm] = gen_item(lat, a * 0.5, b, rng)
                else:
                    cols[nm] = gen_item(th, a, b, rng)
            if esc == "S1" and t == 1:
                for k, b in enumerate(np.linspace(-2, 1.5, 12)):
                    cols[f"x1_{k}"] = gen_item(th, 1.0, np.array([b]), rng)
            if esc in ("S2", "S4"):
                if t == 0:
                    for k, b in enumerate(np.linspace(-1.2, 1.2, 8)):
                        cols[f"cult_{k}"] = gen_item(ka, 1.6, np.array([b]), rng)
                else:
                    for k, b in enumerate([-3.0, -2.5, -2.0, 1.0, 1.5, 2.0]):
                        cols[f"loc_{k}"] = gen_item(0.3 * th + 0.95 * z, 0.9, np.array([b]), rng)
            filas.append(pd.DataFrame(dict(c=c, t=t, th=th, Y=Y)))
            Xs.append(pd.DataFrame(cols))
    return pd.concat(filas, ignore_index=True), pd.concat(Xs, ignore_index=True)


def hueco(x, Y, c, t):
    """cambio del hueco Q4-Q1 (media simple de paises) con cuartiles dentro de pais x ola."""
    gaps = {0: [], 1: []}
    for cc in range(NP):
        for tt in (0, 1):
            m = (c == cc) & (t == tt) & np.isfinite(x)
            q = pd.qcut(pd.Series(x[m]).rank(method="first"), 4, labels=False).to_numpy()
            y = Y[m]
            gaps[tt].append(y[q == 3].mean() - y[q == 0].mean())
    return np.mean(gaps[1]) - np.mean(gaps[0]), np.mean(gaps[0]), np.mean(gaps[1])


def ajusta(Xdf, grupos, items, pcm, pi):
    X = Xdf[items].fillna(-1).to_numpy().astype(np.int16)
    K = [int(Xdf[i].max()) + 1 for i in items]
    Gn = grupos.max() + 1
    mod = G.Modelo(items, K, Gn, pcm=pcm)
    for j in range(len(items)):
        gs = np.unique(grupos[X[:, j] >= 0])
        s = mod.nuevo_set(j, etiqueta=items[j]); mod.M[j, gs] = s
        v = X[X[:, j] >= 0, j]
        cum = np.clip(np.cumsum(np.bincount(v, minlength=K[j])[::-1])[::-1][1:] / len(v), 1e-3, 1 - 1e-3)
        mod.sets[s]["b"] = -np.log(cum / (1 - cum))
    dat = G.Datos(X, grupos, np.ones(len(X)))
    G.calibra_pi(mod, dat, umbral=0.25 if pi else 99, n_min=150, log=None, max_iter=300)
    lib = sum(1 for st in mod.sets if "UP" in st["etiqueta"] or "DOWN" in st["etiqueta"])
    return G.puntua(mod, dat, "wle"), lib


t0 = time.time()
hechas = {int(x["rep"]) for x in res}
for r in range(R):
    if r in hechas:
        continue
    rng = np.random.default_rng(1000 + r + 100 * int(ESC[1]))
    P, Xdf = simula(rng, ESC)
    grupos = (P.c * 2 + P.t).to_numpy()
    Y, c, t = P.Y.to_numpy(), P.c.to_numpy(), P.t.to_numpy()
    verdad, v0, v1 = hueco(P.th.to_numpy(), Y, c, t)
    fila = dict(esc=ESC, rep=r, verdad=verdad, verdad_g0=v0, verdad_g1=v1)
    todos = list(Xdf.columns)
    for nom, items, pcm, pi in [("C-GPCM", NOM_COM, False, False), ("A-GPCM-PI", todos, False, True),
                                ("A-PCM-PI", todos, True, True), ("A-GPCM", todos, False, False)]:
        if ESC in ("S0",) and nom != "C-GPCM" and nom != "A-GPCM-PI":
            continue
        x, lib = ajusta(Xdf, grupos, items, pcm, pi)
        dg, g0, g1 = hueco(x, Y, c, t)
        fila[f"{nom}"] = dg; fila[f"{nom}_g0"] = g0; fila[f"{nom}_g1"] = g1; fila[f"{nom}_lib"] = lib
        # fiabilidad por ola: r^2 con theta verdadera
        for tt in (0, 1):
            mm = (t == tt) & np.isfinite(x)
            fila[f"{nom}_rel{tt}"] = np.mean([np.corrcoef(x[mm & (c == cc)], P.th.to_numpy()[mm & (c == cc)])[0, 1] ** 2
                                              for cc in range(NP)])
    res.append(fila)
    print(f"[{time.time()-t0:6.0f}s] {ESC} rep {r}: verdad {verdad:+.2f}  " +
          "  ".join(f"{k} {v:+.2f}" for k, v in fila.items() if k in ("C-GPCM", "A-GPCM-PI", "A-PCM-PI", "A-GPCM")),
          flush=True)
    pd.DataFrame(res).to_csv(f"output/tables/metodos/sim_{ESC}.csv", index=False)
