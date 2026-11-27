"""GPCM / PCM multigrupo con invarianza parcial: la maquinaria de escalado de HOMEPOS de la OCDE
(Technical Report PISA 2022 cap. 19 y 2025 cap. 22) y de Lee y von Davier (2020), que es lo que
Avvisati (2020, LSAE 8:8, Tabla 6) y Avvisati y Wuyts (2024) proponen para la tendencia.

Parametrizacion (la del repo `lib/irt.py`): para un item de categorias 0..m,
    log P(X=k) / P(X=k-1) = a (theta - b_k),   k = 1..m.
Los parametros de la OCDE (slope, location, step_k, con log P_k/P_{k-1} = slope*theta - location - step_k)
se pasan con b_k = (location + step_k) / slope.

Estructura del modelo
- items j = 0..J-1, con K[j] categorias;
- grupos g = 0..G-1 (pais x lengua x ola), cada uno con su N(mu_g, sd_g);
- conjuntos de parametros s: cada uno con (a, b) y una bandera `fijo`;
- M[j, g] = conjunto que usa el grupo g para el item j (-1 si el item no se administra al grupo).
Liberar un item para un grupo es apuntar M[j, g] a un conjunto nuevo.
"""
import numpy as np
import scipy.sparse as sp
from scipy.optimize import minimize

NODES = np.linspace(-6, 6, 49)
Q = len(NODES)


def probs(a, b, th=NODES):
    """(K, T) probabilidad de cada categoria en cada theta."""
    th = np.asarray(th, "float64")
    S = np.concatenate([[0.0], np.cumsum(b)])
    k = np.arange(len(S))
    z = a * (k[:, None] * th[None, :] - S[:, None])
    z -= z.max(axis=0, keepdims=True)
    P = np.exp(z)
    return P / P.sum(axis=0, keepdims=True)


def desde_ocde(slope, location, steps):
    steps = [s for s in steps if s is not None and np.isfinite(s)] or [0.0]
    return float(slope), np.array([(location + s) / slope for s in steps])


class Modelo:
    def __init__(self, items, K, G, pcm=False):
        self.items, self.K, self.G, self.pcm = list(items), list(K), G, pcm
        self.J = len(items)
        self.M = -np.ones((self.J, G), np.int32)
        self.sets = []            # dicts: j, a, b, fijo, etiqueta
        self.mu, self.sd = np.zeros(G), np.ones(G)

    def nuevo_set(self, j, a=1.0, b=None, fijo=False, etiqueta=""):
        if b is None:
            b = np.linspace(-1, 1, self.K[j] - 1) if self.K[j] > 2 else np.zeros(1)
        self.sets.append(dict(j=j, a=float(a), b=np.asarray(b, float), fijo=fijo, etiqueta=etiqueta))
        return len(self.sets) - 1

    def params_grupo(self, g):
        """Listas (idx items, A, B) que usa el grupo g."""
        js = np.where(self.M[:, g] >= 0)[0]
        return js, [self.sets[self.M[j, g]]["a"] for j in js], [self.sets[self.M[j, g]]["b"] for j in js]


# ---------------------------------------------------------------- estado de datos
class Datos:
    """X (n,J) int8 con -1 ausente, g (n,) grupo, w (n,) peso de calibracion."""
    def __init__(self, X, g, w):
        self.X, self.g, self.w = X.astype(np.int16), g.astype(np.int32), w.astype("float64")
        self.n, self.J = X.shape
        self.obs = self.X >= 0


def _filas_por_set(mod, dat):
    """Estructura por item: filas que responden con set asignado, set local de cada fila, codigo de
    respuesta, lista de sets y matriz dispersa (S_j*K_j, n) de pesos para los conteos del paso M."""
    out = []
    for j in range(mod.J):
        sidx = mod.M[j][dat.g]
        ok = dat.obs[:, j] & (sidx >= 0)
        filas = np.where(ok)[0]
        sets, sloc = np.unique(sidx[filas], return_inverse=True)
        x = dat.X[filas, j].astype(np.int64)
        Kj = mod.K[j]
        S = sp.csr_matrix((dat.w[filas], (sloc * Kj + x, filas)), shape=(len(sets) * Kj, dat.n))
        out.append(dict(filas=filas, sloc=sloc, x=x, sets=sets, S=S))
    return out


def _logprior(mod):
    z = (NODES[None, :] - mod.mu[:, None]) / mod.sd[:, None]
    lp = -0.5 * z ** 2
    return lp - np.log(np.exp(lp).sum(axis=1, keepdims=True))


def posterior(mod, dat, filas=None, devuelve_ll=False):
    if filas is None:
        filas = _filas_por_set(mod, dat)
    LL = _logprior(mod)[dat.g]
    for j in range(mod.J):
        f = filas[j]
        if not len(f["filas"]):
            continue
        Kj = mod.K[j]
        LP = np.stack([np.log(np.clip(probs(mod.sets[s]["a"], mod.sets[s]["b"]), 1e-300, 1)) for s in f["sets"]])
        LP = LP.reshape(len(f["sets"]) * Kj, Q)
        LL[f["filas"]] += LP[f["sloc"] * Kj + f["x"]]
    mx = LL.max(axis=1, keepdims=True)
    P = np.exp(LL - mx)
    tot = P.sum(axis=1, keepdims=True)
    P /= tot
    if devuelve_ll:
        return P, float((dat.w * (np.log(tot[:, 0]) + mx[:, 0])).sum())
    return P


def _mstep(R, a0, b0, fija_a=False):
    K = R.shape[0]
    k = np.arange(K)[:, None]
    below = (np.arange(1, K)[:, None] <= k.T).astype(float)      # (m, K)
    N = R.sum(axis=0)

    def f(p):
        if fija_a:
            la, b = np.log(a0), p
        else:
            la, b = p[0], p[1:]
        a = np.exp(la)
        S = np.concatenate([[0.0], np.cumsum(b)])
        P = probs(a, b)
        nll = -(R * np.log(np.clip(P, 1e-300, 1))).sum()
        D = R - N[None, :] * P                                   # (K,Q)
        U = k * NODES[None, :] - S[:, None]
        g_la = -(D * U).sum() * a
        g_b = a * (below @ D.sum(axis=1))
        g = g_b if fija_a else np.concatenate([[g_la], g_b])
        return nll, g

    x0 = np.asarray(b0, float) if fija_a else np.concatenate([[np.log(a0)], b0])
    bnds = [(-12, 12)] * (K - 1) if fija_a else [(-3.0, 2.3)] + [(-12, 12)] * (K - 1)
    r = minimize(f, x0, jac=True, method="L-BFGS-B", bounds=bnds,
                 options=dict(maxiter=300, ftol=1e-11, gtol=1e-7))
    return (a0, r.x) if fija_a else (float(np.exp(r.x[0])), r.x[1:])


def em(mod, dat, max_iter=300, tol=1e-3, log=print, cada=10, identifica=True):
    filas = _filas_por_set(mod, dat)
    W_g = np.bincount(dat.g, weights=dat.w, minlength=mod.G)
    hay_fijos = any(st["fijo"] for st in mod.sets)
    ll_prev = -np.inf
    for it in range(max_iter):
        post, ll = posterior(mod, dat, filas, devuelve_ll=True)
        cambio = 0.0
        # --- items: conteos esperados de todos los sets del item con una sola matriz dispersa
        for j in range(mod.J):
            f = filas[j]
            if not len(f["filas"]):
                continue
            Kj = mod.K[j]
            C = np.asarray(f["S"] @ post).reshape(len(f["sets"]), Kj, Q)
            for i, s in enumerate(f["sets"]):
                st = mod.sets[s]
                if st["fijo"]:
                    continue
                a1, b1 = _mstep(C[i] + 1e-9, st["a"], st["b"], fija_a=mod.pcm)
                cambio = max(cambio, abs(a1 - st["a"]), float(np.abs(b1 - st["b"]).max()))
                st["a"], st["b"] = a1, b1
        # --- distribuciones de grupo
        m1 = np.zeros(mod.G); m2 = np.zeros(mod.G)
        pw = post * dat.w[:, None]
        E1 = pw @ NODES; E2 = pw @ NODES ** 2
        m1 = np.bincount(dat.g, weights=E1, minlength=mod.G) / np.maximum(W_g, 1e-12)
        m2 = np.bincount(dat.g, weights=E2, minlength=mod.G) / np.maximum(W_g, 1e-12)
        sd = np.sqrt(np.maximum(m2 - m1 ** 2, 1e-4))
        cambio = max(cambio, float(np.abs(m1 - mod.mu).max()), float(np.abs(sd - mod.sd).max()))
        mod.mu, mod.sd = m1, sd
        # --- identificacion: pool (peso de calibracion) con media 0 y DT 1
        if identifica and not hay_fijos:
            m = (W_g * mod.mu).sum() / W_g.sum()
            if mod.pcm:
                mod.mu = mod.mu - m
                for st in mod.sets:
                    st["b"] = st["b"] - m
            else:
                v = (W_g * (mod.sd ** 2 + mod.mu ** 2)).sum() / W_g.sum() - m ** 2
                s = np.sqrt(v)
                mod.mu, mod.sd = (mod.mu - m) / s, mod.sd / s
                for st in mod.sets:
                    st["a"], st["b"] = st["a"] * s, (st["b"] - m) / s
        if log and (it % cada == 0):
            log(f"    EM {it:3d}  logL {ll:,.1f}  cambio {cambio:.5f}")
        if cambio < tol or abs(ll - ll_prev) < 1e-9 * abs(ll):
            if log:
                log(f"    EM convergido en {it} iteraciones  logL {ll:,.1f}")
            break
        ll_prev = ll
    return mod


# ---------------------------------------------------------------- ajuste: RMSD
def rmsd(mod, dat, post=None, n_min=150):
    """Tabla (j, g, set, n, rmsd, md). RMSD = max_k sqrt(sum_q f_g(q) (Pobs_gk - Pexp_gk)^2);
    en items dicotomicos coincide con la definicion estandar. MD = desviacion media de la
    puntuacion esperada del item, reescalada a [0,1], con signo (UP = el grupo puntua mas alto de lo
    que predice el modelo)."""
    if post is None:
        post = posterior(mod, dat)
    pw = post * dat.w[:, None]
    fil = []
    n_g = None
    for j in range(mod.J):
        Kj = mod.K[j]
        ok = dat.obs[:, j] & (mod.M[j][dat.g] >= 0)
        r = np.where(ok)[0]
        if not len(r):
            continue
        gi, x = dat.g[r], dat.X[r, j]
        ind = sp.csr_matrix((np.ones(len(r)), (gi * Kj + x, np.arange(len(r)))), shape=(mod.G * Kj, len(r)))
        C = np.asarray((ind @ pw[r]))                    # (G*K, Q)
        C = C.reshape(mod.G, Kj, Q)
        nun = np.bincount(gi, minlength=mod.G)
        for g in np.where(nun > 0)[0]:
            s = mod.M[j, g]
            tot = C[g].sum(axis=0)                        # (Q,)
            f = tot / tot.sum()
            with np.errstate(invalid="ignore", divide="ignore"):
                Pobs = np.where(tot > 0, C[g] / tot, 0.0)
            Pexp = probs(mod.sets[s]["a"], mod.sets[s]["b"])
            dev = np.sqrt((f[None, :] * (Pobs - Pexp) ** 2).sum(axis=1))
            k = np.arange(Kj)[:, None]
            md = (f * ((Pobs - Pexp) * k).sum(axis=0)).sum() / (Kj - 1)
            fil.append((j, g, s, int(nun[g]), float(dev.max()), float(md)))
    import pandas as pd
    return pd.DataFrame(fil, columns=["j", "g", "set", "n", "rmsd", "md"])


def libera(mod, tabla, umbral=0.25, n_min=150, no_liberables=()):
    """Aplica una ronda de liberacion. Los grupos que no ajustan en un item se reparten por el signo de
    MD (UP/DOWN) y, con 4 o mas grupos del mismo signo, en dos tramos por |MD| (1 = menor, 2 = mayor):
    el esquema UP1/UP2/DOWN1/DOWN2 de las tablas de la OCDE. Devuelve el numero de pares liberados."""
    t = tabla[(tabla.rmsd > umbral) & (tabla.n > n_min) & (~tabla.j.isin(no_liberables))]
    n_lib = 0
    for (j, s_old), sub in t.groupby(["j", "set"]):
        st = mod.sets[s_old]
        if st["fijo"] and not st.get("liberable_si_fijo", True):
            continue
        # si el set ya es de un unico grupo, no hay nada que liberar
        grupos_en_set = np.where(mod.M[j] == s_old)[0]
        if len(grupos_en_set) <= 1:
            continue
        for signo, sb in sub.groupby(np.sign(sub.md)):
            sb = sb.sort_values("md", key=np.abs)
            h = len(sb) // 2
            tramos = [sb] if len(sb) < 4 else [sb.iloc[:h], sb.iloc[h:]]
            for i, tr in enumerate(tramos):
                grupos = tr.g.to_numpy()
                if len(grupos) == len(grupos_en_set):
                    continue
                nombre = f"{'UP' if signo > 0 else 'DOWN'}{i+1}"
                s_new = mod.nuevo_set(j, st["a"], st["b"].copy(), etiqueta=f"{st['etiqueta']}_{nombre}")
                mod.M[j, grupos] = s_new
                n_lib += len(grupos)
    return n_lib


def calibra_pi(mod, dat, umbral=0.25, n_min=150, rondas=8, log=print, no_liberables=(), **kw):
    """EM + rondas de liberacion hasta que ningun item x grupo supere el umbral."""
    em(mod, dat, log=log, **kw)
    historial = []
    for r in range(rondas):
        t = rmsd(mod, dat, n_min=n_min)
        malos = int(((t.rmsd > umbral) & (t.n > n_min)).sum())
        historial.append(dict(ronda=r, pares_malos=malos, rmsd_max=float(t[t.n > n_min].rmsd.max())))
        if log:
            log(f"  ronda {r}: item x grupo con RMSD > {umbral}: {malos}  (max {historial[-1]['rmsd_max']:.3f})")
        if malos == 0 or umbral >= 9:
            break
        n = libera(mod, t, umbral, n_min, no_liberables)
        if n == 0:
            break
        em(mod, dat, log=log, **kw)
    return mod, historial


# ---------------------------------------------------------------- puntuacion
def wle_grupo(X, A, B, min_items=3, max_iter=40, tol=1e-6):
    """WLE de Warm (Newton desde el EAP con prior N(0,1)). X (n,J) con -1 ausente."""
    n, J = X.shape
    obs = X >= 0
    # EAP de arranque
    L = np.zeros((n, Q))
    for j in range(J):
        m = obs[:, j]
        L[m] += np.log(np.clip(probs(A[j], B[j]), 1e-300, 1))[X[m, j]]
    L += -0.5 * NODES ** 2
    L = np.exp(L - L.max(axis=1, keepdims=True)); L /= L.sum(axis=1, keepdims=True)
    theta = L @ NODES
    active = obs.sum(axis=1) >= min_items
    for it in range(max_iter):
        idx = np.where(active)[0]
        if not len(idx):
            break
        th = theta[idx]
        score, info, dinfo, d2info = (np.zeros(len(idx)) for _ in range(4))
        for j in range(J):
            m = obs[idx, j]
            if not m.any():
                continue
            P = probs(A[j], B[j], th[m])
            k = np.arange(P.shape[0])[:, None]
            mu = (P * k).sum(axis=0)
            c2, c3, c4 = ((P * (k - mu) ** p).sum(axis=0) for p in (2, 3, 4))
            a = A[j]
            score[m] += a * (X[idx[m], j] - mu)
            info[m] += a ** 2 * c2
            dinfo[m] += a ** 3 * c3
            d2info[m] += a ** 4 * (c4 - 3 * c2 ** 2)
        info = np.maximum(info, 1e-10)
        g = score + dinfo / (2 * info)
        h = -info + (d2info * info - dinfo ** 2) / (2 * info ** 2)
        h = np.where(h < -1e-8, h, -1e-8)
        step = np.clip(-g / h, -1.0, 1.0)
        theta[idx] = np.clip(th + step, -10, 10)
        active[idx[np.abs(step) < tol]] = False
    return np.where(obs.sum(axis=1) >= min_items, theta, np.nan)


def eap_grupo(X, A, B, mu=0.0, sd=1.0, min_items=3):
    n, J = X.shape
    obs = X >= 0
    L = np.zeros((n, Q))
    for j in range(J):
        m = obs[:, j]
        L[m] += np.log(np.clip(probs(A[j], B[j]), 1e-300, 1))[X[m, j]]
    L += -0.5 * ((NODES - mu) / sd) ** 2
    L = np.exp(L - L.max(axis=1, keepdims=True)); L /= L.sum(axis=1, keepdims=True)
    e = L @ NODES
    v = L @ NODES ** 2 - e ** 2
    return np.where(obs.sum(axis=1) >= min_items, e, np.nan), np.where(obs.sum(axis=1) >= min_items, v, np.nan)


def puntua(mod, dat, metodo="wle", min_items=3):
    out = np.full(dat.n, np.nan)
    for g in range(mod.G):
        r = np.where(dat.g == g)[0]
        if not len(r):
            continue
        js, A, B = mod.params_grupo(g)
        X = dat.X[np.ix_(r, js)]
        if metodo == "wle":
            out[r] = wle_grupo(X, A, B, min_items)
        else:
            out[r] = eap_grupo(X, A, B, mod.mu[g], mod.sd[g], min_items)[0]
    return out


def informacion(mod, g, th=NODES):
    """Informacion del test del grupo g en cada theta (suma sobre sus items)."""
    js, A, B = mod.params_grupo(g)
    I = np.zeros(len(th))
    for a, b in zip(A, B):
        P = probs(a, b, th)
        k = np.arange(P.shape[0])[:, None]
        mu = (P * k).sum(axis=0)
        I += a ** 2 * (P * (k - mu) ** 2).sum(axis=0)
    return I


def resumen_sets(mod, grupos_nombre=None):
    import pandas as pd
    fil = []
    for s, st in enumerate(mod.sets):
        gs = np.where(mod.M[st["j"]] == s)[0]
        fil.append(dict(set=s, item=mod.items[st["j"]], etiqueta=st["etiqueta"], a=st["a"],
                        b=";".join(f"{x:.4f}" for x in st["b"]), fijo=st["fijo"], n_grupos=len(gs),
                        grupos=";".join(grupos_nombre[g] for g in gs) if grupos_nombre is not None and len(gs) < 40 else ""))
    return pd.DataFrame(fil)


def se_wle(mod, dat, theta):
    """Error tipico de cada WLE: 1/sqrt(I(theta_hat)) con los items que respondio el alumno."""
    out = np.full(dat.n, np.nan)
    for g in range(mod.G):
        r = np.where((dat.g == g) & np.isfinite(theta))[0]
        if not len(r):
            continue
        js, A, B = mod.params_grupo(g)
        th = theta[r]; I = np.zeros(len(r))
        for j, a, b in zip(js, A, B):
            m = dat.obs[r, j]
            if not m.any():
                continue
            P = probs(a, b, th[m]); k = np.arange(P.shape[0])[:, None]
            mu = (P * k).sum(axis=0)
            I[m] += a ** 2 * (P * (k - mu) ** 2).sum(axis=0)
        out[r] = 1 / np.sqrt(np.maximum(I, 1e-10))
    return out


def info_grupos(mod, grupos=None):
    """(G, Q) informacion del test de cada grupo con TODOS sus items."""
    grupos = range(mod.G) if grupos is None else grupos
    return np.array([informacion(mod, g) for g in grupos])
