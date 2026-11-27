"""Motor BRR: cuartiles recalculados en cada replica, 10 PV, Fay 0,5."""
import numpy as np, pandas as pd

N_REP, N_PV = 80, 10
FAY = 1.0 / (N_REP * 0.25)
REPS = [f"w_fsturwt{i}" for i in range(1, N_REP + 1)]


def _pesos(ola, llaves):
    """Replicate weights from the repository extract (data/extract/stu_<ola>.parquet + wts_<ola>.npy, Git LFS)."""
    import os
    ext = os.environ.get("PISA_EXTRACT", "../data/extract")
    p = pd.read_parquet(f"{ext}/stu_{ola}.parquet", columns=["wave", "cnt", "cntstuid"])
    p["cntstuid"] = p.cntstuid.astype("int64")
    w = np.load(f"{ext}/wts_{ola}.npy")
    assert len(w) == len(p)
    return pd.concat([p, pd.DataFrame(w, columns=REPS)], axis=1)


def matriz_pesos(d, cache={}):
    """(N, 81): columna 0 el peso final, 1..80 los replicados."""
    partes = []
    for ola in sorted(d.wave.unique()):
        if ola not in cache:
            cache[ola] = _pesos(ola, None)
        partes.append(cache[ola])
    p = pd.concat(partes, ignore_index=True)
    n0 = len(d)
    m = d[["wave", "cnt", "cntstuid"]].merge(p, on=["wave", "cnt", "cntstuid"],
                                             how="left")
    assert len(m) == n0
    return np.column_stack([d["w_fstuwt"].to_numpy("float64"),
                            m[REPS].to_numpy("float64")])


def _grupo_por_replica(x, W, rng, k=4):
    """(n,81) de codigos 0..k-1: el grupo de peso igual se recalcula con cada peso."""
    n, R = W.shape
    orden = np.lexsort((rng.random(n), x))
    xs, Ws = x[orden], W[orden]
    cum = np.cumsum(Ws, axis=0)
    tot = cum[-1]
    frac = cum / tot                       # (n,R)
    cod = np.zeros((n, R), np.int8)
    for j in range(1, k):
        cod += (frac > j / k)
    # el corte pertenece al grupo de abajo: desplazamos un puesto
    cod = np.vstack([np.zeros((1, R), np.int8), cod[:-1]])
    out = np.empty_like(cod)
    out[orden] = cod
    return out


def _cuartil_por_replica(x, W, rng):
    """(n,81) de codigos 0..3: el cuartil se recalcula con cada peso."""
    return _grupo_por_replica(x, W, rng, 4)


def _agrega(cod, W, PV, k):
    """theta[g, r, m] = media ponderada de PV_m en el grupo g bajo el peso r."""
    R, M = W.shape[1], PV.shape[1]
    th = np.full((k, R, M), np.nan)
    for r in range(R):
        c = cod[:, r] if cod.ndim == 2 else cod
        w = W[:, r]
        den = np.bincount(c, w, minlength=k)
        for m in range(M):
            num = np.bincount(c, w * PV[:, m], minlength=k)
            with np.errstate(invalid="ignore", divide="ignore"):
                th[:, r, m] = num / den
    return th


def theta_cuantiles(x, W, PV, k=4, semilla=7):
    """(k, R=81, M=10) de medias por grupo de peso igual (cuartiles, deciles...)."""
    rng = np.random.default_rng(semilla)
    cod = _grupo_por_replica(x, W, rng, k)
    return _agrega(cod, W, PV, k)


def theta_cuartiles(x, W, PV, semilla=7):
    """Devuelve (k=4, R=81, M=10) de medias por cuartil."""
    return theta_cuantiles(x, W, PV, 4, semilla)


def theta_grupos(cod, W, PV, k):
    """Grupos fijos (no dependen de la replica)."""
    return _agrega(np.ascontiguousarray(cod), W, PV, k)


def theta_percentiles(x, W, ps=(10, 25, 50, 75, 90)):
    """(len(ps), R, 1): percentiles ponderados de x bajo cada peso."""
    n, R = W.shape
    orden = np.argsort(x, kind="mergesort")
    xs, Ws = x[orden], W[orden]
    cum = np.cumsum(Ws, axis=0) - 0.5 * Ws
    tot = Ws.sum(axis=0)
    out = np.empty((len(ps), R, 1))
    for r in range(R):
        f = cum[:, r] / tot[r]
        out[:, r, 0] = np.interp([p / 100 for p in ps], f, xs)
    return out


def resume(th):
    """theta (k,R,M) -> (estimacion, ee, var_muestral, var_imputacion)."""
    t0 = th[:, 0, :]                                  # (k,M) peso final
    tr = th[:, 1:, :]                                 # (k,R-1,M)
    v_mue = (FAY * ((tr - t0[:, None, :]) ** 2).sum(axis=1)).mean(axis=1)
    M = t0.shape[1]
    v_imp = (1 + 1 / M) * t0.var(axis=1, ddof=1) if M > 1 else 0.0
    return t0.mean(axis=1), np.sqrt(v_mue + v_imp), v_mue, v_imp


def resume_lineal(th, c):
    """Combinacion lineal c (k,) de los grupos, con su error BRR."""
    return resume(np.einsum("k,krm->krm", c, th).sum(axis=0)[None, ...])
