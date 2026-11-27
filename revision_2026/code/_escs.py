"""Construccion del ESCS con la regla de la OCDE.

Fuente: PISA 2025 Technical Report, cap. 22 ("Scaling Procedures and Construct
Validation of Context Questionnaire Data"), seccion "Computation of ESCS", pp. 20-21,
identica a la de PISA 2022 (Avvisati, 2020; OECD, 2020; OECD, 2024):

  1. IMPUTACION. "For students with missing data on one out of the three components, the
     missing component was imputed using a regression equation which was created for each
     country/economy using data from students without any missing components. For each
     student with a missing component, this regression equation was used to predict the
     missing component with the two non-missing components and a random value was added to
     the predicted value to reflect the error of the regression model." La nota 4 precisa
     que el valor aleatorio se extrae de una normal de media 0 y desviacion tipica igual a
     la de los residuos de la regresion de ese pais. "If a student had missing data on more
     than one component, the ESCS score was not computed."
  2. ESTANDARIZACION. Cada componente (imputados incluidos) a media 0 y DT 1 sobre los
     paises de la OCDE, con cada pais pesando aproximadamente igual (senate weights).
  3. MEDIA ARITMETICA de los tres componentes estandarizados (ESCS preliminar).
  4. REESTANDARIZACION del ESCS preliminar sobre la OCDE, otra vez con senate weights.

La UNICA diferencia con la OCDE en este proyecto es el ambito de la estandarizacion: aqui
el pool son las dos (o cuatro) olas juntas, para que la metrica no se reancle en cada
ciclo. El ESCS publicado se reancla, y por eso no es comparable entre olas.
"""
import numpy as np
import pandas as pd


def senate(d, claves=("cnt", "wave"), peso="w_fstuwt"):
    """Peso de senado: cada celda de `claves` (pais x ola) suma 1."""
    w = d[peso].to_numpy("float64").copy()
    g = d.groupby(list(claves), sort=False).ngroup().to_numpy()
    return w / np.bincount(g, w)[g]


def zpool(x, w):
    """Estandariza sobre TODO el pool con el peso w (media 0, DT 1)."""
    x = np.asarray(x, "float64")
    m = np.isfinite(x) & np.isfinite(w)
    mu = np.average(x[m], weights=w[m])
    sd = np.sqrt(np.average((x[m] - mu) ** 2, weights=w[m]))
    return (x - mu) / sd


def imputar(X, grupo, semilla=20252022, pesos=None, ruido=True):
    """Imputacion OCDE del componente que falta cuando solo falta UNO.

    X       (n, 3) con NaN en lo que falta, en la metrica original del componente.
    grupo   (n,) identificador de pais/economia x ola: la regresion se estima en cada uno.
    pesos   si se pasan, la regresion es ponderada (la OCDE no lo especifica; por defecto
            OLS sin ponderar sobre los casos completos, como en el codigo de referencia).
    ruido   False suprime el residuo aleatorio (variante determinista, para robustez).

    Devuelve (Xi, imputado, n_falta) donde `imputado` marca las celdas rellenadas y
    `n_falta` cuenta componentes ausentes por alumno ANTES de imputar.
    """
    X = np.array(X, dtype="float64", copy=True)
    n, k = X.shape
    assert k == 3, "la regla de la OCDE es para los tres componentes de ESCS"
    falta = ~np.isfinite(X)
    n_falta = falta.sum(axis=1)
    imputado = np.zeros_like(falta)
    rng = np.random.default_rng(semilla)
    grupo = np.asarray(grupo)

    # numpy levanta avisos espurios de BLAS en matmul con matrices grandes; los valores
    # resultantes se comprueban finitos mas abajo.
    for g in np.unique(grupo):
        sel = grupo == g
        Xg = X[sel]
        fg = ~np.isfinite(Xg)
        completo = fg.sum(axis=1) == 0
        if completo.sum() < 30:                      # sin base para estimar: no se imputa
            continue
        wg = None if pesos is None else np.asarray(pesos)[sel][completo]
        for j in range(3):
            obj = (fg.sum(axis=1) == 1) & fg[:, j]   # falta solo el componente j
            if not obj.any():
                continue
            otros = [c for c in range(3) if c != j]
            A = np.column_stack([np.ones(completo.sum()), Xg[completo][:, otros]])
            y = Xg[completo][:, j]
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                if wg is None:
                    b, *_ = np.linalg.lstsq(A, y, rcond=None)
                    res = y - A @ b
                    sd = res.std(ddof=A.shape[1])
                else:
                    sw = np.sqrt(wg)
                    b, *_ = np.linalg.lstsq(A * sw[:, None], y * sw, rcond=None)
                    res = y - A @ b
                    sd = np.sqrt(np.average(res ** 2, weights=wg) * len(y) / (len(y) - A.shape[1]))
                A2 = np.column_stack([np.ones(obj.sum()), Xg[obj][:, otros]])
                pred = A2 @ b
            if ruido:
                pred = pred + rng.normal(0.0, sd, obj.sum())
            assert np.isfinite(pred).all(), f"imputacion no finita en el grupo {g}"
            idx = np.where(sel)[0][obj]
            X[idx, j] = pred
            imputado[idx, j] = True
    X[n_falta >= 2] = np.nan                          # dos o mas ausentes: sin ESCS
    return X, imputado, n_falta


def escs(X, grupo, w, semilla=20252022, imputacion="regresion", pesos_reg=None, ruido=True):
    """ESCS por la regla de la OCDE. Devuelve (escs, Z, info).

    imputacion: "regresion" (OCDE), "media" (media de los componentes disponibles, sin
    imputar), o "completos" (solo alumnos con los tres componentes).
    ruido: solo con "regresion"; False usa la prediccion sin el residuo aleatorio.
    """
    X = np.asarray(X, "float64")
    n_falta = (~np.isfinite(X)).sum(axis=1)
    if imputacion == "regresion":
        Xi, imp, n_falta = imputar(X, grupo, semilla, pesos_reg, ruido)
        Z = np.column_stack([zpool(Xi[:, j], w) for j in range(3)])
        prelim = Z.mean(axis=1)
    elif imputacion == "media":
        Z = np.column_stack([zpool(X[:, j], w) for j in range(3)])
        with np.errstate(invalid="ignore"):
            prelim = np.where(n_falta <= 1, np.nanmean(Z, axis=1), np.nan)
        imp = np.zeros_like(X, bool)
    elif imputacion == "completos":
        Z = np.column_stack([zpool(X[:, j], w) for j in range(3)])
        prelim = np.where(n_falta == 0, Z.mean(axis=1), np.nan)
        imp = np.zeros_like(X, bool)
    else:
        raise ValueError(imputacion)
    prelim = np.where(np.isfinite(prelim), prelim, np.nan)
    info = dict(n_falta=n_falta, imputado=imp,
                pc_imputado=float(np.mean(imp.any(axis=1)) * 100),
                cobertura=float(np.mean(np.isfinite(prelim)) * 100))
    return zpool(prelim, w), Z, info
