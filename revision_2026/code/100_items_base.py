"""Base de items de HOMEPOS en version de tendencia, cuatro olas, paises OCDE (muestra principal).

Salida: data/metodos/items.parquet — una fila por alumno con los slots recodificados, los items propios
de pais (respuesta + prefijo de adaptacion), la lengua del cuestionario, los indices publicados y, para
2015 y 2018, el HOMEPOS de tendencia oficial de la OCDE (escs_trend.csv, reescalado a 2022 por enlace
de parametros fijos: TR 2022 cap. 19).
"""
import sys, numpy as np, pandas as pd
import os
sys.path.insert(0, "code")
import _load as pisa, _items_homepos as I

# Cycles to read (default: the four). The paper's Section 4 only needs 2022 and 2025:
#     python code/100_items_base.py 2022 2025
OLAS = [int(a) for a in sys.argv[1:]] or list(I.OLAS)
# OECD trend file for 2015 and 2018 (HOMEPOS/ESCS rescaled to 2022, released with the PISA 2022 data).
# Optional: only the 2015-2018 and 2018-2022 windows use it, not the paper.
TREND = os.environ.get("PISA_TREND_CSV", "")
os.makedirs("data/metodos", exist_ok=True)

partes = []
for ola in OLAS:
    brutos = sorted({v for s in I.SLOTS.values() for o, (v, f) in s.items() if o == ola}
                    | {v for v, K in I.PROPIOS[ola]} | {"langtest_qqq", "st255q01ja"}
                    | ({"st253q01ja"} if ola == 2022 else set()))
    d = pisa.cargar(ola=ola, ocde=True, vars=brutos)
    base = d[["wave", "cnt", "region", "cntschid", "cntstuid", "w_fstuwt", "senwt", "escs", "hisei", "paredint",
              "homepos"] + [f"pv{i}{m}" for m in ("math", "read", "scie") for i in range(1, 11)]].copy()
    base["lang"] = d["langtest_qqq"].astype("string").fillna("NA").str.replace(r"\.0$", "", regex=True)
    X = I.recodifica(d, ola)
    P = I.propios(d, ola)
    extra = [I.amplia2022(d)] if ola == 2022 else []
    out = pd.concat([base, X, P] + extra, axis=1)
    print(f"{ola}: {out.cnt.nunique()} paises OCDE, {len(out):,} alumnos, {X.shape[1]} slots, "
          f"{len(I.PROPIOS[ola])} propios")
    cob = X.notna().mean().mul(100).round(1)
    print("   cobertura media de los slots: %.1f%%  (min %s %.1f%%)" % (cob.mean(), cob.idxmin(), cob.min()))
    partes.append(out)

d = pd.concat(partes, ignore_index=True)
# Fixed row order (cycle, country, school, student): the random tie-breaks of the quartiles and the random
# residual of the ESCS imputation follow it, so the results only reproduce exactly with this order.
d["cntstuid"] = d["cntstuid"].astype("int64")
d["schoolid"] = (d["cntschid"].fillna(-1) % 100000).astype("int64")
d = d.sort_values(["wave", "cnt", "schoolid", "cntstuid"]).reset_index(drop=True)

# --- HOMEPOS de tendencia oficial (OCDE 2022) para 2015 y 2018
d["homepos_trend"] = np.nan; d["escs_trend"] = np.nan
if {2015, 2018} & set(OLAS) and not os.path.exists(TREND):
    print("  PISA_TREND_CSV not set: homepos_trend/escs_trend left missing for 2015 and 2018")
if {2015, 2018} & set(OLAS) and os.path.exists(TREND):
  d = d.drop(columns=["homepos_trend", "escs_trend"])
  t = pd.read_csv(TREND, usecols=["cycle", "cnt", "schoolid", "studentid", "homepos_trend", "escs_trend",
                                "hisei_trend", "paredint_trend"])
  t = t[t.cycle.isin([6, 7])].copy()
  t["wave"] = t.cycle.map({6: 2015, 7: 2018})
  t = t.drop(columns="cycle")
  # Los IDs del fichero de tendencia no son los del PUF en 2015 (en 2018 si: cntstuid mod 1e5), pero
  # los centros si (schoolid = cntschid mod 1e5) y el orden de alumnos dentro de cada centro coincide:
  # se casa por (ola, pais, centro, rango del alumno). Verificado: tamanos de centro identicos en el
  # 100% y r(hisei, hisei_trend) = 0,998 alumno a alumno.
  d["cntstuid"] = d["cntstuid"].astype("int64")
  d["schoolid"] = (d["cntschid"].fillna(-1) % 100000).astype("int64")
  d = d.sort_values(["wave", "cnt", "schoolid", "cntstuid"]).reset_index(drop=True)
  t = t.sort_values(["wave", "cnt", "schoolid", "studentid"])
  d["rk"] = d.groupby(["wave", "cnt", "schoolid"]).cumcount()
  t["rk"] = t.groupby(["wave", "cnt", "schoolid"]).cumcount()
  tam = pd.concat([d[d.wave.isin([2015, 2018])].groupby(["wave", "cnt", "schoolid"]).size(),
                   t.groupby(["wave", "cnt", "schoolid"]).size()], axis=1).dropna()
  print(f"  centros con el mismo numero de alumnos en PUF y tendencia: {(tam[0] == tam[1]).mean()*100:.2f}%")
  n0 = len(d)
  d = d.merge(t.drop(columns=["studentid", "oecd"], errors="ignore"),
              on=["wave", "cnt", "schoolid", "rk"], how="left", validate="1:1")
  assert len(d) == n0
  d = d.drop(columns=["rk"])
  for ola in (2015, 2018):
      m = d.wave == ola
      print(f"  tendencia oficial {ola}: casados {d.loc[m, 'homepos_trend'].notna().mean()*100:.1f}% "
            f"(homepos publicado {d.loc[m, 'homepos'].notna().mean()*100:.1f}%)  "
            f"r(publicado, tendencia) = {d.loc[m, ['homepos', 'homepos_trend']].corr().iloc[0, 1]:.4f}")
m = d.wave == 2022
d.loc[m, "homepos_trend"] = d.loc[m, "homepos"]          # 2022 es la referencia de la tendencia oficial
d.loc[m, "escs_trend"] = d.loc[m, "escs"]

d.to_parquet("data/metodos/items.parquet", index=False)
print(f"\nguardado data/metodos/items.parquet  {d.shape}")
print(d.groupby("wave").cnt.nunique().to_dict())
