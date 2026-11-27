"""Catalogo de items de HOMEPOS por ola, en su 'version de tendencia'.

Fuentes: Avvisati y Wuyts (2024), Anexo A, Tabla A.1 (correspondencia 2015/18 <-> 2022);
TR PISA 2022 cap. 19 (recodificaciones de tendencia: todo a las categorias de 2022);
TR PISA 2025 cap. 22, Tabla 22.A.17 (recodificaciones de 2025).

Codigos brutos: en ST251/ST254/ST012 el 1 es "ninguno"; en ST254 el 5 es "no lo se" (-> ausente);
en ST256 1 = ninguno, 2 = 1-5, 3 = 6-10, 4 = mas de 10, 5 = no lo se.
Un `slot` es el concepto: dos variables de olas distintas que comparten slot son el mismo item de
tendencia. Los items nacionales (propios de cada pais) no son slots: cada pais tiene el suyo.
"""
import numpy as np
import pandas as pd

OLAS = (2015, 2018, 2022, 2025)


def yn(s):                      # 1 si, 2 no
    return s.map({1.0: 1, 2.0: 0})

def cnt4(s):                    # 1..4 -> 0..3 ; 5 (no lo se) -> ausente
    return s.where(s.between(1, 4)) - 1

def tri(s):                     # 2022/25 ST254: ninguno / 1-2 / 3-5 / >5 (5 = no lo se) -> ninguno / 1-2 / 3 o mas
    return np.minimum(s.where(s.between(1, 4)) - 1, 2)

def tri_old(s):                 # 2015/18 ST012: ninguno / uno / dos / tres o mas -> ninguno / 1-2 / 3 o mas
    return s.where(s.between(1, 4)).map({1.0: 0.0, 2.0: 1.0, 3.0: 1.0, 4.0: 2.0})

def algun(s):                   # "cuantos" -> binario, como en la tendencia de 2022
    return (s.where(s.between(1, 4)) >= 2).astype(float).where(s.between(1, 4))

def libros13(s):                # ST013 (1..6) -> 0..5
    return s.where(s.between(1, 6)) - 1

def libros255(s):               # ST255 (1..7) -> 0..5: "no hay" y "1-10" se juntan
    return np.maximum(s.where(s.between(1, 7)) - 2, 0)

def ocho(s):                    # ST253 (1..8) -> 0..7
    return s.where(s.between(1, 8)) - 1


# slot -> {ola: (variable, recodificacion)}
SLOTS = {
    # --- de tendencia en las cuatro olas (el nucleo de 13 del paper, mas los de 2022/25)
    "room":        {2015: ("st011q02ta", yn), 2018: ("st011q02ta", yn), 2022: ("st250q01ja", yn), 2025: ("st250q01ja", yn)},
    "computer":    {2015: ("st011q04ta", yn), 2018: ("st011q04ta", yn), 2022: ("st250q02ja", yn), 2025: ("st250q02ja", yn)},
    "software":    {2015: ("st011q05ta", yn), 2018: ("st011q05ta", yn), 2022: ("st250q03ja", yn), 2025: ("st250q03ja", yn)},
    "internet":    {2015: ("st011q06ta", yn), 2018: ("st011q06ta", yn), 2022: ("st250q05ja", yn), 2025: ("st250q05ja", yn)},
    "art":         {2015: ("st011q09ta", yn), 2018: ("st011q09ta", yn), 2022: ("st251q07ja", algun), 2025: ("st251q07ja", algun)},
    "tvs":         {2015: ("st012q01ta", tri_old), 2018: ("st012q01ta", tri_old), 2022: ("st254q01ja", tri), 2025: ("st254q01ja", tri)},
    "cars":        {2015: ("st012q02ta", cnt4), 2018: ("st012q02ta", cnt4), 2022: ("st251q01ja", cnt4), 2025: ("st251q01ja", cnt4)},
    "bathrooms":   {2015: ("st012q03ta", cnt4), 2018: ("st012q03ta", cnt4), 2022: ("st251q03ja", cnt4), 2025: ("st251q03ja", cnt4)},
    "tablets":     {2015: ("st012q07na", tri_old), 2018: ("st012q07na", tri_old), 2022: ("st254q04ja", tri), 2025: ("st254q04ja", tri)},
    "ereaders":    {2015: ("st012q08na", tri_old), 2018: ("st012q08na", tri_old), 2022: ("st254q05ja", tri), 2025: ("st254q05ja", tri)},
    "instruments": {2015: ("st012q09na", cnt4), 2018: ("st012q09na", cnt4), 2022: ("st251q06ja", cnt4), 2025: ("st251q06ja", cnt4)},
    "books":       {2015: ("st013q01ta", libros13), 2018: ("st013q01ta", libros13), 2022: ("st255q01ja", libros255), 2025: ("st255q01ja", libros255)},
    # --- de tendencia 2015/18 <-> 2022 (desaparecen en 2025)
    "classic_lit": {2015: ("st011q07ta", yn), 2018: ("st011q07ta", yn), 2022: ("st256q02ja", algun)},
    "books_school":{2015: ("st011q10ta", yn), 2018: ("st011q10ta", yn), 2022: ("st256q10ja", algun)},
    "tech_books":  {2015: ("st011q11ta", yn), 2018: ("st011q11ta", yn), 2022: ("st256q08ja", algun)},
    "dictionary":  {2015: ("st011q12ta", yn), 2018: ("st011q12ta", yn), 2022: ("st256q09ja", algun)},
    "art_books":   {2015: ("st011q16na", yn), 2018: ("st011q16na", yn), 2022: ("st256q07ja", algun)},
    "smartphones": {2015: ("st012q05na", tri_old), 2018: ("st012q05na", tri_old), 2022: ("st254q06ja", tri)},
    # --- solo 2015/18
    "desk":        {2015: ("st011q01ta", yn), 2018: ("st011q01ta", yn)},
    "quiet_place": {2015: ("st011q03ta", yn), 2018: ("st011q03ta", yn)},
    "poetry":      {2015: ("st011q08ta", yn), 2018: ("st011q08ta", yn)},
    "computers_n": {2015: ("st012q06na", cnt4), 2018: ("st012q06na", cnt4)},
    # --- 2022 y 2025
    "own_phone":   {2022: ("st250q04ja", yn), 2025: ("st250q04ja", yn)},
    "toilets":     {2022: ("st251q04ja", cnt4), 2025: ("st251q04ja", cnt4)},
    "desktops":    {2022: ("st254q02ja", cnt4), 2025: ("st254q02ja", cnt4)},
    "laptops":     {2022: ("st254q03ja", cnt4), 2025: ("st254q03ja", cnt4)},
    # --- solo 2022
    "mopeds":      {2022: ("st251q02ja", cnt4)},
    "screens":     {2022: ("st253q01ja", ocho)},
    "religious":   {2022: ("st256q01ja", algun)},
    "st256q03":    {2022: ("st256q03ja", algun)},
    "st256q06":    {2022: ("st256q06ja", algun)},
}
# --- solo 2025: los 20 items de seleccion nacional (parametros internacionales en la OCDE)
NAC25 = ["st250q%02dda" % i for i in list(range(8, 17)) + list(range(18, 29))]
NOMBRES_NAC25 = {
    "st250q08da": "electric_light", "st250q09da": "plumbed_water", "st250q10da": "parking",
    "st250q11da": "garbage", "st250q12da": "stove", "st250q13da": "meal_table",
    "st250q14da": "vacuum", "st250q15da": "fridge", "st250q16da": "sewer", "st250q18da": "aircon_heating",
    "st250q19da": "guest_room", "st250q20da": "quiet_room", "st250q21da": "pool", "st250q22da": "alarm",
    "st250q23da": "smart_tv", "st250q24da": "dishwasher", "st250q25da": "own_tablet",
    "st250q26da": "tv_subscription", "st250q27da": "domestic_workers", "st250q28da": "newspaper"}
for v, nm in NOMBRES_NAC25.items():
    SLOTS[nm] = {2025: (v, yn)}

# items propios de cada pais (cadena: prefijo de adaptacion de 6 digitos + respuesta)
PROPIOS = {2015: [("st011d17ta", 2), ("st011d18ta", 2), ("st011d19ta", 2)],
           2018: [("st011d17ta", 2), ("st011d18ta", 2), ("st011d19ta", 2)],
           2022: [("st250d06ja", 2), ("st250d07ja", 2), ("st251d08ja", 4), ("st251d09ja", 4)],
           2025: [("st251d08ja", 4), ("st251d09ja", 4)]}

# Auditoria de comparabilidad de los 13 items de cuatro olas (enunciado, categorias, filtro; ver 112):
#   nivel 1 identicos: room, bathrooms, instruments, cars
#   nivel 2 categorias que anidan exactamente: books, tvs, tablets, ereaders
#   nivel 3 cambio de enunciado o de formato: computer (+tablet), software (+apps), internet (sin smartphone),
#           art (si/no -> cuantos), computers_n (una pregunta -> sobremesa + portatil)
NIVEL1 = ["room", "bathrooms", "instruments", "cars"]
NIVEL2 = ["books", "tvs", "tablets", "ereaders"]
NIVEL3 = ["computer", "software", "internet", "art", "computers_n"]

# bloques para las ablaciones
BLOQUES = {
    "st256": ["classic_lit", "books_school", "tech_books", "dictionary", "art_books",
                                    "religious", "st256q03", "st256q06"],
    "solo22": ["mopeds", "screens", "smartphones"],
    "nac25": list(NOMBRES_NAC25.values()),
    "propios": ["__propios__"],
    "solo1518": ["desk", "quiet_place", "poetry", "computers_n"],
}


def slots_de(ola):
    return [s for s, d in SLOTS.items() if ola in d]


def comunes(olas):
    return [s for s, d in SLOTS.items() if all(o in d for o in olas)]


def recodifica(df, ola):
    """DataFrame de slots (float, NaN = ausente) para una ola, con los saltos de filtro de 2022."""
    out = pd.DataFrame(index=df.index)
    for s, d in SLOTS.items():
        if ola in d:
            v, f = d[ola]
            out[s] = f(df[v].astype("float64")).astype("float64")
    if ola == 2022:
        sin_libros = df["st255q01ja"] == 1            # filtro: sin libros no ven ST256 -> ninguno
        for s in ["classic_lit", "books_school", "tech_books", "dictionary", "art_books",
                  "religious", "st256q03", "st256q06"]:
            out.loc[sin_libros & out[s].isna(), s] = 0.0
        sin_pantallas = df["st253q01ja"] == 1         # filtro: sin pantallas no ven ST254 -> ninguno
        for s in ["tvs", "tablets", "ereaders", "smartphones", "desktops", "laptops"]:
            out.loc[sin_pantallas & out[s].isna(), s] = 0.0
    return out


ST256 = {"classic_lit": "st256q02ja", "books_school": "st256q10ja", "tech_books": "st256q08ja", "dictionary": "st256q09ja",
         "art_books": "st256q07ja", "religious": "st256q01ja", "st256q03": "st256q03ja", "st256q06": "st256q06ja"}


def amplia2022(df):
    """Version 'amplia' de 2022: ST256 con sus 4 categorias (ninguno, 1-5, 6-10, >10) y smartphones con 4
    (ninguno, 1-2, 3-5, >5), en vez de colapsar a la version de tendencia. Columnas 'x4_<slot>'."""
    out = pd.DataFrame(index=df.index)
    sin_libros = df["st255q01ja"] == 1
    for s, v in ST256.items():
        x = cnt4(df[v].astype("float64"))
        out["x4_" + s] = x.where(~(sin_libros & x.isna()), 0.0)
    x = cnt4(df["st254q06ja"].astype("float64"))
    out["x4_smartphones"] = x.where(~((df["st253q01ja"] == 1) & x.isna()), 0.0)
    return out


def propios(df, ola):
    """Items propios de pais en formato largo-ancho: columnas '<var>' con respuesta recodificada y
    '<var>_pref' con el prefijo de adaptacion (clave del conjunto de parametros)."""
    out = pd.DataFrame(index=df.index)
    for v, K in PROPIOS[ola]:
        s = df[v].astype("string").str.strip()
        valido = s.str.len().fillna(0) >= 7
        pref = s.where(valido).str.slice(0, 6)
        resp = pd.to_numeric(s.where(valido).str.slice(6), errors="coerce")
        if K == 2:
            r = resp.map({1: 1.0, 2: 0.0})
        else:
            r = (resp.where(resp.between(1, 4)) - 1).astype("float64")
        out[v] = r
        out[v + "_pref"] = pref.where(r.notna())
    return out

ETIQUETAS_BLOQUES = {"st256": "ST256 book types (2022)", "solo22": "Other 2022-only (mopeds, screens, smartphones)",
                     "nac25": "National-selection items 2025 (ST250Q08-28DA)", "propios": "Country-specific items",
                     "solo1518": "2015/18-only (desk, quiet place, poetry, computers)"}
