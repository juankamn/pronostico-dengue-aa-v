"""Attempt to rebuild the learned-rule horizon base.

It does not reproduce the notebook: H1 2014-2023 is about 3519 here and 3386
there, and 2025 does not match seleccion 2. Do not submit this series.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

import nivel_h1 as nivel

RAIZ = nivel.RAIZ
BASES = ["b_med_all", "b_mean_all", "b_rec6", "b_med5", "b_rec3", "b_end_direct", "b_end_scaled"]
REGLA = (1.8, 0.45, 0.4)


def suavizar(serie: pd.Series) -> pd.Series:
    tabla = serie.unstack("semana")
    columnas = [semana for semana in range(1, 53) if semana in tabla.columns]
    tabla = tabla[columnas]
    extendida = pd.concat([tabla[columnas[-2:]], tabla, tabla[columnas[:2]]], axis=1)
    suave = extendida.T.rolling(5, center=True, min_periods=1).mean().T.iloc[:, 2:-2]
    suave.columns = pd.Index(columnas, name="semana")
    return suave.stack()


def marco_bases(hist: pd.DataFrame, cur: pd.DataFrame) -> pd.DataFrame:
    anio = int(cur.anio.iloc[0])
    mediana_6 = hist[hist.anio >= anio - 6].groupby("id_departamento").casos_dengue.median()
    mediana_hist = hist.groupby("id_departamento").casos_dengue.median()
    referencia = np.maximum(mediana_6, mediana_hist).replace(0, 1.0)
    historia_baja = hist.copy()
    historia_alta = hist.copy()
    historia_baja["casos_dengue"] = np.minimum(hist.casos_dengue, hist.id_departamento.map(referencia * 2.0))
    historia_alta["casos_dengue"] = np.minimum(hist.casos_dengue, hist.id_departamento.map(referencia * 2.8))
    perfiles = {
        "b_med_all": suavizar(hist.groupby(["id_departamento", "semana"]).casos_dengue.median()),
        "b_mean_all": suavizar(historia_baja.groupby(["id_departamento", "semana"]).casos_dengue.mean()),
        "b_rec6": suavizar(historia_baja[historia_baja.anio >= anio - 6].groupby(["id_departamento", "semana"]).casos_dengue.mean()),
        "b_med5": suavizar(hist[hist.anio >= anio - 5].groupby(["id_departamento", "semana"]).casos_dengue.median()),
        "b_rec3": suavizar(historia_alta[historia_alta.anio >= anio - 3].groupby(["id_departamento", "semana"]).casos_dengue.mean()),
    }
    previo = hist[hist.anio == anio - 1]
    cierre_2 = previo[previo.semana >= 51].groupby("id_departamento").casos_dengue.mean()
    cierre_6 = previo[previo.semana >= 47].groupby("id_departamento").casos_dengue.mean()
    cierre_12 = previo[previo.semana >= 41].groupby("id_departamento").casos_dengue.mean()
    cierre = 0.50 * cierre_2 + 0.35 * cierre_6 + 0.15 * cierre_12
    mitad_1 = previo[previo.semana <= 26].groupby("id_departamento").casos_dengue.mean()
    trimestre_2 = previo[previo.semana.between(14, 26)].groupby("id_departamento").casos_dengue.mean()
    media_previa = previo.groupby("id_departamento").casos_dengue.mean()
    nivel = perfiles["b_rec6"].unstack("semana").mean(axis=1).replace(0, 1.0)
    nivel_cierre = perfiles["b_rec6"].unstack("semana")[[47, 48, 49, 50, 51, 52]].mean(axis=1).replace(0, 1.0)
    pico = np.maximum(mitad_1, trimestre_2).replace(0, 1.0)
    razon = (cierre / nivel_cierre).fillna(1.0)
    razon_suave = pd.Series(
        np.where(razon > 1.15, 1.15 + np.log1p(np.maximum(razon - 1.15, 0.0)), np.clip(razon, 0.25, 1.15)),
        index=razon.index,
    )
    indice = pd.MultiIndex.from_arrays([cur.id_departamento, cur.semana])
    salida = cur[["id_departamento", "anio", "semana"]].copy()
    for nombre, serie in perfiles.items():
        salida[nombre] = serie.reindex(indice).fillna(0.0).to_numpy()
    cierre_fila = cur.id_departamento.map(cierre).fillna(0.0).to_numpy()
    salida["b_end_direct"] = np.minimum(cierre_fila, 2.2 * salida["b_rec6"])
    salida["b_end_scaled"] = salida["b_rec6"].to_numpy() * cur.id_departamento.map(razon_suave).fillna(1.0).to_numpy()
    salida["exceso_previo"] = cur.id_departamento.map(media_previa / nivel).fillna(0.0).to_numpy()
    salida["cierre_sobre_pico"] = cur.id_departamento.map(cierre / pico).fillna(1.0).to_numpy()
    if "casos_dengue" in cur.columns and cur.casos_dengue.notna().all():
        salida["y"] = cur.casos_dengue.to_numpy(float)
        media_actual = cur.groupby("id_departamento").casos_dengue.mean()
        salida["brote"] = cur.id_departamento.map(media_actual > 2.3 * nivel).fillna(False).to_numpy()
    else:
        salida["y"] = np.nan
        salida["brote"] = False
    return salida.reset_index(drop=True)


def aplicar_regla(marco: pd.DataFrame, regla: tuple[float, float, float]) -> pd.DataFrame:
    salida = marco.copy()
    exceso, caida, razon = regla
    m = ((salida.exceso_previo > exceso) & (salida.cierre_sobre_pico < caida)).to_numpy()
    salida.loc[m, "b_end_direct"] = salida.loc[m, "b_med_all"].to_numpy()
    salida.loc[m, "b_end_scaled"] = salida.loc[m, "b_rec6"].to_numpy() * razon
    return salida


def pesos_horizonte(marcos: list[pd.DataFrame], semana_min: int, semana_max: int) -> np.ndarray:
    entrenamiento = pd.concat(
        [t.loc[t.semana.between(semana_min, semana_max) & ~t.brote] for t in marcos],
        ignore_index=True,
    )
    matriz = entrenamiento[BASES].to_numpy(float)
    objetivo = entrenamiento.y.to_numpy(float)
    escala = float(np.mean(matriz**2))
    centro = np.full(len(BASES), 1.0 / len(BASES))

    def costo(pesos: np.ndarray) -> float:
        return float(np.mean((matriz @ pesos - objetivo) ** 2) + 0.05 * escala * np.sum((pesos - centro) ** 2))

    return minimize(costo, centro, bounds=[(0.0, 1.0)] * len(BASES)).x


def aplicar_pesos(tabla: pd.DataFrame, peso_q1: np.ndarray, peso_q2: np.ndarray, peso_h2: np.ndarray) -> np.ndarray:
    matriz = tabla[BASES].to_numpy(float)
    semana = tabla.semana.to_numpy()
    return np.where(semana <= 13, matriz @ peso_q1, np.where(semana <= 26, matriz @ peso_q2, matriz @ peso_h2))


def serie() -> pd.DataFrame:
    train = pd.read_parquet(RAIZ / "data/train.parquet")
    test = pd.read_parquet(RAIZ / "data/test.parquet")
    train = train[train.semana <= 52]
    test = test[test.semana <= 52]
    crudos = {}
    for anio in range(2010, 2025):
        actual = train[(train.anio == anio) & (train.semana <= 52)]
        crudos[anio] = marco_bases(train[train.anio < anio], actual)
    crudos[2025] = marco_bases(train, test)
    marcos = {anio: aplicar_regla(marco, REGLA) for anio, marco in crudos.items()}
    piezas = []
    for anio in range(2011, 2026):
        previos = [marcos[a] for a in range(2010, anio)]
        pesos = [pesos_horizonte(previos, a, b) for a, b in ((1, 13), (14, 26), (27, 52))]
        tabla = marcos[anio][["id_departamento", "anio", "semana", "y"]].copy()
        tabla["b"] = np.clip(aplicar_pesos(marcos[anio], *pesos), 0, None)
        piezas.append(tabla)
        print("anio", anio, flush=True)
    return pd.concat(piezas, ignore_index=True)


def main() -> None:
    tabla = serie()
    tabla.to_parquet(Path(__file__).resolve().parent / "b_aprendida.parquet", index=False)
    h1 = tabla[tabla.semana <= 26]
    m = h1.anio.between(2014, 2023)
    print("sin nivel", round(float(np.mean((h1.loc[m, "b"] - h1.loc[m, "y"]) ** 2)), 1))
    crudo = pd.read_parquet(RAIZ / "ensayos/juan/2026-11-04-red-dinamica-h1/b_walkforward.parquet")
    h2_k = crudo[crudo.semana >= 27]
    for modo in ("region", "tres"):
        r = nivel.evaluar(tabla, h2_k, modo)
        print(modo, "2014", round(r["2014"], 1), "2016", round(r["2016"], 1), "params", r["params_2025"])


if __name__ == "__main__":
    main()
