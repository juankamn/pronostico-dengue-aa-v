"""Bases por horizonte (B) con y sin la regla de caida, en walk-forward 2011-2025.

La regla de caida es un "si pasa X, usar tal valor": si el ano anterior supero 2,2 veces lo normal y el
cierre quedo bajo el 65 % del pico, la base de cierre pasa a la mediana historica y la razon a 0,85. Aqui
se compara B tal como esta con B sin esa regla, dejando que los pesos ajustados por minimos cuadrados decidan.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

CARPETA = Path(__file__).resolve().parent
RAIZ = CARPETA.parents[2]
BASES_B = ["b_med_all", "b_mean_all", "b_rec6", "b_med5", "b_rec3", "b_end_direct", "b_end_scaled"]


def suavizar(serie):
    tabla = serie.unstack("semana")
    columnas = [semana for semana in range(1, 53) if semana in tabla.columns]
    tabla = tabla[columnas]
    extendida = pd.concat([tabla[columnas[-2:]], tabla, tabla[columnas[:2]]], axis=1)
    suave = extendida.T.rolling(5, center=True, min_periods=1).mean().T.iloc[:, 2:-2]
    suave.columns = pd.Index(columnas, name="semana")
    return suave.stack()


def marco_bases(hist, cur, regla_caida=True):
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
    es_caida = (media_previa > 2.2 * nivel) & (cierre < 0.65 * pico)
    razon = (cierre / nivel_cierre).fillna(1.0)
    razon_suave = pd.Series(np.where(razon > 1.15, 1.15 + np.log1p(np.maximum(razon - 1.15, 0.0)), np.clip(razon, 0.25, 1.15)), index=razon.index)
    indice = pd.MultiIndex.from_arrays([cur.id_departamento, cur.semana])
    salida = cur[["id_departamento", "anio", "semana"]].copy()
    for nombre, serie in perfiles.items():
        salida[nombre] = serie.reindex(indice).fillna(0.0).to_numpy()
    caida = cur.id_departamento.map(es_caida).fillna(False).astype(float).to_numpy() if regla_caida else np.zeros(len(cur))
    razon_fila = np.where(caida > 0.5, 0.85, cur.id_departamento.map(razon_suave).fillna(1.0).to_numpy())
    cierre_fila = cur.id_departamento.map(cierre).fillna(0.0).to_numpy()
    salida["b_end_direct"] = np.where(caida > 0.5, salida["b_med_all"], np.minimum(cierre_fila, 2.2 * salida["b_rec6"]))
    salida["b_end_scaled"] = salida["b_rec6"].to_numpy() * razon_fila
    if "casos_dengue" in cur.columns and cur.casos_dengue.notna().all():
        salida["y"] = cur.casos_dengue.to_numpy(float)
        media_actual = cur.groupby("id_departamento").casos_dengue.mean()
        salida["brote"] = cur.id_departamento.map(media_actual > 2.3 * nivel).fillna(False).to_numpy()
    else:
        salida["y"] = np.nan
        salida["brote"] = False
    return salida.reset_index(drop=True)


def pesos_horizonte(marcos, semana_min, semana_max):
    entrenamiento = pd.concat([t.loc[t.semana.between(semana_min, semana_max) & ~t.brote] for t in marcos], ignore_index=True)
    matriz = entrenamiento[BASES_B].to_numpy(float)
    objetivo = entrenamiento.y.to_numpy(float)
    escala = float(np.mean(matriz**2))
    centro = np.full(len(BASES_B), 1.0 / len(BASES_B))
    costo = lambda p: float(np.mean((matriz @ p - objetivo) ** 2) + 0.05 * escala * np.sum((p - centro) ** 2))
    return minimize(costo, centro, bounds=[(0.0, 1.0)] * len(BASES_B)).x


def aplicar_pesos(tabla, q1, q2, h2):
    matriz = tabla[BASES_B].to_numpy(float)
    semana = tabla.semana.to_numpy()
    return np.where(semana <= 13, matriz @ q1, np.where(semana <= 26, matriz @ q2, matriz @ h2))


def caminar(regla_caida: bool) -> pd.DataFrame:
    train = pd.read_parquet(RAIZ / "data/train.parquet").sort_values(["id_departamento", "anio", "semana"]).reset_index(drop=True)
    test = pd.read_parquet(RAIZ / "data/test.parquet").sort_values(["id_departamento", "anio", "semana"]).reset_index(drop=True)
    marcos = {a: marco_bases(train[train.anio < a], train[(train.anio == a) & (train.semana <= 52)].reset_index(drop=True), regla_caida) for a in range(2010, 2025)}
    marcos[2025] = marco_bases(train, test, regla_caida)
    piezas = []
    for anio in range(2011, 2026):
        previos = [marcos[a] for a in range(2010, anio)]
        pesos = [pesos_horizonte(previos, a, b) for a, b in ((1, 13), (14, 26), (27, 52))]
        tabla = marcos[anio][["id_departamento", "anio", "semana", "y"]].copy()
        tabla["b"] = np.clip(aplicar_pesos(marcos[anio], *pesos), 0, None)
        piezas.append(tabla)
    return pd.concat(piezas, ignore_index=True)


if __name__ == "__main__":
    for regla in (True, False):
        salida = caminar(regla)
        destino = CARPETA / f"b_walkforward_regla_{'si' if regla else 'no'}.parquet"
        salida.to_parquet(destino)
        h = salida[salida.anio.between(2014, 2023)]
        print(f"regla de caida {'si' if regla else 'no'}: H1 2014-2023 {np.mean((h[h.semana <= 26].b - h[h.semana <= 26].y) ** 2):.0f}"
              f" | H2 {np.mean((h[h.semana >= 27].b - h[h.semana >= 27].y) ** 2):.0f}", flush=True)
