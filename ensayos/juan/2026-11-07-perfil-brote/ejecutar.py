"""Outbreak profile versus calm profile, gated on the first semester.

History is not censored at 2 or 2.8 times the median. A department uses the
outbreak profile only when last year was high and the year still closed high.
A crash uses the calm profile. The mix with horizon B is one weight, chosen
on earlier years only. 2024 does not choose that weight.
"""

from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[3]
EXCESO = 1.8
CIERRE_ALTO = 1.0
CIERRE_CAIDA = 0.45
PESOS = (0.0, 0.25, 0.5, 0.75, 1.0)


def suavizar(valores: np.ndarray) -> np.ndarray:
    extendida = np.concatenate([valores[-2:], valores, valores[:2]])
    nucleo = np.ones(5) / 5
    # Window of 5 on a series padded by 2 weeks is already aligned to the 52 weeks.
    return np.convolve(extendida, nucleo, mode="valid")


def perfil_semanal(casos: pd.DataFrame) -> np.ndarray:
    tabla = np.zeros(52)
    if casos.empty:
        return tabla
    for semana, grupo in casos.groupby("semana"):
        if 1 <= int(semana) <= 52:
            tabla[int(semana) - 1] = float(grupo.casos_dengue.median())
    return suavizar(tabla)


def armar(hist: pd.DataFrame, previo: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return calm profile, outbreak profile, recent profile, and the year-before state.

    Profiles are shaped (departamento, semana) with weeks 1..52 in columns 0..51.
    State columns are excess and close-over-peak for the previous year.
    """
    anual = hist.groupby(["id_departamento", "anio"]).casos_dengue.mean()
    referencia = anual.groupby("id_departamento").median().replace(0, 1.0)
    deptos = sorted(hist.id_departamento.unique())
    calmo = np.zeros((max(deptos) + 1, 52))
    brote = np.zeros_like(calmo)
    reciente = np.zeros_like(calmo)
    anio_max = int(hist.anio.max())
    for depto in deptos:
        ref = float(referencia.loc[depto])
        serie = anual.xs(depto, level=0)
        anos_brote = [int(anio) for anio, media in serie.items() if media >= EXCESO * ref]
        anos_calmo = [int(anio) for anio, media in serie.items() if media <= ref]
        propios = hist[hist.id_departamento == depto]
        calmo[depto] = perfil_semanal(propios[propios.anio.isin(anos_calmo)])
        fuente_brote = propios[propios.anio.isin(anos_brote)]
        brote[depto] = perfil_semanal(fuente_brote if len(anos_brote) >= 2 else propios)
        reciente[depto] = perfil_semanal(propios[propios.anio >= anio_max - 5])
    estado = np.ones((max(deptos) + 1, 2))
    for depto, grupo in previo.groupby("id_departamento"):
        semana = grupo.semana.to_numpy()
        casos = grupo.casos_dengue.to_numpy(float)
        h1 = casos[semana <= 26]
        mitad = casos[(semana >= 14) & (semana <= 26)]
        pico = max(float(h1.mean()) if len(h1) else 1.0, float(mitad.mean()) if len(mitad) else 1.0, 1.0)
        cierre = (
            0.50 * casos[semana >= 51].mean()
            + 0.35 * casos[semana >= 47].mean()
            + 0.15 * casos[semana >= 41].mean()
        )
        ref = float(referencia.loc[depto]) if depto in referencia.index else 1.0
        estado[int(depto), 0] = float(casos.mean()) / ref
        estado[int(depto), 1] = float(cierre) / pico
    return calmo, brote, reciente, estado


def elegir(exceso: float, cierre: float, calmo: np.ndarray, brote: np.ndarray, reciente: np.ndarray) -> np.ndarray:
    if exceso >= EXCESO and cierre >= CIERRE_ALTO:
        return brote
    if exceso >= EXCESO and cierre < CIERRE_CAIDA:
        return calmo
    return reciente


def predecir_anio(train: pd.DataFrame, anio: int) -> pd.DataFrame:
    hist = train[train.anio < anio]
    previo = train[train.anio == anio - 1]
    calmo, brote, reciente, estado = armar(hist, previo)
    actual = train[train.anio == anio][["id_departamento", "anio", "semana", "casos_dengue"]].copy()
    perfil = np.empty(len(actual))
    deptos = actual.id_departamento.to_numpy()
    semanas = actual.semana.to_numpy()
    for i, (depto, semana) in enumerate(zip(deptos, semanas)):
        curva = elegir(estado[depto, 0], estado[depto, 1], calmo[depto], brote[depto], reciente[depto])
        perfil[i] = curva[int(semana) - 1] if 1 <= int(semana) <= 52 else 0.0
    actual["perfil"] = np.maximum(perfil, 0.0)
    actual["exceso"] = estado[deptos, 0]
    actual["cierre"] = estado[deptos, 1]
    return actual


def mejor_peso(b: np.ndarray, perfil: np.ndarray, y: np.ndarray) -> float:
    mejor = None
    for peso in PESOS:
        err = float(np.mean((peso * perfil + (1 - peso) * b - y) ** 2))
        if mejor is None or err < mejor[0]:
            mejor = (err, peso)
    return mejor[1]


def main() -> None:
    train = pd.read_parquet(RAIZ / "data/train.parquet")
    train = train[train.semana <= 52]
    base = pd.read_parquet(RAIZ / "ensayos/juan/2026-11-04-red-dinamica-h1/b_walkforward.parquet")
    piezas = []
    for anio in range(2014, 2025):
        print("perfil", anio, flush=True)
        piezas.append(predecir_anio(train, anio))
    perfiles = pd.concat(piezas, ignore_index=True)
    tabla = perfiles.merge(base, on=["id_departamento", "anio", "semana"], suffixes=("", "_b"))
    h1 = tabla[tabla.semana <= 26].dropna(subset=["casos_dengue", "b"]).copy()
    pesos = {}
    pred = np.empty(len(h1))
    y = h1.casos_dengue.to_numpy(float)
    b = h1.b.to_numpy(float)
    perfil = h1.perfil.to_numpy(float)
    anios = h1.anio.to_numpy(int)
    for anio in range(2014, 2025):
        previo = (anios >= 2014) & (anios < anio) & (anios != 2024)
        if previo.sum() == 0:
            pesos[anio] = 0.0
        else:
            pesos[anio] = mejor_peso(b[previo], perfil[previo], y[previo])
        actual = anios == anio
        pred[actual] = pesos[anio] * perfil[actual] + (1 - pesos[anio]) * b[actual]
    h1 = h1.copy()
    h1["pred"] = pred
    print("pesos", pesos)
    for desde in (2014, 2016):
        m = h1.anio.between(desde, 2023)
        print(
            f"{desde}-2023 perfil {np.mean((h1.loc[m, 'pred'] - h1.loc[m, 'casos_dengue']) ** 2):.1f}",
            f"B {np.mean((h1.loc[m, 'b'] - h1.loc[m, 'casos_dengue']) ** 2):.1f}",
        )
    for anio in range(2014, 2025):
        m = h1.anio == anio
        if h1.loc[m, "casos_dengue"].isna().all():
            continue
        yy = h1.loc[m, "casos_dengue"]
        print(
            anio,
            "peso", pesos[anio],
            "pred", round(float(np.mean((h1.loc[m, "pred"] - yy) ** 2)), 1),
            "B", round(float(np.mean((h1.loc[m, "b"] - yy) ** 2)), 1),
        )
    print("filas H1", len(h1))
    h1.to_parquet(Path(__file__).resolve().parent / "walkforward_h1.parquet", index=False)


if __name__ == "__main__":
    main()
