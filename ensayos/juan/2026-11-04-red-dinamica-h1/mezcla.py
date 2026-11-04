"""Mezcla walk-forward de la red con B: el peso de cada bloque de semanas sale solo de anos anteriores."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

CARPETA = Path(__file__).resolve().parent
BLOQUES = [(1, 4), (5, 8), (9, 13), (14, 19), (20, 26)]
PRIMER_ANIO_PESOS = 2011


def tabla(etiquetas: list[str]) -> pd.DataFrame:
    B = pd.read_parquet(CARPETA / "b_walkforward.parquet")
    B = B[B.semana <= 26]
    red = pd.concat([pd.read_parquet(CARPETA / f"predicciones_{e}.parquet") for e in etiquetas])
    red = red.groupby(["anio", "id_departamento", "semana"], as_index=False).pred.mean()
    return red.merge(B[["anio", "id_departamento", "semana", "b", "y"]], on=["anio", "id_departamento", "semana"])


def peso_bloque(previo: pd.DataFrame) -> float:
    """Minimos cuadrados de y - b = w (red - b), con w en [0, 1]."""
    x = (previo.pred - previo.b).to_numpy()
    r = (previo.y - previo.b).to_numpy()
    if np.dot(x, x) == 0:
        return 0.0
    return float(np.clip(np.dot(x, r) / np.dot(x, x), 0.0, 1.0))


def caminar(T: pd.DataFrame, desde: int = 2014) -> pd.DataFrame:
    filas = []
    for anio in sorted(T.anio.unique()):
        if anio < desde:
            continue
        actual = T[T.anio == anio].copy()
        previo = T[(T.anio >= PRIMER_ANIO_PESOS) & (T.anio < anio) & (T.anio != 2024)]
        actual["mezcla"] = actual.b
        pesos = []
        for lo, hi in BLOQUES:
            w = peso_bloque(previo[previo.semana.between(lo, hi)])
            mascara = actual.semana.between(lo, hi)
            actual.loc[mascara, "mezcla"] = w * actual.loc[mascara, "pred"] + (1 - w) * actual.loc[mascara, "b"]
            pesos.append(round(w, 2))
        fila = {"anio": anio, "pesos": pesos}
        if actual.y.notna().all():
            for col in ("b", "pred", "mezcla"):
                fila[col] = float(np.mean((actual[col] - actual.y) ** 2))
            fila["mitad"] = float(np.mean((0.5 * actual.pred + 0.5 * actual.b - actual.y) ** 2))
        filas.append(fila)
    return pd.DataFrame(filas).set_index("anio")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("etiquetas", nargs="+")
    args = parser.parse_args()
    R = caminar(tabla(args.etiquetas))
    print(R.round(0).to_string())
    m = R.loc[2014:2023]
    print(f"media 2014-2023: B {m.b.mean():.0f} | red {m.pred.mean():.0f} | mezcla {m.mezcla.mean():.0f} | 50/50 {m.mitad.mean():.0f} | mezcla gana {(m.mezcla < m.b).sum()}/10")
