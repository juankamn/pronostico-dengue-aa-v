"""Que par de selecciones finales habria rendido mejor en el pasado.

Kaggle se queda con el mejor de los dos envios elegidos. Para cada ano 2014-2023 se calcula el MSE del primer
semestre de cada candidato y, para cada par, el minimo de los dos. La media de ese minimo estima cuanto vale
el par. C es el escenario de continuidad construido igual que en 2025 (0,8 del nivel de diciembre, coseno
hasta la semana 27, solo sube).
"""

import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

CARPETA = Path(__file__).resolve().parent
RAIZ = CARPETA.parents[2]
ETIQUETA = sys.argv[1] if len(sys.argv) > 1 else "final5"
FRACCION = 0.8


def continuidad(tabla: pd.DataFrame, columna: str, nivel: pd.Series) -> np.ndarray:
    apertura = tabla[tabla.semana <= 4].groupby("id_departamento")[columna].mean()
    brecha = (np.log1p(FRACCION * nivel) - np.log1p(apertura)).clip(lower=0.0)
    peso = 0.5 * (1 + np.cos(np.pi * (tabla.semana.to_numpy() - 1) / 26))
    return np.expm1(np.log1p(tabla[columna].to_numpy()) + tabla.id_departamento.map(brecha).to_numpy() * peso)


def main() -> None:
    train = pd.read_parquet(RAIZ / "data/train.parquet")
    B = pd.read_parquet(CARPETA / "b_walkforward.parquet")
    B = B[B.semana <= 26][["anio", "id_departamento", "semana", "b", "y"]]
    red = pd.read_parquet(CARPETA / f"predicciones_{ETIQUETA}.parquet")
    ancho = red.pivot_table(index=["anio", "id_departamento", "semana"], columns="familia", values="pred").reset_index()
    T = ancho.merge(B, on=["anio", "id_departamento", "semana"])
    T["ens"] = T[["mlp", "rnn", "lstm", "gru", "tcn"]].mean(axis=1)
    T["tcn_b"] = 0.5 * T.tcn + 0.5 * T.b
    T["mlp_b"] = 0.5 * T.mlp + 0.5 * T.b
    T["tm_b"] = 0.25 * T.tcn + 0.25 * T.mlp + 0.5 * T.b
    T["ens_b"] = 0.5 * T.ens + 0.5 * T.b

    filas = []
    for anio in range(2014, 2024):
        t = T[T.anio == anio].sort_values(["id_departamento", "semana"]).copy()
        nivel = train[(train.anio == anio - 1) & train.semana.between(45, 52)].groupby("id_departamento").casos_dengue.median()
        t["C"] = continuidad(t, "b", nivel)
        t["C_tm"] = continuidad(t, "tm_b", nivel)
        fila = {"anio": anio}
        for col in ("b", "tcn_b", "mlp_b", "tm_b", "ens_b", "C", "C_tm"):
            fila[col] = float(np.mean((t[col] - t.y) ** 2))
        filas.append(fila)
    R = pd.DataFrame(filas).set_index("anio")
    print(R.round(0).to_string())
    print("\nmedia individual:", R.mean().round(0).to_dict())
    pares = []
    for a, b in itertools.combinations(R.columns, 2):
        minimo = np.minimum(R[a], R[b])
        pares.append({"par": f"{a} + {b}", "media_min": minimo.mean(), "gana_primero": int((R[a] < R[b]).sum())})
    print(pd.DataFrame(pares).sort_values("media_min").round(0).head(12).to_string(index=False))


if __name__ == "__main__":
    main()
