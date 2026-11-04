"""Tabla final de F3: cada familia sola, en mezcla 50/50 con B y con pesos walk-forward, y su lectura de 2025."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

CARPETA = Path(__file__).resolve().parent
RAIZ = CARPETA.parents[2]
sys.path.insert(0, str(CARPETA))
from mezcla import caminar  # noqa: E402

ETIQUETA = sys.argv[1] if len(sys.argv) > 1 else "final5"


def cargar() -> pd.DataFrame:
    B = pd.read_parquet(CARPETA / "b_walkforward.parquet")
    B = B[B.semana <= 26][["anio", "id_departamento", "semana", "b", "y"]]
    red = pd.read_parquet(CARPETA / f"predicciones_{ETIQUETA}.parquet")
    ensamble = red.groupby(["anio", "id_departamento", "semana"], as_index=False).pred.mean().assign(familia="ensamble")
    red = pd.concat([red, ensamble], ignore_index=True)
    return red.merge(B, on=["anio", "id_departamento", "semana"])


def main() -> None:
    T = cargar()
    filas, por_anio = [], {}
    for familia, bloque in T.groupby("familia", sort=False):
        R = caminar(bloque.drop(columns="familia"))
        m = R.loc[2014:2023]
        por_anio[familia] = R.pred.round(0)
        filas.append({
            "familia": familia,
            "red_2014_2023": m.pred.mean(),
            "mitad_con_B": m.mitad.mean(),
            "mezcla_wf": m.mezcla.mean(),
            "B": m.b.mean(),
            "gana_a_B": int((m.pred < m.b).sum()),
            "red_2011_2013": R.loc[2011:2013].pred.mean() if 2011 in R.index else np.nan,
            "red_2024": R.loc[2024, "pred"] if 2024 in R.index else np.nan,
            "B_2024": R.loc[2024, "b"] if 2024 in R.index else np.nan,
            "pesos_2025": R.loc[2025, "pesos"] if 2025 in R.index else None,
        })
    tabla = pd.DataFrame(filas)
    print(tabla.round(0).to_string(index=False))
    anual = pd.DataFrame(por_anio)
    anual["B"] = caminar(T[T.familia == "gru"].drop(columns="familia")).b.round(0)
    print("\nMSE del primer semestre por ano")
    print(anual.to_string())

    # Lectura de 2025
    C = pd.read_csv(RAIZ / "ensayos/juan/2026-11-04-h1-continuidad/submission.csv")
    partes = C.id.str.split("_", expand=True).astype(int)
    C = C.assign(id_departamento=partes[0], semana=partes[2])
    C = C[C.semana <= 26].rename(columns={"dengue": "c"})[["id_departamento", "semana", "c"]]
    actual = T[T.anio == 2025].merge(C, on=["id_departamento", "semana"])
    nombres = pd.read_parquet(RAIZ / "data/train.parquet").drop_duplicates("id_departamento").set_index("id_departamento").departamento
    print("\n2025, primer semestre (media por departamento y semana)")
    filas = []
    for familia, bloque in actual.groupby("familia", sort=False):
        filas.append({
            "familia": familia,
            "w1_4": bloque[bloque.semana <= 4].pred.mean(),
            "q1": bloque[bloque.semana <= 13].pred.mean(),
            "q2": bloque[bloque.semana > 13].pred.mean(),
            "h1": bloque.pred.mean(),
            "dist_B": float(np.mean((bloque.pred - bloque.b) ** 2)),
            "dist_C": float(np.mean((bloque.pred - bloque.c) ** 2)),
        })
    ref = actual[actual.familia == "ensamble"]
    filas.append({"familia": "B", "w1_4": ref[ref.semana <= 4].b.mean(), "q1": ref[ref.semana <= 13].b.mean(), "q2": ref[ref.semana > 13].b.mean(), "h1": ref.b.mean()})
    filas.append({"familia": "C", "w1_4": ref[ref.semana <= 4].c.mean(), "q1": ref[ref.semana <= 13].c.mean(), "q2": ref[ref.semana > 13].c.mean(), "h1": ref.c.mean()})
    print(pd.DataFrame(filas).round(1).to_string(index=False))
    q1 = ref[ref.semana <= 13].groupby("id_departamento")[["pred", "b", "c"]].mean()
    q1["departamento"] = nombres
    print("\nPrimer trimestre 2025 por departamento (ensamble de redes, B, C)")
    print(q1.sort_values("c", ascending=False).head(14).round(0).to_string())


if __name__ == "__main__":
    main()
