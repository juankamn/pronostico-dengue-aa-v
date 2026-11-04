"""Escenario de continuidad para el primer semestre de 2025.

El envio de horizontes (2048,49606) abre 2025 muy por debajo del nivel de diciembre de 2024: un colapso
inmediato. Este archivo cubre el escenario contrario. Donde B abre por debajo de FRACCION veces la
mediana de las semanas 45-52 de 2024, el primer semestre arranca en ese nivel y baja con una curva coseno
hasta tocar a B en la semana 27. El segundo semestre es identico a B, asi que el tablero publico debe dar
exactamente 2048,49606.
"""

from pathlib import Path

import numpy as np
import pandas as pd

CARPETA = Path("ensayos/juan/2026-11-04-h1-continuidad")
ENVIO_BASE = Path("ensayos/juan/2026-10-31-horizonte-bases/submission.csv")
FRACCION = 0.8
SEMANA_ATERRIZAJE = 27


def raiz_repo() -> Path:
    aqui = Path.cwd().resolve()
    for carpeta in [aqui, *list(aqui.parents)[:5]]:
        if (carpeta / "data" / "train.parquet").exists():
            return carpeta
    raise FileNotFoundError("No se encontro data/train.parquet.")


def nivel_diciembre(train: pd.DataFrame) -> pd.Series:
    """Mediana de las semanas 45-52 de 2024. La mediana deja por fuera los picos de reporte mensual."""
    cierre = train[(train.anio == 2024) & train.semana.between(45, 52)]
    return cierre.groupby("id_departamento").casos_dengue.median()


def puente(base: pd.DataFrame, nivel: pd.Series) -> np.ndarray:
    apertura = base[base.semana <= 4].groupby("id_departamento").dengue.mean()
    brecha = (np.log1p(FRACCION * nivel) - np.log1p(apertura)).clip(lower=0.0)
    semana = base.semana.to_numpy()
    peso = np.where(
        semana < SEMANA_ATERRIZAJE,
        0.5 * (1 + np.cos(np.pi * (semana - 1) / (SEMANA_ATERRIZAJE - 1))),
        0.0,
    )
    salto = base.id_departamento.map(brecha).fillna(0.0).to_numpy()
    return np.expm1(np.log1p(base.dengue.to_numpy()) + salto * peso)


def main() -> None:
    raiz = raiz_repo()
    train = pd.read_parquet(raiz / "data" / "train.parquet")
    muestra = pd.read_csv(raiz / "data" / "sample_submission.csv")
    base = pd.read_csv(raiz / ENVIO_BASE)
    base = muestra[["id"]].merge(base, on="id", how="left")
    partes = base.id.str.split("_", expand=True).astype(int)
    base["id_departamento"] = partes[0]
    base["semana"] = partes[2]

    nivel = nivel_diciembre(train)
    final = base[["id"]].copy()
    final["dengue"] = puente(base, nivel)

    h1 = base.semana.to_numpy() <= 26
    if len(final) != 1612 or final.dengue.isna().any() or (final.dengue < 0).any():
        raise RuntimeError("El envio no respeta el formato de la muestra.")
    if not np.allclose(final.dengue[~h1], base.dengue[~h1]):
        raise RuntimeError("El segundo semestre cambio y no debia.")
    if (final.dengue[h1] < base.dengue[h1] - 1e-9).any():
        raise RuntimeError("El escenario de continuidad solo puede subir el primer semestre.")

    nombres = train.drop_duplicates("id_departamento").set_index("id_departamento").departamento
    resumen = pd.DataFrame({"departamento": nombres, "diciembre": nivel})
    for etiqueta, desde, hasta in (("w1_4", 1, 4), ("q1", 1, 13), ("q2", 14, 26)):
        mascara = base.semana.between(desde, hasta)
        resumen[f"B_{etiqueta}"] = base[mascara].groupby("id_departamento").dengue.mean()
        resumen[f"C_{etiqueta}"] = final[mascara].groupby(base.id_departamento[mascara]).dengue.mean()
    print(resumen.sort_values("diciembre", ascending=False).head(16).round(0).to_string())

    semanal = pd.DataFrame({"B": base.dengue, "C": final.dengue, "semana": base.semana}).groupby("semana").mean()
    print("\nNacional B w1-26:", semanal.B.loc[1:26].round(0).astype(int).tolist())
    print("Nacional C w1-26:", semanal.C.loc[1:26].round(0).astype(int).tolist())
    print("Diciembre 2024 sin picos (media nacional de las medianas):", round(float(nivel.mean()), 1))
    print(
        "H1 B", round(float(base.dengue[h1].mean()), 1),
        "H1 C", round(float(final.dengue[h1].mean()), 1),
        "| Q1 B", round(float(base.dengue[base.semana <= 13].mean()), 1),
        "Q1 C", round(float(final.dengue[base.semana <= 13].mean()), 1),
    )
    distancia = float(np.mean((final.dengue[h1].to_numpy() - base.dengue[h1].to_numpy()) ** 2))
    print("MSE entre B y C en el primer semestre:", round(distancia, 1))
    destino = raiz / CARPETA / "submission.csv"
    final.to_csv(destino, index=False)
    print("archivo", destino)


if __name__ == "__main__":
    main()
