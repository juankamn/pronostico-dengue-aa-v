"""Sensor del tablero publico: el patron mensual de reporte de 2024 aplicado al segundo semestre de 2025.

Parte del envio de horizontes (2048,49606). El primer semestre queda igual. En el segundo semestre,
dentro de cada mes, se mueven casos entre la semana del primer miercoles y el resto de semanas del
mes, sin cambiar el total del mes. Si el puntaje publico baja, el patron de 2024 siguio en 2025.
"""

from pathlib import Path

import numpy as np
import pandas as pd

CARPETA = Path("ensayos/juan/2026-11-04-sensor-picos")
ENVIO_BASE = Path("ensayos/juan/2026-10-31-horizonte-bases/submission.csv")
PUNTAJE_BASE = 2048.49606
AMPLITUD = 0.5
MESES_2024 = (202409, 202410, 202411, 202412)
MESES_SONDA = (202508, 202509, 202510, 202511, 202512)
NIVEL_MINIMO = 60.0
DESVIO_MINIMO = 0.3


def raiz_repo() -> Path:
    aqui = Path.cwd().resolve()
    for carpeta in [aqui, *list(aqui.parents)[:5]]:
        if (carpeta / "data" / "train.parquet").exists():
            return carpeta
    raise FileNotFoundError("No se encontro data/train.parquet.")


def marcar_meses(tabla: pd.DataFrame) -> pd.DataFrame:
    """Cada semana pertenece al mes de su miercoles; la posicion 0 es la semana del primer miercoles."""
    tabla = tabla.sort_values(["id_departamento", "semana_inicio"]).copy()
    miercoles = tabla.semana_inicio + pd.Timedelta(days=2)
    tabla["mes"] = miercoles.dt.year * 100 + miercoles.dt.month
    tabla["posicion"] = tabla.groupby(["id_departamento", "mes"]).cumcount()
    tabla["semanas_mes"] = tabla.groupby(["id_departamento", "mes"]).semana.transform("size")
    return tabla


def patron_2024(train: pd.DataFrame) -> pd.DataFrame:
    """Razon entre la semana del primer miercoles y la media de su mes, septiembre a diciembre de 2024."""
    tabla = marcar_meses(train)
    tabla = tabla[tabla.mes.isin(MESES_2024)].copy()
    tabla["media_mes"] = tabla.groupby(["id_departamento", "mes"]).casos_dengue.transform("mean")
    primera = tabla[tabla.posicion == 0].copy()
    primera["razon"] = primera.casos_dengue / primera.media_mes.replace(0, np.nan)
    resumen = primera.groupby(["id_departamento", "departamento"]).razon.agg(["mean", "min", "max"]).reset_index()
    resumen["nivel"] = resumen.id_departamento.map(tabla.groupby("id_departamento").casos_dengue.mean())
    # Solo departamentos con nivel suficiente y un signo estable en los cuatro meses.
    estable = (resumen["min"] > 1.0) | (resumen["max"] < 1.0)
    resumen["usar"] = (resumen.nivel >= NIVEL_MINIMO) & ((resumen["mean"] - 1).abs() >= DESVIO_MINIMO) & estable
    return resumen


def sonda(test: pd.DataFrame, base: pd.Series, patron: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Devuelve el envio con el patron y el desplazamiento completo (amplitud 1) para leer el resultado."""
    tabla = marcar_meses(test)
    tabla["id"] = tabla.ID
    tabla["base"] = base.loc[tabla.id].to_numpy()
    razon = patron.loc[patron.usar].set_index("id_departamento")["mean"]
    tabla["exceso"] = tabla.id_departamento.map(razon - 1.0).fillna(0.0)
    tabla.loc[~tabla.mes.isin(MESES_SONDA), "exceso"] = 0.0
    media_mes = tabla.groupby(["id_departamento", "mes"]).base.transform("mean")
    reparto = np.where(tabla.posicion == 0, 1.0, -1.0 / (tabla.semanas_mes - 1))
    completo = pd.Series((tabla.exceso * media_mes * reparto).to_numpy(), index=tabla.id)
    nuevo = (base.loc[completo.index] + AMPLITUD * completo).clip(lower=0.0)
    return nuevo, completo


def lectura(completo: pd.Series, semanas: pd.Series) -> pd.DataFrame:
    """Puntaje publico esperado segun la fraccion beta del patron de 2024 que siga en 2025.

    MSE nuevo - MSE base = (A^2 - 2 A beta) * suma(d^2) / n, con A la amplitud y d el desplazamiento completo,
    si el error de la base es independiente del patron dentro del mes.
    """
    publico = semanas.loc[completo.index] >= 27
    q = float((completo[publico] ** 2).sum() / publico.sum())
    filas = [{"beta": beta, "puntaje_esperado": PUNTAJE_BASE + (AMPLITUD**2 - 2 * AMPLITUD * beta) * q} for beta in (-0.5, 0.0, 0.25, 0.5, 0.75, 1.0)]
    return pd.DataFrame(filas), q


def main() -> None:
    raiz = raiz_repo()
    train = pd.read_parquet(raiz / "data" / "train.parquet")
    test = pd.read_parquet(raiz / "data" / "test.parquet")
    muestra = pd.read_csv(raiz / "data" / "sample_submission.csv")
    base = pd.read_csv(raiz / ENVIO_BASE).set_index("id").dengue

    patron = patron_2024(train)
    print(patron.sort_values("nivel", ascending=False).round(2).to_string(index=False))
    nuevo, completo = sonda(test, base, patron)

    final = muestra[["id"]].copy()
    final["dengue"] = nuevo.loc[final.id].to_numpy()
    semanas = final.id.str.split("_").str[2].astype(int)
    semanas.index = final.id
    if len(final) != 1612 or final.dengue.isna().any() or (final.dengue < 0).any():
        raise RuntimeError("El envio no respeta el formato de la muestra.")
    if not np.allclose(final.dengue[semanas.to_numpy() <= 26], base.loc[final.id[semanas.to_numpy() <= 26]]):
        raise RuntimeError("El primer semestre cambio y no debia.")
    mensual = pd.DataFrame({"id": final.id, "delta": final.dengue.to_numpy() - base.loc[final.id].to_numpy()})
    mensual["depto"] = mensual.id.str.split("_").str[0].astype(int)
    if mensual.groupby("depto").delta.sum().abs().max() > 1e-6:
        raise RuntimeError("El patron cambio el total de algun departamento.")

    tabla, q = lectura(completo, semanas)
    print(f"\nsuma d^2 / n (publico) = {q:.1f}")
    print(tabla.round({"beta": 2, "puntaje_esperado": 1}).to_string(index=False))
    print(f"beta estimado = ({AMPLITUD} - (puntaje - {PUNTAJE_BASE}) / {AMPLITUD * q:.1f}) / 2")
    destino = raiz / CARPETA / "submission.csv"
    final.to_csv(destino, index=False)
    print("\nmedia", round(final.dengue.mean(), 2), "H2", round(final.dengue[semanas.to_numpy() >= 27].mean(), 2), "archivo", destino)


if __name__ == "__main__":
    main()
