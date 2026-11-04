"""Finalistas con red neuronal.

1. `2026-11-04-finalista-red`: primer semestre = 25 % TCN + 25 % MLP + 50 % B, con las redes de
   `2026-11-04-red-dinamica-h1` entrenadas hasta 2024. Es el mejor modelo del walk-forward 2014-2023.
2. `2026-11-04-finalista-continuidad-red`: el mismo archivo con el escenario de continuidad encima (arranca en
   0,8 veces el nivel de diciembre de 2024 donde la mezcla abre por debajo y baja con un coseno hasta la
   semana 27).

Los dos llevan el segundo semestre de B y el patron mensual de reporte con la amplitud que midio el sensor.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

CARPETA = Path("ensayos/juan/2026-11-04-finalista-red")
CARPETA_CONTINUIDAD = Path("ensayos/juan/2026-11-04-finalista-continuidad-red")
ENVIO_B = Path("ensayos/juan/2026-10-31-horizonte-bases/submission.csv")
ENVIO_C = Path("ensayos/juan/2026-11-04-h1-continuidad/submission.csv")
RED = Path("ensayos/juan/2026-11-04-red-dinamica-h1/predicciones_final2025.parquet")
PATRON = Path("ensayos/juan/2026-11-04-finalista-colapso/armar.py")
PESOS_RED = {"tcn": 0.25, "mlp": 0.25}
FRACCION = 0.8
SEMANA_ATERRIZAJE = 27


def raiz_repo() -> Path:
    aqui = Path.cwd().resolve()
    for carpeta in [aqui, *list(aqui.parents)[:5]]:
        if (carpeta / "data" / "train.parquet").exists():
            return carpeta
    raise FileNotFoundError("No se encontro data/train.parquet.")


def modulo_patron(raiz: Path):
    """Reutiliza las funciones del finalista de colapso para que el patron sea identico."""
    spec = importlib.util.spec_from_file_location("patron_reporte", raiz / PATRON)
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    return modulo


def partes(ids: pd.Index) -> tuple[np.ndarray, np.ndarray]:
    trozos = ids.str.split("_", expand=True)
    return trozos.get_level_values(0).astype(int).to_numpy(), trozos.get_level_values(2).astype(int).to_numpy()


def mezcla_red(base: pd.Series, red: pd.DataFrame) -> pd.Series:
    ancho = red[red.anio == 2025].pivot_table(index=["id_departamento", "semana"], columns="familia", values="pred")
    if len(ancho) != 31 * 26 or ancho[list(PESOS_RED)].isna().any().any():
        raise RuntimeError("Faltan pronosticos de la red para el primer semestre de 2025.")
    ids = [f"{d}_2025_{s}" for d, s in ancho.index]
    salida = base.copy()
    peso_b = 1.0 - sum(PESOS_RED.values())
    salida.loc[ids] = peso_b * base.loc[ids].to_numpy() + sum(w * ancho[f].to_numpy() for f, w in PESOS_RED.items())
    return salida


def continuidad(serie: pd.Series, train: pd.DataFrame) -> pd.Series:
    nivel = train[(train.anio == 2024) & train.semana.between(45, 52)].groupby("id_departamento").casos_dengue.median()
    depto, semana = partes(serie.index)
    tabla = pd.DataFrame({"depto": depto, "semana": semana, "valor": serie.to_numpy()})
    apertura = tabla[tabla.semana <= 4].groupby("depto").valor.mean()
    brecha = (np.log1p(FRACCION * nivel) - np.log1p(apertura)).clip(lower=0.0)
    peso = np.where(semana < SEMANA_ATERRIZAJE, 0.5 * (1 + np.cos(np.pi * (semana - 1) / (SEMANA_ATERRIZAJE - 1))), 0.0)
    salto = pd.Series(depto).map(brecha).fillna(0.0).to_numpy()
    return pd.Series(np.expm1(np.log1p(serie.to_numpy()) + salto * peso), index=serie.index)


def con_patron(serie: pd.Series, test: pd.DataFrame, patron, razon: pd.Series, amplitud: float) -> pd.Series:
    d = patron.desplazamiento(test, serie, razon, patron.MESES_FINAL)
    return (serie + amplitud * d.loc[serie.index]).clip(lower=0.0)


def resumen(nombre: str, serie: pd.Series, nombres: pd.Series) -> None:
    depto, semana = partes(serie.index)
    h1 = semana <= 26
    print(f"{nombre}: H1 {serie[h1].mean():.1f} | Q1 {serie[semana <= 13].mean():.1f} | w1-4 {serie[semana <= 4].mean():.1f} | H2 {serie[~h1].mean():.1f}")
    q1 = pd.Series(serie[semana <= 13].to_numpy()).groupby(depto[semana <= 13]).mean()
    top = q1.sort_values(ascending=False).head(8)
    print("   Q1:", {nombres[d]: round(v) for d, v in top.items()})


def guardar(serie: pd.Series, muestra: pd.DataFrame, destino: Path) -> pd.DataFrame:
    final = muestra[["id"]].copy()
    final["dengue"] = serie.loc[final.id].to_numpy()
    if len(final) != 1612 or final.dengue.isna().any() or (final.dengue < 0).any() or not final.id.equals(muestra.id):
        raise RuntimeError("El envio no respeta el formato de la muestra.")
    destino.parent.mkdir(parents=True, exist_ok=True)
    final.to_csv(destino, index=False)
    return final


def main() -> None:
    raiz = raiz_repo()
    train = pd.read_parquet(raiz / "data" / "train.parquet")
    test = pd.read_parquet(raiz / "data" / "test.parquet")
    muestra = pd.read_csv(raiz / "data" / "sample_submission.csv")
    B = pd.read_csv(raiz / ENVIO_B).set_index("id").dengue.loc[muestra.id]
    C = pd.read_csv(raiz / ENVIO_C).set_index("id").dengue.loc[muestra.id]
    red = pd.read_parquet(raiz / RED)
    nombres = train.drop_duplicates("id_departamento").set_index("id_departamento").departamento

    patron = modulo_patron(raiz)
    razon = patron.patron_2024(train)
    amplitud, esperado = patron.amplitud_optima(test, razon, B)
    print(f"patron: {len(razon)} departamentos, amplitud {amplitud:.4f}, publico esperado agosto-diciembre {esperado:.2f}")

    mezcla = mezcla_red(B, red)
    alta = continuidad(mezcla, train)
    _, semana = partes(B.index)
    if np.abs((mezcla - B)[semana >= 27]).max() > 1e-9 or np.abs((alta - B)[semana >= 27]).max() > 1e-9:
        raise RuntimeError("El segundo semestre debe ser el de B antes del patron.")
    if (alta < mezcla - 1e-9).any():
        raise RuntimeError("La continuidad solo puede subir.")

    for nombre, serie in (("B", B), ("C", C), ("mezcla con red", mezcla), ("continuidad sobre la mezcla", alta)):
        resumen(nombre, serie, nombres)
    h1 = semana <= 26
    print("distancia H1 (MSE): mezcla-B", round(float(np.mean((mezcla - B)[h1] ** 2)), 1), "| continuidad-mezcla", round(float(np.mean((alta - mezcla)[h1] ** 2)), 1), "| continuidad-C", round(float(np.mean((alta - C)[h1] ** 2)), 1))

    final_red = guardar(con_patron(mezcla, test, patron, razon, amplitud), muestra, raiz / CARPETA / "submission.csv")
    final_alta = guardar(con_patron(alta, test, patron, razon, amplitud), muestra, raiz / CARPETA_CONTINUIDAD / "submission.csv")
    enero = semana <= 4
    if np.abs(final_red.dengue.to_numpy()[enero] - mezcla.to_numpy()[enero]).max() > 1e-9:
        raise RuntimeError("Enero cambio con el patron.")
    colapso = pd.read_csv(raiz / "ensayos/juan/2026-11-04-finalista-colapso/submission.csv").dengue.to_numpy()
    print("max |H2 finalista-red - H2 finalista-colapso|:", round(float(np.abs(final_red.dengue.to_numpy()[~h1] - colapso[~h1]).max()), 3))
    print("archivos:", raiz / CARPETA / "submission.csv", "|", raiz / CARPETA_CONTINUIDAD / "submission.csv")


if __name__ == "__main__":
    main()
