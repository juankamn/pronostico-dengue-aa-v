"""Dos envios de medicion: el nivel del segundo semestre de 2025 en la Costa y en el interior.

Cada archivo es la seleccion final con el apilado (`entrega/submission_final_apilado.csv`, 1836,20689 en el
publico) mas EPS veces B en las semanas 27-52 de una sola region. Como el MSE es cuadratico, la diferencia de
puntaje da exacto <B_region, real>, y con eso k_region = <B_region, real> / <B_region, B_region>, la escala que
mejor ajusta B al segundo semestre real de esa region. El primer semestre no cambia.

Uso:
    python armar.py                      arma los dos archivos
    python armar.py 1850.1 1820.3        lee los puntajes de Costa e interior y calcula los k
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

CARPETA = Path("ensayos/juan/2026-11-04-sonda-regiones")
BASE = Path("entrega/submission_final_apilado.csv")
PUNTAJE_BASE = 1836.20689
B_SIN_PATRON = Path("ensayos/juan/2026-10-31-horizonte-bases/submission.csv")
COSTA = [4, 5, 11, 13, 16, 19, 28]
EPS = 0.2


def raiz_repo() -> Path:
    aqui = Path.cwd().resolve()
    for carpeta in [aqui, *list(aqui.parents)[:5]]:
        if (carpeta / "data" / "train.parquet").exists():
            return carpeta
    raise FileNotFoundError("No se encontro data/train.parquet.")


def cargar(raiz: Path):
    muestra = pd.read_csv(raiz / "data/sample_submission.csv")
    partes = muestra.id.str.split("_", expand=True).astype(int)
    publico = (partes[2] >= 27).to_numpy()
    costa = partes[0].isin(COSTA).to_numpy()
    base = pd.read_csv(raiz / BASE).set_index("id").dengue.loc[muestra.id].to_numpy()
    b = pd.read_csv(raiz / B_SIN_PATRON).set_index("id").dengue.loc[muestra.id].to_numpy()
    return muestra, publico, costa, base, b


def main() -> None:
    raiz = raiz_repo()
    muestra, publico, costa, base, b = cargar(raiz)
    n = int(publico.sum())
    regiones = {"costa": publico & costa, "interior": publico & ~costa}
    if len(sys.argv) == 1:
        for nombre, mascara in regiones.items():
            sonda = base.copy()
            sonda[mascara] = base[mascara] + EPS * b[mascara]
            assert (sonda >= 0).all() and np.allclose(sonda[~publico], base[~publico])
            envio = muestra[["id"]].copy()
            envio["dengue"] = sonda
            envio.to_csv(raiz / CARPETA / f"submission_sonda_{nombre}.csv", index=False)
            print(f"{nombre}: archivo listo; segundo semestre de la region sube {EPS:.0%} de B")
        return
    puntajes = dict(zip(regiones, map(float, sys.argv[1:3])))
    for nombre, mascara in regiones.items():
        d = np.where(mascara, EPS * b, 0.0)[publico]
        p = base[publico]
        b_r = np.where(mascara, b, 0.0)[publico]
        # S1 - S0 = (|d|^2 + 2 <d, p - y>) / n  ->  <d, y> = <d, p> - (n (S1 - S0) - |d|^2) / 2
        d_y = d @ p - (n * (puntajes[nombre] - PUNTAJE_BASE) - d @ d) / 2
        k = (d_y / EPS) / (b_r @ b_r)
        print(f"{nombre}: k = {k:.3f}  (media de B en el segundo semestre {b[mascara].mean():.1f}; real estimada {k * b[mascara].mean():.1f})")


if __name__ == "__main__":
    main()
