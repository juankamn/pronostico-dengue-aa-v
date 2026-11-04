"""Redes de 52 semanas: el segundo semestre de las redes contra B, ano por ano y en el tablero publico.

Reutiliza el pipeline de `2026-11-04-red-dinamica-h1` con un horizonte de 52 semanas. Para cada ano Y de
2011 a 2025 entrena solo con ventanas cuyo objetivo termina antes del 1 de enero de Y y pronostica las
52 semanas de Y desde la ultima semana de Y-1. Guarda las predicciones; `evaluar.py` las compara con B.
"""

import importlib.util
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

CARPETA = Path("ensayos/juan/2026-11-04-red-segundo-semestre")
PIPELINE = Path("ensayos/juan/2026-11-04-red-dinamica-h1/pipeline.py")
FAMILIAS = ("tcn", "mlp")
EPOCAS = 12
SEMILLAS = [0, 1]
SEMILLAS_2025 = [0, 1, 2, 3, 4]


def raiz_repo() -> Path:
    aqui = Path.cwd().resolve()
    for carpeta in [aqui, *list(aqui.parents)[:5]]:
        if (carpeta / "data" / "train.parquet").exists():
            return carpeta
    raise FileNotFoundError("No se encontro data/train.parquet.")


def cargar_pipeline(raiz: Path):
    spec = importlib.util.spec_from_file_location("pipeline_red", raiz / PIPELINE)
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    return modulo


def main() -> None:
    warnings.filterwarnings("ignore")
    raiz = raiz_repo()
    pipe = cargar_pipeline(raiz)
    pipe.HORIZONTE = 52
    panel = pipe.Panel(pipe.cargar(raiz))
    V = pipe.ventanas(panel)
    X = pipe.Tensores(V)
    print("ventanas", V["enc"].shape[0] * V["enc"].shape[1], "| futuro", V["dec"].shape[2:], flush=True)
    filas = []
    for familia in FAMILIAS:
        cfg = pipe.CONFIGS[familia]
        for anio in range(2011, 2026):
            inicio = time.time()
            semillas = SEMILLAS_2025 if anio == 2025 else SEMILLAS
            pred, info = pipe.correr_anio(cfg, panel, V, X, anio, semillas, EPOCAS)
            for d, depto in enumerate(panel.deptos):
                for h in range(pipe.HORIZONTE):
                    filas.append((familia, anio, int(depto), h + 1, float(pred[d, h])))
            print(familia, anio, info, round(time.time() - inicio, 1), "s", flush=True)
    salida = pd.DataFrame(filas, columns=["familia", "anio", "id_departamento", "semana", "pred"])
    salida.to_parquet(raiz / CARPETA / "predicciones_52.parquet")
    print("listo", len(salida), flush=True)


if __name__ == "__main__":
    main()
