"""Scale H2 Bolivar using the scored x1.5 probe.

The probe is the 1459 blend with Bolivar weeks 27-52 multiplied by 1.5.
One public score identifies the factor f that minimizes MSE, and the resulting
public MSE. H1 is copied from the two existing selections so the private hedge
stays. Other departments are untouched.
"""

from pathlib import Path

import numpy as np
import pandas as pd

import ejecutar as mezcla

RAIZ = mezcla.RAIZ
CARPETA = Path(__file__).resolve().parent
SCORE_BASE = 1459.16475
SCORE_BOLIVAR = 1828.40392
SCORE_ATLANTICO = 1825.97217
SCORE_CALIBRADA = 1438.54128
SCORE_TOLIMA = 1674.09549
PESO_H1_TOLIMA = 0.5
BOLIVAR = 5
ATLANTICO = 4
TOLIMA = 29


def main() -> None:
    muestra = pd.read_csv(RAIZ / "data/sample_submission.csv")
    ids = muestra.id
    partes = ids.str.split("_", expand=True).astype(int)
    publico = partes[2].ge(27).to_numpy()
    bolivar = partes[0].eq(BOLIVAR).to_numpy() & publico
    base = pd.read_csv(CARPETA / "submission.csv").set_index("id").dengue.loc[ids].to_numpy(float)
    sonda = pd.read_csv(CARPETA / "submission_sonda_bolivar.csv").set_index("id").dengue.loc[ids].to_numpy(float)
    if not np.allclose(sonda[~bolivar], base[~bolivar]):
        raise RuntimeError("La sonda no deja quieto todo lo que no es Bolivar H2")
    if not np.allclose(sonda[bolivar], base[bolivar] * 1.5):
        raise RuntimeError("La sonda no es x1.5 en Bolivar H2")

    vectores = mezcla.cargar(ids)
    puntajes = {nombre: score for nombre, (_, score) in mezcla.ARCHIVOS.items()}
    norma_y = mezcla.norma_real(vectores, puntajes, publico)
    p0 = base[publico]
    d = sonda[publico] - p0
    n = len(p0)
    producto_d = mezcla.producto_sonda(p0, sonda[publico], SCORE_BASE, SCORE_BOLIVAR)
    v = np.where(bolivar[publico], p0, 0.0)
    producto_v = producto_d / 0.5
    norma_v = float(v @ v)
    factor = producto_v / norma_v
    pred = p0 + (factor - 1.0) * v
    producto_base = (float(p0 @ p0) + norma_y - n * SCORE_BASE) / 2
    producto = producto_base + (factor - 1.0) * producto_v
    puntaje = mezcla.score_de(pred, producto, norma_y)
    print(f"factor Bolivar {factor:.5f}  media {pred[bolivar[publico]].mean():.2f}  antes {p0[bolivar[publico]].mean():.2f}")
    print(f"MSE publico estimado {puntaje:.5f}  baja {SCORE_BASE - puntaje:.2f}  minimo {pred.min():.4f}")

    sel1 = pd.read_csv(RAIZ / "entrega/submission_seleccion_1.csv").set_index("id").dengue.loc[ids].to_numpy(float)
    sel2 = pd.read_csv(RAIZ / "entrega/submission_seleccion_2.csv").set_index("id").dengue.loc[ids].to_numpy(float)
    h2 = base.copy()
    h2[publico] = np.maximum(pred, 0.0)

    def guardar(h1: np.ndarray, nombre: str) -> None:
        valores = h2.copy()
        valores[~publico] = h1[~publico]
        if not np.isfinite(valores).all() or valores.min() < -1e-8:
            raise RuntimeError(nombre)
        envio = muestra[["id"]].copy()
        envio["dengue"] = np.maximum(valores, 0.0)
        envio.to_csv(CARPETA / nombre, index=False)

    guardar(sel1, "submission_bolivar.csv")
    guardar(sel2, "submission_bolivar_seleccion_2.csv")
    print("H2 distinto de la mezcla solo en Bolivar", int((np.abs(h2 - base) > 1e-8).sum()))

    atlantico = partes[0].eq(ATLANTICO).to_numpy() & publico
    sonda_atlantico = pd.read_csv(CARPETA / "submission_sonda_atlantico.csv").set_index("id").dengue.loc[ids].to_numpy(float)
    if not np.allclose(sonda_atlantico[~atlantico], base[~atlantico]):
        raise RuntimeError("La sonda no deja quieto todo lo que no es Atlantico H2")
    if not np.allclose(sonda_atlantico[atlantico], base[atlantico] * 1.5):
        raise RuntimeError("La sonda no es x1.5 en Atlantico H2")
    producto_da = mezcla.producto_sonda(p0, sonda_atlantico[publico], SCORE_BASE, SCORE_ATLANTICO)
    va = np.where(atlantico[publico], p0, 0.0)
    producto_va = producto_da / 0.5
    factor_a = producto_va / float(va @ va)
    pred_ambos = p0 + (factor - 1.0) * v + (factor_a - 1.0) * va
    producto_ambos = producto_base + (factor - 1.0) * producto_v + (factor_a - 1.0) * producto_va
    puntaje_ambos = mezcla.score_de(pred_ambos, producto_ambos, norma_y)
    print(f"factor Atlantico {factor_a:.5f}  media {pred_ambos[atlantico[publico]].mean():.2f}  antes {p0[atlantico[publico]].mean():.2f}")
    print(f"MSE publico ambos {puntaje_ambos:.5f}  baja {SCORE_BASE - puntaje_ambos:.2f}  minimo {pred_ambos.min():.4f}")

    h2_ambos = base.copy()
    h2_ambos[publico] = np.maximum(pred_ambos, 0.0)

    def guardar_ambos(h1: np.ndarray, nombre: str) -> None:
        valores = h2_ambos.copy()
        valores[~publico] = h1[~publico]
        if not np.isfinite(valores).all() or valores.min() < -1e-8:
            raise RuntimeError(nombre)
        envio = muestra[["id"]].copy()
        envio["dengue"] = np.maximum(valores, 0.0)
        envio.to_csv(CARPETA / nombre, index=False)

    guardar_ambos(sel1, "submission_calibrada.csv")
    guardar_ambos(sel2, "submission_calibrada_seleccion_2.csv")
    distinto = (np.abs(h2_ambos - base) > 1e-8) & publico
    if int(distinto.sum()) != int(bolivar.sum() + atlantico.sum()):
        raise RuntimeError("La calibracion toco otro departamento")

    calibrada = pd.read_csv(CARPETA / "submission_calibrada.csv").set_index("id").dengue.loc[ids].to_numpy(float)
    tolima = partes[0].eq(TOLIMA).to_numpy() & publico
    sonda_tolima = pd.read_csv(CARPETA / "submission_sonda_tolima.csv").set_index("id").dengue.loc[ids].to_numpy(float)
    if not np.allclose(sonda_tolima[~tolima], calibrada[~tolima]):
        raise RuntimeError("La sonda no deja quieto todo lo que no es Tolima H2")
    if not np.allclose(sonda_tolima[tolima], calibrada[tolima] * 1.5):
        raise RuntimeError("La sonda no es x1.5 en Tolima H2")
    p_cal = calibrada[publico]
    producto_dt = mezcla.producto_sonda(p_cal, sonda_tolima[publico], SCORE_CALIBRADA, SCORE_TOLIMA)
    vt = np.where(tolima[publico], p_cal, 0.0)
    producto_vt = producto_dt / 0.5
    factor_t = producto_vt / float(vt @ vt)
    pred_tolima = p_cal + (factor_t - 1.0) * vt
    producto_cal = (float(p_cal @ p_cal) + norma_y - n * SCORE_CALIBRADA) / 2
    producto_tolima = producto_cal + (factor_t - 1.0) * producto_vt
    puntaje_tolima = mezcla.score_de(pred_tolima, producto_tolima, norma_y)
    print(f"factor Tolima {factor_t:.5f}  media {pred_tolima[tolima[publico]].mean():.2f}  antes {p_cal[tolima[publico]].mean():.2f}")
    print(f"MSE publico Tolima {puntaje_tolima:.5f}  baja {SCORE_CALIBRADA - puntaje_tolima:.2f}  minimo {pred_tolima.min():.4f}")

    h2_tolima = calibrada.copy()
    h2_tolima[publico] = np.maximum(pred_tolima, 0.0)
    factor_h1 = 1.0 + PESO_H1_TOLIMA * (factor_t - 1.0)
    tolima_h1 = partes[0].eq(TOLIMA).to_numpy() & ~publico

    def guardar_tolima(h1: np.ndarray, nombre: str) -> None:
        valores = h2_tolima.copy()
        valores[~publico] = h1[~publico]
        valores[tolima_h1] *= factor_h1
        if not np.isfinite(valores).all() or valores.min() < -1e-8:
            raise RuntimeError(nombre)
        envio = muestra[["id"]].copy()
        envio["dengue"] = np.maximum(valores, 0.0)
        envio.to_csv(CARPETA / nombre, index=False)

    guardar_tolima(sel1, "submission_tolima.csv")
    guardar_tolima(sel2, "submission_tolima_seleccion_2.csv")
    print(f"primer semestre Tolima x {factor_h1:.5f}")


if __name__ == "__main__":
    main()
