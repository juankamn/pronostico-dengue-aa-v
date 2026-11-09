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
SCORE_VIGENTE = 1398.01602
SCORE_NORTE = 1366.44918
SCORE_NORTE_ARCHIVO = 1356.02336
SCORE_META = 1326.70176
SCORE_META_ARCHIVO = 1313.36105
SCORE_CORDOBA = 1292.95694
SCORE_CORDOBA_ARCHIVO = 1276.39544
SCORE_ANTIOQUIA = 1383.21656
SCORE_SUCRE = 1464.39444
SCORE_SUCRE_ARCHIVO = 1222.72250
SCORE_SANTANDER = 1418.34102
SCORE_SANTANDER_ARCHIVO = 1221.71529
SCORE_HUILA = 1329.83604
SCORE_HUILA_ARCHIVO = 1190.74829
SCORE_CESAR = 1162.65830
SCORE_CESAR_ARCHIVO = 1162.39485
SCORE_GUAJIRA = 1170.31288
SCORE_GUAJIRA_ARCHIVO = 1160.41638
SCORE_CUNDINAMARCA = 1203.76053
PESO_H1_CESAR = 0.5
PESO_H1_SUCRE = 0.5
PESO_H1_META = 1.0
PESO_H1_TOLIMA = 0.5
BOLIVAR = 5
ATLANTICO = 4
TOLIMA = 29
NORTE = 22
META = 20
CORDOBA = 13
ANTIOQUIA = 2
SUCRE = 28
SANTANDER = 27
HUILA = 18
CESAR = 11
GUAJIRA = 16
CUNDINAMARCA = 14
PUTUMAYO = 23


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

    vigente = pd.read_csv(CARPETA / "submission_tolima.csv").set_index("id").dengue.loc[ids].to_numpy(float)
    norte = partes[0].eq(NORTE).to_numpy() & publico
    sonda_norte = pd.read_csv(CARPETA / "submission_sonda_norte.csv").set_index("id").dengue.loc[ids].to_numpy(float)
    if not np.allclose(sonda_norte[~norte], vigente[~norte]):
        raise RuntimeError("La sonda no deja quieto todo lo que no es Norte H2")
    if not np.allclose(sonda_norte[norte], vigente[norte] * 1.5):
        raise RuntimeError("La sonda no es x1.5 en Norte H2")
    p_vig = vigente[publico]
    producto_dn = mezcla.producto_sonda(p_vig, sonda_norte[publico], SCORE_VIGENTE, SCORE_NORTE)
    vn = np.where(norte[publico], p_vig, 0.0)
    producto_vn = producto_dn / 0.5
    factor_n = producto_vn / float(vn @ vn)
    pred_norte = p_vig + (factor_n - 1.0) * vn
    producto_vig = (float(p_vig @ p_vig) + norma_y - n * SCORE_VIGENTE) / 2
    producto_norte = producto_vig + (factor_n - 1.0) * producto_vn
    puntaje_norte = mezcla.score_de(pred_norte, producto_norte, norma_y)
    print(f"factor Norte {factor_n:.5f}  media {pred_norte[norte[publico]].mean():.2f}  antes {p_vig[norte[publico]].mean():.2f}")
    print(f"MSE publico Norte {puntaje_norte:.5f}  baja {SCORE_VIGENTE - puntaje_norte:.2f}  minimo {pred_norte.min():.4f}")

    def guardar_norte(origen: str, nombre: str) -> None:
        valores = pd.read_csv(CARPETA / origen).set_index("id").dengue.loc[ids].to_numpy(float)
        valores[publico] = np.maximum(pred_norte, 0.0)
        if not np.isfinite(valores).all() or valores.min() < -1e-8:
            raise RuntimeError(nombre)
        envio = muestra[["id"]].copy()
        envio["dengue"] = np.maximum(valores, 0.0)
        envio.to_csv(CARPETA / nombre, index=False)

    guardar_norte("submission_tolima.csv", "submission_norte.csv")
    guardar_norte("submission_tolima_seleccion_2.csv", "submission_norte_seleccion_2.csv")

    norte_h2 = pd.read_csv(CARPETA / "submission_norte.csv").set_index("id").dengue.loc[ids].to_numpy(float)
    meta = partes[0].eq(META).to_numpy() & publico
    sonda_meta = pd.read_csv(CARPETA / "submission_sonda_meta.csv").set_index("id").dengue.loc[ids].to_numpy(float)
    if not np.allclose(sonda_meta[~meta], norte_h2[~meta]):
        raise RuntimeError("La sonda no deja quieto todo lo que no es Meta H2")
    if not np.allclose(sonda_meta[meta], norte_h2[meta] * 1.5):
        raise RuntimeError("La sonda no es x1.5 en Meta H2")
    p_n = norte_h2[publico]
    producto_dm = mezcla.producto_sonda(p_n, sonda_meta[publico], SCORE_NORTE_ARCHIVO, SCORE_META)
    vm = np.where(meta[publico], p_n, 0.0)
    producto_vm = producto_dm / 0.5
    factor_m = producto_vm / float(vm @ vm)
    pred_meta = p_n + (factor_m - 1.0) * vm
    producto_n = (float(p_n @ p_n) + norma_y - n * SCORE_NORTE_ARCHIVO) / 2
    producto_meta = producto_n + (factor_m - 1.0) * producto_vm
    puntaje_meta = mezcla.score_de(pred_meta, producto_meta, norma_y)
    print(f"factor Meta {factor_m:.5f}  media {pred_meta[meta[publico]].mean():.2f}  antes {p_n[meta[publico]].mean():.2f}")
    print(f"MSE publico Meta {puntaje_meta:.5f}  baja {SCORE_NORTE_ARCHIVO - puntaje_meta:.2f}  minimo {pred_meta.min():.4f}")

    factor_h1_m = 1.0 + PESO_H1_META * (factor_m - 1.0)
    meta_h1 = partes[0].eq(META).to_numpy() & ~publico

    def guardar_meta(origen: str, nombre: str) -> None:
        valores = pd.read_csv(CARPETA / origen).set_index("id").dengue.loc[ids].to_numpy(float)
        valores[publico] = np.maximum(pred_meta, 0.0)
        valores[meta_h1] *= factor_h1_m
        if not np.isfinite(valores).all() or valores.min() < -1e-8:
            raise RuntimeError(nombre)
        envio = muestra[["id"]].copy()
        envio["dengue"] = np.maximum(valores, 0.0)
        envio.to_csv(CARPETA / nombre, index=False)

    guardar_meta("submission_norte.csv", "submission_meta.csv")
    guardar_meta("submission_norte_seleccion_2.csv", "submission_meta_seleccion_2.csv")
    print(f"primer semestre Meta x {factor_h1_m:.5f}")

    meta_h2 = pd.read_csv(CARPETA / "submission_meta.csv").set_index("id").dengue.loc[ids].to_numpy(float)
    cordoba = partes[0].eq(CORDOBA).to_numpy() & publico
    sonda_cordoba = pd.read_csv(CARPETA / "submission_sonda_cordoba.csv").set_index("id").dengue.loc[ids].to_numpy(float)
    if not np.allclose(sonda_cordoba[~cordoba], meta_h2[~cordoba]):
        raise RuntimeError("La sonda no deja quieto todo lo que no es Cordoba H2")
    if not np.allclose(sonda_cordoba[cordoba], meta_h2[cordoba] * 1.5):
        raise RuntimeError("La sonda no es x1.5 en Cordoba H2")
    p_m = meta_h2[publico]
    producto_dc = mezcla.producto_sonda(p_m, sonda_cordoba[publico], SCORE_META_ARCHIVO, SCORE_CORDOBA)
    vc = np.where(cordoba[publico], p_m, 0.0)
    producto_vc = producto_dc / 0.5
    factor_c = producto_vc / float(vc @ vc)
    pred_cordoba = p_m + (factor_c - 1.0) * vc
    producto_m = (float(p_m @ p_m) + norma_y - n * SCORE_META_ARCHIVO) / 2
    producto_cordoba = producto_m + (factor_c - 1.0) * producto_vc
    puntaje_cordoba = mezcla.score_de(pred_cordoba, producto_cordoba, norma_y)
    print(f"factor Cordoba {factor_c:.5f}  media {pred_cordoba[cordoba[publico]].mean():.2f}  antes {p_m[cordoba[publico]].mean():.2f}")
    print(f"MSE publico Cordoba {puntaje_cordoba:.5f}  baja {SCORE_META_ARCHIVO - puntaje_cordoba:.2f}  minimo {pred_cordoba.min():.4f}")

    def guardar_cordoba(origen: str, nombre: str) -> None:
        valores = pd.read_csv(CARPETA / origen).set_index("id").dengue.loc[ids].to_numpy(float)
        valores[publico] = np.maximum(pred_cordoba, 0.0)
        if not np.isfinite(valores).all() or valores.min() < -1e-8:
            raise RuntimeError(nombre)
        envio = muestra[["id"]].copy()
        envio["dengue"] = np.maximum(valores, 0.0)
        envio.to_csv(CARPETA / nombre, index=False)

    guardar_cordoba("submission_meta.csv", "submission_cordoba.csv")
    guardar_cordoba("submission_meta_seleccion_2.csv", "submission_cordoba_seleccion_2.csv")

    cordoba_h2 = pd.read_csv(CARPETA / "submission_cordoba.csv").set_index("id").dengue.loc[ids].to_numpy(float)
    antioquia = partes[0].eq(ANTIOQUIA).to_numpy() & publico
    sonda_antioquia = pd.read_csv(CARPETA / "submission_sonda_antioquia.csv").set_index("id").dengue.loc[ids].to_numpy(float)
    if not np.allclose(sonda_antioquia[~antioquia], cordoba_h2[~antioquia]):
        raise RuntimeError("La sonda no deja quieto todo lo que no es Antioquia H2")
    if not np.allclose(sonda_antioquia[antioquia], cordoba_h2[antioquia] * 1.5):
        raise RuntimeError("La sonda no es x1.5 en Antioquia H2")
    p_c = cordoba_h2[publico]
    producto_da = mezcla.producto_sonda(p_c, sonda_antioquia[publico], SCORE_CORDOBA_ARCHIVO, SCORE_ANTIOQUIA)
    va = np.where(antioquia[publico], p_c, 0.0)
    producto_va = producto_da / 0.5
    factor_a = producto_va / float(va @ va)
    pred_antioquia = p_c + (factor_a - 1.0) * va
    producto_c = (float(p_c @ p_c) + norma_y - n * SCORE_CORDOBA_ARCHIVO) / 2
    producto_ant = producto_c + (factor_a - 1.0) * producto_va
    puntaje_ant = mezcla.score_de(pred_antioquia, producto_ant, norma_y)
    print(f"factor Antioquia {factor_a:.5f}  media {pred_antioquia[antioquia[publico]].mean():.2f}  antes {p_c[antioquia[publico]].mean():.2f}")
    print(f"MSE publico Antioquia {puntaje_ant:.5f}  baja {SCORE_CORDOBA_ARCHIVO - puntaje_ant:.2f}")

    sucre = partes[0].eq(SUCRE).to_numpy() & publico
    sonda_sucre = pd.read_csv(CARPETA / "submission_sonda_sucre.csv").set_index("id").dengue.loc[ids].to_numpy(float)
    if not np.allclose(sonda_sucre[~sucre], cordoba_h2[~sucre]):
        raise RuntimeError("La sonda no deja quieto todo lo que no es Sucre H2")
    if not np.allclose(sonda_sucre[sucre], cordoba_h2[sucre] * 1.5):
        raise RuntimeError("La sonda no es x1.5 en Sucre H2")
    producto_ds = mezcla.producto_sonda(p_c, sonda_sucre[publico], SCORE_CORDOBA_ARCHIVO, SCORE_SUCRE)
    vs = np.where(sucre[publico], p_c, 0.0)
    producto_vs = producto_ds / 0.5
    factor_s = producto_vs / float(vs @ vs)
    pred_sucre = p_c + (factor_s - 1.0) * vs + (factor_a - 1.0) * va
    producto_sucre = producto_c + (factor_s - 1.0) * producto_vs + (factor_a - 1.0) * producto_va
    puntaje_sucre = mezcla.score_de(pred_sucre, producto_sucre, norma_y)
    print(f"factor Sucre {factor_s:.5f}  media {pred_sucre[sucre[publico]].mean():.2f}  antes {p_c[sucre[publico]].mean():.2f}")
    print(f"MSE publico Sucre {puntaje_sucre:.5f}  baja {SCORE_CORDOBA_ARCHIVO - puntaje_sucre:.2f}  minimo {pred_sucre.min():.4f}")

    factor_h1_s = 1.0 + PESO_H1_SUCRE * (factor_s - 1.0)
    sucre_h1 = partes[0].eq(SUCRE).to_numpy() & ~publico

    def guardar_sucre(origen: str, nombre: str) -> None:
        valores = pd.read_csv(CARPETA / origen).set_index("id").dengue.loc[ids].to_numpy(float)
        valores[publico] = np.maximum(pred_sucre, 0.0)
        valores[sucre_h1] *= factor_h1_s
        if not np.isfinite(valores).all() or valores.min() < -1e-8:
            raise RuntimeError(nombre)
        envio = muestra[["id"]].copy()
        envio["dengue"] = np.maximum(valores, 0.0)
        envio.to_csv(CARPETA / nombre, index=False)

    guardar_sucre("submission_cordoba.csv", "submission_sucre.csv")
    guardar_sucre("submission_cordoba_seleccion_2.csv", "submission_sucre_seleccion_2.csv")
    print(f"primer semestre Sucre x {factor_h1_s:.5f}")

    sucre_h2 = pd.read_csv(CARPETA / "submission_sucre.csv").set_index("id").dengue.loc[ids].to_numpy(float)
    santander = partes[0].eq(SANTANDER).to_numpy() & publico
    sonda_santander = pd.read_csv(CARPETA / "submission_sonda_santander.csv").set_index("id").dengue.loc[ids].to_numpy(float)
    if not np.allclose(sonda_santander[~santander], sucre_h2[~santander]):
        raise RuntimeError("La sonda no deja quieto todo lo que no es Santander H2")
    if not np.allclose(sonda_santander[santander], sucre_h2[santander] * 1.5):
        raise RuntimeError("La sonda no es x1.5 en Santander H2")
    p_s = sucre_h2[publico]
    producto_dn = mezcla.producto_sonda(p_s, sonda_santander[publico], SCORE_SUCRE_ARCHIVO, SCORE_SANTANDER)
    vn = np.where(santander[publico], p_s, 0.0)
    producto_vn = producto_dn / 0.5
    factor_n = producto_vn / float(vn @ vn)
    pred_santander = p_s + (factor_n - 1.0) * vn
    producto_archivo = (float(p_s @ p_s) + norma_y - n * SCORE_SUCRE_ARCHIVO) / 2
    producto_santander = producto_archivo + (factor_n - 1.0) * producto_vn
    puntaje_santander = mezcla.score_de(pred_santander, producto_santander, norma_y)
    print(f"factor Santander {factor_n:.5f}  media {pred_santander[santander[publico]].mean():.2f}  antes {p_s[santander[publico]].mean():.2f}")
    print(f"MSE publico Santander {puntaje_santander:.5f}  baja {SCORE_SUCRE_ARCHIVO - puntaje_santander:.2f}  minimo {pred_santander.min():.4f}")

    def guardar_santander(origen: str, nombre: str) -> None:
        valores = pd.read_csv(CARPETA / origen).set_index("id").dengue.loc[ids].to_numpy(float)
        valores[publico] = np.maximum(pred_santander, 0.0)
        if not np.isfinite(valores).all() or valores.min() < -1e-8:
            raise RuntimeError(nombre)
        envio = muestra[["id"]].copy()
        envio["dengue"] = np.maximum(valores, 0.0)
        envio.to_csv(CARPETA / nombre, index=False)

    guardar_santander("submission_sucre.csv", "submission_santander.csv")
    guardar_santander("submission_sucre_seleccion_2.csv", "submission_santander_seleccion_2.csv")

    santander_h2 = pd.read_csv(CARPETA / "submission_santander.csv").set_index("id").dengue.loc[ids].to_numpy(float)
    huila = partes[0].eq(HUILA).to_numpy() & publico
    sonda_huila = pd.read_csv(CARPETA / "submission_sonda_huila.csv").set_index("id").dengue.loc[ids].to_numpy(float)
    if not np.allclose(sonda_huila[~huila], santander_h2[~huila]):
        raise RuntimeError("La sonda no deja quieto todo lo que no es Huila H2")
    if not np.allclose(sonda_huila[huila], santander_h2[huila] * 1.5):
        raise RuntimeError("La sonda no es x1.5 en Huila H2")
    p_h = santander_h2[publico]
    producto_dh = mezcla.producto_sonda(p_h, sonda_huila[publico], SCORE_SANTANDER_ARCHIVO, SCORE_HUILA)
    vh = np.where(huila[publico], p_h, 0.0)
    producto_vh = producto_dh / 0.5
    factor_h = producto_vh / float(vh @ vh)
    pred_huila = p_h + (factor_h - 1.0) * vh
    producto_san = (float(p_h @ p_h) + norma_y - n * SCORE_SANTANDER_ARCHIVO) / 2
    producto_huila = producto_san + (factor_h - 1.0) * producto_vh
    puntaje_huila = mezcla.score_de(pred_huila, producto_huila, norma_y)
    print(f"factor Huila {factor_h:.5f}  media {pred_huila[huila[publico]].mean():.2f}  antes {p_h[huila[publico]].mean():.2f}")
    print(f"MSE publico Huila {puntaje_huila:.5f}  baja {SCORE_SANTANDER_ARCHIVO - puntaje_huila:.2f}  minimo {pred_huila.min():.4f}")

    def guardar_huila(origen: str, nombre: str) -> None:
        valores = pd.read_csv(CARPETA / origen).set_index("id").dengue.loc[ids].to_numpy(float)
        valores[publico] = np.maximum(pred_huila, 0.0)
        if not np.isfinite(valores).all() or valores.min() < -1e-8:
            raise RuntimeError(nombre)
        envio = muestra[["id"]].copy()
        envio["dengue"] = np.maximum(valores, 0.0)
        envio.to_csv(CARPETA / nombre, index=False)

    guardar_huila("submission_santander.csv", "submission_huila.csv")
    guardar_huila("submission_santander_seleccion_2.csv", "submission_huila_seleccion_2.csv")

    huila_h2 = pd.read_csv(CARPETA / "submission_huila.csv").set_index("id").dengue.loc[ids].to_numpy(float)
    cesar = partes[0].eq(CESAR).to_numpy() & publico
    sonda_cesar = pd.read_csv(CARPETA / "submission_sonda_cesar.csv").set_index("id").dengue.loc[ids].to_numpy(float)
    if not np.allclose(sonda_cesar[~cesar], huila_h2[~cesar]):
        raise RuntimeError("La sonda no deja quieto todo lo que no es Cesar H2")
    if not np.allclose(sonda_cesar[cesar], huila_h2[cesar] * 1.5):
        raise RuntimeError("La sonda no es x1.5 en Cesar H2")
    p_e = huila_h2[publico]
    producto_de = mezcla.producto_sonda(p_e, sonda_cesar[publico], SCORE_HUILA_ARCHIVO, SCORE_CESAR)
    ve = np.where(cesar[publico], p_e, 0.0)
    producto_ve = producto_de / 0.5
    factor_e = producto_ve / float(ve @ ve)
    pred_cesar = p_e + (factor_e - 1.0) * ve
    producto_hu = (float(p_e @ p_e) + norma_y - n * SCORE_HUILA_ARCHIVO) / 2
    producto_cesar = producto_hu + (factor_e - 1.0) * producto_ve
    puntaje_cesar = mezcla.score_de(pred_cesar, producto_cesar, norma_y)
    print(f"factor Cesar {factor_e:.5f}  media {pred_cesar[cesar[publico]].mean():.2f}  antes {p_e[cesar[publico]].mean():.2f}")
    print(f"MSE publico Cesar {puntaje_cesar:.5f}  baja {SCORE_HUILA_ARCHIVO - puntaje_cesar:.2f}  minimo {pred_cesar.min():.4f}")

    factor_h1_e = 1.0 + PESO_H1_CESAR * (factor_e - 1.0)
    cesar_h1 = partes[0].eq(CESAR).to_numpy() & ~publico

    def guardar_cesar(origen: str, nombre: str) -> None:
        valores = pd.read_csv(CARPETA / origen).set_index("id").dengue.loc[ids].to_numpy(float)
        valores[publico] = np.maximum(pred_cesar, 0.0)
        valores[cesar_h1] *= factor_h1_e
        if not np.isfinite(valores).all() or valores.min() < -1e-8:
            raise RuntimeError(nombre)
        envio = muestra[["id"]].copy()
        envio["dengue"] = np.maximum(valores, 0.0)
        envio.to_csv(CARPETA / nombre, index=False)

    guardar_cesar("submission_huila.csv", "submission_cesar.csv")
    guardar_cesar("submission_huila_seleccion_2.csv", "submission_cesar_seleccion_2.csv")
    print(f"primer semestre Cesar x {factor_h1_e:.5f}")

    cesar_h2 = pd.read_csv(CARPETA / "submission_cesar.csv").set_index("id").dengue.loc[ids].to_numpy(float)
    guajira = partes[0].eq(GUAJIRA).to_numpy() & publico
    sonda_guajira = pd.read_csv(CARPETA / "submission_sonda_guajira.csv").set_index("id").dengue.loc[ids].to_numpy(float)
    if not np.allclose(sonda_guajira[~guajira], cesar_h2[~guajira]):
        raise RuntimeError("La sonda no deja quieto todo lo que no es Guajira H2")
    if not np.allclose(sonda_guajira[guajira], cesar_h2[guajira] * 1.5):
        raise RuntimeError("La sonda no es x1.5 en Guajira H2")
    p_g = cesar_h2[publico]
    producto_dg = mezcla.producto_sonda(p_g, sonda_guajira[publico], SCORE_CESAR_ARCHIVO, SCORE_GUAJIRA)
    vg = np.where(guajira[publico], p_g, 0.0)
    producto_vg = producto_dg / 0.5
    factor_g = producto_vg / float(vg @ vg)
    pred_guajira = p_g + (factor_g - 1.0) * vg
    producto_ce = (float(p_g @ p_g) + norma_y - n * SCORE_CESAR_ARCHIVO) / 2
    producto_guajira = producto_ce + (factor_g - 1.0) * producto_vg
    puntaje_guajira = mezcla.score_de(pred_guajira, producto_guajira, norma_y)
    print(f"factor Guajira {factor_g:.5f}  media {pred_guajira[guajira[publico]].mean():.2f}  antes {p_g[guajira[publico]].mean():.2f}")
    print(f"MSE publico Guajira {puntaje_guajira:.5f}  baja {SCORE_CESAR_ARCHIVO - puntaje_guajira:.2f}  minimo {pred_guajira.min():.4f}")

    def guardar_guajira(origen: str, nombre: str) -> None:
        valores = pd.read_csv(CARPETA / origen).set_index("id").dengue.loc[ids].to_numpy(float)
        valores[publico] = np.maximum(pred_guajira, 0.0)
        if not np.isfinite(valores).all() or valores.min() < -1e-8:
            raise RuntimeError(nombre)
        envio = muestra[["id"]].copy()
        envio["dengue"] = np.maximum(valores, 0.0)
        envio.to_csv(CARPETA / nombre, index=False)

    guardar_guajira("submission_cesar.csv", "submission_guajira.csv")
    guardar_guajira("submission_cesar_seleccion_2.csv", "submission_guajira_seleccion_2.csv")

    guajira_h2 = pd.read_csv(CARPETA / "submission_guajira.csv").set_index("id").dengue.loc[ids].to_numpy(float)
    cundinamarca = partes[0].eq(CUNDINAMARCA).to_numpy() & publico
    sonda_cundi = pd.read_csv(CARPETA / "submission_sonda_cundinamarca.csv").set_index("id").dengue.loc[ids].to_numpy(float)
    if not np.allclose(sonda_cundi[~cundinamarca], guajira_h2[~cundinamarca]):
        raise RuntimeError("La sonda no deja quieto todo lo que no es Cundinamarca H2")
    if not np.allclose(sonda_cundi[cundinamarca], guajira_h2[cundinamarca] * 1.5):
        raise RuntimeError("La sonda no es x1.5 en Cundinamarca H2")
    p_u = guajira_h2[publico]
    producto_du = mezcla.producto_sonda(p_u, sonda_cundi[publico], SCORE_GUAJIRA_ARCHIVO, SCORE_CUNDINAMARCA)
    vu = np.where(cundinamarca[publico], p_u, 0.0)
    producto_vu = producto_du / 0.5
    factor_u = producto_vu / float(vu @ vu)
    pred_cundi = p_u + (factor_u - 1.0) * vu
    producto_gj = (float(p_u @ p_u) + norma_y - n * SCORE_GUAJIRA_ARCHIVO) / 2
    producto_cundi = producto_gj + (factor_u - 1.0) * producto_vu
    puntaje_cundi = mezcla.score_de(pred_cundi, producto_cundi, norma_y)
    print(f"factor Cundinamarca {factor_u:.5f}  media {pred_cundi[cundinamarca[publico]].mean():.2f}  antes {p_u[cundinamarca[publico]].mean():.2f}")
    print(f"MSE publico Cundinamarca {puntaje_cundi:.5f}  baja {SCORE_GUAJIRA_ARCHIVO - puntaje_cundi:.2f}  minimo {pred_cundi.min():.4f}")

    def guardar_cundi(origen: str, nombre: str) -> None:
        valores = pd.read_csv(CARPETA / origen).set_index("id").dengue.loc[ids].to_numpy(float)
        valores[publico] = np.maximum(pred_cundi, 0.0)
        if not np.isfinite(valores).all() or valores.min() < -1e-8:
            raise RuntimeError(nombre)
        envio = muestra[["id"]].copy()
        envio["dengue"] = np.maximum(valores, 0.0)
        envio.to_csv(CARPETA / nombre, index=False)

    guardar_cundi("submission_guajira.csv", "submission_cundinamarca.csv")
    guardar_cundi("submission_guajira_seleccion_2.csv", "submission_cundinamarca_seleccion_2.csv")

    putumayo = partes[0].eq(PUTUMAYO).to_numpy() & publico
    sonda_putumayo = pd.read_csv(CARPETA / "submission_cundinamarca_seleccion_2.csv").set_index("id").dengue.loc[ids].to_numpy(float)
    if not np.allclose(sonda_putumayo[publico], pred_cundi):
        raise RuntimeError("La seleccion 2 no trae el segundo semestre calibrado")
    sonda_putumayo[putumayo] *= 1.5
    if sonda_putumayo.min() < -1e-8 or not np.isfinite(sonda_putumayo).all():
        raise RuntimeError("Sonda de Putumayo no valida")
    envio_putumayo = muestra[["id"]].copy()
    envio_putumayo["dengue"] = sonda_putumayo
    envio_putumayo.to_csv(CARPETA / "submission_sonda_putumayo.csv", index=False)
    distancia_putumayo = float(np.mean((sonda_putumayo[publico] - pred_cundi) ** 2))
    print(
        f"sonda Putumayo x1.5 sobre seleccion 2  distancia MSE {distancia_putumayo:.6f}"
        f"  media {pred_cundi[putumayo[publico]].mean():.2f}  filas {int(putumayo.sum())}"
    )


if __name__ == "__main__":
    main()
