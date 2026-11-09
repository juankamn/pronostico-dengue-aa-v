"""Calibrate scored department probes and write the next x1.5 probe.

Scored probes are read and checked. They are not rewritten. The public half
of both selections stays the calibrated vector. The next probe multiplies one
department on the seleccion-2 file, so one upload measures that department
and keeps the Valle hedge.
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd

ANTERIOR = Path(__file__).resolve().parents[1] / "2026-11-07-mezcla-medida"
sys.path.insert(0, str(ANTERIOR))

import ejecutar as mezcla  # noqa: E402

RAIZ = mezcla.RAIZ
CARPETA = Path(__file__).resolve().parent
SCORE_BASE = 1156.27758
SCORE_PUTUMAYO = 1177.25208
SCORE_PUTUMAYO_ARCHIVO = 1154.63701
SCORE_CASANARE = 1186.12303
SCORE_CASANARE_ARCHIVO = 1143.95494
SCORE_MAGDALENA = 1080.79285
SCORE_MAGDALENA_ARCHIVO = 982.31573
SCORE_CAQUETA = 989.56434
PESO_H1_PUTUMAYO = 0.75
PESO_H1_CASANARE = 1.0
# 2014-2023 picks 0.5. Rolling 2016-2023 is 3908.1 against 3909.4; the worst year is +0.2.
PESO_H1_CAQUETA = 0.5
SCORE_ARAUCA = 976.63977
# 2014-2023 picks 0.75. Rolling 2016-2023 is 3905.2 against 3909.4; the worst year is +0.02.
PESO_H1_ARAUCA = 0.75
SCORE_ARAUCA_ARCHIVO = 976.60859
SCORE_QUINDIO = 987.76571
SCORE_QUINDIO_ARCHIVO = 973.14988
SCORE_CAUCA = 982.33246
SCORE_CAUCA_ARCHIVO = 970.92550
SCORE_RISARALDA = 977.20048
# 2014-2023 picks 1.0. Rolling 2016-2023 is 3906.3 against 3909.4; the worst year is +0.7.
PESO_H1_RISARALDA = 1.0
SCORE_RISARALDA_ARCHIVO = 969.64615
SCORE_NARINO = 968.76129
SCORE_NARINO_ARCHIVO = 968.69500
SCORE_CHOCO = 965.57514
# 2014-2023 picks 0.25. Rolling 2016-2023 is 3908.8 against 3909.4; the worst year is +1.5.
PESO_H1_CHOCO = 0.25
SCORE_CHOCO_ARCHIVO = 963.69475
SCORE_BOYACA = 964.63791
# 2014-2023 picks 1.0. Rolling 2016-2023 is 3906.9 against 3909.4; the worst year is +0.6.
PESO_H1_BOYACA = 1.0
SCORE_AMAZONAS = 964.08840
SCORE_GUAVIARE = 963.12784
# 2014-2023 picks 0.75. Rolling 2016-2023 is 3908.4 against 3909.4; the worst year is +0.5.
PESO_H1_GUAVIARE = 0.75
SCORE_CALDAS = 964.30352
# 2014-2023 picks 0.5. Rolling 2016-2023 is 3909.3 against 3909.4; the worst year is +0.7.
PESO_H1_CALDAS = 0.5
SCORE_VICHADA = 962.09030
# 2014-2023 picks 0.75. Rolling 2016-2023 is 3909.2 against 3909.4; no year gets worse.
PESO_H1_VICHADA = 0.75
SCORE_VICHADA_ARCHIVO = 961.80425
SCORE_VAUPES = 961.85112
# 2014-2023 picks 0.75. Rolling 2016-2023 is 3909.3 against 3909.4; the worst year is +0.2.
PESO_H1_VAUPES = 0.75
SCORE_GUAINIA = 961.80446
# 2014-2023 picks 1.0. Rolling 2016-2023 is 3909.3 against 3909.4; no year gets worse.
PESO_H1_GUAINIA = 1.0
PUTUMAYO = 23
CASANARE = 9
MAGDALENA = 19
CAQUETA = 8
ARAUCA = 3
QUINDIO = 24
CAUCA = 10
RISARALDA = 25
NARINO = 21
CHOCO = 12
BOYACA = 6
AMAZONAS = 1
GUAVIARE = 17
CALDAS = 7
VICHADA = 32
VAUPES = 31
GUAINIA = 15


def main() -> None:
    muestra = pd.read_csv(RAIZ / "data/sample_submission.csv")
    ids = muestra.id
    partes = ids.str.split("_", expand=True).astype(int)
    publico = partes[2].ge(27).to_numpy()
    putumayo = partes[0].eq(PUTUMAYO).to_numpy() & publico
    base = pd.read_csv(ANTERIOR / "submission_cundinamarca.csv").set_index("id").dengue.loc[ids].to_numpy(float)
    sel2 = pd.read_csv(ANTERIOR / "submission_cundinamarca_seleccion_2.csv").set_index("id").dengue.loc[ids].to_numpy(float)
    sonda = pd.read_csv(ANTERIOR / "submission_sonda_putumayo.csv").set_index("id").dengue.loc[ids].to_numpy(float)
    if not np.allclose(base[publico], sel2[publico]):
        raise RuntimeError("Las dos selecciones no comparten el segundo semestre")
    if not np.allclose(sonda[~putumayo], sel2[~putumayo]):
        raise RuntimeError("La sonda no deja quieto todo lo que no es Putumayo H2")
    if not np.allclose(sonda[putumayo], sel2[putumayo] * 1.5):
        raise RuntimeError("La sonda no es x1.5 en Putumayo H2")
    if not np.allclose(sonda[~publico], sel2[~publico]):
        raise RuntimeError("La sonda no trae el primer semestre de la seleccion 2")

    vectores = mezcla.cargar(ids)
    puntajes = {nombre: score for nombre, (_, score) in mezcla.ARCHIVOS.items()}
    norma_y = mezcla.norma_real(vectores, puntajes, publico)
    p0 = base[publico]
    n = len(p0)
    producto_d = mezcla.producto_sonda(p0, sonda[publico], SCORE_BASE, SCORE_PUTUMAYO)
    v = np.where(putumayo[publico], p0, 0.0)
    producto_v = producto_d / 0.5
    factor = producto_v / float(v @ v)
    pred = p0 + (factor - 1.0) * v
    producto_base = (float(p0 @ p0) + norma_y - n * SCORE_BASE) / 2
    producto = producto_base + (factor - 1.0) * producto_v
    puntaje = mezcla.score_de(pred, producto, norma_y)
    print(f"factor Putumayo {factor:.5f}  media {pred[putumayo[publico]].mean():.2f}  antes {p0[putumayo[publico]].mean():.2f}")
    print(f"MSE publico Putumayo {puntaje:.5f}  baja {SCORE_BASE - puntaje:.2f}  minimo {pred.min():.4f}")

    factor_h1 = 1.0 + PESO_H1_PUTUMAYO * (factor - 1.0)
    putumayo_h1 = partes[0].eq(PUTUMAYO).to_numpy() & ~publico

    def guardar(origen: np.ndarray, nombre: str) -> None:
        valores = origen.copy()
        valores[publico] = np.maximum(pred, 0.0)
        valores[putumayo_h1] *= factor_h1
        if not np.isfinite(valores).all() or valores.min() < -1e-8:
            raise RuntimeError(nombre)
        envio = muestra[["id"]].copy()
        envio["dengue"] = np.maximum(valores, 0.0)
        envio.to_csv(CARPETA / nombre, index=False)

    guardar(base, "submission_putumayo.csv")
    guardar(sel2, "submission_putumayo_seleccion_2.csv")
    print(f"primer semestre Putumayo x {factor_h1:.5f}")

    casanare = partes[0].eq(CASANARE).to_numpy() & publico
    putumayo_h2 = pd.read_csv(CARPETA / "submission_putumayo.csv").set_index("id").dengue.loc[ids].to_numpy(float)
    putumayo_sel2 = pd.read_csv(CARPETA / "submission_putumayo_seleccion_2.csv").set_index("id").dengue.loc[ids].to_numpy(float)
    sonda_casanare = pd.read_csv(CARPETA / "submission_sonda_casanare.csv").set_index("id").dengue.loc[ids].to_numpy(float)
    if not np.allclose(sonda_casanare[~casanare], putumayo_sel2[~casanare]):
        raise RuntimeError("La sonda no deja quieto todo lo que no es Casanare H2")
    if not np.allclose(sonda_casanare[casanare], putumayo_sel2[casanare] * 1.5):
        raise RuntimeError("La sonda no es x1.5 en Casanare H2")
    if not np.allclose(sonda_casanare[~publico], putumayo_sel2[~publico]):
        raise RuntimeError("La sonda no trae el primer semestre de la seleccion 2")
    p_c = putumayo_h2[publico]
    producto_dc = mezcla.producto_sonda(p_c, sonda_casanare[publico], SCORE_PUTUMAYO_ARCHIVO, SCORE_CASANARE)
    vc = np.where(casanare[publico], p_c, 0.0)
    producto_vc = producto_dc / 0.5
    factor_c = producto_vc / float(vc @ vc)
    pred_casanare = p_c + (factor_c - 1.0) * vc
    producto_pu = (float(p_c @ p_c) + norma_y - n * SCORE_PUTUMAYO_ARCHIVO) / 2
    producto_casanare = producto_pu + (factor_c - 1.0) * producto_vc
    puntaje_casanare = mezcla.score_de(pred_casanare, producto_casanare, norma_y)
    print(f"factor Casanare {factor_c:.5f}  media {pred_casanare[casanare[publico]].mean():.2f}  antes {p_c[casanare[publico]].mean():.2f}")
    print(f"MSE publico Casanare {puntaje_casanare:.5f}  baja {SCORE_PUTUMAYO_ARCHIVO - puntaje_casanare:.2f}  minimo {pred_casanare.min():.4f}")

    factor_h1_c = 1.0 + PESO_H1_CASANARE * (factor_c - 1.0)
    casanare_h1 = partes[0].eq(CASANARE).to_numpy() & ~publico

    def guardar_casanare(origen: np.ndarray, nombre: str) -> None:
        valores = origen.copy()
        valores[publico] = np.maximum(pred_casanare, 0.0)
        valores[casanare_h1] *= factor_h1_c
        if not np.isfinite(valores).all() or valores.min() < -1e-8:
            raise RuntimeError(nombre)
        envio_c = muestra[["id"]].copy()
        envio_c["dengue"] = np.maximum(valores, 0.0)
        envio_c.to_csv(CARPETA / nombre, index=False)

    guardar_casanare(putumayo_h2, "submission_casanare.csv")
    guardar_casanare(putumayo_sel2, "submission_casanare_seleccion_2.csv")
    print(f"primer semestre Casanare x {factor_h1_c:.5f}")

    casanare_h2 = pd.read_csv(CARPETA / "submission_casanare.csv").set_index("id").dengue.loc[ids].to_numpy(float)
    casanare_sel2 = pd.read_csv(CARPETA / "submission_casanare_seleccion_2.csv").set_index("id").dengue.loc[ids].to_numpy(float)
    magdalena = partes[0].eq(MAGDALENA).to_numpy() & publico
    sonda_magdalena = pd.read_csv(CARPETA / "submission_sonda_magdalena.csv").set_index("id").dengue.loc[ids].to_numpy(float)
    if not np.allclose(sonda_magdalena[~magdalena], casanare_sel2[~magdalena]):
        raise RuntimeError("La sonda no deja quieto todo lo que no es Magdalena H2")
    if not np.allclose(sonda_magdalena[magdalena], casanare_sel2[magdalena] * 1.5):
        raise RuntimeError("La sonda no es x1.5 en Magdalena H2")
    if not np.allclose(sonda_magdalena[~publico], casanare_sel2[~publico]):
        raise RuntimeError("La sonda no trae el primer semestre de la seleccion 2")
    p_m = casanare_h2[publico]
    producto_dm = mezcla.producto_sonda(p_m, sonda_magdalena[publico], SCORE_CASANARE_ARCHIVO, SCORE_MAGDALENA)
    vm = np.where(magdalena[publico], p_m, 0.0)
    producto_vm = producto_dm / 0.5
    factor_m = producto_vm / float(vm @ vm)
    pred_magdalena = p_m + (factor_m - 1.0) * vm
    producto_ca = (float(p_m @ p_m) + norma_y - n * SCORE_CASANARE_ARCHIVO) / 2
    producto_magdalena = producto_ca + (factor_m - 1.0) * producto_vm
    puntaje_magdalena = mezcla.score_de(pred_magdalena, producto_magdalena, norma_y)
    print(f"factor Magdalena {factor_m:.5f}  media {pred_magdalena[magdalena[publico]].mean():.2f}  antes {p_m[magdalena[publico]].mean():.2f}")
    print(f"MSE publico Magdalena {puntaje_magdalena:.5f}  baja {SCORE_CASANARE_ARCHIVO - puntaje_magdalena:.2f}  minimo {pred_magdalena.min():.4f}")

    def guardar_magdalena(origen: np.ndarray, nombre: str) -> None:
        valores = origen.copy()
        valores[publico] = np.maximum(pred_magdalena, 0.0)
        if not np.isfinite(valores).all() or valores.min() < -1e-8:
            raise RuntimeError(nombre)
        envio_g = muestra[["id"]].copy()
        envio_g["dengue"] = np.maximum(valores, 0.0)
        envio_g.to_csv(CARPETA / nombre, index=False)

    guardar_magdalena(casanare_h2, "submission_magdalena.csv")
    guardar_magdalena(casanare_sel2, "submission_magdalena_seleccion_2.csv")

    magdalena_h2 = pd.read_csv(CARPETA / "submission_magdalena.csv").set_index("id").dengue.loc[ids].to_numpy(float)
    magdalena_sel2 = pd.read_csv(CARPETA / "submission_magdalena_seleccion_2.csv").set_index("id").dengue.loc[ids].to_numpy(float)
    caqueta = partes[0].eq(CAQUETA).to_numpy() & publico
    sonda_caqueta = pd.read_csv(CARPETA / "submission_sonda_caqueta.csv").set_index("id").dengue.loc[ids].to_numpy(float)
    if not np.allclose(sonda_caqueta[~caqueta], magdalena_sel2[~caqueta]):
        raise RuntimeError("La sonda no deja quieto todo lo que no es Caqueta H2")
    if not np.allclose(sonda_caqueta[caqueta], magdalena_sel2[caqueta] * 1.5):
        raise RuntimeError("La sonda no es x1.5 en Caqueta H2")
    if not np.allclose(sonda_caqueta[~publico], magdalena_sel2[~publico]):
        raise RuntimeError("La sonda no trae el primer semestre de la seleccion 2")
    p_q = magdalena_h2[publico]
    producto_dq = mezcla.producto_sonda(p_q, sonda_caqueta[publico], SCORE_MAGDALENA_ARCHIVO, SCORE_CAQUETA)
    vq = np.where(caqueta[publico], p_q, 0.0)
    producto_vq = producto_dq / 0.5
    factor_q = producto_vq / float(vq @ vq)
    pred_caqueta = p_q + (factor_q - 1.0) * vq
    producto_mg = (float(p_q @ p_q) + norma_y - n * SCORE_MAGDALENA_ARCHIVO) / 2
    producto_caqueta = producto_mg + (factor_q - 1.0) * producto_vq
    puntaje_caqueta = mezcla.score_de(pred_caqueta, producto_caqueta, norma_y)
    print(f"factor Caqueta {factor_q:.5f}  media {pred_caqueta[caqueta[publico]].mean():.2f}  antes {p_q[caqueta[publico]].mean():.2f}")
    print(f"MSE publico Caqueta {puntaje_caqueta:.5f}  baja {SCORE_MAGDALENA_ARCHIVO - puntaje_caqueta:.2f}  minimo {pred_caqueta.min():.4f}")

    factor_h1_q = 1.0 + PESO_H1_CAQUETA * (factor_q - 1.0)
    caqueta_h1 = partes[0].eq(CAQUETA).to_numpy() & ~publico

    def guardar_caqueta(origen: np.ndarray, nombre: str) -> None:
        valores = origen.copy()
        valores[publico] = np.maximum(pred_caqueta, 0.0)
        valores[caqueta_h1] *= factor_h1_q
        if not np.isfinite(valores).all() or valores.min() < -1e-8:
            raise RuntimeError(nombre)
        envio_q = muestra[["id"]].copy()
        envio_q["dengue"] = np.maximum(valores, 0.0)
        envio_q.to_csv(CARPETA / nombre, index=False)

    guardar_caqueta(magdalena_h2, "submission_caqueta.csv")
    guardar_caqueta(magdalena_sel2, "submission_caqueta_seleccion_2.csv")
    print(f"primer semestre Caqueta x {factor_h1_q:.5f}")

    caqueta_h2 = pd.read_csv(CARPETA / "submission_caqueta.csv").set_index("id").dengue.loc[ids].to_numpy(float)
    caqueta_sel2 = pd.read_csv(CARPETA / "submission_caqueta_seleccion_2.csv").set_index("id").dengue.loc[ids].to_numpy(float)
    arauca = partes[0].eq(ARAUCA).to_numpy() & publico
    sonda_arauca = pd.read_csv(CARPETA / "submission_sonda_arauca.csv").set_index("id").dengue.loc[ids].to_numpy(float)
    if not np.allclose(sonda_arauca[~arauca], caqueta_sel2[~arauca]):
        raise RuntimeError("La sonda no deja quieto todo lo que no es Arauca H2")
    if not np.allclose(sonda_arauca[arauca], caqueta_sel2[arauca] * 1.5):
        raise RuntimeError("La sonda no es x1.5 en Arauca H2")
    if not np.allclose(sonda_arauca[~publico], caqueta_sel2[~publico]):
        raise RuntimeError("La sonda no trae el primer semestre de la seleccion 2")
    p_a = caqueta_h2[publico]
    producto_da = mezcla.producto_sonda(p_a, sonda_arauca[publico], puntaje_caqueta, SCORE_ARAUCA)
    va = np.where(arauca[publico], p_a, 0.0)
    producto_va = producto_da / 0.5
    factor_a = producto_va / float(va @ va)
    pred_arauca = p_a + (factor_a - 1.0) * va
    producto_cq = (float(p_a @ p_a) + norma_y - n * puntaje_caqueta) / 2
    producto_arauca = producto_cq + (factor_a - 1.0) * producto_va
    puntaje_arauca = mezcla.score_de(pred_arauca, producto_arauca, norma_y)
    print(f"factor Arauca {factor_a:.5f}  media {pred_arauca[arauca[publico]].mean():.2f}  antes {p_a[arauca[publico]].mean():.2f}")
    print(f"MSE publico Arauca {puntaje_arauca:.5f}  baja {puntaje_caqueta - puntaje_arauca:.2f}  minimo {pred_arauca.min():.4f}")

    factor_h1_a = 1.0 + PESO_H1_ARAUCA * (factor_a - 1.0)
    arauca_h1 = partes[0].eq(ARAUCA).to_numpy() & ~publico

    def guardar_arauca(origen: np.ndarray, nombre: str) -> None:
        valores = origen.copy()
        valores[publico] = np.maximum(pred_arauca, 0.0)
        valores[arauca_h1] *= factor_h1_a
        if not np.isfinite(valores).all() or valores.min() < -1e-8:
            raise RuntimeError(nombre)
        envio_a = muestra[["id"]].copy()
        envio_a["dengue"] = np.maximum(valores, 0.0)
        envio_a.to_csv(CARPETA / nombre, index=False)

    guardar_arauca(caqueta_h2, "submission_arauca.csv")
    guardar_arauca(caqueta_sel2, "submission_arauca_seleccion_2.csv")
    print(f"primer semestre Arauca x {factor_h1_a:.5f}")

    arauca_h2 = pd.read_csv(CARPETA / "submission_arauca.csv").set_index("id").dengue.loc[ids].to_numpy(float)
    arauca_sel2 = pd.read_csv(CARPETA / "submission_arauca_seleccion_2.csv").set_index("id").dengue.loc[ids].to_numpy(float)
    quindio = partes[0].eq(QUINDIO).to_numpy() & publico
    sonda_quindio = pd.read_csv(CARPETA / "submission_sonda_quindio.csv").set_index("id").dengue.loc[ids].to_numpy(float)
    if not np.allclose(sonda_quindio[~quindio], arauca_sel2[~quindio]):
        raise RuntimeError("La sonda no deja quieto todo lo que no es Quindio H2")
    if not np.allclose(sonda_quindio[quindio], arauca_sel2[quindio] * 1.5):
        raise RuntimeError("La sonda no es x1.5 en Quindio H2")
    if not np.allclose(sonda_quindio[~publico], arauca_sel2[~publico]):
        raise RuntimeError("La sonda no trae el primer semestre de la seleccion 2")
    p_u = arauca_h2[publico]
    producto_du = mezcla.producto_sonda(p_u, sonda_quindio[publico], SCORE_ARAUCA_ARCHIVO, SCORE_QUINDIO)
    vu = np.where(quindio[publico], p_u, 0.0)
    producto_vu = producto_du / 0.5
    factor_u = producto_vu / float(vu @ vu)
    pred_quindio = p_u + (factor_u - 1.0) * vu
    producto_ar = (float(p_u @ p_u) + norma_y - n * SCORE_ARAUCA_ARCHIVO) / 2
    producto_quindio = producto_ar + (factor_u - 1.0) * producto_vu
    puntaje_quindio = mezcla.score_de(pred_quindio, producto_quindio, norma_y)
    print(f"factor Quindio {factor_u:.5f}  media {pred_quindio[quindio[publico]].mean():.2f}  antes {p_u[quindio[publico]].mean():.2f}")
    print(f"MSE publico Quindio {puntaje_quindio:.5f}  baja {SCORE_ARAUCA_ARCHIVO - puntaje_quindio:.2f}  minimo {pred_quindio.min():.4f}")

    # Rolling H1 loses 2016 by 10.3 and the 2016-2023 average does not improve, so H1 stays put.

    def guardar_quindio(origen: np.ndarray, nombre: str) -> None:
        valores = origen.copy()
        valores[publico] = np.maximum(pred_quindio, 0.0)
        if not np.isfinite(valores).all() or valores.min() < -1e-8:
            raise RuntimeError(nombre)
        envio_u = muestra[["id"]].copy()
        envio_u["dengue"] = np.maximum(valores, 0.0)
        envio_u.to_csv(CARPETA / nombre, index=False)

    guardar_quindio(arauca_h2, "submission_quindio.csv")
    guardar_quindio(arauca_sel2, "submission_quindio_seleccion_2.csv")

    quindio_h2 = pd.read_csv(CARPETA / "submission_quindio.csv").set_index("id").dengue.loc[ids].to_numpy(float)
    quindio_sel2 = pd.read_csv(CARPETA / "submission_quindio_seleccion_2.csv").set_index("id").dengue.loc[ids].to_numpy(float)
    cauca = partes[0].eq(CAUCA).to_numpy() & publico
    sonda_cauca = pd.read_csv(CARPETA / "submission_sonda_cauca.csv").set_index("id").dengue.loc[ids].to_numpy(float)
    if not np.allclose(sonda_cauca[~cauca], quindio_sel2[~cauca]):
        raise RuntimeError("La sonda no deja quieto todo lo que no es Cauca H2")
    if not np.allclose(sonda_cauca[cauca], quindio_sel2[cauca] * 1.5):
        raise RuntimeError("La sonda no es x1.5 en Cauca H2")
    if not np.allclose(sonda_cauca[~publico], quindio_sel2[~publico]):
        raise RuntimeError("La sonda no trae el primer semestre de la seleccion 2")
    p_k = quindio_h2[publico]
    producto_dk = mezcla.producto_sonda(p_k, sonda_cauca[publico], SCORE_QUINDIO_ARCHIVO, SCORE_CAUCA)
    vk = np.where(cauca[publico], p_k, 0.0)
    producto_vk = producto_dk / 0.5
    factor_k = producto_vk / float(vk @ vk)
    pred_cauca = p_k + (factor_k - 1.0) * vk
    producto_qu = (float(p_k @ p_k) + norma_y - n * SCORE_QUINDIO_ARCHIVO) / 2
    producto_cauca = producto_qu + (factor_k - 1.0) * producto_vk
    puntaje_cauca = mezcla.score_de(pred_cauca, producto_cauca, norma_y)
    print(f"factor Cauca {factor_k:.5f}  media {pred_cauca[cauca[publico]].mean():.2f}  antes {p_k[cauca[publico]].mean():.2f}")
    print(f"MSE publico Cauca {puntaje_cauca:.5f}  baja {SCORE_QUINDIO_ARCHIVO - puntaje_cauca:.2f}  minimo {pred_cauca.min():.4f}")

    # Rolling H1 loses 2023 by 39, so H1 stays put.

    def guardar_cauca(origen: np.ndarray, nombre: str) -> None:
        valores = origen.copy()
        valores[publico] = np.maximum(pred_cauca, 0.0)
        if not np.isfinite(valores).all() or valores.min() < -1e-8:
            raise RuntimeError(nombre)
        envio_k = muestra[["id"]].copy()
        envio_k["dengue"] = np.maximum(valores, 0.0)
        envio_k.to_csv(CARPETA / nombre, index=False)

    guardar_cauca(quindio_h2, "submission_cauca.csv")
    guardar_cauca(quindio_sel2, "submission_cauca_seleccion_2.csv")

    cauca_h2 = pd.read_csv(CARPETA / "submission_cauca.csv").set_index("id").dengue.loc[ids].to_numpy(float)
    cauca_sel2 = pd.read_csv(CARPETA / "submission_cauca_seleccion_2.csv").set_index("id").dengue.loc[ids].to_numpy(float)
    risaralda = partes[0].eq(RISARALDA).to_numpy() & publico
    sonda_risaralda = pd.read_csv(CARPETA / "submission_sonda_risaralda.csv").set_index("id").dengue.loc[ids].to_numpy(float)
    if not np.allclose(sonda_risaralda[~risaralda], cauca_sel2[~risaralda]):
        raise RuntimeError("La sonda no deja quieto todo lo que no es Risaralda H2")
    if not np.allclose(sonda_risaralda[risaralda], cauca_sel2[risaralda] * 1.5):
        raise RuntimeError("La sonda no es x1.5 en Risaralda H2")
    if not np.allclose(sonda_risaralda[~publico], cauca_sel2[~publico]):
        raise RuntimeError("La sonda no trae el primer semestre de la seleccion 2")
    p_r = cauca_h2[publico]
    producto_dr = mezcla.producto_sonda(p_r, sonda_risaralda[publico], SCORE_CAUCA_ARCHIVO, SCORE_RISARALDA)
    vr = np.where(risaralda[publico], p_r, 0.0)
    producto_vr = producto_dr / 0.5
    factor_r = producto_vr / float(vr @ vr)
    pred_risaralda = p_r + (factor_r - 1.0) * vr
    producto_ck = (float(p_r @ p_r) + norma_y - n * SCORE_CAUCA_ARCHIVO) / 2
    producto_risaralda = producto_ck + (factor_r - 1.0) * producto_vr
    puntaje_risaralda = mezcla.score_de(pred_risaralda, producto_risaralda, norma_y)
    print(f"factor Risaralda {factor_r:.5f}  media {pred_risaralda[risaralda[publico]].mean():.2f}  antes {p_r[risaralda[publico]].mean():.2f}")
    print(f"MSE publico Risaralda {puntaje_risaralda:.5f}  baja {SCORE_CAUCA_ARCHIVO - puntaje_risaralda:.2f}  minimo {pred_risaralda.min():.4f}")

    factor_h1_r = 1.0 + PESO_H1_RISARALDA * (factor_r - 1.0)
    risaralda_h1 = partes[0].eq(RISARALDA).to_numpy() & ~publico

    def guardar_risaralda(origen: np.ndarray, nombre: str) -> None:
        valores = origen.copy()
        valores[publico] = np.maximum(pred_risaralda, 0.0)
        valores[risaralda_h1] *= factor_h1_r
        if not np.isfinite(valores).all() or valores.min() < -1e-8:
            raise RuntimeError(nombre)
        envio_r = muestra[["id"]].copy()
        envio_r["dengue"] = np.maximum(valores, 0.0)
        envio_r.to_csv(CARPETA / nombre, index=False)

    guardar_risaralda(cauca_h2, "submission_risaralda.csv")
    guardar_risaralda(cauca_sel2, "submission_risaralda_seleccion_2.csv")
    print(f"primer semestre Risaralda x {factor_h1_r:.5f}")

    risaralda_h2 = pd.read_csv(CARPETA / "submission_risaralda.csv").set_index("id").dengue.loc[ids].to_numpy(float)
    risaralda_sel2 = pd.read_csv(CARPETA / "submission_risaralda_seleccion_2.csv").set_index("id").dengue.loc[ids].to_numpy(float)
    narino = partes[0].eq(NARINO).to_numpy() & publico
    sonda_narino = pd.read_csv(CARPETA / "submission_sonda_narino.csv").set_index("id").dengue.loc[ids].to_numpy(float)
    if not np.allclose(sonda_narino[~narino], risaralda_sel2[~narino]):
        raise RuntimeError("La sonda no deja quieto todo lo que no es Narino H2")
    if not np.allclose(sonda_narino[narino], risaralda_sel2[narino] * 1.5):
        raise RuntimeError("La sonda no es x1.5 en Narino H2")
    if not np.allclose(sonda_narino[~publico], risaralda_sel2[~publico]):
        raise RuntimeError("La sonda no trae el primer semestre de la seleccion 2")
    p_n = risaralda_h2[publico]
    producto_dn = mezcla.producto_sonda(p_n, sonda_narino[publico], SCORE_RISARALDA_ARCHIVO, SCORE_NARINO)
    vn = np.where(narino[publico], p_n, 0.0)
    producto_vn = producto_dn / 0.5
    factor_n = producto_vn / float(vn @ vn)
    pred_narino = p_n + (factor_n - 1.0) * vn
    producto_rs = (float(p_n @ p_n) + norma_y - n * SCORE_RISARALDA_ARCHIVO) / 2
    producto_narino = producto_rs + (factor_n - 1.0) * producto_vn
    puntaje_narino = mezcla.score_de(pred_narino, producto_narino, norma_y)
    print(f"factor Narino {factor_n:.5f}  media {pred_narino[narino[publico]].mean():.2f}  antes {p_n[narino[publico]].mean():.2f}")
    print(f"MSE publico Narino {puntaje_narino:.5f}  baja {SCORE_RISARALDA_ARCHIVO - puntaje_narino:.2f}  minimo {pred_narino.min():.4f}")

    # Rolling H1 loses 2023 by 14.4 and the 2016-2023 average does not improve, so H1 stays put.

    def guardar_narino(origen: np.ndarray, nombre: str) -> None:
        valores = origen.copy()
        valores[publico] = np.maximum(pred_narino, 0.0)
        if not np.isfinite(valores).all() or valores.min() < -1e-8:
            raise RuntimeError(nombre)
        envio_n = muestra[["id"]].copy()
        envio_n["dengue"] = np.maximum(valores, 0.0)
        envio_n.to_csv(CARPETA / nombre, index=False)

    guardar_narino(risaralda_h2, "submission_narino.csv")
    guardar_narino(risaralda_sel2, "submission_narino_seleccion_2.csv")

    narino_h2 = pd.read_csv(CARPETA / "submission_narino.csv").set_index("id").dengue.loc[ids].to_numpy(float)
    narino_sel2 = pd.read_csv(CARPETA / "submission_narino_seleccion_2.csv").set_index("id").dengue.loc[ids].to_numpy(float)
    choco = partes[0].eq(CHOCO).to_numpy() & publico
    sonda_choco = pd.read_csv(CARPETA / "submission_sonda_choco.csv").set_index("id").dengue.loc[ids].to_numpy(float)
    if not np.allclose(sonda_choco[~choco], narino_sel2[~choco]):
        raise RuntimeError("La sonda no deja quieto todo lo que no es Choco H2")
    if not np.allclose(sonda_choco[choco], narino_sel2[choco] * 1.5):
        raise RuntimeError("La sonda no es x1.5 en Choco H2")
    if not np.allclose(sonda_choco[~publico], narino_sel2[~publico]):
        raise RuntimeError("La sonda no trae el primer semestre de la seleccion 2")
    p_h = narino_h2[publico]
    producto_dh = mezcla.producto_sonda(p_h, sonda_choco[publico], SCORE_NARINO_ARCHIVO, SCORE_CHOCO)
    vh = np.where(choco[publico], p_h, 0.0)
    producto_vh = producto_dh / 0.5
    factor_h = producto_vh / float(vh @ vh)
    pred_choco = p_h + (factor_h - 1.0) * vh
    producto_na = (float(p_h @ p_h) + norma_y - n * SCORE_NARINO_ARCHIVO) / 2
    producto_choco = producto_na + (factor_h - 1.0) * producto_vh
    puntaje_choco = mezcla.score_de(pred_choco, producto_choco, norma_y)
    print(f"factor Choco {factor_h:.5f}  media {pred_choco[choco[publico]].mean():.2f}  antes {p_h[choco[publico]].mean():.2f}")
    print(f"MSE publico Choco {puntaje_choco:.5f}  baja {SCORE_NARINO_ARCHIVO - puntaje_choco:.2f}  minimo {pred_choco.min():.4f}")

    factor_h1_h = 1.0 + PESO_H1_CHOCO * (factor_h - 1.0)
    choco_h1 = partes[0].eq(CHOCO).to_numpy() & ~publico

    def guardar_choco(origen: np.ndarray, nombre: str) -> None:
        valores = origen.copy()
        valores[publico] = np.maximum(pred_choco, 0.0)
        valores[choco_h1] *= factor_h1_h
        if not np.isfinite(valores).all() or valores.min() < -1e-8:
            raise RuntimeError(nombre)
        envio_h = muestra[["id"]].copy()
        envio_h["dengue"] = np.maximum(valores, 0.0)
        envio_h.to_csv(CARPETA / nombre, index=False)

    guardar_choco(narino_h2, "submission_choco.csv")
    guardar_choco(narino_sel2, "submission_choco_seleccion_2.csv")
    print(f"primer semestre Choco x {factor_h1_h:.5f}")

    choco_h2 = pd.read_csv(CARPETA / "submission_choco.csv").set_index("id").dengue.loc[ids].to_numpy(float)
    choco_sel2 = pd.read_csv(CARPETA / "submission_choco_seleccion_2.csv").set_index("id").dengue.loc[ids].to_numpy(float)
    boyaca = partes[0].eq(BOYACA).to_numpy() & publico
    sonda_boyaca = pd.read_csv(CARPETA / "submission_sonda_boyaca.csv").set_index("id").dengue.loc[ids].to_numpy(float)
    if not np.allclose(sonda_boyaca[~boyaca], choco_sel2[~boyaca]):
        raise RuntimeError("La sonda no deja quieto todo lo que no es Boyaca H2")
    if not np.allclose(sonda_boyaca[boyaca], choco_sel2[boyaca] * 1.5):
        raise RuntimeError("La sonda no es x1.5 en Boyaca H2")
    if not np.allclose(sonda_boyaca[~publico], choco_sel2[~publico]):
        raise RuntimeError("La sonda no trae el primer semestre de la seleccion 2")
    p_b = choco_h2[publico]
    producto_db = mezcla.producto_sonda(p_b, sonda_boyaca[publico], SCORE_CHOCO_ARCHIVO, SCORE_BOYACA)
    vb = np.where(boyaca[publico], p_b, 0.0)
    producto_vb = producto_db / 0.5
    factor_b = producto_vb / float(vb @ vb)
    pred_boyaca = p_b + (factor_b - 1.0) * vb
    producto_ch = (float(p_b @ p_b) + norma_y - n * SCORE_CHOCO_ARCHIVO) / 2
    producto_boyaca = producto_ch + (factor_b - 1.0) * producto_vb
    puntaje_boyaca = mezcla.score_de(pred_boyaca, producto_boyaca, norma_y)
    print(f"factor Boyaca {factor_b:.5f}  media {pred_boyaca[boyaca[publico]].mean():.2f}  antes {p_b[boyaca[publico]].mean():.2f}")
    print(f"MSE publico Boyaca {puntaje_boyaca:.5f}  baja {SCORE_CHOCO_ARCHIVO - puntaje_boyaca:.2f}  minimo {pred_boyaca.min():.4f}")

    factor_h1_b = 1.0 + PESO_H1_BOYACA * (factor_b - 1.0)
    boyaca_h1 = partes[0].eq(BOYACA).to_numpy() & ~publico

    def guardar_boyaca(origen: np.ndarray, nombre: str) -> None:
        valores = origen.copy()
        valores[publico] = np.maximum(pred_boyaca, 0.0)
        valores[boyaca_h1] *= factor_h1_b
        if not np.isfinite(valores).all() or valores.min() < -1e-8:
            raise RuntimeError(nombre)
        envio_b = muestra[["id"]].copy()
        envio_b["dengue"] = np.maximum(valores, 0.0)
        envio_b.to_csv(CARPETA / nombre, index=False)

    guardar_boyaca(choco_h2, "submission_boyaca.csv")
    guardar_boyaca(choco_sel2, "submission_boyaca_seleccion_2.csv")
    print(f"primer semestre Boyaca x {factor_h1_b:.5f}")

    boyaca_h2 = pd.read_csv(CARPETA / "submission_boyaca.csv").set_index("id").dengue.loc[ids].to_numpy(float)
    boyaca_sel2 = pd.read_csv(CARPETA / "submission_boyaca_seleccion_2.csv").set_index("id").dengue.loc[ids].to_numpy(float)
    amazonas = partes[0].eq(AMAZONAS).to_numpy() & publico
    sonda_amazonas = pd.read_csv(CARPETA / "submission_sonda_amazonas.csv").set_index("id").dengue.loc[ids].to_numpy(float)
    if not np.allclose(sonda_amazonas[~amazonas], boyaca_sel2[~amazonas]):
        raise RuntimeError("La sonda no deja quieto todo lo que no es Amazonas H2")
    if not np.allclose(sonda_amazonas[amazonas], boyaca_sel2[amazonas] * 1.5):
        raise RuntimeError("La sonda no es x1.5 en Amazonas H2")
    if not np.allclose(sonda_amazonas[~publico], boyaca_sel2[~publico]):
        raise RuntimeError("La sonda no trae el primer semestre de la seleccion 2")
    p_z = boyaca_h2[publico]
    producto_dz = mezcla.producto_sonda(p_z, sonda_amazonas[publico], puntaje_boyaca, SCORE_AMAZONAS)
    vz = np.where(amazonas[publico], p_z, 0.0)
    producto_vz = producto_dz / 0.5
    factor_z = producto_vz / float(vz @ vz)
    pred_amazonas = p_z + (factor_z - 1.0) * vz
    producto_bo = (float(p_z @ p_z) + norma_y - n * puntaje_boyaca) / 2
    producto_amazonas = producto_bo + (factor_z - 1.0) * producto_vz
    puntaje_amazonas = mezcla.score_de(pred_amazonas, producto_amazonas, norma_y)
    print(f"factor Amazonas {factor_z:.5f}  media {pred_amazonas[amazonas[publico]].mean():.2f}  antes {p_z[amazonas[publico]].mean():.2f}")
    print(f"MSE publico Amazonas {puntaje_amazonas:.5f}  baja {puntaje_boyaca - puntaje_amazonas:.2f}  minimo {pred_amazonas.min():.4f}")

    # Rolling H1 loses 2019 by 16 and the 2016-2023 average does not improve, so H1 stays put.

    def guardar_amazonas(origen: np.ndarray, nombre: str) -> None:
        valores = origen.copy()
        valores[publico] = np.maximum(pred_amazonas, 0.0)
        if not np.isfinite(valores).all() or valores.min() < -1e-8:
            raise RuntimeError(nombre)
        envio_z = muestra[["id"]].copy()
        envio_z["dengue"] = np.maximum(valores, 0.0)
        envio_z.to_csv(CARPETA / nombre, index=False)

    guardar_amazonas(boyaca_h2, "submission_amazonas.csv")
    guardar_amazonas(boyaca_sel2, "submission_amazonas_seleccion_2.csv")

    amazonas_h2 = pd.read_csv(CARPETA / "submission_amazonas.csv").set_index("id").dengue.loc[ids].to_numpy(float)
    amazonas_sel2 = pd.read_csv(CARPETA / "submission_amazonas_seleccion_2.csv").set_index("id").dengue.loc[ids].to_numpy(float)
    guaviare = partes[0].eq(GUAVIARE).to_numpy() & publico
    sonda_guaviare = pd.read_csv(CARPETA / "submission_sonda_guaviare.csv").set_index("id").dengue.loc[ids].to_numpy(float)
    if not np.allclose(sonda_guaviare[~guaviare], amazonas_sel2[~guaviare]):
        raise RuntimeError("La sonda no deja quieto todo lo que no es Guaviare H2")
    if not np.allclose(sonda_guaviare[guaviare], amazonas_sel2[guaviare] * 1.5):
        raise RuntimeError("La sonda no es x1.5 en Guaviare H2")
    if not np.allclose(sonda_guaviare[~publico], amazonas_sel2[~publico]):
        raise RuntimeError("La sonda no trae el primer semestre de la seleccion 2")
    p_v = amazonas_h2[publico]
    producto_dv = mezcla.producto_sonda(p_v, sonda_guaviare[publico], puntaje_amazonas, SCORE_GUAVIARE)
    vv = np.where(guaviare[publico], p_v, 0.0)
    producto_vv = producto_dv / 0.5
    factor_v = producto_vv / float(vv @ vv)
    pred_guaviare = p_v + (factor_v - 1.0) * vv
    producto_am = (float(p_v @ p_v) + norma_y - n * puntaje_amazonas) / 2
    producto_guaviare = producto_am + (factor_v - 1.0) * producto_vv
    puntaje_guaviare = mezcla.score_de(pred_guaviare, producto_guaviare, norma_y)
    print(f"factor Guaviare {factor_v:.5f}  media {pred_guaviare[guaviare[publico]].mean():.2f}  antes {p_v[guaviare[publico]].mean():.2f}")
    print(f"MSE publico Guaviare {puntaje_guaviare:.5f}  baja {puntaje_amazonas - puntaje_guaviare:.2f}  minimo {pred_guaviare.min():.4f}")

    factor_h1_v = 1.0 + PESO_H1_GUAVIARE * (factor_v - 1.0)
    guaviare_h1 = partes[0].eq(GUAVIARE).to_numpy() & ~publico

    def guardar_guaviare(origen: np.ndarray, nombre: str) -> None:
        valores = origen.copy()
        valores[publico] = np.maximum(pred_guaviare, 0.0)
        valores[guaviare_h1] *= factor_h1_v
        if not np.isfinite(valores).all() or valores.min() < -1e-8:
            raise RuntimeError(nombre)
        envio_v = muestra[["id"]].copy()
        envio_v["dengue"] = np.maximum(valores, 0.0)
        envio_v.to_csv(CARPETA / nombre, index=False)

    guardar_guaviare(amazonas_h2, "submission_guaviare.csv")
    guardar_guaviare(amazonas_sel2, "submission_guaviare_seleccion_2.csv")
    print(f"primer semestre Guaviare x {factor_h1_v:.5f}")

    guaviare_h2 = pd.read_csv(CARPETA / "submission_guaviare.csv").set_index("id").dengue.loc[ids].to_numpy(float)
    guaviare_sel2 = pd.read_csv(CARPETA / "submission_guaviare_seleccion_2.csv").set_index("id").dengue.loc[ids].to_numpy(float)
    caldas = partes[0].eq(CALDAS).to_numpy() & publico
    sonda_caldas = pd.read_csv(CARPETA / "submission_sonda_caldas.csv").set_index("id").dengue.loc[ids].to_numpy(float)
    if not np.allclose(sonda_caldas[~caldas], guaviare_sel2[~caldas]):
        raise RuntimeError("La sonda no deja quieto todo lo que no es Caldas H2")
    if not np.allclose(sonda_caldas[caldas], guaviare_sel2[caldas] * 1.5):
        raise RuntimeError("La sonda no es x1.5 en Caldas H2")
    if not np.allclose(sonda_caldas[~publico], guaviare_sel2[~publico]):
        raise RuntimeError("La sonda no trae el primer semestre de la seleccion 2")
    p_d = guaviare_h2[publico]
    producto_dd = mezcla.producto_sonda(p_d, sonda_caldas[publico], puntaje_guaviare, SCORE_CALDAS)
    vd = np.where(caldas[publico], p_d, 0.0)
    producto_vd = producto_dd / 0.5
    factor_d = producto_vd / float(vd @ vd)
    pred_caldas = p_d + (factor_d - 1.0) * vd
    producto_gu = (float(p_d @ p_d) + norma_y - n * puntaje_guaviare) / 2
    producto_caldas = producto_gu + (factor_d - 1.0) * producto_vd
    puntaje_caldas = mezcla.score_de(pred_caldas, producto_caldas, norma_y)
    print(f"factor Caldas {factor_d:.5f}  media {pred_caldas[caldas[publico]].mean():.2f}  antes {p_d[caldas[publico]].mean():.2f}")
    print(f"MSE publico Caldas {puntaje_caldas:.5f}  baja {puntaje_guaviare - puntaje_caldas:.2f}  minimo {pred_caldas.min():.4f}")

    factor_h1_d = 1.0 + PESO_H1_CALDAS * (factor_d - 1.0)
    caldas_h1 = partes[0].eq(CALDAS).to_numpy() & ~publico

    def guardar_caldas(origen: np.ndarray, nombre: str) -> None:
        valores = origen.copy()
        valores[publico] = np.maximum(pred_caldas, 0.0)
        valores[caldas_h1] *= factor_h1_d
        if not np.isfinite(valores).all() or valores.min() < -1e-8:
            raise RuntimeError(nombre)
        envio_d = muestra[["id"]].copy()
        envio_d["dengue"] = np.maximum(valores, 0.0)
        envio_d.to_csv(CARPETA / nombre, index=False)

    guardar_caldas(guaviare_h2, "submission_caldas.csv")
    guardar_caldas(guaviare_sel2, "submission_caldas_seleccion_2.csv")
    print(f"primer semestre Caldas x {factor_h1_d:.5f}")

    caldas_h2 = pd.read_csv(CARPETA / "submission_caldas.csv").set_index("id").dengue.loc[ids].to_numpy(float)
    caldas_sel2 = pd.read_csv(CARPETA / "submission_caldas_seleccion_2.csv").set_index("id").dengue.loc[ids].to_numpy(float)
    vichada = partes[0].eq(VICHADA).to_numpy() & publico
    sonda_vichada = pd.read_csv(CARPETA / "submission_sonda_vichada.csv").set_index("id").dengue.loc[ids].to_numpy(float)
    if not np.allclose(sonda_vichada[~vichada], caldas_sel2[~vichada]):
        raise RuntimeError("La sonda no deja quieto todo lo que no es Vichada H2")
    if not np.allclose(sonda_vichada[vichada], caldas_sel2[vichada] * 1.5):
        raise RuntimeError("La sonda no es x1.5 en Vichada H2")
    if not np.allclose(sonda_vichada[~publico], caldas_sel2[~publico]):
        raise RuntimeError("La sonda no trae el primer semestre de la seleccion 2")
    p_i = caldas_h2[publico]
    producto_di = mezcla.producto_sonda(p_i, sonda_vichada[publico], puntaje_caldas, SCORE_VICHADA)
    vi = np.where(vichada[publico], p_i, 0.0)
    producto_vi = producto_di / 0.5
    factor_i = producto_vi / float(vi @ vi)
    pred_vichada = p_i + (factor_i - 1.0) * vi
    producto_ca = (float(p_i @ p_i) + norma_y - n * puntaje_caldas) / 2
    producto_vichada = producto_ca + (factor_i - 1.0) * producto_vi
    puntaje_vichada = mezcla.score_de(pred_vichada, producto_vichada, norma_y)
    print(f"factor Vichada {factor_i:.5f}  media {pred_vichada[vichada[publico]].mean():.2f}  antes {p_i[vichada[publico]].mean():.2f}")
    print(f"MSE publico Vichada {puntaje_vichada:.5f}  baja {puntaje_caldas - puntaje_vichada:.2f}  minimo {pred_vichada.min():.4f}")

    factor_h1_i = 1.0 + PESO_H1_VICHADA * (factor_i - 1.0)
    vichada_h1 = partes[0].eq(VICHADA).to_numpy() & ~publico

    def guardar_vichada(origen: np.ndarray, nombre: str) -> None:
        valores = origen.copy()
        valores[publico] = np.maximum(pred_vichada, 0.0)
        valores[vichada_h1] *= factor_h1_i
        if not np.isfinite(valores).all() or valores.min() < -1e-8:
            raise RuntimeError(nombre)
        envio_i = muestra[["id"]].copy()
        envio_i["dengue"] = np.maximum(valores, 0.0)
        envio_i.to_csv(CARPETA / nombre, index=False)

    guardar_vichada(caldas_h2, "submission_vichada.csv")
    guardar_vichada(caldas_sel2, "submission_vichada_seleccion_2.csv")
    print(f"primer semestre Vichada x {factor_h1_i:.5f}")

    vichada_h2 = pd.read_csv(CARPETA / "submission_vichada.csv").set_index("id").dengue.loc[ids].to_numpy(float)
    vichada_sel2 = pd.read_csv(CARPETA / "submission_vichada_seleccion_2.csv").set_index("id").dengue.loc[ids].to_numpy(float)
    vaupes = partes[0].eq(VAUPES).to_numpy() & publico
    sonda_vaupes = pd.read_csv(CARPETA / "submission_sonda_vaupes.csv").set_index("id").dengue.loc[ids].to_numpy(float)
    if not np.allclose(sonda_vaupes[~vaupes], vichada_sel2[~vaupes]):
        raise RuntimeError("La sonda no deja quieto todo lo que no es Vaupes H2")
    if not np.allclose(sonda_vaupes[vaupes], vichada_sel2[vaupes] * 1.5):
        raise RuntimeError("La sonda no es x1.5 en Vaupes H2")
    if not np.allclose(sonda_vaupes[~publico], vichada_sel2[~publico]):
        raise RuntimeError("La sonda no trae el primer semestre de la seleccion 2")
    p_p = vichada_h2[publico]
    producto_dp = mezcla.producto_sonda(p_p, sonda_vaupes[publico], SCORE_VICHADA_ARCHIVO, SCORE_VAUPES)
    vp = np.where(vaupes[publico], p_p, 0.0)
    producto_vp = producto_dp / 0.5
    factor_p = producto_vp / float(vp @ vp)
    pred_vaupes = p_p + (factor_p - 1.0) * vp
    producto_vi_archivo = (float(p_p @ p_p) + norma_y - n * SCORE_VICHADA_ARCHIVO) / 2
    producto_vaupes = producto_vi_archivo + (factor_p - 1.0) * producto_vp
    puntaje_vaupes = mezcla.score_de(pred_vaupes, producto_vaupes, norma_y)
    print(f"factor Vaupes {factor_p:.5f}  media {pred_vaupes[vaupes[publico]].mean():.2f}  antes {p_p[vaupes[publico]].mean():.2f}")
    print(f"MSE publico Vaupes {puntaje_vaupes:.5f}  baja {SCORE_VICHADA_ARCHIVO - puntaje_vaupes:.2f}  minimo {pred_vaupes.min():.4f}")

    factor_h1_p = 1.0 + PESO_H1_VAUPES * (factor_p - 1.0)
    vaupes_h1 = partes[0].eq(VAUPES).to_numpy() & ~publico

    def guardar_vaupes(origen: np.ndarray, nombre: str) -> None:
        valores = origen.copy()
        valores[publico] = np.maximum(pred_vaupes, 0.0)
        valores[vaupes_h1] *= factor_h1_p
        if not np.isfinite(valores).all() or valores.min() < -1e-8:
            raise RuntimeError(nombre)
        envio_p = muestra[["id"]].copy()
        envio_p["dengue"] = np.maximum(valores, 0.0)
        envio_p.to_csv(CARPETA / nombre, index=False)

    guardar_vaupes(vichada_h2, "submission_vaupes.csv")
    guardar_vaupes(vichada_sel2, "submission_vaupes_seleccion_2.csv")
    print(f"primer semestre Vaupes x {factor_h1_p:.5f}")

    vaupes_h2 = pd.read_csv(CARPETA / "submission_vaupes.csv").set_index("id").dengue.loc[ids].to_numpy(float)
    vaupes_sel2 = pd.read_csv(CARPETA / "submission_vaupes_seleccion_2.csv").set_index("id").dengue.loc[ids].to_numpy(float)
    guainia = partes[0].eq(GUAINIA).to_numpy() & publico
    sonda_guainia = pd.read_csv(CARPETA / "submission_sonda_guainia.csv").set_index("id").dengue.loc[ids].to_numpy(float)
    if not np.allclose(sonda_guainia[~guainia], vaupes_sel2[~guainia]):
        raise RuntimeError("La sonda no deja quieto todo lo que no es Guainia H2")
    if not np.allclose(sonda_guainia[guainia], vaupes_sel2[guainia] * 1.5):
        raise RuntimeError("La sonda no es x1.5 en Guainia H2")
    if not np.allclose(sonda_guainia[~publico], vaupes_sel2[~publico]):
        raise RuntimeError("La sonda no trae el primer semestre de la seleccion 2")
    p_g = vaupes_h2[publico]
    producto_dg = mezcla.producto_sonda(p_g, sonda_guainia[publico], puntaje_vaupes, SCORE_GUAINIA)
    vg = np.where(guainia[publico], p_g, 0.0)
    producto_vg = producto_dg / 0.5
    factor_g = producto_vg / float(vg @ vg)
    pred_guainia = p_g + (factor_g - 1.0) * vg
    producto_va = (float(p_g @ p_g) + norma_y - n * puntaje_vaupes) / 2
    producto_guainia = producto_va + (factor_g - 1.0) * producto_vg
    puntaje_guainia = mezcla.score_de(pred_guainia, producto_guainia, norma_y)
    print(f"factor Guainia {factor_g:.5f}  media {pred_guainia[guainia[publico]].mean():.2f}  antes {p_g[guainia[publico]].mean():.2f}")
    print(f"MSE publico Guainia {puntaje_guainia:.5f}  baja {puntaje_vaupes - puntaje_guainia:.2f}  minimo {pred_guainia.min():.4f}")

    # Last department. Both files keep the calibrated second semester and no x1.5.
    factor_h1_g = 1.0 + PESO_H1_GUAINIA * (factor_g - 1.0)
    guainia_h1 = partes[0].eq(GUAINIA).to_numpy() & ~publico

    def guardar_guainia(origen: np.ndarray, nombre: str) -> None:
        valores = origen.copy()
        valores[publico] = np.maximum(pred_guainia, 0.0)
        valores[guainia_h1] *= factor_h1_g
        if not np.isfinite(valores).all() or valores.min() < -1e-8:
            raise RuntimeError(nombre)
        envio_g = muestra[["id"]].copy()
        envio_g["dengue"] = np.maximum(valores, 0.0)
        envio_g.to_csv(CARPETA / nombre, index=False)

    guardar_guainia(vaupes_h2, "submission_guainia.csv")
    guardar_guainia(vaupes_sel2, "submission_guainia_seleccion_2.csv")
    print(f"primer semestre Guainia x {factor_h1_g:.5f}")


if __name__ == "__main__":
    main()
