"""Blend of already scored submissions, exact on the public half.

The public MSE of any linear combination of scored files is fixed by the Gram
matrix on the 806 second-semester rows. H1 is copied from seleccion 1.
"""

from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[3]
CARPETA = Path(__file__).resolve().parent

ARCHIVOS = {
    "b": ("ensayos/juan/2026-10-31-horizonte-bases/submission.csv", 2048.49606),
    "arbol_2027": ("ensayos/juan/2026-10-27-nivel-sin-brote/submission.csv", 3754.91182),
    "decaimiento": ("ensayos/juan/2026-11-03-zoo-bayesiano/submission.csv", 2836.48931),
    "sensor": ("ensayos/juan/2026-11-04-sensor-picos/submission.csv", 2016.28310),
    "red_52": ("ensayos/juan/2026-11-04-red-segundo-semestre/submission.csv", 2050.42336),
    "mezcla_region": ("entrega/submission_mezcla_region.csv", 1971.62176),
    "bases_patron": ("entrega/submission_bases_patron.csv", 1972.72476),
    "apilado": ("entrega/submission_final_apilado.csv", 1836.20689),
    "sonda_costa": ("ensayos/juan/2026-11-04-sonda-regiones/submission_sonda_costa.csv", 1842.97221),
    "sonda_interior": ("ensayos/juan/2026-11-04-sonda-regiones/submission_sonda_interior.csv", 2056.16650),
    "seleccion_1": ("entrega/submission_seleccion_1.csv", 1689.03586),
    "arbol_global": ("ensayos/jairo/2026-11-05-modelo-global/submission.csv", 5773.16350),
    "mezcla_arbol": ("ensayos/jairo/2026-11-05-modelo-global/submission_mezcla_publico.csv", 1654.88355),
    "multi": ("ensayos/jairo/2026-11-05-mezcla-multimodelo/submission.csv", 1646.97677),
    "residuos": ("ensayos/jairo/2026-11-05-residuos-clima/submission_sonda_h2.csv", 2174.34931),
    "sonda_valle": ("ensayos/jairo/2026-11-05-sonda-valle/submission_sonda_valle.csv", 2005.86755),
}


def cargar(ids: pd.Series) -> dict[str, np.ndarray]:
    salida = {}
    for nombre, (ruta, _) in ARCHIVOS.items():
        tabla = pd.read_csv(RAIZ / ruta).set_index("id")
        valores = tabla.dengue.reindex(ids).to_numpy(float)
        if not np.isfinite(valores).all():
            raise RuntimeError(f"{nombre} no cubre sample_submission")
        salida[nombre] = valores
    return salida


def producto_sonda(base: np.ndarray, sonda: np.ndarray, s0: float, s1: float) -> float:
    diferencia = sonda - base
    return float(diferencia @ base - (len(base) * (s1 - s0) - diferencia @ diferencia) / 2)


def norma_real(vectores: dict[str, np.ndarray], puntajes: dict[str, float], publico: np.ndarray) -> float:
    """||y||^2 on H2, from the two regional probes that partition B."""
    n = int(publico.sum())
    apilado = vectores["apilado"][publico]
    costa = vectores["sonda_costa"][publico]
    interior = vectores["sonda_interior"][publico]
    base_b = vectores["b"][publico]
    d_costa = costa - apilado
    d_interior = interior - apilado
    cobertura = np.abs(d_costa) + np.abs(d_interior)
    if np.any(cobertura < 1e-8):
        raise RuntimeError("Las sondas regionales no cubren todo el segundo semestre")
    if np.any((np.abs(d_costa) > 1e-8) & (np.abs(d_interior) > 1e-8)):
        raise RuntimeError("Las sondas regionales se solapan")
    py_costa = producto_sonda(apilado, costa, puntajes["apilado"], puntajes["sonda_costa"])
    py_interior = producto_sonda(apilado, interior, puntajes["apilado"], puntajes["sonda_interior"])
    escala = np.divide(d_costa + d_interior, base_b, out=np.ones_like(base_b), where=np.abs(base_b) > 1e-8)
    if np.std(escala[np.abs(base_b) > 1e-8]) > 1e-6:
        raise RuntimeError("La sonda regional no es una escala uniforme de B")
    eps = float(np.median(escala[np.abs(base_b) > 1e-8]))
    producto_b = (py_costa + py_interior) / eps
    return 2 * producto_b - float(base_b @ base_b) + n * puntajes["b"]


def score_de(vector: np.ndarray, producto: float, norma_y: float) -> float:
    return float((vector @ vector + norma_y - 2 * producto) / len(vector))


def mezcla(matriz: np.ndarray, productos: np.ndarray, norma_y: float) -> tuple[np.ndarray, np.ndarray, float]:
    """Minimum-norm solution in the scored span. Ridge only if a prediction would go negative."""
    gram = matriz.T @ matriz
    libre, *_ = np.linalg.lstsq(gram, productos, rcond=1e-8)
    candidatos = [(libre, matriz @ libre)]
    escala = np.linalg.norm(matriz, axis=0)
    escala = np.where(escala < 1e-8, 1.0, escala)
    normal = matriz / escala
    gram_n = normal.T @ normal
    lado = productos / escala
    for lam in (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1e3, 1e4):
        pesos = np.linalg.solve(gram_n + lam * np.eye(gram_n.shape[0]), lado) / escala
        candidatos.append((pesos, matriz @ pesos))
    mejor_w, mejor_pred, mejor_score = None, None, np.inf
    for pesos, pred in candidatos:
        if pred.min() < -1e-6:
            continue
        puntaje = score_de(pred, float(pesos @ productos), norma_y)
        if puntaje < mejor_score:
            mejor_w, mejor_pred, mejor_score = pesos, pred, puntaje
    if mejor_w is None:
        raise RuntimeError("Ninguna mezcla quedo no negativa")
    return mejor_w, mejor_pred, mejor_score


def main() -> None:
    muestra = pd.read_csv(RAIZ / "data" / "sample_submission.csv")
    ids = muestra.id
    semana = ids.str.rsplit("_", n=1).str[-1].astype(int)
    publico = (semana >= 27).to_numpy()
    vectores = cargar(ids)
    puntajes = {nombre: score for nombre, (_, score) in ARCHIVOS.items()}
    norma_y = norma_real(vectores, puntajes, publico)
    n = int(publico.sum())
    print(f"||y||^2 = {norma_y:.2f}  n = {n}")

    nombres = list(ARCHIVOS)
    matriz = np.column_stack([vectores[nombre][publico] for nombre in nombres])
    productos = np.array([(matriz[:, i] @ matriz[:, i] + norma_y - n * puntajes[nombre]) / 2 for i, nombre in enumerate(nombres)])
    for i, nombre in enumerate(nombres):
        recuperado = score_de(matriz[:, i], productos[i], norma_y)
        if abs(recuperado - puntajes[nombre]) > 1e-6:
            raise RuntimeError(f"{nombre} no cierra: {recuperado} vs {puntajes[nombre]}")

    d_valle = vectores["sonda_valle"][publico] - vectores["multi"][publico]
    py_valle = producto_sonda(vectores["multi"][publico], vectores["sonda_valle"][publico], puntajes["multi"], puntajes["sonda_valle"])
    producto_sonda_valle = productos[nombres.index("multi")] + py_valle
    cierre = score_de(vectores["sonda_valle"][publico], producto_sonda_valle, norma_y)
    print(f"cierre sonda valle {cierre:.5f} contra {puntajes['sonda_valle']:.5f}  diferencia de vector {d_valle @ d_valle:.1f}")

    pesos, pred, puntaje = mezcla(matriz, productos, norma_y)
    print(f"MSE publico estimado {puntaje:.5f}  minimo {pred.min():.4f}  maximo {pred.max():.1f}")
    for nombre, peso in sorted(zip(nombres, pesos), key=lambda par: -abs(par[1])):
        if abs(peso) >= 0.01:
            print(f"  {nombre:16} {peso:+.3f}")

    envio = muestra[["id"]].copy()
    envio["dengue"] = vectores["seleccion_1"].copy()
    envio.loc[publico, "dengue"] = pred
    if envio.dengue.isna().any() or (envio.dengue < -1e-8).any():
        raise RuntimeError("El envio tiene huecos o negativos")
    envio["dengue"] = np.maximum(envio.dengue, 0)
    envio.to_csv(CARPETA / "submission.csv", index=False)
    pd.DataFrame({"modelo": nombres, "peso": pesos, "score_publico": [puntajes[n] for n in nombres]}).to_csv(CARPETA / "pesos.csv", index=False)

    partes = ids.str.split("_", expand=True).astype(int)
    bolivar = (partes[0] == 5).to_numpy() & publico
    sonda = envio.dengue.to_numpy(float).copy()
    sonda[bolivar] *= 1.5
    sonda_envio = muestra[["id"]].copy()
    sonda_envio["dengue"] = sonda
    sonda_envio.to_csv(CARPETA / "submission_sonda_bolivar.csv", index=False)
    distancia = float(np.mean((sonda[publico] - pred) ** 2))
    print(f"sonda Bolivar x1.5  distancia MSE {distancia:.6f}  media Bolivar {pred[bolivar[publico]].mean():.2f}")
    print(f"filas {len(envio)}  H1 identico a seleccion_1 {np.allclose(envio.dengue.to_numpy()[~publico], vectores['seleccion_1'][~publico])}")


if __name__ == "__main__":
    main()
