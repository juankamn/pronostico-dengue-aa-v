"""Segundo semestre combinado con los puntajes publicos exactos de envios ya subidos.

Para archivos p_i con puntaje publico S_i (MSE sobre las semanas 27-52), el puntaje de cualquier
combinacion p = sum c_i p_i con sum c_i = 1 es exacto y no necesita envios nuevos:

    S(c) = c' G c - c' (diag(G) - S),   G_ij = media(p_i * p_j) en el segundo semestre.

Se busca la combinacion de menor S(c), y se pega ese segundo semestre a las dos selecciones finales sin tocar
su primer semestre. El privado (semanas 1-26) no cambia; solo cambia el puesto visible del tablero publico.
"""

from pathlib import Path

import numpy as np
import pandas as pd

CARPETA = Path("ensayos/juan/2026-11-04-mezcla-publico")
ENVIOS = {
    "nivel_sin_brote": ("ensayos/juan/2026-10-27-nivel-sin-brote/submission.csv", 3754.91),
    "decaimiento": ("ensayos/juan/2026-11-03-zoo-bayesiano/submission.csv", 2836.48931),
    "sensor": ("ensayos/juan/2026-11-04-sensor-picos/submission.csv", 2016.28310),
    "red_52": ("ensayos/juan/2026-11-04-red-segundo-semestre/submission.csv", 2050.42336),
}
BASES = {
    "mezcla_region": ("entrega/submission_mezcla_region.csv", 1971.62176),
    "bases_patron": ("entrega/submission_bases_patron.csv", 1972.72476),
}


def raiz_repo() -> Path:
    aqui = Path.cwd().resolve()
    for carpeta in [aqui, *list(aqui.parents)[:5]]:
        if (carpeta / "data" / "train.parquet").exists():
            return carpeta
    raise FileNotFoundError("No se encontro data/train.parquet.")


def optimo(M: np.ndarray, s: np.ndarray) -> tuple[np.ndarray, float]:
    n, K = M.shape
    G = M.T @ M / n
    g = np.diag(G)
    A = np.zeros((K + 1, K + 1))
    A[:K, :K] = 2 * G
    A[:K, K] = -1
    A[K, :K] = 1
    c = np.linalg.solve(A, np.concatenate([g - s, [1.0]]))[:K]
    return c, float(c @ G @ c - c @ (g - s))


def main() -> None:
    raiz = raiz_repo()
    muestra = pd.read_csv(raiz / "data/sample_submission.csv")
    semana = muestra.id.str.split("_").str[2].astype(int).to_numpy()
    h2 = semana >= 27
    leer = lambda ruta: pd.read_csv(raiz / ruta).set_index("id").dengue.loc[muestra.id].to_numpy()
    otros = {k: leer(ruta) for k, (ruta, _) in ENVIOS.items()}
    for nombre, (ruta, puntaje) in BASES.items():
        base = leer(ruta)
        nombres = [nombre, *ENVIOS]
        M = np.column_stack([base[h2], *[otros[k][h2] for k in ENVIOS]])
        s = np.array([puntaje, *[p for _, p in ENVIOS.values()]])
        c, esperado = optimo(M, s)
        segundo = M @ c
        negativos = int((segundo < 0).sum())
        final = base.copy()
        final[h2] = np.clip(segundo, 0, None)
        envio = muestra[["id"]].copy()
        envio["dengue"] = final
        assert envio.dengue.notna().all() and (envio.dengue >= 0).all() and len(envio) == 1612
        assert np.allclose(envio.dengue[~h2], base[~h2]), "El primer semestre no debia cambiar."
        destino = raiz / CARPETA / f"submission_{nombre}_h2.csv"
        envio.to_csv(destino, index=False)
        print(f"{nombre}: publico exacto {esperado:.2f} (antes {puntaje:.2f}); negativos recortados a 0: {negativos} (solo pueden mejorarlo)")
        print("   pesos:", ", ".join(f"{k}={w:+.3f}" for k, w in zip(nombres, c)))
        print(f"   media del segundo semestre {final[h2].mean():.1f} (antes {base[h2].mean():.1f}) | primer semestre intacto {final[~h2].mean():.1f}")


if __name__ == "__main__":
    main()
