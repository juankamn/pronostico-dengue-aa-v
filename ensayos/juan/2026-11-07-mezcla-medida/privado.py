"""H1 that the walk-forward actually improves, spliced onto the measured H2 blend.

On 2016-2023 the stacked model with the regional level correction sits near 3356.
Pulling that same stack toward a coast / Valle / rest scale makes it worse.
The mix that does improve is a weight, learned only on earlier years, between
that stack and horizon-B corrected with the three-way scale. H2 is copied
from submission.csv, so the public score stays the measured blend.
"""

from pathlib import Path

import numpy as np
import pandas as pd

import nivel_h1 as nivel

RAIZ = nivel.RAIZ
CARPETA = Path(__file__).resolve().parent
COSTA = nivel.COSTA
VALLE = nivel.VALLE


def k_grupos(depto: np.ndarray, b: np.ndarray, proyeccion: np.ndarray) -> dict[str, float]:
    grupos = {
        "costa": np.isin(depto, list(COSTA)),
        "valle": depto == VALLE,
        "resto": ~np.isin(depto, list(COSTA | {VALLE})),
    }
    salida = {}
    for nombre, mascara in grupos.items():
        bb = float(b[mascara] @ b[mascara])
        salida[nombre] = float(b[mascara] @ proyeccion[mascara] / bb) if bb > 0 else 1.0
    return salida


def corregir_b(crudo: pd.DataFrame, h2_k: pd.DataFrame) -> pd.DataFrame:
    h1 = crudo[crudo.semana <= 26].dropna(subset=["b", "y"]).copy()
    k = nivel.asignar_k(h1.id_departamento.to_numpy(), h1.anio.to_numpy(), h2_k, "tres")
    pred = h1.b.to_numpy(float)
    y = h1.y.to_numpy(float)
    semana = h1.semana.to_numpy(int)
    anios = h1.anio.to_numpy(int)
    out = pred.copy()
    for anio in range(2014, 2024):
        previo = (anios >= 2012) & (anios < anio)
        inicio, fuerza = nivel.ajustar(semana[previo], pred[previo], y[previo], k[previo])
        actual = anios == anio
        out[actual] = pred[actual] * nivel.factor(semana[actual], k[actual], inicio, fuerza)
    h1["pred"] = out
    return h1


def peso_mezcla(junto: pd.DataFrame, desde: int, hasta: int) -> float:
    bloque = junto[junto.anio.between(desde, hasta)]
    mejor = None
    for paso in range(0, 21):
        w = paso / 20
        err = float(np.mean((w * bloque.pred_a + (1 - w) * bloque.pred_b - bloque.y) ** 2))
        if mejor is None or err < mejor[0]:
            mejor = (err, w)
    return mejor[1]


def main() -> None:
    crudo = pd.read_parquet(RAIZ / "ensayos/juan/2026-11-04-red-dinamica-h1/b_walkforward.parquet")
    h2_k = crudo[crudo.semana >= 27]
    apilado = nivel.apilado(crudo)
    h1_a = apilado[apilado.semana <= 26].dropna(subset=["apilado", "y"]).copy()
    k_a = nivel.asignar_k(h1_a.id_departamento.to_numpy(), h1_a.anio.to_numpy(), h2_k, "region")
    pred = h1_a.apilado.to_numpy(float)
    y = h1_a.y.to_numpy(float)
    semana = h1_a.semana.to_numpy(int)
    anios = h1_a.anio.to_numpy(int)
    out = pred.copy()
    for anio in range(2014, 2024):
        previo = (anios >= 2012) & (anios < anio)
        inicio, fuerza = nivel.ajustar(semana[previo], pred[previo], y[previo], k_a[previo])
        actual = anios == anio
        out[actual] = pred[actual] * nivel.factor(semana[actual], k_a[actual], inicio, fuerza)
    h1_a["pred"] = out
    h1_b = corregir_b(crudo, h2_k)
    junto = h1_a.merge(
        h1_b[["anio", "id_departamento", "semana", "pred"]],
        on=["anio", "id_departamento", "semana"],
        suffixes=("_a", "_b"),
    )
    w_16 = peso_mezcla(junto, 2016, 2023)
    w_14 = peso_mezcla(junto, 2014, 2023)
    pred_wf = np.empty(len(junto))
    anios_j = junto.anio.to_numpy()
    for anio in range(2014, 2024):
        w_anio = peso_mezcla(junto, 2012, anio - 1)
        pred_wf[anios_j == anio] = (
            w_anio * junto.loc[junto.anio == anio, "pred_a"] + (1 - w_anio) * junto.loc[junto.anio == anio, "pred_b"]
        )
    junto = junto.copy()
    junto["pred_wf"] = pred_wf
    bloque = junto[junto.anio.between(2016, 2023)]
    print(f"peso del apilado en 2025 {w_14:.2f}  si solo se usan 2016-2023 {w_16:.2f}")
    print(
        "MSE H1 2016-2023",
        f"mezcla {np.mean((bloque.pred_wf - bloque.y) ** 2):.1f}",
        f"apilado {np.mean((bloque.pred_a - bloque.y) ** 2):.1f}",
        f"B con k fino {np.mean((bloque.pred_b - bloque.y) ** 2):.1f}",
    )
    params = nivel.evaluar(crudo, h2_k, "tres")
    inicio_b, fuerza_b = params["params_2025"]
    print(f"nivel de B para 2025 inicio {inicio_b} fuerza {fuerza_b}")

    muestra = pd.read_csv(RAIZ / "data/sample_submission.csv")
    ids = muestra.id
    partes = ids.str.split("_", expand=True).astype(int)
    depto = partes[0].to_numpy()
    semana_envio = partes[2].to_numpy()
    publico = semana_envio >= 27
    blend = pd.read_csv(CARPETA / "submission.csv").set_index("id").dengue.loc[ids].to_numpy(float)
    b_2025 = pd.read_csv(RAIZ / "ensayos/juan/2026-10-31-horizonte-bases/submission.csv").set_index("id").dengue.loc[ids].to_numpy(float)
    sel1 = pd.read_csv(RAIZ / "entrega/submission_seleccion_1.csv").set_index("id").dengue.loc[ids].to_numpy(float)
    sel2 = pd.read_csv(RAIZ / "entrega/submission_seleccion_2.csv").set_index("id").dengue.loc[ids].to_numpy(float)
    ks = k_grupos(depto[publico], b_2025[publico], blend[publico])
    print("k 2025", {nombre: round(valor, 4) for nombre, valor in ks.items()})
    k_fila = np.where(np.isin(depto, list(COSTA)), ks["costa"], np.where(depto == VALLE, ks["valle"], ks["resto"]))
    b_corr = b_2025 * nivel.factor(semana_envio, k_fila, int(inicio_b), float(fuerza_b))
    # Same rule as the walk-forward: weight fit on every earlier year, 2024 left out.
    h1_mix = w_14 * sel1 + (1 - w_14) * b_corr
    h1_mix = np.maximum(h1_mix, 0)

    def armar(h1: np.ndarray, nombre: str) -> None:
        envio = muestra[["id"]].copy()
        valores = blend.copy()
        valores[~publico] = h1[~publico]
        if not np.isfinite(valores).all() or valores.min() < -1e-8:
            raise RuntimeError(f"{nombre} tiene huecos o negativos")
        envio["dengue"] = np.maximum(valores, 0)
        envio.to_csv(CARPETA / nombre, index=False)

    armar(h1_mix, "submission_privado.csv")
    armar(sel2, "submission_seleccion_2.csv")
    h1 = ~publico
    valle_q1 = h1 & (depto == VALLE) & (semana_envio <= 13)
    print(f"Valle Q1  seleccion_1 {sel1[valle_q1].mean():.1f}  mezcla {h1_mix[valle_q1].mean():.1f}  seleccion_2 {sel2[valle_q1].mean():.1f}  B corregida {b_corr[valle_q1].mean():.1f}")
    print(f"media H1  seleccion_1 {sel1[h1].mean():.2f}  mezcla {h1_mix[h1].mean():.2f}  seleccion_2 {sel2[h1].mean():.2f}")
    guardado = pd.read_csv(CARPETA / "submission_privado.csv").set_index("id").dengue.loc[ids].to_numpy()
    print(f"H2 identico al blend {np.allclose(blend[publico], guardado[publico])}")


if __name__ == "__main__":
    main()
