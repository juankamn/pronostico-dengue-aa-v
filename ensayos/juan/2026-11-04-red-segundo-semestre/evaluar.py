"""Compara el segundo semestre de las redes de 52 semanas con B y arma el archivo de prueba para el publico.

El archivo de prueba conserva exacto el primer semestre de `entrega/submission_mezcla_region.csv` (ya subido,
1971,62176) y cambia solo las semanas 27-52: mitad red y mitad B, con el mismo patron de reporte semana a
semana que tenia el archivo subido. Asi la diferencia de puntaje publico mide solo la red.
"""

from pathlib import Path

import numpy as np
import pandas as pd

CARPETA = Path(__file__).resolve().parent
RAIZ = CARPETA.parents[2]
COSTA = [4, 5, 11, 13, 16, 19, 28]
EVAL = list(range(2014, 2024))


def peso(previo: pd.DataFrame) -> float:
    x = (previo.red - previo.b).to_numpy()
    r = (previo.y - previo.b).to_numpy()
    return float(np.clip(x @ r / (x @ x), 0, 1)) if x @ x > 0 else 0.0


def mse(g: pd.DataFrame, col: str) -> float:
    return float(np.mean((g[col] - g.y) ** 2))


def main() -> None:
    red = pd.read_parquet(CARPETA / "predicciones_52.parquet")
    B = pd.read_parquet(RAIZ / "ensayos/juan/2026-11-04-red-dinamica-h1/b_walkforward.parquet")
    W = red.pivot_table(index=["anio", "id_departamento", "semana"], columns="familia", values="pred").reset_index()
    W = W.merge(B[["anio", "id_departamento", "semana", "b", "y"]], on=["anio", "id_departamento", "semana"])
    W["red"] = 0.5 * W.tcn + 0.5 * W.mlp
    W["costa"] = W.id_departamento.isin(COSTA)
    W["mitad"] = np.where(W.semana <= 26, "H1", "H2")

    # pesos por region y semestre, solo con anos anteriores (sin 2024)
    W["region"] = np.nan
    pesos = {}
    for anio in range(2012, 2026):
        previo = W[(W.anio >= 2011) & (W.anio < anio) & (W.anio != 2024)]
        for mitad in ("H1", "H2"):
            for es_costa in (True, False):
                p = previo[(previo.mitad == mitad) & (previo.costa == es_costa)]
                w = peso(p)
                pesos[(anio, mitad, "Costa" if es_costa else "Interior")] = w
                m = (W.anio == anio) & (W.mitad == mitad) & (W.costa == es_costa)
                W.loc[m, "region"] = w * W.loc[m, "red"] + (1 - w) * W.loc[m, "b"]
    W["mitad_b"] = 0.5 * W.red + 0.5 * W.b

    filas = []
    for (anio, mitad), g in W[W.anio.isin(EVAL + [2024])].groupby(["anio", "mitad"]):
        filas.append({"anio": anio, "mitad": mitad, **{c: mse(g, c) for c in ("b", "tcn", "mlp", "red", "mitad_b", "region")}})
    R = pd.DataFrame(filas)
    for mitad in ("H1", "H2"):
        r = R[(R.mitad == mitad) & R.anio.isin(EVAL)].set_index("anio")
        print(f"\n== {mitad}: MSE medio 2014-2023 | 2016-2023")
        for c in ("b", "tcn", "mlp", "red", "mitad_b", "region"):
            print(f"  {c:8s} {r[c].mean():8.0f} {r.loc[2016:, c].mean():8.0f}   gana a B: {int((r[c] < r.b).sum())}/10")
        print("  2024:", {c: round(float(R[(R.mitad == mitad) & (R.anio == 2024)][c].iloc[0])) for c in ("b", "red", "region")})
    print("\nPesos 2025 (H1 y H2):", {k[1:]: round(v, 2) for k, v in pesos.items() if k[0] == 2025})

    # archivo de prueba: H1 del envio subido, H2 = mezcla por region con el patron del envio subido
    subido = pd.read_csv(RAIZ / "entrega/submission_mezcla_region.csv")
    partes = subido.id.str.split("_", expand=True).astype(int)
    subido["id_departamento"], subido["semana"] = partes[0], partes[2]
    actual = W[W.anio == 2025][["id_departamento", "semana", "b", "red", "mitad_b"]]
    t = subido.merge(actual, on=["id_departamento", "semana"], how="left")
    h2 = t.semana >= 27
    razon = np.where(t.b > 0.01, t.dengue / t.b.clip(lower=0.01), 1.0)
    nuevo = t.dengue.to_numpy().copy()
    # En el segundo semestre gana la mitad fija (4954 frente a 5164 de B); los pesos por region no ayudan.
    nuevo[h2] = (t.mitad_b * razon)[h2]
    final = subido[["id"]].copy()
    final["dengue"] = np.clip(nuevo, 0, None)
    muestra = pd.read_csv(RAIZ / "data/sample_submission.csv")
    assert final.id.equals(muestra.id) and final.dengue.notna().all() and len(final) == 1612
    assert np.allclose(final.dengue[~h2], subido.dengue[~h2])
    final.to_csv(CARPETA / "submission.csv", index=False)

    nombres = pd.read_parquet(RAIZ / "data/train.parquet").drop_duplicates("id_departamento").set_index("id_departamento").departamento
    resumen = t[h2].assign(nuevo=nuevo[h2]).groupby("id_departamento")[["dengue", "nuevo", "b", "red"]].mean()
    resumen["departamento"] = nombres
    print("\nSegundo semestre 2025 por departamento (subido, prueba, B sin patron, red sola)")
    print(resumen.sort_values("dengue", ascending=False).head(14).round(0).to_string())
    print("media H2 subido", round(float(t.dengue[h2].mean()), 2), "| prueba", round(float(nuevo[h2].mean()), 2), "| red sola", round(float(t.red[h2].mean()), 2))
    print("MSE entre el H2 subido y el de prueba:", round(float(np.mean((nuevo[h2] - t.dengue[h2]) ** 2)), 1))
    R.to_csv(CARPETA / "errores_por_anio.csv", index=False)


if __name__ == "__main__":
    main()
