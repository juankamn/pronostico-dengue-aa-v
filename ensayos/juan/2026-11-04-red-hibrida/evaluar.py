"""Compara la red hibrida con B y con la mezcla vigente, y arma el archivo de todo 2025.

Regla fijada antes de subir: el archivo reemplaza a la primera seleccion solo si (a) en el walk-forward
2016-2023 su primer semestre no es peor que el de la mezcla por region vigente y (b) su puntaje publico baja
de 1971,62. Si mejora el publico pero no el primer semestre, el primer semestre se queda como esta.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

CARPETA = Path(__file__).resolve().parent
RAIZ = CARPETA.parents[2]
RED_H1 = RAIZ / "ensayos/juan/2026-11-04-red-dinamica-h1"
COSTA = [4, 5, 11, 13, 16, 19, 28]
EVAL = list(range(2014, 2024))


def peso(previo, col):
    x = (previo[col] - previo.b).to_numpy()
    r = (previo.y - previo.b).to_numpy()
    return float(np.clip(x @ r / (x @ x), 0, 1)) if x @ x > 0 else 0.0


def region_wf(W, col, salida):
    """Mezcla con B usando pesos por region y semestre estimados solo con anos anteriores (sin 2024)."""
    W[salida] = np.nan
    pesos = {}
    for anio in range(2012, 2026):
        previo = W[(W.anio >= 2011) & (W.anio < anio) & (W.anio != 2024)]
        for mitad in ("H1", "H2"):
            for es_costa in (True, False):
                p = previo[(previo.mitad == mitad) & (previo.costa == es_costa)]
                w = peso(p, col)
                pesos[(anio, mitad, es_costa)] = w
                m = (W.anio == anio) & (W.mitad == mitad) & (W.costa == es_costa)
                W.loc[m, salida] = w * W.loc[m, col] + (1 - w) * W.loc[m, "b"]
    return pesos


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--etiqueta", default="hibrida")
    parser.add_argument("--tipo", default="tcn_lstm")
    args = parser.parse_args()

    B = pd.read_parquet(RED_H1 / "b_walkforward.parquet")[["anio", "id_departamento", "semana", "b", "y"]]
    hib = pd.read_parquet(CARPETA / f"predicciones_{args.etiqueta}.parquet")
    hib = hib[hib.familia == args.tipo].rename(columns={"pred": "hib"}).drop(columns="familia")
    vig = pd.read_parquet(RED_H1 / "predicciones_final5.parquet")
    vig = vig[vig.familia.isin(["tcn", "mlp"])].pivot_table(index=["anio", "id_departamento", "semana"], columns="familia", values="pred").reset_index()
    vig["red26"] = 0.5 * vig.tcn + 0.5 * vig.mlp

    W = hib.merge(B, on=["anio", "id_departamento", "semana"]).merge(vig[["anio", "id_departamento", "semana", "red26"]], on=["anio", "id_departamento", "semana"], how="left")
    W["costa"] = W.id_departamento.isin(COSTA)
    W["mitad"] = np.where(W.semana <= 26, "H1", "H2")
    W["hib_mitad"] = 0.5 * W.hib + 0.5 * W.b
    pesos_hib = region_wf(W, "hib", "hib_region")
    w26 = W[W.mitad == "H1"].copy()
    region_wf(w26, "red26", "vigente")  # mezcla por region vigente (redes de 26 semanas), solo primer semestre
    W = W.merge(w26[["anio", "id_departamento", "semana", "vigente"]], on=["anio", "id_departamento", "semana"], how="left")

    filas = []
    for (anio, mitad), g in W[W.anio.isin(EVAL + [2024])].groupby(["anio", "mitad"]):
        fila = {"anio": anio, "mitad": mitad}
        for c in ("b", "hib", "hib_mitad", "hib_region", "vigente"):
            if g[c].notna().all():
                fila[c] = float(np.mean((g[c] - g.y) ** 2))
        filas.append(fila)
    R = pd.DataFrame(filas)
    for mitad in ("H1", "H2"):
        r = R[(R.mitad == mitad) & R.anio.isin(EVAL)].set_index("anio")
        print(f"\n== {args.tipo} {mitad}: MSE medio 2014-2023 | 2016-2023 | gana a B")
        for c in [c for c in ("b", "hib", "hib_mitad", "hib_region", "vigente") if c in r]:
            print(f"  {c:11s} {r[c].mean():8.0f} {r.loc[2016:, c].mean():8.0f}   {int((r[c] < r.b).sum())}/10")
        r24 = R[(R.mitad == mitad) & (R.anio == 2024)]
        print("  2024:", {c: round(float(r24[c].iloc[0])) for c in ("b", "hib", "hib_mitad") if c in r24})
    print("\npesos 2025 de la hibrida (semestre, costa):", {(k[1], k[2]): round(v, 2) for k, v in pesos_hib.items() if k[0] == 2025})
    R.to_csv(CARPETA / f"errores_{args.etiqueta}_{args.tipo}.csv", index=False)

    # Archivo de todo 2025: mitad hibrida, mitad B en los dos semestres, con el patron del envio vigente
    subido = pd.read_csv(RAIZ / "entrega/submission_mezcla_region.csv")
    bases = pd.read_csv(RAIZ / "entrega/submission_bases_patron.csv").rename(columns={"dengue": "b_patron"})
    partes = subido.id.str.split("_", expand=True).astype(int)
    t = subido[["id"]].assign(id_departamento=partes[0], semana=partes[2]).merge(bases, on="id")
    actual = W[W.anio == 2025][["id_departamento", "semana", "b", "hib", "hib_mitad"]]
    t = t.merge(actual, on=["id_departamento", "semana"], how="left")
    razon = np.where(t.b > 0.01, t.b_patron / t.b.clip(lower=0.01), 1.0)
    final = t[["id"]].copy()
    final["dengue"] = np.clip(t.hib_mitad * razon, 0, None)
    muestra = pd.read_csv(RAIZ / "data/sample_submission.csv")
    assert final.id.equals(muestra.id) and final.dengue.notna().all() and len(final) == 1612
    final.to_csv(CARPETA / f"submission_{args.tipo}.csv", index=False)
    h1 = t.semana <= 26
    print(f"\nArchivo {args.tipo}: primer semestre {final.dengue[h1].mean():.1f} | segundo {final.dengue[~h1].mean():.1f}"
          f" | vigente: {subido.dengue[h1].mean():.1f} y {subido.dengue[~h1].mean():.1f}")
    print("MSE entre este archivo y la mezcla por region: H1", round(float(np.mean((final.dengue - subido.dengue)[h1] ** 2)), 1),
          "| H2", round(float(np.mean((final.dengue - subido.dengue)[~h1] ** 2)), 1))


if __name__ == "__main__":
    main()
