"""Walk-forward of the H1 level correction with the k the public probes can actually see.

The correction already used in the two selections pulls the end of H1 toward
k = <B, y> / ||B||^2 of H2. Those files used one k for the coast and one for
the interior. The Valle probe splits the interior. This script checks whether
that finer k lowers H1 error on 2014-2023 before any 2025 file is written.
"""

from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[3]
COSTA = {4, 5, 11, 13, 16, 19, 28}
VALLE = 30
INICIOS = (1, 4, 8, 13, 17)


def factor(semana: np.ndarray, k: np.ndarray, inicio: int, fuerza: float) -> np.ndarray:
    rampa = np.clip((semana - inicio) / (26 - inicio), 0, 1)
    return 1 + fuerza * (k - 1) * rampa


def ajustar(semana, pred, y, k) -> tuple[int, float]:
    mejor = None
    for inicio in INICIOS:
        rampa = np.clip((semana - inicio) / (26 - inicio), 0, 1)
        x = pred * (k - 1) * rampa
        denom = float(x @ x)
        fuerza = float(np.clip(x @ (y - pred) / denom, 0, 1.5)) if denom > 0 else 0.0
        mse = float(np.mean((pred + fuerza * x - y) ** 2))
        if mejor is None or mse < mejor[0]:
            mejor = (mse, inicio, fuerza)
    return mejor[1], mejor[2]


def k_de(grupo: np.ndarray, b: np.ndarray, y: np.ndarray) -> float:
    bb = float(b @ b)
    return float(b @ y / bb) if bb > 0 else 1.0


def asignar_k(depto: np.ndarray, anio: np.ndarray, tabla_h2: pd.DataFrame, modo: str) -> np.ndarray:
    claves = {}
    for (an, dep), g in tabla_h2.groupby(["anio", "id_departamento"]):
        if modo == "region":
            clave = (an, dep in COSTA)
        elif modo == "tres":
            clave = (an, "costa" if dep in COSTA else "valle" if dep == VALLE else "resto")
        else:
            clave = (an, dep)
        claves.setdefault(clave, []).append(g)
    mapa = {}
    for clave, partes in claves.items():
        bloque = pd.concat(partes)
        mapa[clave] = k_de(bloque.b.to_numpy(), bloque.b.to_numpy(), bloque.y.to_numpy())
    salida = np.empty(len(depto))
    for i, (an, dep) in enumerate(zip(anio, depto)):
        if modo == "region":
            clave = (an, dep in COSTA)
        elif modo == "tres":
            clave = (an, "costa" if dep in COSTA else "valle" if dep == VALLE else "resto")
        else:
            clave = (an, dep)
        salida[i] = mapa[clave]
    return salida


def evaluar(tabla: pd.DataFrame, h2_k: pd.DataFrame, modo: str, columna: str = "b") -> dict[str, float]:
    h1 = tabla[tabla.semana <= 26].dropna(subset=[columna, "y"]).copy()
    k_h1 = asignar_k(h1.id_departamento.to_numpy(), h1.anio.to_numpy(), h2_k, modo)
    pred = h1[columna].to_numpy(float)
    y = h1.y.to_numpy(float)
    semana = h1.semana.to_numpy(int)
    anios = h1.anio.to_numpy(int)
    corregido = np.empty_like(pred)
    params = {}
    for anio in list(range(2014, 2024)) + [2025]:
        previo = (anios >= 2012) & (anios < anio) & (anios != 2024)
        inicio, fuerza = ajustar(semana[previo], pred[previo], y[previo], k_h1[previo])
        params[anio] = (inicio, round(fuerza, 3))
        actual = anios == anio
        if actual.any() and anio <= 2023:
            corregido[actual] = pred[actual] * factor(semana[actual], k_h1[actual], inicio, fuerza)
    pesos = {"params_2025": params[2025]}
    for desde in (2014, 2016):
        m = (anios >= desde) & (anios <= 2023)
        pesos[str(desde)] = float(np.mean((corregido[m] - y[m]) ** 2))
        por_anio = []
        for anio in range(desde, 2024):
            a = anios == anio
            por_anio.append(float(np.mean((corregido[a] - y[a]) ** 2)))
        pesos[f"anios_{desde}"] = por_anio
    return pesos


def apilado(crudo: pd.DataFrame) -> pd.DataFrame:
    """Convex B/TCN/MLP mix by coast, weights from earlier H1 years, 2024 left out of the fit."""
    from scipy.optimize import minimize

    redes = pd.read_parquet(RAIZ / "ensayos/juan/2026-11-04-red-dinamica-h1/predicciones_final5.parquet")
    ancho = redes.pivot_table(index=["anio", "id_departamento", "semana"], columns="familia", values="pred").reset_index()
    ref = crudo[["anio", "id_departamento", "semana", "b", "y"]]
    ancho = ancho.merge(ref, on=["anio", "id_departamento", "semana"])
    h1 = ancho[ancho.semana <= 26].copy()
    h1["costa"] = h1.id_departamento.isin(COSTA)
    h1["apilado"] = np.nan
    for anio in range(2012, 2026):
        previo = h1[(h1.anio >= 2011) & (h1.anio < anio) & (h1.anio != 2024)].dropna(subset=["y"])
        for es_costa in (True, False):
            bloque = previo[previo.costa == es_costa]
            x = bloque[["b", "tcn", "mlp"]].to_numpy(float)
            y = bloque.y.to_numpy(float)

            def objetivo(w, x=x, y=y):
                return float(np.mean((x @ w - y) ** 2))

            ajuste = minimize(objetivo, np.full(3, 1 / 3), method="SLSQP", bounds=[(0, 1)] * 3,
                              constraints={"type": "eq", "fun": lambda w: w.sum() - 1})
            if anio == 2025:
                print("pesos 2025", "costa" if es_costa else "interior", np.round(ajuste.x, 3))
            m = (h1.anio == anio) & (h1.costa == es_costa)
            h1.loc[m, "apilado"] = h1.loc[m, ["b", "tcn", "mlp"]].to_numpy(float) @ ajuste.x
    return h1


def main() -> None:
    crudo = pd.read_parquet(RAIZ / "ensayos/juan/2026-11-04-red-dinamica-h1/b_walkforward.parquet")
    h2_k = crudo[crudo.semana >= 27]
    for desde in (2014, 2016):
        m = crudo.anio.between(desde, 2023) & crudo.semana.le(26)
        print("b sin nivel", desde, round(float(np.mean((crudo.loc[m, "b"] - crudo.loc[m, "y"]) ** 2)), 1))
    for modo in ("region", "tres", "depto"):
        r = evaluar(crudo, h2_k, modo)
        print("b", modo, "2014", round(r["2014"], 1), "2016", round(r["2016"], 1), "params", r["params_2025"])
    mezcla = apilado(crudo)
    for modo in ("region", "tres"):
        r = evaluar(mezcla, h2_k, modo, columna="apilado")
        print("apilado", modo, "2014", round(r["2014"], 1), "2016", round(r["2016"], 1), "params", r["params_2025"])
        regional = evaluar(mezcla, h2_k, "region", columna="apilado")
        if modo == "tres":
            print("apilado tres gana anios 2016", int(np.sum(np.array(r["anios_2016"]) < np.array(regional["anios_2016"]))), "de 8")


if __name__ == "__main__":
    main()
