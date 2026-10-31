"""Pronostico de competencia por horizonte. Bases causales y residuo solo si gana el backtest."""

import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.optimize import minimize
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

CLIMA = [
    "precip_chirps_mm",
    "temp_media_C",
    "temp_min_C",
    "temp_max_C",
    "humedad_suelo_capa1",
    "evapotranspiracion_mm",
    "punto_rocio_C",
    "radiacion_solar_MJ_m2",
]
REZAGOS = ["precip_chirps_mm", "temp_min_C", "humedad_suelo_capa1", "punto_rocio_C"]
BASES = [
    "b_med_all",
    "b_mean_all",
    "b_rec6",
    "b_med5",
    "b_rec3",
    "b_end_direct",
    "b_end_scaled",
]
ANIOS_VALIDACION = list(range(2014, 2025))
ANIOS_MODERADOS = [2014, 2015, 2017, 2018, 2020, 2021, 2022]
ANIOS_POST = [2014, 2017, 2020, 2021]
ARBOL_H2 = {
    2014: 5431,
    2015: 4812,
    2016: 12220,
    2017: 702,
    2018: 1409,
    2019: 10172,
    2020: 4320,
    2021: 4231,
    2022: 2672,
    2023: 17407,
    2024: 75048,
}
CARPETA_ENSAYO = Path("ensayos/juan/2026-10-31-horizonte-bases")
ENVIO_ANTERIOR = Path("ensayos/juan/2026-10-27-nivel-sin-brote/submission.csv")


# %% datos
def raiz_repo() -> Path:
    aqui = Path.cwd().resolve()
    for carpeta in [aqui, *list(aqui.parents)[:5]]:
        if (carpeta / "data" / "train.parquet").exists():
            return carpeta
    return aqui


def carpeta_datos() -> Path:
    candidatos = [
        Path("/kaggle/input/competitions/aa-v-2026-ii-pronosticos-nn-rnn-lstm-tcn"),
        Path("/kaggle/input/aa-v-2026-ii-pronosticos-nn-rnn-lstm-tcn"),
        raiz_repo() / "data",
    ]
    for candidato in candidatos:
        if (candidato / "train.parquet").exists():
            return candidato
    raise FileNotFoundError("No se encontro train.parquet.")


def cargar():
    datos = carpeta_datos()
    train = pd.read_parquet(datos / "train.parquet")
    test = pd.read_parquet(datos / "test.parquet")
    muestra = pd.read_csv(datos / "sample_submission.csv")
    train = train.sort_values(["id_departamento", "anio", "semana"]).reset_index(drop=True)
    test = test.sort_values(["id_departamento", "anio", "semana"]).reset_index(drop=True)
    return train, test, muestra


# %% bases
def suavizar(serie: pd.Series) -> pd.Series:
    tabla = serie.unstack("semana")
    columnas = [semana for semana in range(1, 53) if semana in tabla.columns]
    tabla = tabla[columnas]
    extendida = pd.concat([tabla[columnas[-2:]], tabla, tabla[columnas[:2]]], axis=1)
    suave = extendida.T.rolling(5, center=True, min_periods=1).mean().T.iloc[:, 2:-2]
    suave.columns = pd.Index(columnas, name="semana")
    return suave.stack()


def marco(hist: pd.DataFrame, cur: pd.DataFrame) -> pd.DataFrame:
    anio = int(cur.anio.iloc[0])
    mediana_6 = hist[hist.anio >= anio - 6].groupby("id_departamento").casos_dengue.median()
    mediana_hist = hist.groupby("id_departamento").casos_dengue.median()
    referencia = np.maximum(mediana_6, mediana_hist).replace(0, 1.0)
    tope_bajo = hist.id_departamento.map(referencia * 2.0)
    tope_alto = hist.id_departamento.map(referencia * 2.8)
    historia_baja = hist.copy()
    historia_alta = hist.copy()
    historia_baja["casos_dengue"] = np.minimum(historia_baja.casos_dengue, tope_bajo)
    historia_alta["casos_dengue"] = np.minimum(historia_alta.casos_dengue, tope_alto)

    perfiles = {
        "b_med_all": suavizar(hist.groupby(["id_departamento", "semana"]).casos_dengue.median()),
        "b_mean_all": suavizar(historia_baja.groupby(["id_departamento", "semana"]).casos_dengue.mean()),
        "b_rec6": suavizar(
            historia_baja[historia_baja.anio >= anio - 6].groupby(["id_departamento", "semana"]).casos_dengue.mean()
        ),
        "b_med5": suavizar(
            hist[hist.anio >= anio - 5].groupby(["id_departamento", "semana"]).casos_dengue.median()
        ),
        "b_rec3": suavizar(
            historia_alta[historia_alta.anio >= anio - 3].groupby(["id_departamento", "semana"]).casos_dengue.mean()
        ),
    }

    previo = hist[hist.anio == anio - 1]
    cierre_2 = previo[previo.semana >= 51].groupby("id_departamento").casos_dengue.mean()
    cierre_6 = previo[previo.semana >= 47].groupby("id_departamento").casos_dengue.mean()
    cierre_12 = previo[previo.semana >= 41].groupby("id_departamento").casos_dengue.mean()
    cierre = 0.50 * cierre_2 + 0.35 * cierre_6 + 0.15 * cierre_12
    mitad_1 = previo[previo.semana <= 26].groupby("id_departamento").casos_dengue.mean()
    trimestre_2 = previo[previo.semana.between(14, 26)].groupby("id_departamento").casos_dengue.mean()
    media_previa = previo.groupby("id_departamento").casos_dengue.mean()
    nivel = perfiles["b_rec6"].unstack("semana").mean(axis=1).replace(0, 1.0)
    nivel_cierre = perfiles["b_rec6"].unstack("semana")[[47, 48, 49, 50, 51, 52]].mean(axis=1).replace(0, 1.0)
    pico = np.maximum(mitad_1, trimestre_2).replace(0, 1.0)
    es_caida = (media_previa > 2.2 * nivel) & (cierre < 0.65 * pico)
    razon = (cierre / nivel_cierre).fillna(1.0)
    razon_suave = pd.Series(
        np.where(razon > 1.15, 1.15 + np.log1p(np.maximum(razon - 1.15, 0.0)), np.clip(razon, 0.25, 1.15)),
        index=razon.index,
    )

    indice = pd.MultiIndex.from_arrays([cur.id_departamento, cur.semana])
    salida = cur[["id_departamento", "departamento", "anio", "semana", "ID", *CLIMA]].copy()
    for nombre, serie in perfiles.items():
        salida[nombre] = serie.reindex(indice).fillna(0.0).to_numpy()
    salida["is_crash"] = cur.id_departamento.map(es_caida).fillna(False).astype(float).to_numpy()
    salida["dept_scale"] = cur.id_departamento.map(nivel).fillna(1.0).clip(lower=1.0).to_numpy()
    razon_fila = cur.id_departamento.map(razon_suave).fillna(1.0).to_numpy()
    razon_fila = np.where(salida["is_crash"].to_numpy() > 0.5, 0.85, razon_fila)
    cierre_fila = cur.id_departamento.map(cierre).fillna(0.0).to_numpy()
    salida["b_end_direct"] = np.where(salida["is_crash"] > 0.5, salida["b_med_all"], np.minimum(cierre_fila, 2.2 * salida["b_rec6"]))
    salida["b_end_scaled"] = salida["b_rec6"].to_numpy() * razon_fila
    salida["q4_over_h1"] = cur.id_departamento.map((cierre_12 / pico).clip(0.1, 3.0)).fillna(1.0).to_numpy()
    salida["prev_over_norm"] = cur.id_departamento.map((media_previa / nivel).clip(0.1, 5.0)).fillna(1.0).to_numpy()

    clima_media = hist.groupby(["id_departamento", "semana"])[CLIMA].mean()
    clima_std = hist.groupby("id_departamento")[CLIMA].std().replace(0, 1.0)
    for columna in CLIMA:
        centro = clima_media[columna].reindex(indice).fillna(0.0).to_numpy()
        escala = cur.id_departamento.map(clima_std[columna]).fillna(1.0).to_numpy()
        salida[f"z_{columna}"] = (salida[columna].to_numpy() - centro) / np.maximum(escala, 1e-4)

    panel = pd.concat(
        [
            hist.loc[hist.anio == anio - 1, ["id_departamento", "anio", "semana", *REZAGOS]].assign(_parte=0),
            cur[["id_departamento", "anio", "semana", *REZAGOS]].assign(_parte=1),
        ],
        ignore_index=True,
    ).sort_values(["id_departamento", "anio", "semana"])
    grupo = panel.groupby("id_departamento", sort=False)
    for columna in REZAGOS:
        for paso in (4, 8, 12):
            panel[f"zlag_{columna}_{paso}"] = grupo[columna].shift(paso)
    rezagos = panel.loc[panel._parte == 1, [f"zlag_{columna}_{paso}" for columna in REZAGOS for paso in (4, 8, 12)]]
    salida = pd.concat([salida.reset_index(drop=True), rezagos.reset_index(drop=True)], axis=1)
    salida[list(rezagos.columns)] = salida[list(rezagos.columns)].fillna(0.0)
    salida["seno"] = np.sin(2 * np.pi * salida.semana / 52)
    salida["coseno"] = np.cos(2 * np.pi * salida.semana / 52)
    salida["vecinos"] = vecinos(hist, salida)
    if "casos_dengue" in cur.columns:
        salida["y"] = cur.casos_dengue.to_numpy(float)
        media_actual = cur.groupby("id_departamento").casos_dengue.mean()
        salida["brote"] = cur.id_departamento.map(media_actual > 2.3 * nivel).fillna(False).to_numpy()
    return salida


# %% vecinos
def vecinos(hist: pd.DataFrame, filas: pd.DataFrame, k: int = 12) -> np.ndarray:
    anual = hist.groupby(["id_departamento", "anio"]).casos_dengue.mean()
    mediana = hist.groupby("id_departamento").casos_dengue.median().replace(0, 1.0)
    brotes = {(int(depto), int(anio)) for (depto, anio), valor in anual.items() if valor > 2.3 * float(mediana.loc[depto])}
    predicho = np.full(len(filas), np.nan)
    columnas = CLIMA
    for depto, bloque in filas.groupby("id_departamento", sort=False):
        historia = hist[hist.id_departamento == depto]
        if historia.empty:
            continue
        quedarse = [(depto, int(anio)) not in brotes for anio in historia.anio.to_numpy()]
        historia = historia.loc[quedarse]
        if len(historia) < k:
            continue
        origen = historia[columnas].to_numpy(float)
        destino = bloque[columnas].to_numpy(float)
        centro = np.nanmean(origen, axis=0)
        escala = np.nanstd(origen, axis=0)
        escala = np.where(escala < 1e-6, 1.0, escala)
        origen_n = np.nan_to_num((origen - centro) / escala)
        destino_n = np.nan_to_num((destino - centro) / escala)
        distancia = ((destino_n[:, None, :] - origen_n[None, :, :]) ** 2).sum(axis=2)
        salto = np.abs(bloque.semana.to_numpy()[:, None] - historia.semana.to_numpy()[None, :])
        salto = np.minimum(salto, 53 - salto)
        distancia = np.where(salto <= 4, distancia, 1e6)
        cercanos = min(k, len(historia))
        elegidos = np.argpartition(distancia, cercanos - 1, axis=1)[:, :cercanos]
        predicho[bloque.index.to_numpy()] = np.median(historia.casos_dengue.to_numpy(float)[elegidos], axis=1)
    return predicho


# %% horizontes
def pesos_horizonte(marcos: list[pd.DataFrame], semana_min: int, semana_max: int) -> np.ndarray:
    piezas = []
    for tabla in marcos:
        mascara = tabla.semana.between(semana_min, semana_max) & ~tabla.brote
        piezas.append(tabla.loc[mascara])
    entrenamiento = pd.concat(piezas, ignore_index=True)
    matriz = entrenamiento[BASES].to_numpy(float)
    objetivo = entrenamiento.y.to_numpy(float)
    escala = float(np.mean(matriz**2))

    def objetivo_pesos(pesos: np.ndarray) -> float:
        residuo = matriz @ pesos - objetivo
        centro = np.full(len(BASES), 1.0 / len(BASES))
        return float(np.mean(residuo**2) + 0.05 * escala * np.sum((pesos - centro) ** 2))

    resultado = minimize(objetivo_pesos, np.full(len(BASES), 1.0 / len(BASES)), bounds=[(0.0, 1.0)] * len(BASES))
    return resultado.x


def aplicar_pesos(tabla: pd.DataFrame, peso_q1: np.ndarray, peso_q2: np.ndarray, peso_h2: np.ndarray) -> np.ndarray:
    matriz = tabla[BASES].to_numpy(float)
    semana = tabla.semana.to_numpy()
    return np.where(semana <= 13, matriz @ peso_q1, np.where(semana <= 26, matriz @ peso_q2, matriz @ peso_h2))


def columnas_residuales(tabla: pd.DataFrame) -> list[str]:
    columnas = [columna for columna in tabla.columns if columna.startswith("z_") or columna.startswith("zlag_")]
    columnas += ["seno", "coseno", "is_crash", "q4_over_h1", "prev_over_norm"]
    return columnas


# %% residuo
def modelo_residual(entrenamiento: pd.DataFrame, columnas: list[str]):
    escala = np.maximum(entrenamiento.dept_scale.to_numpy(float), 1.0)
    objetivo = (entrenamiento.y.to_numpy(float) - entrenamiento.prior.to_numpy(float)) / escala
    pesos = escala**1.5
    arbol = lgb.LGBMRegressor(
        n_estimators=160,
        learning_rate=0.04,
        num_leaves=8,
        min_child_samples=80,
        subsample=0.85,
        colsample_bytree=0.8,
        reg_lambda=12.0,
        random_state=42,
        verbosity=-1,
    )
    arbol.fit(entrenamiento[columnas], objetivo, sample_weight=pesos)
    bosque = xgb.XGBRegressor(
        n_estimators=120,
        learning_rate=0.04,
        max_depth=3,
        min_child_weight=40,
        subsample=0.85,
        colsample_bytree=0.8,
        reg_lambda=15.0,
        random_state=42,
        verbosity=0,
    )
    bosque.fit(entrenamiento[columnas], objetivo, sample_weight=pesos)
    linea = Ridge(alpha=30.0)
    linea.fit(entrenamiento[columnas], objetivo, sample_weight=pesos)
    red_x = StandardScaler().fit(entrenamiento[columnas])
    red = MLPRegressor(hidden_layer_sizes=(24,), alpha=8.0, max_iter=80, random_state=42)
    red.fit(red_x.transform(entrenamiento[columnas]), objetivo)
    return arbol, bosque, linea, red, red_x


def corregir(modelo, tabla: pd.DataFrame, columnas: list[str], base: np.ndarray) -> np.ndarray:
    arbol, bosque, linea, red, red_x = modelo
    escala = np.maximum(tabla.dept_scale.to_numpy(float), 1.0)
    partes = [
        arbol.predict(tabla[columnas]),
        bosque.predict(tabla[columnas]),
        linea.predict(tabla[columnas]),
        red.predict(red_x.transform(tabla[columnas])),
    ]
    residuo = np.mean(partes, axis=0)
    return np.clip(base + residuo * escala, 0, None)


# %% validacion
_MARCOS: dict[tuple[int, bool], pd.DataFrame] = {}


def preparar_anio(fuente: pd.DataFrame, anio: int, futuro: pd.DataFrame | None = None) -> pd.DataFrame:
    clave = (anio, futuro is not None)
    if clave not in _MARCOS:
        historia = fuente[fuente.anio < anio]
        actual = futuro if futuro is not None else fuente[(fuente.anio == anio) & (fuente.semana <= 52)].copy()
        _MARCOS[clave] = marco(historia, actual.reset_index(drop=True))
    return _MARCOS[clave].copy()


def evaluar(real: pd.DataFrame, predicho: np.ndarray) -> tuple[float, float]:
    semana = real.semana.to_numpy()
    casos = real.y.to_numpy(float)
    return (
        float(mean_squared_error(casos[semana <= 26], predicho[semana <= 26])),
        float(mean_squared_error(casos[semana >= 27], predicho[semana >= 27])),
    )


def media(tabla: dict[int, tuple[float, float]], anios: list[int], mitad: int) -> float:
    return float(np.mean([tabla[anio][mitad] for anio in anios]))


def caminar(
    fuente: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[int, np.ndarray], dict[int, np.ndarray], dict[int, tuple[float, float]], dict[int, tuple[float, float]]]:
    cache: dict[int, pd.DataFrame] = {}
    for anio in range(2010, 2025):
        cache[anio] = preparar_anio(fuente, anio)
    bases: dict[int, tuple[float, float]] = {}
    residuos: dict[int, tuple[float, float]] = {}
    prediccion_base: dict[int, np.ndarray] = {}
    prediccion_residual: dict[int, np.ndarray] = {}
    for anio in ANIOS_VALIDACION:
        previos = [cache[anterior] for anterior in range(2010, anio)]
        peso_q1 = pesos_horizonte(previos, 1, 13)
        peso_q2 = pesos_horizonte(previos, 14, 26)
        peso_h2 = pesos_horizonte(previos, 27, 52)
        piezas = []
        for anterior in range(2010, anio):
            tabla = cache[anterior].copy()
            tabla["prior"] = aplicar_pesos(tabla, peso_q1, peso_q2, peso_h2)
            piezas.append(tabla.loc[~tabla.brote])
        entrenamiento = pd.concat(piezas, ignore_index=True)
        columnas = columnas_residuales(entrenamiento)
        ajuste = modelo_residual(entrenamiento, columnas)
        actual = cache[anio]
        base = np.clip(aplicar_pesos(actual, peso_q1, peso_q2, peso_h2), 0, None)
        corregido = corregir(ajuste, actual, columnas, base)
        bases[anio] = evaluar(actual, base)
        residuos[anio] = evaluar(actual, corregido)
        prediccion_base[anio] = base
        prediccion_residual[anio] = corregido
        print(anio, "base", round(bases[anio][0]), round(bases[anio][1]), "residual", round(residuos[anio][0]), round(residuos[anio][1]), flush=True)
    resumen = pd.DataFrame(
        [
            {
                "modelo": nombre,
                "post_h1": media(tabla, ANIOS_POST, 0),
                "post_h2": media(tabla, ANIOS_POST, 1),
                "mod_h1": media(tabla, ANIOS_MODERADOS, 0),
                "mod_h2": media(tabla, ANIOS_MODERADOS, 1),
            }
            for nombre, tabla in (("bases", bases), ("residual", residuos))
        ]
    )
    return resumen, prediccion_base, prediccion_residual, bases, residuos


# %% envio
def predecir_2025(fuente: pd.DataFrame, futuro: pd.DataFrame, usar_residual: bool) -> pd.DataFrame:
    previos = [preparar_anio(fuente, anio) for anio in range(2010, 2025)]
    peso_q1 = pesos_horizonte(previos, 1, 13)
    peso_q2 = pesos_horizonte(previos, 14, 26)
    peso_h2 = pesos_horizonte(previos, 27, 52)
    actual = preparar_anio(fuente, 2025, futuro=futuro)
    base = np.clip(aplicar_pesos(actual, peso_q1, peso_q2, peso_h2), 0, None)
    if usar_residual:
        piezas = []
        for tabla in previos:
            tabla = tabla.copy()
            tabla["prior"] = aplicar_pesos(tabla, peso_q1, peso_q2, peso_h2)
            piezas.append(tabla.loc[~tabla.brote])
        entrenamiento = pd.concat(piezas, ignore_index=True)
        columnas = columnas_residuales(entrenamiento)
        base = corregir(modelo_residual(entrenamiento, columnas), actual, columnas, base)
    envio = actual[["ID"]].copy()
    envio["dengue"] = base
    envio["semana"] = actual.semana.to_numpy()
    envio["departamento"] = actual.departamento.to_numpy()
    envio["id_departamento"] = actual.id_departamento.to_numpy()
    return envio


def guardar(envio: pd.DataFrame, muestra: pd.DataFrame, destino: Path) -> pd.DataFrame:
    final = muestra[["id"]].merge(envio.rename(columns={"ID": "id"})[["id", "dengue"]], on="id", how="left")
    if final.dengue.isna().any() or len(final) != len(muestra):
        raise RuntimeError("El archivo de envio no cubre todos los identificadores.")
    final["dengue"] = final.dengue.clip(lower=0)
    destino.parent.mkdir(parents=True, exist_ok=True)
    final.to_csv(destino, index=False)
    return final


def tabla_alfas(pred_base: dict[int, np.ndarray], pred_res: dict[int, np.ndarray]) -> pd.DataFrame:
    filas = []
    for alpha in (0.0, 0.15, 0.3, 0.5, 1.0):
        puntajes = {}
        for anio, base in pred_base.items():
            mezcla = (1.0 - alpha) * base + alpha * pred_res[anio]
            real = _MARCOS[(anio, False)]
            puntajes[anio] = evaluar(real, mezcla)
        filas.append(
            {
                "alpha": alpha,
                "mod_h1": media(puntajes, ANIOS_MODERADOS, 0),
                "mod_h2": media(puntajes, ANIOS_MODERADOS, 1),
                "post_h1": media(puntajes, ANIOS_POST, 0),
                "post_h2": media(puntajes, ANIOS_POST, 1),
            }
        )
    return pd.DataFrame(filas)


def auditar(envio: pd.DataFrame, viejo: pd.DataFrame) -> pd.DataFrame:
    junto = envio.merge(viejo, left_on="ID", right_on="id", suffixes=("_nuevo", "_viejo"))
    deptos = [
        "Santander",
        "Valle Del Cauca",
        "Bolivar",
        "Atlantico",
        "Cordoba",
        "Sucre",
        "Antioquia",
        "Tolima",
        "Huila",
        "Meta",
        "Norte De Santander",
    ]
    filas = []
    for depto in deptos:
        mascara = junto.departamento == depto
        filas.append(
            {
                "departamento": depto,
                "h1_nuevo": float(junto.loc[mascara & (junto.semana <= 26), "dengue_nuevo"].mean()),
                "h1_viejo": float(junto.loc[mascara & (junto.semana <= 26), "dengue_viejo"].mean()),
                "h2_nuevo": float(junto.loc[mascara & (junto.semana >= 27), "dengue_nuevo"].mean()),
                "h2_viejo": float(junto.loc[mascara & (junto.semana >= 27), "dengue_viejo"].mean()),
            }
        )
    return pd.DataFrame(filas)


def sesgo_departamental(predicciones: dict[int, np.ndarray], anios: list[int]) -> pd.DataFrame:
    piezas = []
    for anio in anios:
        real = _MARCOS[(anio, False)]
        mitad = real.semana.to_numpy() >= 27
        pieza = real.loc[mitad, ["departamento"]].copy()
        pieza["error"] = predicciones[anio][mitad] - real.y.to_numpy(float)[mitad]
        piezas.append(pieza)
    junto = pd.concat(piezas, ignore_index=True)
    junto["cuadrado"] = np.square(junto.error)
    resumen = junto.groupby("departamento").agg(sesgo=("error", "mean"), mse=("cuadrado", "mean"), semanas=("error", "size"))
    return resumen.sort_values("mse", ascending=False)


def sesgo_por_anio(predicciones: dict[int, np.ndarray], anios: list[int], deptos: list[str]) -> pd.DataFrame:
    filas = []
    for anio in anios:
        real = _MARCOS[(anio, False)]
        mitad = real.semana.to_numpy() >= 27
        error = predicciones[anio][mitad] - real.y.to_numpy(float)[mitad]
        nombres = real.departamento.to_numpy()[mitad]
        fila = {"anio": anio}
        for depto in deptos:
            fila[depto] = float(error[nombres == depto].mean())
        filas.append(fila)
    return pd.DataFrame(filas)


def correr() -> None:
    warnings.filterwarnings("ignore", category=UserWarning)
    train, test, muestra = cargar()
    resumen, pred_base, pred_res, bases, residuos = caminar(train)
    print(resumen.round(1).to_string(index=False), flush=True)
    alfas = tabla_alfas(pred_base, pred_res)
    vista = alfas.copy()
    vista["alpha"] = vista["alpha"].map(lambda valor: f"{valor:.2f}")
    for columna in ("mod_h1", "mod_h2", "post_h1", "post_h2"):
        vista[columna] = vista[columna].map(lambda valor: f"{valor:.1f}")
    print(vista.to_string(index=False), flush=True)
    sesgo = sesgo_departamental(pred_base, ANIOS_MODERADOS)
    print(sesgo.head(8).round(1).to_string(), flush=True)
    detalle = sesgo_por_anio(
        pred_base,
        ANIOS_MODERADOS,
        ["Atlantico", "Bolivar", "Cordoba", "Santander", "Valle Del Cauca"],
    )
    print(detalle.round(1).to_string(index=False), flush=True)
    if float(alfas.loc[alfas.alpha == 0.0, "mod_h2"].iloc[0]) > float(alfas.mod_h2.min()) + 1:
        raise RuntimeError("Un residuo bajo el segundo semestre. Revisar antes de guardar.")
    anios = pd.DataFrame(
        {
            "anio": ANIOS_VALIDACION,
            "h1_bases": [round(bases[anio][0]) for anio in ANIOS_VALIDACION],
            "h2_bases": [round(bases[anio][1]) for anio in ANIOS_VALIDACION],
            "h2_arbol_anterior": [ARBOL_H2[anio] for anio in ANIOS_VALIDACION],
            "h2_residual": [round(residuos[anio][1]) for anio in ANIOS_VALIDACION],
        }
    )
    print(anios.to_string(index=False), flush=True)
    ganancia_mod = float(np.mean([ARBOL_H2[anio] for anio in ANIOS_MODERADOS])) - float(resumen.loc[resumen.modelo == "bases", "mod_h2"].iloc[0])
    ganancia_post = float(np.mean([ARBOL_H2[anio] for anio in ANIOS_POST])) - float(resumen.loc[resumen.modelo == "bases", "post_h2"].iloc[0])
    print("ganancia H2 moderados", round(ganancia_mod), "ganancia H2 post-brote", round(ganancia_post), flush=True)
    if ganancia_mod < 1000 or ganancia_post < 1000:
        raise RuntimeError("La ganancia del backtest no llega a 1000 puntos de MSE.")
    previos = [preparar_anio(train, anio) for anio in range(2010, 2025)]
    for nombre, inicio, fin in (("Q1", 1, 13), ("Q2", 14, 26), ("H2", 27, 52)):
        pesos = pesos_horizonte(previos, inicio, fin)
        print(nombre, {base: round(float(peso), 3) for base, peso in zip(BASES, pesos)}, flush=True)
    envio = predecir_2025(train, test, usar_residual=False)
    destino = raiz_repo() / CARPETA_ENSAYO / "submission.csv"
    final = guardar(envio, muestra, destino)
    if final.id.tolist() != muestra.id.tolist() or final.dengue.isna().any() or (final.dengue < 0).any():
        raise RuntimeError("El envio no respeta el orden ni el dominio del archivo de muestra.")
    viejo = pd.read_csv(raiz_repo() / ENVIO_ANTERIOR)
    comparacion = auditar(envio, viejo)
    print(comparacion.round(1).to_string(index=False), flush=True)
    print(
        "media",
        round(float(final.dengue.mean()), 2),
        "H1",
        round(float(envio.loc[envio.semana <= 26, "dengue"].mean()), 2),
        "H2",
        round(float(envio.loc[envio.semana >= 27, "dengue"].mean()), 2),
        "filas",
        len(final),
        flush=True,
    )
    santander = float(comparacion.loc[comparacion.departamento == "Santander", "h2_nuevo"].iloc[0])
    bolivar = float(comparacion.loc[comparacion.departamento == "Bolivar", "h2_nuevo"].iloc[0])
    atlantico = float(comparacion.loc[comparacion.departamento == "Atlantico", "h2_nuevo"].iloc[0])
    if santander > 130 or bolivar < 140 or atlantico < 120:
        raise RuntimeError("Quedo el sesgo de Santander o de la Costa Caribe.")
    print("archivo", destino, flush=True)


if __name__ == "__main__":
    correr()
