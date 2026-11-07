"""Walk-forward forecast for the 2025 dengue competition.

Every cutoff year Y is fit only on years before Y. Department groupings come from a
Gaussian mixture on climate and seasonality, never from a hand-written id list.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.optimize import minimize, nnls
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from torch import nn

CARPETA = Path(__file__).resolve().parent
RAIZ = CARPETA.parents[2]
ANIOS_EVAL = list(range(2014, 2025))
ANIOS_META = list(range(2014, 2026))
K_MAX = 4
BLOQUES = ((1, 13), (14, 26), (27, 52))
BASES = [
    "b_med_all",
    "b_mean_all",
    "b_rec6",
    "b_med5",
    "b_rec3",
    "b_end_direct",
    "b_end_scaled",
]
CLIMA_Z = ["temp_media_C", "temp_min_C", "precip_chirps_mm", "presion_superficial_hPa"]
VENTANA = 52
HORIZONTE = 52
SEMANALES = 52
DISPOSITIVO = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LOTE = 512
EPOCAS = 12


def _horizonte():
    ruta = RAIZ / "ensayos" / "juan" / "2026-10-31-horizonte-bases" / "pipeline.py"
    spec = importlib.util.spec_from_file_location("horizonte_bases", ruta)
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    return modulo


def suavizar_razon(razon: float) -> float:
    if razon > 1.15:
        return 1.15 + float(np.log1p(razon - 1.15))
    return float(np.clip(razon, 0.25, 1.15))


def estado_epidemiologico(historia: pd.DataFrame, anio: int) -> pd.DataFrame:
    """Continuous epidemic state at the close of anio-1. No department rules."""
    previo = historia[historia.anio == anio - 1]
    anterior = historia[historia.anio == anio - 2]
    nivel = historia.groupby("id_departamento").casos_dengue.median().replace(0, 1.0)
    media_prev = previo.groupby("id_departamento").casos_dengue.mean()
    media_104 = pd.concat([previo, anterior]).groupby("id_departamento").casos_dengue.mean()
    cierre = previo[previo.semana >= 41].groupby("id_departamento").casos_dengue.mean()
    mitad = previo[previo.semana <= 26].groupby("id_departamento").casos_dengue.mean()
    tarde = previo[previo.semana.between(29, 40)].groupby("id_departamento").casos_dengue.mean()
    pico = np.maximum(mitad, previo[previo.semana.between(14, 26)].groupby("id_departamento").casos_dengue.mean())
    pico = pico.replace(0, 1.0)
    nivel_cierre = (
        historia[historia.semana >= 47].groupby("id_departamento").casos_dengue.median().replace(0, 1.0)
    )
    razon = (cierre / nivel_cierre).replace([np.inf, -np.inf], np.nan).fillna(1.0)
    pendiente = ((cierre - tarde) / pico).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    tabla = pd.DataFrame(
        {
            "exceso_52": (media_prev / nivel).clip(0.1, 8.0),
            "exceso_104": (media_104 / nivel).clip(0.1, 8.0),
            "cierre_sobre_pico": (cierre / pico).clip(0.05, 3.0),
            "pendiente_12": pendiente.clip(-3, 3),
            "cierre": cierre.clip(lower=0),
            "razon_cierre": razon.map(suavizar_razon),
        }
    )
    return tabla.fillna(1.0)


def rasgos_cluster(historia: pd.DataFrame) -> pd.DataFrame:
    clima = historia.groupby("id_departamento")[["temp_media_C", "temp_min_C", "presion_superficial_hPa", "precip_chirps_mm"]].mean()
    clima.columns = ["temp_media", "temp_min", "presion", "precip"]
    semanal = historia.groupby(["id_departamento", "semana"]).casos_dengue.median().unstack("semana")
    angulo = 2 * np.pi * (semanal.columns.to_numpy() - 1) / 52.0
    peso = semanal.fillna(0.0).to_numpy()
    seno = peso @ np.sin(angulo)
    coseno = peso @ np.cos(angulo)
    fase = pd.DataFrame(
        {"fase_seno": seno / (np.linalg.norm(np.c_[seno, coseno], axis=1) + 1e-6),
         "fase_coseno": coseno / (np.linalg.norm(np.c_[seno, coseno], axis=1) + 1e-6)},
        index=semanal.index,
    )
    escala = np.log1p(historia.groupby("id_departamento").casos_dengue.median())
    anual = historia.groupby(["id_departamento", "anio"]).casos_dengue.mean().unstack("anio")
    auto = []
    for depto, fila in anual.iterrows():
        valores = fila.dropna().to_numpy()
        if len(valores) < 4:
            auto.append(0.0)
        else:
            auto.append(float(np.corrcoef(valores[:-1], valores[1:])[0, 1]))
    fase["escala"] = escala
    fase["autocorr"] = pd.Series(auto, index=anual.index).fillna(0.0)
    return clima.join(fase).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def agrupar(historia: pd.DataFrame) -> pd.DataFrame:
    rasgos = rasgos_cluster(historia)
    scaler = StandardScaler()
    x = scaler.fit_transform(rasgos.to_numpy(float))
    mejor, bic = None, np.inf
    for k in range(2, K_MAX + 1):
        modelo = GaussianMixture(n_components=k, covariance_type="diag", random_state=0, n_init=3)
        modelo.fit(x)
        if modelo.bic(x) < bic:
            mejor, bic = modelo, modelo.bic(x)
    prob = np.zeros((len(rasgos), K_MAX))
    prob[:, : mejor.n_components] = mejor.predict_proba(x)
    columnas = [f"p{k}" for k in range(K_MAX)]
    return pd.DataFrame(prob, index=rasgos.index, columns=columnas)


def clima_nacional_pc(historia: pd.DataFrame, actual: pd.DataFrame) -> float:
    def perfil(tabla: pd.DataFrame) -> pd.Series:
        nacional = tabla[tabla.semana <= 13].groupby(["anio", "semana"])[["temp_media_C", "precip_chirps_mm"]].mean()
        ancho = nacional.groupby("anio").size()
        completos = ancho[ancho >= 13].index
        plano = nacional.loc[completos].unstack("semana")
        return plano

    pasado = perfil(historia)
    if pasado.shape[0] < 3:
        return 0.0
    futuro = perfil(pd.concat([historia, actual], ignore_index=True))
    columnas = pasado.columns.intersection(futuro.columns)
    pasado = pasado[columnas]
    futuro = futuro[columnas]
    if pasado.shape[1] < 2:
        return 0.0
    scaler = StandardScaler()
    base = scaler.fit_transform(pasado.fillna(pasado.mean()).to_numpy(float))
    pca = PCA(n_components=1, random_state=0)
    pca.fit(base)
    puntajes = pca.transform(base).ravel()
    escala = float(puntajes.std())
    if escala < 1e-6:
        escala = 1.0
    anio = int(actual.anio.iloc[0])
    if anio not in futuro.index:
        return 0.0
    fila = scaler.transform(futuro.loc[[anio]].fillna(pasado.mean()).to_numpy(float))
    z = (float(pca.transform(fila)[0, 0]) - float(puntajes.mean())) / escala
    return float(np.clip(z, -2.0, 2.0))


def marcos(train: pd.DataFrame, test: pd.DataFrame) -> dict[int, pd.DataFrame]:
    horizonte = _horizonte()
    salida = {}
    for anio in range(2011, 2026):
        futuro = test if anio == 2025 else None
        tabla = horizonte.preparar_anio(train, anio, futuro=futuro)
        historia = train[train.anio < anio]
        estado = estado_epidemiologico(historia, anio)
        grupos = agrupar(historia)
        pc = clima_nacional_pc(historia, tabla)
        tabla = tabla.copy()
        for columna in estado.columns:
            tabla[columna] = tabla.id_departamento.map(estado[columna]).astype(float).fillna(1.0)
        for columna in grupos.columns:
            tabla[columna] = tabla.id_departamento.map(grupos[columna]).astype(float).fillna(0.0)
        tabla = tabla.merge(
            pd.concat(
                [
                    train[["id_departamento", "anio", "semana", "semana_inicio"]],
                    test[["id_departamento", "anio", "semana", "semana_inicio"]],
                ]
            ).drop_duplicates(["id_departamento", "anio", "semana"]),
            on=["id_departamento", "anio", "semana"],
            how="left",
        )
        tabla["pc_clima"] = pc
        cierre = tabla["cierre"].to_numpy(float)
        rec6 = tabla["b_rec6"].to_numpy(float)
        tabla["b_end_direct"] = np.minimum(cierre, 2.2 * rec6)
        tabla["b_end_scaled"] = rec6 * tabla["razon_cierre"].to_numpy(float)
        tabla["agot_direct"] = tabla["b_end_direct"] / (1.0 + tabla["exceso_104"])
        tabla["agot_scaled"] = tabla["b_end_scaled"] * tabla["cierre_sobre_pico"]
        tabla["clima_base"] = tabla["b_rec6"] * tabla["pc_clima"]
        salida[anio] = tabla
        print(f"marco {anio} pc={pc:.3f} k={int((grupos.sum() > 0).sum())}", flush=True)
    return salida


def columnas_a() -> list[str]:
    return BASES + ["agot_direct", "agot_scaled"]


def _mezcla_convexa(x: np.ndarray, y: np.ndarray, peso: np.ndarray | None = None) -> np.ndarray:
    n = x.shape[1]
    inicio = np.full(n, 1.0 / n)
    factor = np.ones(len(y)) if peso is None else np.sqrt(np.clip(peso, 0, None))

    def objetivo(pesos: np.ndarray) -> float:
        residuo = (y - x @ pesos) * factor
        return float(residuo @ residuo)

    ajuste = minimize(
        objetivo,
        inicio,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * n,
        constraints={"type": "eq", "fun": lambda pesos: np.sum(pesos) - 1.0},
        options={"maxiter": 200, "ftol": 1e-8},
    )
    return ajuste.x if ajuste.success else inicio


AGOT: dict[int, float] = {}


def _matriz_agotada(tabla: pd.DataFrame, fuerza: float) -> np.ndarray:
    """Pull recent bases toward the long-run median when a long outbreak is already falling."""
    exceso = tabla.exceso_104.to_numpy(float)
    cierre = tabla.cierre_sobre_pico.to_numpy(float)
    crudo = fuerza * np.maximum(exceso - 1.0, 0.0) * np.maximum(1.0 - cierre, 0.0)
    factor = crudo / (1.0 + crudo)
    mediana = tabla.b_med_all.to_numpy(float)
    columnas = []
    for nombre in ("b_med_all", "b_mean_all", "b_med5", "b_rec6", "b_rec3", "b_end_direct", "b_end_scaled", "agot_direct", "agot_scaled"):
        valores = tabla[nombre].to_numpy(float)
        if nombre in ("b_rec6", "b_rec3", "b_end_direct", "b_end_scaled"):
            valores = (1.0 - factor) * valores + factor * mediana
        columnas.append(valores)
    return np.nan_to_num(np.column_stack(columnas), nan=0.0, posinf=0.0, neginf=0.0)


def predecir_a(marcos_anio: dict[int, pd.DataFrame], anio: int) -> np.ndarray:
    fuerza, mejor = 0.0, np.inf
    for candidata in (0.0, 0.4, 0.8, 1.2, 2.0):
        xs, ys = [], []
        for previo in range(2011, anio):
            xs.append(_matriz_agotada(marcos_anio[previo], candidata))
            ys.append(marcos_anio[previo].y.to_numpy(float))
        x = np.vstack(xs)
        y = np.concatenate(ys)
        validos = np.isfinite(y)
        pesos = _mezcla_convexa(x[validos], y[validos])
        error = float(np.mean((y[validos] - x[validos] @ pesos) ** 2))
        if error < mejor:
            fuerza, mejor = candidata, error
    AGOT[anio] = fuerza
    actual = marcos_anio[anio]
    x_actual = _matriz_agotada(actual, fuerza)
    clima_actual = (actual.b_rec6 * actual.pc_clima).to_numpy(float)
    pred = np.zeros(len(actual))
    semana_actual = actual.semana.to_numpy(int)
    for inicio, fin in BLOQUES:
        xs, ys, semanas, climas = [], [], [], []
        for previo in range(2011, anio):
            tabla = marcos_anio[previo]
            xs.append(_matriz_agotada(tabla, fuerza))
            ys.append(tabla.y.to_numpy(float))
            semanas.append(tabla.semana.to_numpy(int))
            climas.append((tabla.b_rec6 * tabla.pc_clima).to_numpy(float))
        x = np.vstack(xs)
        y = np.concatenate(ys)
        semana = np.concatenate(semanas)
        clima = np.concatenate(climas)
        entrenar = (semana >= inicio) & (semana <= fin) & np.isfinite(y)
        aplicar = (semana_actual >= inicio) & (semana_actual <= fin)
        pesos = _mezcla_convexa(x[entrenar], y[entrenar])
        base = x[entrenar] @ pesos
        ajuste = clima[entrenar]
        escala = float(ajuste @ ajuste)
        alpha = 0.0 if escala < 1e-6 else float(np.dot(y[entrenar] - base, ajuste) / escala)
        alpha = float(np.clip(alpha, -0.25, 0.25))
        pred[aplicar] = x_actual[aplicar] @ pesos + alpha * clima_actual[aplicar]
    return np.clip(pred, 0, None)


def matriz_b(tabla: pd.DataFrame, pred_a: np.ndarray) -> pd.DataFrame:
    trabajo = tabla.copy()
    trabajo["base_ml"] = pred_a
    razones = {
        "r_rec3_med": trabajo.b_rec3 / (trabajo.b_med_all + 1),
        "r_end_rec6": trabajo.b_end_direct / (trabajo.b_rec6 + 1),
        "r_med5_mean": trabajo.b_med5 / (trabajo.b_mean_all + 1),
        "r_scaled_med": trabajo.b_end_scaled / (trabajo.b_med_all + 1),
    }
    for nombre, serie in razones.items():
        trabajo[nombre] = serie.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    trabajo["log_base"] = np.log1p(trabajo.base_ml)
    return trabajo


COLUMNAS_B = [
    "r_rec3_med",
    "r_end_rec6",
    "r_med5_mean",
    "r_scaled_med",
    "log_base",
    "exceso_52",
    "exceso_104",
    "pendiente_12",
    "cierre_sobre_pico",
    "pc_clima",
    "seno",
    "coseno",
    "semana",
    "p0",
    "p1",
    "p2",
    "p3",
    "z_temp_media_C",
    "z_temp_min_C",
    "z_precip_chirps_mm",
    "zlag_temp_min_C_4",
    "zlag_temp_min_C_8",
    "zlag_temp_min_C_12",
    "zlag_precip_chirps_mm_4",
    "zlag_precip_chirps_mm_8",
    "zlag_precip_chirps_mm_12",
]


def _modelos_b():
    import lightgbm as lgb
    import xgboost as xgb

    return [
        lgb.LGBMRegressor(
            n_estimators=180,
            learning_rate=0.05,
            num_leaves=15,
            min_child_samples=40,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=0,
            verbosity=-1,
        ),
        xgb.XGBRegressor(
            n_estimators=180,
            learning_rate=0.05,
            max_depth=4,
            min_child_weight=20,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            random_state=0,
            n_jobs=4,
        ),
        HistGradientBoostingRegressor(
            max_iter=180,
            learning_rate=0.05,
            max_leaf_nodes=15,
            min_samples_leaf=40,
            random_state=0,
        ),
    ]


def predecir_b(tablas: dict[int, pd.DataFrame], anio: int) -> np.ndarray:
    marcos_b = []
    for previo in range(2014, anio):
        marcos_b.append(tablas[previo])
    train = pd.concat(marcos_b, ignore_index=True)
    y = train.y.to_numpy(float)
    base = train.base_ml.to_numpy(float)
    objetivo = np.log((y + 1.0) / (base + 1.0))
    peso = base + 1.0
    x = train[COLUMNAS_B].to_numpy(float)
    actual = tablas[anio]
    partes = []
    for modelo in _modelos_b():
        modelo.fit(x, objetivo, sample_weight=peso)
        partes.append(modelo.predict(actual[COLUMNAS_B].to_numpy(float)))
    ratio = np.mean(partes, axis=0)
    return np.clip((actual.base_ml.to_numpy(float) + 1.0) * np.exp(np.clip(ratio, -1.0, 1.0)) - 1.0, 0, None)


class _Bloque(nn.Module):
    def __init__(self, entrada: int, salida: int, kernel: int, dilatacion: int):
        super().__init__()
        self.recorte = (kernel - 1) * dilatacion
        self.conv = nn.Conv1d(entrada, salida, kernel, dilation=dilatacion, padding=self.recorte)
        self.salto = nn.Conv1d(entrada, salida, 1) if entrada != salida else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.conv(x)[..., : -self.recorte] + self.salto(x))


class _Red(nn.Module):
    def __init__(self, familia: str, n_enc: int, n_est: int, n_dec: int):
        super().__init__()
        self.familia = familia
        if familia == "mlp":
            self.cod = nn.Sequential(nn.Flatten(), nn.Linear(VENTANA * n_enc, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU())
            oculto = 32
        elif familia == "gru":
            self.cod = nn.GRU(n_enc, 32, batch_first=True)
            oculto = 32
        else:
            self.cod = nn.Sequential(_Bloque(n_enc, 16, 3, 1), _Bloque(16, 16, 3, 2), _Bloque(16, 32, 3, 4))
            oculto = 32
        self.cabeza = nn.Sequential(nn.Linear(oculto + n_est + n_dec, 32), nn.ReLU(), nn.Linear(32, 2))

    def codificar(self, x: torch.Tensor) -> torch.Tensor:
        if self.familia == "mlp":
            return self.cod(x)
        if self.familia == "gru":
            salida, _ = self.cod(x)
            return salida[:, -1]
        return self.cod(x.transpose(1, 2))[:, :, -1]

    def forward(self, enc: torch.Tensor, est: torch.Tensor, dec: torch.Tensor) -> torch.Tensor:
        z = self.codificar(enc)
        pasos = dec.shape[1]
        u = torch.cat([z[:, None].expand(-1, pasos, -1), est[:, None].expand(-1, pasos, -1), dec], dim=2)
        bruto = self.cabeza(u)
        return torch.sigmoid(bruto[..., 0]) * dec[..., 0] + 0.5 * torch.tanh(bruto[..., 1])


def _panel(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    futuro = test.copy()
    futuro["casos_dengue"] = np.nan
    full = pd.concat([train, futuro], ignore_index=True)
    full = full[full.semana <= 52].sort_values(["id_departamento", "anio", "semana"]).reset_index(drop=True)
    conteo = full.groupby(["id_departamento", "anio"]).size()
    if conteo.min() != 52 or conteo.max() != 52:
        raise RuntimeError(f"La grilla semanal no es 52x31: {conteo.min()} {conteo.max()}")
    return full


def _ventanas(full: pd.DataFrame) -> dict[str, np.ndarray]:
    deptos = np.sort(full.id_departamento.unique())
    primero = full[full.id_departamento == deptos[0]]
    anio = primero.anio.to_numpy()
    semana = primero.semana.to_numpy()
    y = full.casos_dengue.to_numpy(float).reshape(len(deptos), -1)
    clima_cols = ["precip_chirps_mm", "temp_media_C", "temp_min_C"]
    clima = full[clima_cols].to_numpy(float).reshape(len(deptos), -1, len(clima_cols))
    base = np.isin(anio, list(range(2007, 2014)))
    z = np.zeros_like(clima)
    for d in range(len(deptos)):
        escala = np.std(clima[d, base], axis=0)
        escala = np.where(escala < 1e-6, 1.0, escala)
        for s in range(1, 53):
            media = clima[d, base & (semana == s)].mean(axis=0)
            filas = semana == s
            z[d, filas] = (clima[d, filas] - media) / escala
    z = np.clip(z, -4, 4)
    origenes = np.arange(VENTANA - 1, y.shape[1] - HORIZONTE)
    n = len(origenes)
    enc = np.zeros((len(deptos), n, VENTANA, 1 + 2 + len(clima_cols)), np.float32)
    dec = np.zeros((len(deptos), n, HORIZONTE, 1 + 2 + len(clima_cols) + 1), np.float32)
    obj = np.zeros((len(deptos), n, HORIZONTE), np.float32)
    nivel = np.zeros((len(deptos), n), np.float32)
    anio_obj = anio[origenes + 1]
    logy = np.log1p(np.nan_to_num(y, nan=0.0))
    for j, t in enumerate(origenes):
        pasado = slice(t - VENTANA + 1, t + 1)
        futuro = slice(t + 1, t + 1 + HORIZONTE)
        nivel_d = np.median(y[:, t - 7 : t + 1], axis=1)
        logl = np.log1p(np.clip(nivel_d, 0, None))
        ang_p = 2 * np.pi * (semana[pasado] - 1) / 52.0
        ang_f = 2 * np.pi * (semana[futuro] - 1) / 52.0
        enc[:, j, :, 0] = logy[:, pasado] - logl[:, None]
        enc[:, j, :, 1] = np.sin(ang_p)
        enc[:, j, :, 2] = np.cos(ang_p)
        enc[:, j, :, 3:] = z[:, pasado]
        dec[:, j, :, 1] = np.sin(ang_f)
        dec[:, j, :, 2] = np.cos(ang_f)
        dec[:, j, :, 3:6] = z[:, futuro]
        dec[:, j, :, 6] = np.arange(1, HORIZONTE + 1) / HORIZONTE
        obj[:, j] = np.nan_to_num(y[:, futuro], nan=0.0)
        nivel[:, j] = logl
    return {
        "deptos": deptos,
        "origenes": origenes,
        "anio_obj": anio_obj,
        "anio": anio,
        "enc": enc,
        "dec": dec,
        "obj": obj,
        "nivel": nivel,
        "y": y,
        "observable": np.isfinite(y[:, origenes + HORIZONTE]),
    }


def _estaticas(ventanas: dict, marcos_anio: dict[int, pd.DataFrame]) -> np.ndarray:
    deptos = ventanas["deptos"]
    n = len(ventanas["origenes"])
    est = np.zeros((len(deptos), n, 3 + K_MAX), np.float32)
    carga = {}
    for anio, tabla in marcos_anio.items():
        fila = tabla.drop_duplicates("id_departamento").set_index("id_departamento")
        carga[anio] = fila
    for j, anio in enumerate(ventanas["anio_obj"]):
        if anio not in carga:
            continue
        fila = carga[int(anio)].reindex(deptos)
        est[:, j, 0] = np.log1p(fila.exceso_104.to_numpy(float))
        est[:, j, 1] = fila.pc_clima.to_numpy(float) / 5.0
        est[:, j, 2] = fila.cierre_sobre_pico.to_numpy(float)
        est[:, j, 3:] = fila[[f"p{k}" for k in range(K_MAX)]].to_numpy(float)
    return np.nan_to_num(est)


def _anclar_nivel(ventanas: dict, est: np.ndarray) -> None:
    """Move the persistence anchor toward the seasonal profile only if the outbreak is long and already falling."""
    nivel = ventanas["nivel"]
    log_perfil = nivel + np.nan_to_num(ventanas["dec"][..., 0].mean(axis=2))
    exceso = np.expm1(est[..., 0])
    cierre = est[..., 2]
    for j, anio in enumerate(ventanas["anio_obj"]):
        fuerza = AGOT.get(int(anio), 0.0)
        if fuerza == 0.0:
            continue
        crudo = fuerza * np.maximum(exceso[:, j] - 1.0, 0.0) * np.maximum(1.0 - cierre[:, j], 0.0)
        factor = (crudo / (1.0 + crudo)).astype(np.float32)
        nuevo = (1.0 - factor) * nivel[:, j] + factor * log_perfil[:, j]
        ventanas["dec"][:, j, :, 0] = ventanas["dec"][:, j, :, 0] + nivel[:, j, None] - nuevo[:, None]
        ventanas["nivel"][:, j] = nuevo.astype(np.float32)


def _perfil_en_dec(ventanas: dict, full: pd.DataFrame) -> None:
    deptos = ventanas["deptos"]
    anio = ventanas["anio"]
    semana = full[full.id_departamento == deptos[0]].semana.to_numpy()
    y = ventanas["y"]
    for j, t in enumerate(ventanas["origenes"]):
        anio_corte = int(ventanas["anio_obj"][j])
        mascara = anio < anio_corte
        perfil = np.zeros((len(deptos), 53))
        for s in range(1, 53):
            perfil[:, s] = np.nanmedian(y[:, mascara & (semana == s)], axis=1)
        futuro = slice(t + 1, t + 1 + HORIZONTE)
        logl = ventanas["nivel"][:, j]
        nivel_perfil = np.nan_to_num(perfil[:, semana[futuro]], nan=0.0)
        ventanas["dec"][:, j, :, 0] = np.log1p(nivel_perfil) - logl[:, None]


def predecir_redes(full: pd.DataFrame, marcos_anio: dict[int, pd.DataFrame], anios: list[int]) -> dict[int, dict[str, np.ndarray]]:
    print(f"redes en {DISPOSITIVO}", flush=True)
    ventanas = _ventanas(full)
    _perfil_en_dec(ventanas, full)
    est = _estaticas(ventanas, marcos_anio)
    _anclar_nivel(ventanas, est)
    ventanas["est"] = est
    deptos = np.arange(len(ventanas["deptos"]))
    origenes = ventanas["origenes"]
    anio_fin = ventanas["anio"][origenes + HORIZONTE]
    predicciones = {}
    for anio in anios:
        t_origen = int(np.where(ventanas["anio"] == anio - 1)[0].max())
        j_origen = int(np.where(origenes == t_origen)[0][0])
        entrenar = np.where(anio_fin < anio)[0]
        rejilla = np.array([[d, j] for j in entrenar for d in deptos if ventanas["observable"][d, j]])
        predicciones[anio] = {}
        for familia in ("mlp", "tcn", "gru"):
            t0 = time.time()
            torch.manual_seed(0)
            modelo = _Red(familia, ventanas["enc"].shape[-1], est.shape[-1], ventanas["dec"].shape[-1]).to(DISPOSITIVO)
            opt = torch.optim.AdamW(modelo.parameters(), lr=1e-3, weight_decay=1e-4)
            x_enc = torch.tensor(ventanas["enc"], device=DISPOSITIVO)
            x_est = torch.tensor(est, device=DISPOSITIVO)
            x_dec = torch.tensor(ventanas["dec"], device=DISPOSITIVO)
            x_obj = torch.tensor(ventanas["obj"], device=DISPOSITIVO)
            x_niv = torch.tensor(ventanas["nivel"], device=DISPOSITIVO)
            for _ in range(EPOCAS):
                orden = rejilla[np.random.permutation(len(rejilla))]
                modelo.train()
                for inicio in range(0, len(orden), LOTE):
                    lote = orden[inicio : inicio + LOTE]
                    d = torch.tensor(lote[:, 0], device=DISPOSITIVO)
                    j = torch.tensor(lote[:, 1], device=DISPOSITIVO)
                    ratio = modelo(x_enc[d, j], x_est[d, j], x_dec[d, j])
                    eta = ratio + x_niv[d, j][:, None]
                    perdida = (torch.exp(eta) - x_obj[d, j] * eta).mean()
                    opt.zero_grad()
                    perdida.backward()
                    nn.utils.clip_grad_norm_(modelo.parameters(), 1.0)
                    opt.step()
            modelo.eval()
            with torch.no_grad():
                d = torch.tensor(deptos, device=DISPOSITIVO)
                j = torch.full((len(deptos),), j_origen, device=DISPOSITIVO)
                ratio = modelo(x_enc[d, j], x_est[d, j], x_dec[d, j])
                pred = torch.exp(ratio + x_niv[d, j][:, None]).cpu().numpy()
            serie = pd.DataFrame(pred, index=ventanas["deptos"], columns=range(1, 53))
            largo = serie.stack().rename("prediccion").reset_index()
            largo.columns = ["id_departamento", "semana", "prediccion"]
            orden = marcos_anio[anio][["id_departamento", "semana"]].merge(largo, on=["id_departamento", "semana"], how="left")
            predicciones[anio][familia] = orden.prediccion.to_numpy(float)
            print(f"red {familia} {anio} {time.time() - t0:.1f}s", flush=True)
    return predicciones


def _mse(y: np.ndarray, pred: np.ndarray, semana: np.ndarray, mitad: str) -> float:
    if mitad == "h1":
        mascara = semana <= 26
    else:
        mascara = semana >= 27
    return float(np.mean((y[mascara] - pred[mascara]) ** 2))


def _apilar(pred: dict[str, np.ndarray], prob: np.ndarray) -> np.ndarray:
    nombres = list(pred)
    columnas = []
    for nombre in nombres:
        for k in range(K_MAX):
            columnas.append(pred[nombre] * prob[:, k])
    return np.column_stack(columnas)


def ajustar_meta(historial: list[tuple[dict, np.ndarray, np.ndarray]], nombres: list[str]) -> np.ndarray:
    """Non-negative mixture of families. One mixture per cluster, rows weighted by P(cluster)."""
    pesos = []
    for k in range(K_MAX):
        xs, ys, ws = [], [], []
        for pred, y, prob in historial:
            xs.append(np.column_stack([pred[nombre] for nombre in nombres]))
            ys.append(y)
            ws.append(prob[:, k])
        x = np.nan_to_num(np.vstack(xs), nan=0.0, posinf=0.0, neginf=0.0)
        y = np.concatenate(ys)
        w = np.concatenate(ws)
        if float(w.sum()) < 1.0:
            pesos.append(np.full(len(nombres), 1.0 / len(nombres)))
        else:
            pesos.append(_mezcla_convexa(x, y, w))
    return np.vstack(pesos)


def aplicar_meta(pesos: np.ndarray, pred: dict[str, np.ndarray], prob: np.ndarray, nombres: list[str]) -> np.ndarray:
    x = np.nan_to_num(np.column_stack([pred[nombre] for nombre in nombres]), nan=0.0, posinf=0.0, neginf=0.0)
    salida = np.zeros(len(x))
    for k in range(K_MAX):
        salida += prob[:, k] * (x @ pesos[k])
    return np.clip(salida, 0, None)


def empalme(historial: list[dict], actual: dict) -> np.ndarray:
    """Learn how much of the predicted H2/H1 level ratio to apply on H1."""
    mejor, mejor_mse = (1, 0.0), np.inf
    for inicio in range(1, 9):
        for fuerza in np.linspace(0, 1, 11):
            errores = []
            for fila in historial:
                pred = _aplicar_empalme(fila, inicio, fuerza)
                semana = fila["semana"]
                errores.append(np.mean((fila["y"][semana <= 26] - pred[semana <= 26]) ** 2))
            media = float(np.mean(errores)) if errores else np.inf
            if media < mejor_mse:
                mejor, mejor_mse = (inicio, float(fuerza)), media
    inicio, fuerza = mejor
    return _aplicar_empalme(actual, inicio, fuerza), {"inicio": inicio, "fuerza": fuerza, "mse": None if not historial else mejor_mse}


def _aplicar_empalme(fila: dict, inicio: int, fuerza: float) -> np.ndarray:
    pred = fila["pred"].copy()
    semana = fila["semana"]
    depto = fila["depto"]
    for valor in np.unique(depto):
        mascara = depto == valor
        h1 = mascara & (semana <= 26)
        h2 = mascara & (semana >= 27)
        if h1.sum() == 0 or h2.sum() == 0:
            continue
        nivel_h1 = max(float(pred[h1].mean()), 1.0)
        nivel_h2 = max(float(pred[h2].mean()), 0.0)
        razon = np.clip(nivel_h2 / nivel_h1, 0.4, 1.6)
        aplicar = h1 & (semana >= inicio)
        pred[aplicar] = (1 - fuerza) * pred[aplicar] + fuerza * pred[aplicar] * razon
    return pred


def bayes_calendario(train: pd.DataFrame, tabla: pd.DataFrame, pred: np.ndarray, anio: int) -> np.ndarray:
    historia = train[(train.anio < anio) & (train.semana <= 52)].copy()
    historia["miercoles"] = pd.to_datetime(historia.semana_inicio) + pd.Timedelta(days=2)
    historia["lote"] = historia.miercoles.dt.day <= 7
    trabajo = tabla.reset_index(drop=True)
    ajustado = pred.copy()
    for _, bloque in trabajo.groupby("id_departamento"):
        depto = int(bloque.id_departamento.iloc[0])
        pasado = historia[historia.id_departamento == depto]
        if pasado.empty:
            continue
        razon_mes = []
        for _, grupo in pasado.groupby(pasado.miercoles.dt.month):
            lote = grupo.loc[grupo.lote, "casos_dengue"]
            resto = grupo.loc[~grupo.lote, "casos_dengue"]
            if len(lote) < 4 or len(resto) < 4 or resto.mean() <= 0:
                continue
            razon_mes.append(float(lote.mean() / resto.mean()))
        if not razon_mes:
            continue
        razon = float(np.median(razon_mes))
        varianza = float(np.var(razon_mes))
        peso = 1.0 / (1.0 + varianza)
        factor = float(np.clip(1.0 + peso * (razon - 1.0), 0.85, 1.15))
        fechas = pd.to_datetime(bloque.semana_inicio) + pd.Timedelta(days=2)
        lote = (fechas.dt.day <= 7).to_numpy()
        idx = bloque.index.to_numpy()[lote]
        ajustado[idx] *= factor
    return ajustado


def diagnosticar(envio: pd.DataFrame) -> dict:
    """Project the H2 forecast onto submission directions whose public score is known."""
    archivos = {
        "b": RAIZ / "ensayos/juan/2026-10-31-horizonte-bases/submission.csv",
        "apilado": RAIZ / "entrega/submission_final_apilado.csv",
        "costa": RAIZ / "ensayos/juan/2026-11-04-sonda-regiones/submission_sonda_costa.csv",
        "interior": RAIZ / "ensayos/juan/2026-11-04-sonda-regiones/submission_sonda_interior.csv",
        "valle": RAIZ / "ensayos/jairo/2026-11-05-sonda-valle/submission_sonda_valle.csv",
        "multi": RAIZ / "ensayos/jairo/2026-11-05-mezcla-multimodelo/submission.csv",
    }
    if any(not ruta.exists() for ruta in archivos.values()):
        return {"disponible": False}
    puntajes = {"apilado": 1836.20689, "costa": 1842.97221, "interior": 2056.16650, "valle": 2005.86755, "multi": 1646.97677, "b": 2048.49606}
    tablas = {nombre: pd.read_csv(ruta) for nombre, ruta in archivos.items()}
    orden = tablas["b"]["id"]

    def vector(tabla: pd.DataFrame) -> np.ndarray:
        alineado = tablas["b"][["id"]].merge(tabla, on="id", how="left")
        return alineado.dengue.to_numpy(float)[h2.to_numpy()]

    def semana_de(ident: str) -> int:
        return int(str(ident).rsplit("_", 1)[-1])

    h2 = orden.map(semana_de) >= 27
    base = vector(tablas["apilado"])
    multi = vector(tablas["multi"])
    b = vector(tablas["b"])
    nuestro = vector(envio.rename(columns={"ID": "id"}) if "ID" in envio.columns else envio)
    n = len(base)

    def producto(d: np.ndarray, s0: float, s1: float, p0: np.ndarray) -> float:
        return float(d @ p0 - (n * (s1 - s0) - d @ d) / 2)

    d_costa = vector(tablas["costa"]) - base
    d_int = vector(tablas["interior"]) - base
    d_valle = vector(tablas["valle"]) - multi
    py_costa = producto(d_costa, puntajes["apilado"], puntajes["costa"], base)
    py_int = producto(d_int, puntajes["apilado"], puntajes["interior"], base)
    py_valle = producto(d_valle, puntajes["multi"], puntajes["valle"], multi)
    mascara_costa = np.abs(d_costa) > 1e-8
    mascara_valle = np.abs(d_valle) > 1e-8
    mascara_resto = ~mascara_costa & ~mascara_valle

    def media_region(v: np.ndarray, mascara: np.ndarray) -> float:
        return float(v[mascara].mean()) if mascara.any() else float("nan")

    k_costa = py_costa / (d_costa @ b) if mascara_costa.any() else float("nan")
    return {
        "disponible": True,
        "media_b": {"costa": media_region(b, mascara_costa), "valle": media_region(b, mascara_valle), "resto": media_region(b, mascara_resto)},
        "media_modelo": {"costa": media_region(nuestro, mascara_costa), "valle": media_region(nuestro, mascara_valle), "resto": media_region(nuestro, mascara_resto)},
        "k_costa_sobre_b": k_costa,
        "objetivo_valle_sobre_b": 0.525,
        "objetivo_resto_sobre_b": 1.066,
        "objetivo_costa_sobre_b": 0.95,
    }


def evaluar_referencia(marcos_anio: dict[int, pd.DataFrame]) -> dict[str, float]:
    """Original horizon blend, recomputed with the same weights path as the scored file when present."""
    ruta = RAIZ / "ensayos/juan/2026-11-04-red-dinamica-h1/b_walkforward.parquet"
    if not ruta.exists():
        return {}
    b = pd.read_parquet(ruta)
    salida = {}
    for mitad, anios in (("h1_2014", list(range(2014, 2024))), ("h1_2016", list(range(2016, 2024))), ("h2_2014", list(range(2014, 2024))), ("h2_2016", list(range(2016, 2024)))):
        errores = []
        for anio in anios:
            real = marcos_anio[anio]
            pred = b.loc[b.anio == anio, ["id_departamento", "semana", "b"]].merge(
                real[["id_departamento", "semana", "y"]], on=["id_departamento", "semana"], how="inner"
            )
            errores.append(_mse(pred.y.to_numpy(float), pred.b.to_numpy(float), pred.semana.to_numpy(int), "h1" if mitad.startswith("h1") else "h2"))
        salida[mitad] = float(np.mean(errores))
    return salida


def ejecutar() -> dict:
    horizonte = _horizonte()
    train, test, _ = horizonte.cargar()
    print("armando marcos", flush=True)
    marcos_anio = marcos(train, test)
    predicciones = {anio: {} for anio in ANIOS_META}
    for anio in ANIOS_META:
        t0 = time.time()
        pred_a = predecir_a(marcos_anio, anio)
        tabla = matriz_b(marcos_anio[anio], pred_a)
        predicciones[anio]["a"] = pred_a
        predicciones[anio]["tabla"] = tabla
        print(f"familia A {anio} agot={AGOT[anio]:.1f} {time.time() - t0:.1f}s", flush=True)
    for anio in ANIOS_META:
        if anio < 2015:
            predicciones[anio]["b"] = predicciones[anio]["a"].copy()
            continue
        t0 = time.time()
        tablas = {p: matriz_b(marcos_anio[p], predicciones[p]["a"]) for p in range(2014, anio)}
        tablas[anio] = predicciones[anio]["tabla"]
        predicciones[anio]["b"] = predecir_b(tablas, anio)
        print(f"familia B {anio} {time.time() - t0:.1f}s", flush=True)

    full = _panel(train, test)
    redes = predecir_redes(full, marcos_anio, ANIOS_META)
    for anio, familias in redes.items():
        predicciones[anio].update(familias)

    metricas = {"referencia_b": evaluar_referencia(marcos_anio), "familias": {}, "meta": {}}
    for nombre in ("a", "b", "mlp", "tcn", "gru"):
        metricas["familias"][nombre] = {}
        for mitad, anios in (("h1_2014", range(2014, 2024)), ("h1_2016", range(2016, 2024)), ("h2_2014", range(2014, 2024)), ("h2_2016", range(2016, 2024))):
            errores = []
            for anio in anios:
                tabla = predicciones[anio]["tabla"]
                errores.append(_mse(tabla.y.to_numpy(float), predicciones[anio][nombre], tabla.semana.to_numpy(int), "h1" if mitad.startswith("h1") else "h2"))
            metricas["familias"][nombre][mitad] = float(np.mean(errores))
        print(nombre, metricas["familias"][nombre], flush=True)

    nombres = ["a", "b", "mlp", "tcn", "gru"]
    meta_pred = {}
    empalmes = {}
    for anio in ANIOS_META:
        historial = []
        for previo in range(2014, anio):
            tabla = predicciones[previo]["tabla"]
            prob = tabla[[f"p{k}" for k in range(K_MAX)]].to_numpy(float)
            historial.append(({n: predicciones[previo][n] for n in nombres}, tabla.y.to_numpy(float), prob))
        if len(historial) < 2:
            meta_pred[anio] = predicciones[anio]["a"].copy()
            continue
        pesos = ajustar_meta(historial, nombres)
        tabla = predicciones[anio]["tabla"]
        prob = tabla[[f"p{k}" for k in range(K_MAX)]].to_numpy(float)
        crudo = aplicar_meta(pesos, {n: predicciones[anio][n] for n in nombres}, prob, nombres)
        y_real = tabla.y.to_numpy(float) if "y" in tabla.columns else np.zeros(len(tabla))
        fila = {"pred": crudo, "semana": tabla.semana.to_numpy(int), "depto": tabla.id_departamento.to_numpy(int), "y": y_real}
        pasados = []
        for previo in range(2014, anio):
            tabla_p = predicciones[previo]["tabla"]
            pasados.append(
                {
                    "pred": meta_pred[previo],
                    "semana": tabla_p.semana.to_numpy(int),
                    "depto": tabla_p.id_departamento.to_numpy(int),
                    "y": tabla_p.y.to_numpy(float),
                }
            )
        ajustado, info = empalme(pasados, fila)
        info["anio"] = anio
        empalmes[anio] = info
        meta_pred[anio] = bayes_calendario(train, tabla, ajustado, anio)

    for mitad, anios in (("h1_2014", range(2014, 2024)), ("h1_2016", range(2016, 2024)), ("h2_2014", range(2014, 2024)), ("h2_2016", range(2016, 2024))):
        errores = []
        for anio in anios:
            tabla = predicciones[anio]["tabla"]
            errores.append(_mse(tabla.y.to_numpy(float), meta_pred[anio], tabla.semana.to_numpy(int), "h1" if mitad.startswith("h1") else "h2"))
        metricas["meta"][mitad] = float(np.mean(errores))
    print("meta", metricas["meta"], flush=True)

    tabla_2025 = predicciones[2025]["tabla"]
    pred_2025 = np.clip(meta_pred[2025], 0, None)
    envio = pd.DataFrame({"id": tabla_2025.ID, "dengue": pred_2025})
    muestra = pd.read_csv(RAIZ / "data" / "sample_submission.csv")
    id_muestra = "id" if "id" in muestra.columns else "ID"
    envio = muestra[[id_muestra]].merge(envio, left_on=id_muestra, right_on="id", how="left")
    envio = envio.rename(columns={id_muestra: "id"})[["id", "dengue"]]
    if envio.dengue.isna().any():
        raise RuntimeError("El envio no cubre todas las filas de sample_submission.")
    envio.to_csv(CARPETA / "submission.csv", index=False)
    metricas["diagnostico_2025"] = diagnosticar(envio)
    metricas["empalme"] = {str(k): v for k, v in empalmes.items()}
    metricas["filas"] = int(len(envio))
    metricas["minimo"] = float(envio.dengue.min())
    metricas["media_h1_2025"] = float(pred_2025[tabla_2025.semana.to_numpy(int) <= 26].mean())
    metricas["media_h2_2025"] = float(pred_2025[tabla_2025.semana.to_numpy(int) >= 27].mean())
    (CARPETA / "resultados.json").write_text(json.dumps(metricas, indent=2, default=float), encoding="utf-8")
    piezas = []
    for anio in ANIOS_META:
        base = predicciones[anio]["tabla"]
        tabla = base[["id_departamento", "anio", "semana"]].copy()
        tabla["y"] = base["y"].to_numpy(float) if "y" in base.columns else np.nan
        for nombre in nombres:
            tabla[nombre] = predicciones[anio][nombre]
        tabla["meta"] = meta_pred[anio]
        piezas.append(tabla)
    pd.concat(piezas, ignore_index=True).to_parquet(CARPETA / "predicciones.parquet", index=False)
    print(json.dumps(metricas["meta"], indent=2), flush=True)
    print(json.dumps(metricas["diagnostico_2025"], indent=2), flush=True)
    return metricas


if __name__ == "__main__":
    sys.path.insert(0, str(RAIZ))
    ejecutar()
