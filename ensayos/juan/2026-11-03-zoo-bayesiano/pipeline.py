"""Order-statistic bases, post-peak decay, and a model zoo that is kept only if walk-forward improves."""

import importlib.util
import json
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import xgboost as xgb
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*ill-conditioned matrix.*")
optuna.logging.set_verbosity(optuna.logging.WARNING)

_RUTA_BASE = Path(__file__).resolve().parents[1] / "2026-10-31-horizonte-bases" / "pipeline.py"
_spec = importlib.util.spec_from_file_location("bases_horizonte", _RUTA_BASE)
base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(base)

CARPETA = Path("ensayos/juan/2026-11-03-zoo-bayesiano")
ENVIO_P1 = Path("ensayos/juan/2026-10-27-nivel-sin-brote/submission.csv")
ENVIO_P2 = Path("ensayos/juan/2026-10-31-horizonte-bases/submission.csv")
SEMIILLA = 42
EPOCAS = 5
OCULTO = 24
REDES = ["mlp", "rnn", "lstm", "gru", "tcn"]
TABLAS = ["ridge", "lightgbm", "xgboost"]
BASES_PUERTA = base.BASES + ["b_trough", "b_modern"]
REFERENCIA = {"tau": 14.0, "mezcla_costa": 0.65, "umbral_par": 1.85, "umbral_mega": 99.0, "tope_q4": 1.1}


def perfil_orden(hist: pd.DataFrame, anio: int, ventana: int, conservar: int) -> pd.Series:
    recorte = hist[(hist.anio >= anio - ventana) & (hist.anio < anio) & (hist.semana <= 52)]
    medias = recorte.groupby(["id_departamento", "anio"]).casos_dengue.mean().reset_index()
    pares = []
    for depto, grupo in medias.groupby("id_departamento"):
        orden = grupo.sort_values("casos_dengue")
        corte = max(1, min(conservar, len(orden)))
        for anio_bajo in orden.iloc[:corte].anio:
            pares.append((int(depto), int(anio_bajo)))
    indice = pd.MultiIndex.from_tuples(pares, names=["id_departamento", "anio"])
    elegido = recorte.set_index(["id_departamento", "anio"])
    elegido = elegido.loc[elegido.index.isin(indice)].reset_index()
    return base.suavizar(elegido.groupby(["id_departamento", "semana"]).casos_dengue.mean())


def anio_post_pico(hist: pd.DataFrame, anio: int) -> bool:
    previo = hist[hist.anio == anio - 1]
    if previo.empty:
        return False
    media = float(previo.casos_dengue.mean())
    mitad = float(previo.loc[previo.semana <= 26, "casos_dengue"].mean())
    segunda = float(previo.loc[previo.semana.between(27, 52), "casos_dengue"].mean())
    return media > 45.0 and mitad > 1.15 * max(segunda, 1.0)


def enriquecer(fuente: pd.DataFrame, anio: int, futuro: pd.DataFrame | None = None) -> pd.DataFrame:
    marco = base.preparar_anio(fuente, anio, futuro=futuro)
    historia = fuente[fuente.anio < anio]
    indice = pd.MultiIndex.from_arrays([marco.id_departamento, marco.semana])
    bajo_6 = perfil_orden(historia, anio, 6, 3).reindex(indice).fillna(0.0).to_numpy()
    bajo_4 = perfil_orden(historia, anio, 4, 2).reindex(indice).fillna(0.0).to_numpy()
    marco["b_low3_6"] = bajo_6
    marco["b_low2_4"] = bajo_4
    marco["b_trough"] = np.minimum(bajo_4, bajo_6)
    marco["b_modern"] = np.maximum(bajo_4, marco.b_rec6.to_numpy(float))
    nivel = marco.groupby("id_departamento").b_low3_6.mean().replace(0, 1.0)
    media_1 = historia[historia.anio == anio - 1].groupby("id_departamento").casos_dengue.mean()
    media_2 = historia[historia.anio == anio - 2].groupby("id_departamento").casos_dengue.mean()
    marco["r1_clean"] = marco.id_departamento.map((media_1 / nivel).replace([np.inf, -np.inf], np.nan)).fillna(1.0).to_numpy()
    marco["r2_clean"] = marco.id_departamento.map((media_2 / nivel).replace([np.inf, -np.inf], np.nan)).fillna(1.0).to_numpy()
    marco["nat_post_peak"] = 1.0 if anio_post_pico(historia, anio) else 0.0
    return marco


def estandar(marcos: dict[int, pd.DataFrame], anio: int) -> np.ndarray:
    previos = [marcos[anterior] for anterior in range(2010, anio)]
    peso_q1 = base.pesos_horizonte(previos, 1, 13)
    peso_q2 = base.pesos_horizonte(previos, 14, 26)
    peso_h2 = base.pesos_horizonte(previos, 27, 52)
    return np.clip(base.aplicar_pesos(marcos[anio], peso_q1, peso_q2, peso_h2), 0, None)


def decaer(marco: pd.DataFrame, previo: np.ndarray, tau: float, mezcla_costa: float, umbral_par: float, umbral_mega: float, tope_q4: float) -> np.ndarray:
    if float(marco.nat_post_peak.iloc[0]) < 0.5:
        return previo
    semana = marco.semana.to_numpy(float)
    r1 = marco.r1_clean.to_numpy(float)
    r2 = marco.r2_clean.to_numpy(float)
    cae = (marco.q4_over_h1.to_numpy(float) < tope_q4) & (((r1 > umbral_par) & (r2 > umbral_par)) | (r1 > umbral_mega))
    piso = marco.b_trough.to_numpy(float)
    moderno = marco.b_modern.to_numpy(float)
    crece = marco.b_low2_4.to_numpy(float) > 1.3 * marco.b_med_all.to_numpy(float)
    peso_interior = 1.0 - np.exp(-semana / tau)
    peso_costa = np.where((semana >= 27) & crece, mezcla_costa, 0.0)
    return np.where(cae, (1.0 - peso_interior) * previo + peso_interior * piso, (1.0 - peso_costa) * previo + peso_costa * moderno)


def media(tabla: dict[int, tuple[float, float]], anios: list[int], mitad: int) -> float:
    return float(np.mean([tabla[anio][mitad] for anio in anios]))


def resumir(tabla: dict[int, tuple[float, float]]) -> dict[str, float]:
    return {
        "mod_h1": media(tabla, base.ANIOS_MODERADOS, 0),
        "mod_h2": media(tabla, base.ANIOS_MODERADOS, 1),
        "post_h1": media(tabla, base.ANIOS_POST, 0),
        "post_h2": media(tabla, base.ANIOS_POST, 1),
    }


def columnas_zoo(marco: pd.DataFrame) -> list[str]:
    clima = [columna for columna in marco.columns if columna.startswith("z_") or columna.startswith("zlag_")]
    diferencias = ["d_" + nombre for nombre in BASES_PUERTA]
    return clima + [
        "seno",
        "coseno",
        "is_crash",
        "q4_over_h1",
        "prev_over_norm",
        "r1_clean",
        "r2_clean",
        "semana_norm",
        "es_h2",
        *diferencias,
    ]


def matriz_zoo(marco: pd.DataFrame, previo: np.ndarray) -> pd.DataFrame:
    tabla = marco.copy()
    tabla["prior"] = previo
    escala = np.maximum(tabla.dept_scale.to_numpy(float), 1.0)
    tabla["semana_norm"] = tabla.semana.to_numpy(float) / 52.0
    tabla["es_h2"] = (tabla.semana >= 27).astype(float)
    for nombre in BASES_PUERTA:
        tabla["d_" + nombre] = (tabla[nombre].to_numpy(float) - previo) / escala
    return tabla


def ventanas(tabla: pd.DataFrame, columnas: list[str]) -> tuple[np.ndarray, np.ndarray]:
    trabajo = tabla.reset_index(drop=True)
    rasgos_ancho = len(columnas)
    secuencia = np.zeros((len(trabajo), 8, rasgos_ancho), dtype=np.float32)
    bases = np.zeros((len(trabajo), len(BASES_PUERTA)), dtype=np.float32)
    for _, bloque in trabajo.groupby(["anio", "id_departamento"], sort=False):
        bloque = bloque.sort_values("semana")
        indice = bloque.index.to_numpy()
        rasgos = np.nan_to_num(bloque[columnas].to_numpy(np.float32))
        bases[indice] = np.nan_to_num(bloque[BASES_PUERTA].to_numpy(np.float32))
        for paso, fila in enumerate(indice):
            inicio = max(0, paso - 7)
            pedazo = rasgos[inicio : paso + 1]
            secuencia[fila, 8 - len(pedazo) :, :] = pedazo
    return secuencia, bases


class _ConvolucionCausal(nn.Module):
    def __init__(self, entrada: int, salida: int, nucleo: int, dilatacion: int):
        super().__init__()
        self.izquierda = (nucleo - 1) * dilatacion
        self.conv = nn.Conv1d(entrada, salida, nucleo, dilation=dilatacion)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.conv(F.pad(tensor, (self.izquierda, 0)))


class RedZoo(nn.Module):
    """A bounded gate over causal bases, plus a climate term centered at zero."""

    def __init__(self, kind: str, n_feat: int, n_bases: int):
        super().__init__()
        self.kind = kind
        ancho = OCULTO
        if kind == "mlp":
            self.cuerpo = nn.Sequential(nn.Flatten(), nn.Linear(8 * n_feat, ancho), nn.ReLU(), nn.Linear(ancho, ancho), nn.ReLU())
        elif kind == "rnn":
            self.cuerpo = nn.RNN(n_feat, ancho, batch_first=True)
        elif kind == "lstm":
            self.cuerpo = nn.LSTM(n_feat, ancho, batch_first=True)
        elif kind == "gru":
            self.cuerpo = nn.GRU(n_feat, ancho, batch_first=True)
        else:
            self.conv = nn.Sequential(
                _ConvolucionCausal(n_feat, ancho, 3, 1),
                nn.ReLU(),
                _ConvolucionCausal(ancho, ancho, 3, 2),
                nn.ReLU(),
            )
        self.compuerta = nn.Linear(ancho, n_bases)
        self.mezcla = nn.Linear(ancho, 1)
        self.clima = nn.Linear(ancho, 1)

    def forward(self, secuencia: torch.Tensor, bases: torch.Tensor, previo: torch.Tensor) -> torch.Tensor:
        if self.kind == "mlp":
            oculto = self.cuerpo(secuencia)
        elif self.kind == "tcn":
            oculto = self.conv(secuencia.transpose(1, 2))[:, :, -1]
        else:
            _, estado = self.cuerpo(secuencia)
            oculto = estado[0][-1] if self.kind == "lstm" else estado[-1]
        pesos = torch.softmax(self.compuerta(oculto), dim=1)
        mezcla = 0.35 * torch.sigmoid(self.mezcla(oculto)).squeeze(1)
        clima = 0.15 * torch.tanh(self.clima(oculto)).squeeze(1)
        nivel = (pesos * bases).sum(dim=1)
        return ((1.0 - mezcla) * previo + mezcla * nivel) * (1.0 + clima)


def entrenar_red(kind: str, secuencia: np.ndarray, bases: np.ndarray, previo: np.ndarray, objetivo: np.ndarray, escala: np.ndarray, dispositivo: torch.device) -> RedZoo:
    torch.manual_seed(SEMIILLA)
    red = RedZoo(kind, secuencia.shape[2], bases.shape[1]).to(dispositivo)
    optimizador = torch.optim.Adam(red.parameters(), lr=1e-3)
    x = torch.tensor(secuencia, dtype=torch.float32, device=dispositivo)
    b = torch.tensor(bases, dtype=torch.float32, device=dispositivo)
    p = torch.tensor(previo, dtype=torch.float32, device=dispositivo)
    y = torch.tensor(objetivo, dtype=torch.float32, device=dispositivo)
    s = torch.tensor(np.maximum(escala, 1.0), dtype=torch.float32, device=dispositivo)
    red.train()
    for _ in range(EPOCAS):
        orden = torch.randperm(len(x), device=dispositivo)
        for inicio in range(0, len(x), 512):
            lote = orden[inicio : inicio + 512]
            optimizador.zero_grad()
            predicho = red(x[lote], b[lote], p[lote])
            perdida = (((predicho - y[lote]) / s[lote]) ** 2).mean()
            perdida.backward()
            optimizador.step()
    red.eval()
    return red


def predecir_red(red: RedZoo, secuencia: np.ndarray, bases: np.ndarray, previo: np.ndarray, dispositivo: torch.device) -> np.ndarray:
    salidas = []
    with torch.no_grad():
        for inicio in range(0, len(secuencia), 1024):
            fin = inicio + 1024
            predicho = red(
                torch.tensor(secuencia[inicio:fin], dtype=torch.float32, device=dispositivo),
                torch.tensor(bases[inicio:fin], dtype=torch.float32, device=dispositivo),
                torch.tensor(previo[inicio:fin], dtype=torch.float32, device=dispositivo),
            )
            salidas.append(predicho.cpu().numpy())
    return np.clip(np.concatenate(salidas), 0, None)


def _ajustar_tabla(rasgos: np.ndarray, residuo: np.ndarray, peso: np.ndarray):
    escala_rasgos = StandardScaler()
    rasgos_z = escala_rasgos.fit_transform(rasgos)
    cresta = Ridge(alpha=30.0, fit_intercept=False)
    cresta.fit(rasgos_z, residuo, sample_weight=peso)
    arbol = lgb.LGBMRegressor(
        n_estimators=120,
        learning_rate=0.05,
        num_leaves=7,
        min_child_samples=60,
        subsample=0.85,
        colsample_bytree=0.8,
        reg_lambda=8.0,
        random_state=SEMIILLA,
        verbosity=-1,
    )
    arbol.fit(rasgos, residuo, sample_weight=peso)
    bosque = xgb.XGBRegressor(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=3,
        min_child_weight=30,
        subsample=0.85,
        colsample_bytree=0.8,
        reg_lambda=8.0,
        random_state=SEMIILLA,
        verbosity=0,
    )
    bosque.fit(rasgos, residuo, sample_weight=peso)
    return cresta, escala_rasgos, arbol, bosque


def entrenar_tablas(rasgos: np.ndarray, previo: np.ndarray, objetivo: np.ndarray, escala: np.ndarray, es_h2: np.ndarray):
    modelos = {}
    for mitad, mascara in (("h1", es_h2 < 0.5), ("h2", es_h2 >= 0.5)):
        escala_m = np.maximum(escala[mascara], 1.0)
        modelos[mitad] = _ajustar_tabla(rasgos[mascara], (objetivo[mascara] - previo[mascara]) / escala_m, escala_m**2)
    return modelos


def predecir_tablas(modelos, rasgos: np.ndarray, previo: np.ndarray, escala: np.ndarray, es_h2: np.ndarray) -> dict[str, np.ndarray]:
    salida = {nombre: np.zeros(len(previo), dtype=np.float64) for nombre in TABLAS}
    escala = np.maximum(escala, 1.0)
    for mitad, mascara in (("h1", es_h2 < 0.5), ("h2", es_h2 >= 0.5)):
        cresta, escala_rasgos, arbol, bosque = modelos[mitad]
        salida["ridge"][mascara] = previo[mascara] + cresta.predict(escala_rasgos.transform(rasgos[mascara])) * escala[mascara]
        salida["lightgbm"][mascara] = previo[mascara] + arbol.predict(rasgos[mascara]) * escala[mascara]
        salida["xgboost"][mascara] = previo[mascara] + bosque.predict(rasgos[mascara]) * escala[mascara]
    return {nombre: np.clip(valores, 0, None) for nombre, valores in salida.items()}


def _unir(paquete: dict[int, dict], anios: list[int], mascara_entreno: bool) -> dict[str, np.ndarray]:
    piezas = [paquete[anio] for anio in anios]
    def tomar(clave: str) -> np.ndarray:
        bloques = []
        for pieza in piezas:
            valores = pieza[clave]
            bloques.append(valores[pieza["entrenar"]] if mascara_entreno else valores)
        return np.concatenate(bloques)
    return {clave: tomar(clave) for clave in ("x", "b", "rasgos", "previo", "y", "escala", "h2")}


def predecir_zoo(paquete_train: dict[str, np.ndarray], actual: dict, dispositivo: torch.device) -> dict[str, np.ndarray]:
    predicciones = predecir_tablas(
        entrenar_tablas(paquete_train["rasgos"], paquete_train["previo"], paquete_train["y"], paquete_train["escala"], paquete_train["h2"]),
        actual["rasgos"],
        actual["previo"],
        actual["escala"],
        actual["h2"],
    )
    for kind in REDES:
        red = entrenar_red(kind, paquete_train["x"], paquete_train["b"], paquete_train["previo"], paquete_train["y"], paquete_train["escala"], dispositivo)
        predicciones[kind] = predecir_red(red, actual["x"], actual["b"], actual["previo"], dispositivo)
    return predicciones


def _solo_decaimiento(nombres: list[str]) -> dict[str, float]:
    return {nombre: (1.0 if nombre == "decaimiento" else 0.0) for nombre in nombres}


def mezclar_vector(previo: np.ndarray, miembros: dict[str, np.ndarray], pesos: dict[str, float]) -> np.ndarray:
    predicho = pesos.get("decaimiento", 0.0) * previo
    for nombre, valores in miembros.items():
        predicho = predicho + pesos.get(nombre, 0.0) * valores
    return predicho


def puntaje_mezcla(marcos: dict[int, pd.DataFrame], previos: dict[int, np.ndarray], zoo: dict[int, dict[str, np.ndarray]], pesos: dict[str, float]) -> dict[int, tuple[float, float]]:
    tabla = {}
    for anio in base.ANIOS_VALIDACION:
        tabla[anio] = base.evaluar(marcos[anio], mezclar_vector(previos[anio], zoo[anio], pesos))
    return tabla


def mse_tramo(marcos: dict[int, pd.DataFrame], predicciones: dict[int, np.ndarray], anios: list[int], inicio: int, fin: int) -> float:
    errores = []
    for anio in anios:
        mascara = marcos[anio].semana.between(inicio, fin).to_numpy()
        errores.append(mean_squared_error(marcos[anio].y.to_numpy(float)[mascara], predicciones[anio][mascara]))
    return float(np.mean(errores))


def buscar_tramo(
    nombres: list[str],
    marcos: dict[int, pd.DataFrame],
    previos: dict[int, np.ndarray],
    zoo: dict[int, dict[str, np.ndarray]],
    inicio: int,
    fin: int,
) -> dict[str, float]:
    estudio = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=SEMIILLA))

    def objetivo(trial: optuna.Trial) -> float:
        crudo = np.array([trial.suggest_float(nombre, 0.0, 1.0) for nombre in nombres])
        if float(crudo.sum()) <= 0:
            return 1e9
        pesos = {nombre: float(valor) for nombre, valor in zip(nombres, crudo / crudo.sum())}
        predicho = {anio: mezclar_vector(previos[anio], zoo[anio], pesos) for anio in base.ANIOS_VALIDACION}
        return 0.65 * mse_tramo(marcos, predicho, base.ANIOS_MODERADOS, inicio, fin) + 0.35 * mse_tramo(marcos, predicho, base.ANIOS_POST, inicio, fin)

    estudio.enqueue_trial(_solo_decaimiento(nombres))
    estudio.optimize(objetivo, n_trials=50)
    crudo = np.array([estudio.best_params[nombre] for nombre in nombres])
    pesos = {nombre: float(valor) if float(valor) > 0.01 else 0.0 for nombre, valor in zip(nombres, crudo / crudo.sum())}
    if sum(pesos.values()) <= 0:
        return _solo_decaimiento(nombres)
    pesos = {nombre: valor / sum(pesos.values()) for nombre, valor in pesos.items()}
    predicho = {anio: mezclar_vector(previos[anio], zoo[anio], pesos) for anio in base.ANIOS_VALIDACION}
    base_pred = {anio: previos[anio] for anio in base.ANIOS_VALIDACION}
    if mse_tramo(marcos, predicho, base.ANIOS_MODERADOS, inicio, fin) > mse_tramo(marcos, base_pred, base.ANIOS_MODERADOS, inicio, fin) - 5:
        return _solo_decaimiento(nombres)
    if mse_tramo(marcos, predicho, base.ANIOS_POST, inicio, fin) > mse_tramo(marcos, base_pred, base.ANIOS_POST, inicio, fin) + 20:
        return _solo_decaimiento(nombres)
    return pesos


def aplicar_horizontes(semana: np.ndarray, previo: np.ndarray, miembros: dict[str, np.ndarray], pesos_q1: dict[str, float], pesos_q2: dict[str, float], pesos_h2: dict[str, float]) -> np.ndarray:
    predicho = np.zeros(len(semana), dtype=np.float64)
    tramos = (
        (semana <= 13, pesos_q1),
        ((semana >= 14) & (semana <= 26), pesos_q2),
        (semana >= 27, pesos_h2),
    )
    for mascara, pesos in tramos:
        predicho[mascara] = mezclar_vector(previo[mascara], {nombre: valores[mascara] for nombre, valores in miembros.items()}, pesos)
    return predicho


def correr() -> None:
    dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("dispositivo", dispositivo, flush=True)
    fuente, futuro, muestra = base.cargar()
    marcos = {anio: enriquecer(fuente, anio) for anio in range(2010, 2025)}
    print("marcos", 2010, 2024, flush=True)
    brutos = {2010: np.clip(marcos[2010].b_end_scaled.to_numpy(float), 0, None)}
    for anio in range(2011, 2025):
        brutos[anio] = estandar(marcos, anio)
        print("peso", anio, flush=True)

    def aplicar(anio: int, params: dict) -> np.ndarray:
        return decaer(marcos[anio], brutos[anio], params["tau"], params["mezcla_costa"], params["umbral_par"], params["umbral_mega"], params["tope_q4"])

    referencia = {anio: base.evaluar(marcos[anio], aplicar(anio, REFERENCIA)) for anio in base.ANIOS_VALIDACION}
    print("referencia", {clave: round(valor, 1) for clave, valor in resumir(referencia).items()}, flush=True)
    if resumir(referencia)["mod_h2"] > 2400:
        raise RuntimeError("El decaimiento de referencia no reproduce el backtest.")

    def objetivo(trial: optuna.Trial) -> float:
        params = {
            "tau": trial.suggest_float("tau", 12.0, 22.0),
            "mezcla_costa": trial.suggest_float("mezcla_costa", 0.55, 0.8),
            "umbral_par": trial.suggest_float("umbral_par", 1.7, 2.1),
            "umbral_mega": trial.suggest_float("umbral_mega", 2.6, 6.0),
            "tope_q4": trial.suggest_float("tope_q4", 1.0, 1.15),
        }
        tabla = {anio: base.evaluar(marcos[anio], aplicar(anio, params)) for anio in base.ANIOS_VALIDACION}
        cifras = resumir(tabla)
        return 0.35 * cifras["mod_h2"] + 0.25 * cifras["post_h2"] + 0.25 * cifras["mod_h1"] + 0.15 * cifras["post_h1"]

    estudio = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=SEMIILLA))
    estudio.enqueue_trial({"tau": 14.0, "mezcla_costa": 0.65, "umbral_par": 1.85, "umbral_mega": 6.0, "tope_q4": 1.1})
    estudio.enqueue_trial({"tau": 14.0, "mezcla_costa": 0.75, "umbral_par": 1.85, "umbral_mega": 6.0, "tope_q4": 1.1})
    estudio.enqueue_trial({"tau": 16.0, "mezcla_costa": 0.72, "umbral_par": 1.85, "umbral_mega": 2.8, "tope_q4": 1.1})
    estudio.optimize(objetivo, n_trials=40)
    mejor = estudio.best_params
    print("optuna", {clave: round(valor, 3) for clave, valor in mejor.items()}, "objetivo", round(estudio.best_value, 1), flush=True)
    previos = {anio: aplicar(anio, mejor) for anio in range(2010, 2025)}
    base_puntos = {anio: base.evaluar(marcos[anio], previos[anio]) for anio in base.ANIOS_VALIDACION}
    print("decaimiento", {clave: round(valor, 1) for clave, valor in resumir(base_puntos).items()}, flush=True)

    columnas = columnas_zoo(matriz_zoo(marcos[2014], previos[2014]))
    paquete = {}
    for anio in range(2010, 2025):
        tabla = matriz_zoo(marcos[anio], previos[anio]).reset_index(drop=True)
        secuencia, bases_red = ventanas(tabla, columnas)
        paquete[anio] = {
            "tabla": tabla,
            "x": secuencia,
            "b": bases_red,
            "rasgos": np.nan_to_num(tabla[columnas].to_numpy(np.float32)),
            "previo": tabla.prior.to_numpy(np.float32),
            "y": tabla.y.to_numpy(np.float32),
            "escala": np.maximum(tabla.dept_scale.to_numpy(np.float32), 1.0),
            "entrenar": (~tabla.brote.to_numpy()).astype(bool),
            "h2": tabla.es_h2.to_numpy(np.float32),
        }
        print("paquete", anio, flush=True)

    zoo = {}
    for anio in base.ANIOS_VALIDACION:
        zoo[anio] = predecir_zoo(_unir(paquete, list(range(2010, anio)), True), paquete[anio], dispositivo)
        print(anio, "zoo listo", flush=True)

    nombres = ["decaimiento", *TABLAS, *REDES]
    pesos_q1 = buscar_tramo(nombres, marcos, previos, zoo, 1, 13)
    pesos_q2 = buscar_tramo(nombres, marcos, previos, zoo, 14, 26)
    pesos_h2 = buscar_tramo(nombres, marcos, previos, zoo, 27, 52)
    solo = resumir(base_puntos)
    mezcla_anios = {}
    for anio in base.ANIOS_VALIDACION:
        mezcla_anios[anio] = base.evaluar(
            marcos[anio],
            aplicar_horizontes(marcos[anio].semana.to_numpy(), previos[anio], zoo[anio], pesos_q1, pesos_q2, pesos_h2),
        )
    mezcla = resumir(mezcla_anios)
    print("q1", {nombre: round(valor, 3) for nombre, valor in pesos_q1.items() if valor > 0.01}, flush=True)
    print("q2", {nombre: round(valor, 3) for nombre, valor in pesos_q2.items() if valor > 0.01}, flush=True)
    print("h2", {nombre: round(valor, 3) for nombre, valor in pesos_h2.items() if valor > 0.01}, flush=True)
    print("mezcla", {clave: round(valor, 1) for clave, valor in mezcla.items()}, flush=True)

    marco_2025 = enriquecer(fuente, 2025, futuro=futuro)
    bruto_2025 = estandar({**marcos, 2025: marco_2025}, 2025)
    previo_2025 = decaer(marco_2025, bruto_2025, mejor["tau"], mejor["mezcla_costa"], mejor["umbral_par"], mejor["umbral_mega"], mejor["tope_q4"])
    tabla_2025 = matriz_zoo(marco_2025, previo_2025).reset_index(drop=True)
    secuencia_2025, bases_2025 = ventanas(tabla_2025, columnas)
    actual_2025 = {
        "x": secuencia_2025,
        "b": bases_2025,
        "rasgos": np.nan_to_num(tabla_2025[columnas].to_numpy(np.float32)),
        "previo": tabla_2025.prior.to_numpy(np.float32),
        "escala": np.maximum(tabla_2025.dept_scale.to_numpy(np.float32), 1.0),
        "h2": tabla_2025.es_h2.to_numpy(np.float32),
    }
    zoo_2025 = {}
    if any(pesos[nombre] > 0.01 for pesos in (pesos_q1, pesos_q2, pesos_h2) for nombre in nombres if nombre != "decaimiento"):
        zoo_2025 = predecir_zoo(_unir(paquete, list(range(2010, 2025)), True), actual_2025, dispositivo)
    final = aplicar_horizontes(marco_2025.semana.to_numpy(), previo_2025, zoo_2025, pesos_q1, pesos_q2, pesos_h2)

    envio = marco_2025[["ID", "semana", "departamento"]].copy()
    envio["dengue"] = np.clip(final, 0, None)
    p1 = pd.read_csv(base.raiz_repo() / ENVIO_P1).rename(columns={"id": "ID", "dengue": "p1"})
    p2 = pd.read_csv(base.raiz_repo() / ENVIO_P2).rename(columns={"id": "ID", "dengue": "p2"})
    auditoria = envio.merge(p1, on="ID").merge(p2, on="ID")
    mitad = auditoria.semana >= 27
    delta = auditoria.loc[mitad, "p2"].to_numpy(float) - auditoria.loc[mitad, "p1"].to_numpy(float)
    nuevo = auditoria.loc[mitad, "dengue"].to_numpy(float)
    alpha = float(np.mean((nuevo - auditoria.loc[mitad, "p1"].to_numpy(float)) * delta) / np.mean(delta**2))
    print("alpha", round(alpha, 4), flush=True)
    if not 0.72 <= alpha <= 0.88:
        print("alpha fuera de banda, el segundo semestre queda solo con el decaimiento", flush=True)
        pesos_h2 = _solo_decaimiento(nombres)
        envio["dengue"] = np.clip(aplicar_horizontes(marco_2025.semana.to_numpy(), previo_2025, zoo_2025, pesos_q1, pesos_q2, pesos_h2), 0, None)
        auditoria["dengue"] = envio.dengue.to_numpy()
        nuevo = auditoria.loc[mitad, "dengue"].to_numpy(float)
        alpha = float(np.mean((nuevo - auditoria.loc[mitad, "p1"].to_numpy(float)) * delta) / np.mean(delta**2))
        mezcla_anios = {
            anio: base.evaluar(
                marcos[anio],
                aplicar_horizontes(marcos[anio].semana.to_numpy(), previos[anio], zoo[anio], pesos_q1, pesos_q2, pesos_h2),
            )
            for anio in base.ANIOS_VALIDACION
        }
        mezcla = resumir(mezcla_anios)

    destino = base.raiz_repo() / CARPETA / "submission.csv"
    guardado = base.guardar(envio[["ID", "dengue"]], muestra, destino)
    comparacion = []
    for nombre in ["decaimiento", *TABLAS, *REDES]:
        if nombre == "decaimiento":
            cifras = solo
        else:
            cifras = resumir(puntaje_mezcla(marcos, previos, zoo, {"decaimiento": 0.7, nombre: 0.3}))
        comparacion.append({"modelo": nombre, **{clave: round(valor, 1) for clave, valor in cifras.items()}})
    niveles = (
        auditoria.loc[auditoria.semana >= 27]
        .groupby("departamento")[["p1", "p2", "dengue"]]
        .mean()
        .sort_values("p2", ascending=False)
        .round(1)
    )
    print(niveles.head(12).to_string(), flush=True)
    print("filas", len(guardado), "media", round(float(guardado.dengue.mean()), 2), "h2", round(float(auditoria.loc[mitad, "dengue"].mean()), 2), flush=True)
    resultado = {
        "dispositivo": str(dispositivo),
        "referencia": {clave: round(valor, 2) for clave, valor in resumir(referencia).items()},
        "optuna": {clave: round(float(valor), 4) for clave, valor in mejor.items()},
        "decaimiento": {clave: round(valor, 2) for clave, valor in solo.items()},
        "pesos_q1": {clave: round(valor, 4) for clave, valor in pesos_q1.items()},
        "pesos_q2": {clave: round(valor, 4) for clave, valor in pesos_q2.items()},
        "pesos_h2": {clave: round(valor, 4) for clave, valor in pesos_h2.items()},
        "mezcla": {clave: round(valor, 2) for clave, valor in mezcla.items()},
        "alpha_h2": round(alpha, 4),
        "comparacion": comparacion,
        "anios": [
            {"anio": anio, "h1": round(base_puntos[anio][0], 1), "h2": round(base_puntos[anio][1], 1)}
            for anio in base.ANIOS_VALIDACION
        ],
        "niveles_h2": niveles.reset_index().to_dict(orient="records"),
        "media": round(float(guardado.dengue.mean()), 2),
        "media_h1": round(float(auditoria.loc[~mitad, "dengue"].mean()), 2),
        "media_h2": round(float(auditoria.loc[mitad, "dengue"].mean()), 2),
    }
    (base.raiz_repo() / CARPETA / "resultados.json").write_text(json.dumps(resultado, indent=2), encoding="utf-8")


if __name__ == "__main__":
    correr()
