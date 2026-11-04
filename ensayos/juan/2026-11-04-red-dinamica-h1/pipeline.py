"""Red de dinamica para el primer semestre.

Cada ejemplo es una ventana: las 52 semanas que terminan en una semana cualquiera (el origen) y las 26
semanas siguientes como objetivo. La red ve los casos de la ventana relativos al nivel reciente, el clima
como anomalia, el calendario y el perfil estacional del departamento; predice la razon logaritmica entre
cada semana futura y ese nivel. Se entrena con perdida de Poisson, que apunta a la media condicional, la
misma que premia el MSE.

La evaluacion es walk-forward por ano: para el ano Y solo se entrenan ventanas cuyo objetivo termina antes
del 1 de enero de Y, y se pronostica desde la ultima semana de Y-1, igual que en 2025.
"""

import argparse
import json
import sys
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

CARPETA = Path("ensayos/juan/2026-11-04-red-dinamica-h1")
CARPETA_B = Path("ensayos/juan/2026-10-31-horizonte-bases")
HORIZONTE = 26
VENTANA = 52
CLIMA = ["precip_chirps_mm", "temp_media_C", "temp_min_C", "punto_rocio_C", "humedad_suelo_capa1"]
ANIOS_CLIMATOLOGIA = range(2007, 2014)
ANIOS_PERFIL = 8
DISPOSITIVO = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LOTE = 512


@dataclass
class Config:
    familia: str
    oculto: int = 32
    capas: int = 1
    dropout: float = 0.1
    lr: float = 2e-3
    wd: float = 1e-4
    kernel: int = 3
    cabeza: int = 32
    gamma: float = 0.0
    max_epocas: int = 40
    paciencia: int = 6
    salida: str = "compuerta"
    residuo: float = 0.5
    peso_fin: float = 1.0


CONFIGS = {
    "mlp": Config("mlp", oculto=64, dropout=0.2, lr=1e-3),
    "rnn": Config("rnn", oculto=32),
    "lstm": Config("lstm", oculto=32),
    "gru": Config("gru", oculto=32),
    "tcn": Config("tcn", oculto=32, capas=4, lr=1e-3),
}


# %% datos
def raiz_repo() -> Path:
    aqui = Path.cwd().resolve()
    for carpeta in [aqui, *list(aqui.parents)[:5]]:
        if (carpeta / "data" / "train.parquet").exists():
            return carpeta
    raise FileNotFoundError("No se encontro data/train.parquet.")


def cargar(raiz: Path) -> pd.DataFrame:
    train = pd.read_parquet(raiz / "data" / "train.parquet")
    test = pd.read_parquet(raiz / "data" / "test.parquet").assign(casos_dengue=np.nan)
    return pd.concat([train, test], ignore_index=True).sort_values(["id_departamento", "semana_inicio"]).reset_index(drop=True)


class Panel:
    """Series semanales de los 31 departamentos en una grilla [departamento, tiempo]."""

    def __init__(self, full: pd.DataFrame, calmo: bool = False):
        self.calmo = calmo
        self.deptos = np.sort(full.id_departamento.unique())
        self.D = len(self.deptos)
        self.T = len(full) // self.D
        primera = full[full.id_departamento == self.deptos[0]]
        for d in self.deptos[1:]:
            if not np.array_equal(full.loc[full.id_departamento == d, "semana_inicio"].to_numpy(), primera.semana_inicio.to_numpy()):
                raise RuntimeError("Los departamentos no comparten el calendario.")
        self.anio = primera.anio.to_numpy()
        self.semana = primera.semana.to_numpy()
        self.inicio = pd.to_datetime(primera.semana_inicio.to_numpy())
        self.Y = full.casos_dengue.to_numpy(float).reshape(self.D, self.T)
        self.ids = full.ID.to_numpy().reshape(self.D, self.T)
        self.nombres = full.drop_duplicates("id_departamento").set_index("id_departamento").departamento.loc[self.deptos].to_numpy()
        clima = full[CLIMA].to_numpy(float).reshape(self.D, self.T, len(CLIMA))
        self.calendario = self._calendario()
        self.clima_z = self._anomalias(clima)
        self.perfiles, self.normal = self._perfiles()

    def _calendario(self) -> np.ndarray:
        angulo = 2 * np.pi * (self.semana - 1) / 52.18
        miercoles = self.inicio + pd.Timedelta(days=2)
        lote = (miercoles.day <= 7).astype(float)
        fin_anio = (((self.inicio.month == 12) & (self.inicio.day >= 26)) | ((self.inicio.month == 1) & (self.inicio.day == 1))).astype(float)
        return np.stack([np.sin(angulo), np.cos(angulo), lote, fin_anio], axis=1)

    def _anomalias(self, clima: np.ndarray) -> np.ndarray:
        """Anomalia contra la climatologia 2007-2013 del departamento y la semana. Ningun ano validado entra."""
        semana = np.minimum(self.semana, 52)
        base = np.isin(self.anio, list(ANIOS_CLIMATOLOGIA))
        z = np.zeros_like(clima)
        for d in range(self.D):
            escala = clima[d, base].std(axis=0)
            escala = np.where(escala < 1e-6, 1.0, escala)
            for s in range(1, 53):
                media = clima[d, base & (semana == s)].mean(axis=0)
                filas = semana == s
                z[d, filas] = (clima[d, filas] - media) / escala
        return np.clip(z, -4, 4)

    def _perfiles(self) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
        """Perfil estacional y nivel normal para cada ano de inicio P, solo con anos completos anteriores a P.

        Con perfil calmo, un ano del departamento no entra si su media supera 2 veces la mediana de las medias
        anuales de la ventana: el perfil describe un ano sin brote, que es a donde vuelve la serie.
        """
        perfiles, normal = {}, {}
        semana = np.minimum(self.semana, 52)
        for P in range(2008, int(self.anio.max()) + 1):
            anios = [a for a in range(P - ANIOS_PERFIL, P) if a >= 2007]
            tabla = np.full((self.D, 53), np.nan)
            normal[P] = np.zeros(self.D)
            for d in range(self.D):
                usar = anios
                if self.calmo and len(anios) >= 3:
                    medias = np.array([np.nanmean(self.Y[d, self.anio == a]) for a in anios])
                    usar = [a for a, m in zip(anios, medias) if m <= 2.0 * np.median(medias)] or anios
                mascara = np.isin(self.anio, usar)
                for s in range(1, 53):
                    tabla[d, s] = np.nanmedian(self.Y[d, mascara & (semana == s)])
                normal[P][d] = np.nanmedian(self.Y[d, mascara])
            circular = np.concatenate([tabla[:, 51:53], tabla[:, 1:53], tabla[:, 1:3]], axis=1)
            suave = np.stack([circular[:, i : i + 5].mean(axis=1) for i in range(52)], axis=1)
            perfil = np.zeros((self.D, 54))
            perfil[:, 1:53] = suave
            perfil[:, 53] = suave[:, 51]
            perfiles[P] = perfil
        return perfiles, normal

    def ultimo_indice(self, anio: int) -> int:
        return int(np.where(self.anio == anio)[0].max())


def ventanas(panel: Panel) -> dict[str, np.ndarray]:
    """Todas las ventanas posibles. Las que tienen objetivo desconocido (2025) solo sirven para pronosticar."""
    origenes = np.arange(VENTANA - 1, panel.T - HORIZONTE)
    origenes = origenes[panel.anio[origenes + 1] >= 2008]  # el perfil necesita al menos un ano completo previo
    D, n = panel.D, len(origenes)
    enc = np.zeros((D, n, VENTANA, 1 + panel.calendario.shape[1] + len(CLIMA)), np.float32)
    dec = np.zeros((D, n, HORIZONTE, panel.calendario.shape[1] + len(CLIMA) + 2), np.float32)
    est = np.zeros((D, n, 4), np.float32)
    obj = np.zeros((D, n, HORIZONTE), np.float32)
    nivel = np.zeros((D, n), np.float32)
    logY = np.log1p(panel.Y)
    for j, t in enumerate(origenes):
        pasado = slice(t - VENTANA + 1, t + 1)
        futuro = slice(t + 1, t + 1 + HORIZONTE)
        L = np.median(panel.Y[:, t - 7 : t + 1], axis=1)
        logL = np.log1p(L)
        P = int(panel.anio[t + 1])
        enc[:, j, :, 0] = logY[:, pasado] - logL[:, None]
        enc[:, j, :, 1:5] = panel.calendario[pasado][None]
        enc[:, j, :, 5:] = panel.clima_z[:, pasado]
        dec[:, j, :, 0:4] = panel.calendario[futuro][None]
        dec[:, j, :, 4:9] = panel.clima_z[:, futuro]
        semanas = panel.semana[futuro]
        dec[:, j, :, 9] = np.log1p(panel.perfiles[P][:, semanas]) - logL[:, None]
        dec[:, j, :, 10] = np.arange(1, HORIZONTE + 1)[None] / HORIZONTE
        media4 = pd.DataFrame(panel.Y[:, pasado].T).rolling(4).mean().max().to_numpy()
        normal = panel.normal[P]
        est[:, j, 0] = logL / 7.0
        est[:, j, 1] = logL - np.log1p(normal)
        est[:, j, 2] = np.log1p(media4) - logL
        est[:, j, 3] = np.log1p(panel.Y[:, pasado].sum(axis=1)) - np.log1p(VENTANA * normal)
        obj[:, j] = panel.Y[:, futuro]
        nivel[:, j] = logL
    semana_origen = np.broadcast_to(panel.semana[origenes][None], (D, n)).astype(np.float32)
    return {"origenes": origenes, "enc": enc, "dec": dec, "est": est, "obj": obj, "nivel": nivel, "semana_origen": semana_origen}


# %% modelos
class BloqueTCN(nn.Module):
    def __init__(self, entrada: int, salida: int, kernel: int, dilatacion: int, dropout: float):
        super().__init__()
        self.recorte = (kernel - 1) * dilatacion
        self.c1 = nn.Conv1d(entrada, salida, kernel, dilation=dilatacion, padding=self.recorte)
        self.c2 = nn.Conv1d(salida, salida, kernel, dilation=dilatacion, padding=self.recorte)
        self.residuo = nn.Conv1d(entrada, salida, 1) if entrada != salida else nn.Identity()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        o = self.dropout(torch.relu(self.c1(x)[..., : -self.recorte]))
        o = self.dropout(torch.relu(self.c2(o)[..., : -self.recorte]))
        return torch.relu(o + self.residuo(x))


class Pronosticador(nn.Module):
    def __init__(self, cfg: Config, n_enc: int, n_est: int, n_dec: int):
        super().__init__()
        self.familia = cfg.familia
        if cfg.familia == "mlp":
            self.codificador = nn.Sequential(
                nn.Flatten(), nn.Linear(VENTANA * n_enc, cfg.oculto), nn.ReLU(), nn.Dropout(cfg.dropout), nn.Linear(cfg.oculto, cfg.oculto), nn.ReLU()
            )
        elif cfg.familia in ("rnn", "lstm", "gru"):
            celda = {"rnn": nn.RNN, "lstm": nn.LSTM, "gru": nn.GRU}[cfg.familia]
            self.codificador = celda(n_enc, cfg.oculto, num_layers=cfg.capas, batch_first=True, dropout=cfg.dropout if cfg.capas > 1 else 0.0)
        elif cfg.familia == "tcn":
            bloques, entrada = [], n_enc
            for nivel in range(cfg.capas):
                bloques.append(BloqueTCN(entrada, cfg.oculto, cfg.kernel, 2**nivel, cfg.dropout))
                entrada = cfg.oculto
            self.codificador = nn.Sequential(*bloques)
        else:
            raise ValueError(cfg.familia)
        self.salida = cfg.salida
        self.residuo = cfg.residuo
        self.cabeza = nn.Sequential(
            nn.Linear(cfg.oculto + n_est + n_dec, cfg.cabeza), nn.ReLU(), nn.Dropout(cfg.dropout), nn.Linear(cfg.cabeza, 2)
        )

    def codificar(self, x: torch.Tensor) -> torch.Tensor:
        if self.familia == "mlp":
            return self.codificador(x)
        if self.familia == "tcn":
            return self.codificador(x.transpose(1, 2))[:, :, -1]
        salida, _ = self.codificador(x)
        return salida[:, -1]

    def forward(self, enc: torch.Tensor, est: torch.Tensor, dec: torch.Tensor) -> torch.Tensor:
        z = self.codificar(enc)
        pasos = dec.shape[1]
        u = torch.cat([z.unsqueeze(1).expand(-1, pasos, -1), est.unsqueeze(1).expand(-1, pasos, -1), dec], dim=2)
        salida = self.cabeza(u)
        if self.salida == "libre":
            return 4.0 * torch.tanh(salida[..., 0] / 4.0)
        # Compuerta: entre la persistencia del nivel actual (0) y el perfil normal (dec[..., 9]), mas un residuo acotado.
        perfil = dec[..., 9]
        return torch.sigmoid(salida[..., 0]) * perfil + self.residuo * torch.tanh(salida[..., 1])


# %% entrenamiento
class Tensores:
    def __init__(self, V: dict[str, np.ndarray]):
        D, n = V["nivel"].shape
        aplanar = lambda a: torch.tensor(a.reshape(D * n, *a.shape[2:]), device=DISPOSITIVO)
        self.enc, self.dec, self.est = aplanar(V["enc"]), aplanar(V["dec"]), aplanar(V["est"])
        self.obj = aplanar(np.nan_to_num(V["obj"], nan=0.0))
        self.nivel = aplanar(V["nivel"])
        self.semana_origen = aplanar(np.ascontiguousarray(V["semana_origen"]))
        self.D, self.n = D, n

    def indice(self, d: np.ndarray, j: np.ndarray) -> torch.Tensor:
        return torch.tensor(d * self.n + j, device=DISPOSITIVO, dtype=torch.long)


def perdida(modelo: Pronosticador, X: Tensores, ii: torch.Tensor, cfg: Config) -> torch.Tensor:
    r = modelo(X.enc[ii], X.est[ii], X.dec[ii])
    eta = r + X.nivel[ii, None]
    nll = torch.exp(eta) - X.obj[ii] * eta
    peso = torch.exp(cfg.gamma * X.nivel[ii]) * torch.where(X.semana_origen[ii] >= 48, cfg.peso_fin, 1.0)
    return (nll.mean(dim=1) * peso).sum() / peso.sum()


def ajustar(cfg: Config, X: Tensores, entrenar: torch.Tensor, epocas: int, semilla: int, validar: torch.Tensor | None = None):
    torch.manual_seed(semilla)
    np.random.seed(semilla)
    modelo = Pronosticador(cfg, X.enc.shape[-1], X.est.shape[-1], X.dec.shape[-1]).to(DISPOSITIVO)
    opt = torch.optim.AdamW(modelo.parameters(), lr=cfg.lr, weight_decay=cfg.wd)
    curva, mejor, sin_mejora = [], np.inf, 0
    generador = torch.Generator(device=DISPOSITIVO).manual_seed(semilla)
    for _ in range(epocas):
        modelo.train()
        orden = entrenar[torch.randperm(len(entrenar), device=DISPOSITIVO, generator=generador)]
        for inicio in range(0, len(orden), LOTE):
            ii = orden[inicio : inicio + LOTE]
            loss = perdida(modelo, X, ii, cfg)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(modelo.parameters(), 1.0)
            opt.step()
        if validar is not None:
            modelo.eval()
            with torch.no_grad():
                valor = float(perdida(modelo, X, validar, cfg))
            curva.append(valor)
            if valor < mejor - 1e-4:
                mejor, sin_mejora = valor, 0
            else:
                sin_mejora += 1
                if sin_mejora >= cfg.paciencia:
                    break
    return modelo, curva


def predecir(modelo: Pronosticador, X: Tensores, ii: torch.Tensor) -> np.ndarray:
    modelo.eval()
    with torch.no_grad():
        r = modelo(X.enc[ii], X.est[ii], X.dec[ii])
        return torch.exp(r + X.nivel[ii, None]).cpu().numpy()


# %% walk-forward
def correr_anio(cfg: Config, panel: Panel, V: dict, X: Tensores, anio: int, semillas: list[int], epocas_fijas: int | None = None) -> tuple[np.ndarray, dict]:
    """Pronostico de las semanas 1-26 de `anio` desde la ultima semana de anio-1, sin ver nada posterior."""
    origenes = V["origenes"]
    fin_previo = panel.ultimo_indice(anio - 1)
    fin_inner = panel.ultimo_indice(anio - 2)
    deptos = np.arange(panel.D)
    rejilla_d, rejilla_j = np.meshgrid(deptos, np.arange(len(origenes)), indexing="ij")
    ultimo_objetivo = origenes[rejilla_j] + HORIZONTE
    todo = X.indice(rejilla_d[ultimo_objetivo <= fin_previo], rejilla_j[ultimo_objetivo <= fin_previo])
    inner = X.indice(rejilla_d[ultimo_objetivo <= fin_inner], rejilla_j[ultimo_objetivo <= fin_inner])
    val_mask = (ultimo_objetivo > fin_inner) & (ultimo_objetivo <= fin_previo)
    validar = X.indice(rejilla_d[val_mask], rejilla_j[val_mask])
    j_origen = int(np.where(origenes == fin_previo)[0][0])
    pronostico = X.indice(deptos, np.full(panel.D, j_origen))

    if epocas_fijas is None:
        _, curva = ajustar(cfg, X, inner, cfg.max_epocas, semillas[0], validar)
        epocas = int(np.argmin(curva)) + 1
    else:
        curva, epocas = [np.nan], epocas_fijas
    predicciones = []
    for semilla in semillas:
        modelo, _ = ajustar(cfg, X, todo, epocas, semilla)
        predicciones.append(predecir(modelo, X, pronostico))
    return np.mean(predicciones, axis=0), {"epocas": epocas, "ventanas": int(len(todo)), "val": round(min(curva), 4)}


def real_h1(panel: Panel, anio: int) -> np.ndarray:
    inicio = panel.ultimo_indice(anio - 1) + 1
    return panel.Y[:, inicio : inicio + HORIZONTE]


def bases_b(raiz: Path) -> pd.DataFrame:
    """Prediccion walk-forward del envio de horizontes (B), la referencia a batir."""
    cache = raiz / CARPETA / "b_walkforward.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    import os

    sys.path.insert(0, str(raiz / CARPETA_B))
    os.chdir(raiz)
    import pipeline as pb  # noqa: E402

    train, test, _ = pb.cargar()
    marcos = {a: pb.preparar_anio(train, a) for a in range(2010, 2025)}
    piezas = []
    for anio in range(2011, 2026):
        previos = [marcos[a] for a in range(2010, anio)]
        pesos = [pb.pesos_horizonte(previos, a, b) for a, b in ((1, 13), (14, 26), (27, 52))]
        actual = marcos[anio] if anio < 2025 else pb.preparar_anio(train, 2025, futuro=test)
        tabla = actual[["id_departamento", "anio", "semana"]].copy()
        tabla["b"] = np.clip(pb.aplicar_pesos(actual, *pesos), 0, None)
        piezas.append(tabla)
    salida = pd.concat(piezas, ignore_index=True)
    salida.to_parquet(cache)
    return salida


def matriz_b(B: pd.DataFrame, panel: Panel, anio: int) -> np.ndarray:
    tabla = B[(B.anio == anio) & (B.semana <= HORIZONTE)].pivot(index="id_departamento", columns="semana", values="b")
    return tabla.loc[panel.deptos].to_numpy(float)


def main() -> None:
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser()
    parser.add_argument("--familias", nargs="+", default=list(CONFIGS))
    parser.add_argument("--anios", nargs="+", type=int, default=list(range(2011, 2026)))
    parser.add_argument("--semillas", nargs="+", type=int, default=[0, 1])
    parser.add_argument("--gamma", type=float, default=None)
    parser.add_argument("--etiqueta", default="base")
    parser.add_argument("--epocas", type=int, default=None)
    parser.add_argument("--residuo", type=float, default=None)
    parser.add_argument("--peso_fin", type=float, default=None)
    parser.add_argument("--perfil", choices=["mediana", "calmo"], default="mediana")
    parser.add_argument("--salida", choices=["compuerta", "libre"], default=None)
    args = parser.parse_args()

    raiz = raiz_repo()
    panel = Panel(cargar(raiz), calmo=args.perfil == "calmo")
    V = ventanas(panel)
    X = Tensores(V)
    B = bases_b(raiz)
    print(f"dispositivo {DISPOSITIVO} | ventanas {V['enc'].shape[0] * V['enc'].shape[1]} | entrada {V['enc'].shape[-1]} | futuro {V['dec'].shape[-1]}", flush=True)

    filas, pronosticos = [], []
    for familia in args.familias:
        cfg = CONFIGS[familia]
        cambios = {k: v for k, v in (("gamma", args.gamma), ("residuo", args.residuo), ("peso_fin", args.peso_fin), ("salida", args.salida)) if v is not None}
        cfg = Config(**{**asdict(cfg), **cambios})
        for anio in args.anios:
            inicio = time.time()
            pred, info = correr_anio(cfg, panel, V, X, anio, args.semillas, args.epocas)
            fila = {"familia": familia, "anio": anio, **info, "seg": round(time.time() - inicio, 1)}
            if anio < 2025:
                real = real_h1(panel, anio)
                fila["mse_red"] = float(np.mean((pred - real) ** 2))
                fila["mse_b"] = float(np.mean((matriz_b(B, panel, anio) - real) ** 2))
            filas.append(fila)
            print(fila, flush=True)
            for d in range(panel.D):
                for h in range(HORIZONTE):
                    pronosticos.append({"familia": familia, "anio": anio, "id_departamento": int(panel.deptos[d]), "semana": h + 1, "pred": float(pred[d, h])})
    destino = raiz / CARPETA
    pd.DataFrame(filas).to_csv(destino / f"walkforward_{args.etiqueta}.csv", index=False)
    pd.DataFrame(pronosticos).to_parquet(destino / f"predicciones_{args.etiqueta}.parquet")
    with open(destino / f"config_{args.etiqueta}.json", "w", encoding="utf-8") as archivo:
        json.dump({"familias": args.familias, "cli": vars(args), "configs": {f: asdict(CONFIGS[f]) for f in args.familias}}, archivo, indent=1)


if __name__ == "__main__":
    main()
