"""Red hibrida autoregresiva para todo 2025.

Codificador mixto sobre las 52 semanas previas:
- `tcn_lstm`: convoluciones causales dilatadas (TCN) y encima una LSTM que lee esa secuencia;
- `gru_lstm`: una GRU y encima una LSTM.
Decodificador autoregresivo: una celda LSTM avanza semana a semana durante 52 semanas. En cada paso recibe su
propia prediccion de la semana anterior, el clima y el calendario de esa semana, el perfil estacional y las
variables estaticas. La salida usa la misma compuerta entre persistencia y perfil que la etapa 4, y la perdida
es Poisson. Se entrena en modo libre (siempre con sus propias predicciones), igual que pronostica.

Usa el pipeline de `2026-11-04-red-dinamica-h1` para los datos y la evaluacion walk-forward.
"""

import argparse
import importlib.util
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

CARPETA = Path("ensayos/juan/2026-11-04-red-hibrida")
PIPELINE = Path("ensayos/juan/2026-11-04-red-dinamica-h1/pipeline.py")


def raiz_repo() -> Path:
    aqui = Path.cwd().resolve()
    for carpeta in [aqui, *list(aqui.parents)[:5]]:
        if (carpeta / "data" / "train.parquet").exists():
            return carpeta
    raise FileNotFoundError("No se encontro data/train.parquet.")


def cargar_pipeline(raiz: Path):
    spec = importlib.util.spec_from_file_location("pipeline_red", raiz / PIPELINE)
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    return modulo


class Hibrida(nn.Module):
    def __init__(self, pipe, tipo: str, n_enc: int, n_est: int, n_dec: int, oculto: int = 32, dropout: float = 0.1, residuo: float = 0.5):
        super().__init__()
        self.tipo, self.residuo = tipo, residuo
        if tipo == "tcn_lstm":
            bloques, entrada = [], n_enc
            for nivel in range(4):
                bloques.append(pipe.BloqueTCN(entrada, oculto, 3, 2**nivel, dropout))
                entrada = oculto
            self.convoluciones = nn.Sequential(*bloques)
            self.lstm = nn.LSTM(oculto, oculto, batch_first=True)
        elif tipo == "gru_lstm":
            self.gru = nn.GRU(n_enc, oculto, batch_first=True)
            self.lstm = nn.LSTM(oculto, oculto, batch_first=True)
        else:
            raise ValueError(tipo)
        self.celda = nn.LSTMCell(1 + n_dec + n_est, oculto)
        self.cabeza = nn.Sequential(nn.Linear(oculto + n_dec + n_est, 32), nn.ReLU(), nn.Dropout(dropout), nn.Linear(32, 2))

    def codificar(self, enc: torch.Tensor):
        if self.tipo == "tcn_lstm":
            secuencia = self.convoluciones(enc.transpose(1, 2)).transpose(1, 2)
        else:
            secuencia, _ = self.gru(enc)
        _, (h, c) = self.lstm(secuencia)
        return h[-1], c[-1]

    def forward(self, enc: torch.Tensor, est: torch.Tensor, dec: torch.Tensor) -> torch.Tensor:
        h, c = self.codificar(enc)
        previo = enc[:, -1, 0:1]  # log del ultimo conteo observado frente al nivel actual
        salidas = []
        for paso in range(dec.shape[1]):
            entrada = torch.cat([previo, dec[:, paso], est], dim=1)
            h, c = self.celda(entrada, (h, c))
            crudo = self.cabeza(torch.cat([h, dec[:, paso], est], dim=1))
            r = torch.sigmoid(crudo[:, 0]) * dec[:, paso, 9] + self.residuo * torch.tanh(crudo[:, 1])
            salidas.append(r)
            previo = r.unsqueeze(1)
        return torch.stack(salidas, dim=1)


def ajustar(pipe, tipo, X, entrenar, epocas, semilla, lr=2e-3):
    torch.manual_seed(semilla)
    np.random.seed(semilla)
    cfg = pipe.Config("gru", lr=lr)
    modelo = Hibrida(pipe, tipo, X.enc.shape[-1], X.est.shape[-1], X.dec.shape[-1]).to(pipe.DISPOSITIVO)
    opt = torch.optim.AdamW(modelo.parameters(), lr=lr, weight_decay=1e-4)
    generador = torch.Generator(device=pipe.DISPOSITIVO).manual_seed(semilla)
    for _ in range(epocas):
        modelo.train()
        orden = entrenar[torch.randperm(len(entrenar), device=pipe.DISPOSITIVO, generator=generador)]
        for inicio in range(0, len(orden), pipe.LOTE):
            loss = pipe.perdida(modelo, X, orden[inicio : inicio + pipe.LOTE], cfg)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(modelo.parameters(), 1.0)
            opt.step()
    return modelo


def pronosticar(pipe, tipo, panel, V, X, anio, semillas, epocas):
    origenes = V["origenes"]
    fin_previo = panel.ultimo_indice(anio - 1)
    rejilla_d, rejilla_j = np.meshgrid(np.arange(panel.D), np.arange(len(origenes)), indexing="ij")
    valido = origenes[rejilla_j] + pipe.HORIZONTE <= fin_previo
    entrenar = X.indice(rejilla_d[valido], rejilla_j[valido])
    j = int(np.where(origenes == fin_previo)[0][0])
    destino = X.indice(np.arange(panel.D), np.full(panel.D, j))
    salidas = []
    for semilla in semillas:
        modelo = ajustar(pipe, tipo, X, entrenar, epocas, semilla)
        salidas.append(pipe.predecir(modelo, X, destino))
    return np.mean(salidas, axis=0)


def main() -> None:
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser()
    parser.add_argument("--tipos", nargs="+", default=["tcn_lstm", "gru_lstm"])
    parser.add_argument("--anios", nargs="+", type=int, default=list(range(2011, 2026)))
    parser.add_argument("--epocas", type=int, default=12)
    parser.add_argument("--etiqueta", default="hibrida")
    args = parser.parse_args()
    raiz = raiz_repo()
    pipe = cargar_pipeline(raiz)
    pipe.HORIZONTE = 52
    panel = pipe.Panel(pipe.cargar(raiz))
    V = pipe.ventanas(panel)
    X = pipe.Tensores(V)
    filas = []
    for tipo in args.tipos:
        for anio in args.anios:
            inicio = time.time()
            semillas = [0, 1, 2, 3, 4] if anio == 2025 else [0, 1]
            pred = pronosticar(pipe, tipo, panel, V, X, anio, semillas, args.epocas)
            for d, depto in enumerate(panel.deptos):
                for h in range(pipe.HORIZONTE):
                    filas.append((tipo, anio, int(depto), h + 1, float(pred[d, h])))
            print(tipo, anio, round(time.time() - inicio, 1), "s", flush=True)
            pd.DataFrame(filas, columns=["familia", "anio", "id_departamento", "semana", "pred"]).to_parquet(raiz / CARPETA / f"predicciones_{args.etiqueta}.parquet")
    print("listo", flush=True)


if __name__ == "__main__":
    main()
