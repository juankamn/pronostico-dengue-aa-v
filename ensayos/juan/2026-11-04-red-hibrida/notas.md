# Red híbrida autoregresiva para todo 2025

La idea: un solo modelo para las 52 semanas de 2025, con un codificador mixto y un decodificador autoregresivo, para que el puntaje público (segundo semestre) diga algo del mismo modelo que pronostica el privado (primer semestre).

- **TCN→LSTM.** Cuatro bloques de convolución causal dilatada (1, 2, 4, 8) y encima una LSTM que lee esa secuencia.
- **GRU→LSTM.** Una GRU y encima una LSTM.
- **Decodificador.** Una celda LSTM que avanza semana a semana durante 52 semanas y recibe su propia predicción anterior, el clima y el calendario de esa semana, el perfil estacional y las variables estáticas. Misma compuerta entre persistencia y perfil, pérdida Poisson, entrenamiento en modo libre. 12 épocas, 2 semillas por año y 5 para 2025.

`hibrida.py` entrena y `evaluar.py` compara.

## Walk-forward (MSE medio)

| | Primer semestre 2014-2023 | 2016-2023 | Segundo semestre 2014-2023 |
| --- | --- | --- | --- |
| B | 3534 | 3909 | 5164 |
| Mezcla por región vigente (redes de 26 semanas) | 3492 | 3659 | — |
| TCN→LSTM sola | 4299 | 4631 | 6814 |
| TCN→LSTM mitad con B | 3667 | 4036 | 5516 |
| GRU→LSTM sola | 4650 | 5145 | 6749 |
| GRU→LSTM mitad con B | 3697 | 4157 | 5383 |
| TCN + MLP de 52 semanas mitad con B (referencia) | 3565 | 3992 | 4954 |

Las dos híbridas pierden en los dos semestres: frente a la mezcla vigente en el primero y frente a B y a la TCN + MLP simple en el segundo. En 2017, el año posterior a un brote más parecido a 2025, la TCN→LSTM sola da 3037 en el primer semestre, contra 1120 de B: el decodificador autoregresivo arrastra el nivel alto de diciembre durante meses. En 2024 sí corta el error de B (51.245 frente a 152.278), pero ese año no se parece a 2025.

## Decisión

La regla fijada antes era reemplazar la primera selección solo si el primer semestre no empeoraba en 2016-2023 y el público bajaba de 1971,62. Falla la primera condición, así que no se sube. `submission_tcn_lstm.csv` y `submission_gru_lstm.csv` quedan como referencia.
