# Avances

Resumen para no abrir cada ensayo. Lo nuevo se agrega arriba.

## 2026-11-03 — Zoo y decaimiento post-pico

`ensayos/juan/2026-11-03-zoo-bayesiano/submission.csv`. Media 45,0 (semestre 1: 51,1; semestre 2: 39,0). Tablero público: 2836,48931. Quedó peor que 2048,50, así que no reemplaza ese envío.

El archivo no lleva zoo: MLP, RNN, LSTM, GRU, TCN, LightGBM, XGBoost y Ridge quedaron en peso cero. Lo que se subió es el decaimiento hacia el piso de años bajos. En el segundo semestre de 2025 eso cortó de más el interior (Antioquia 115 → 37, Valle 193 → 118, Meta 130 → 59, Tolima 121 → 52, Santander 100 → 33). El 92 % de esa distancia al envío de 2048,50 es ortogonal a la recta de los dos envíos anteriores; el backtest solo mejoraba unos 50 puntos de MSE y el corte real fue de unos 1100. El primero del público sigue en 1353,89. El mejor envío público nuestro sigue siendo el de horizontes.

## 2026-10-31 — Horizontes y bases recientes

`ensayos/juan/2026-10-31-horizonte-bases/submission.csv`. Media 61,9 (semestre 1: 65,8; semestre 2: 58,1). Fue el segundo envío público. Ya no es el archivo vigente.

Tablero público, equipo Pandebono Predictors: puesto 4, score 2048,49606. El anterior estaba en 3754,91. El tercero va en 2042,42 y el primero en 1808,52.

El público de 3754,91 coincidía con el régimen post-brote del árbol anterior. Bolívar, Atlántico y Córdoba quedaban en el nivel de 2007, y Santander llegaba a 201 por doble conteo. Con bases recortadas de los últimos años, el cierre de las semanas 41–53 y un peso distinto para cada horizonte, el backtest del segundo semestre baja más de 1000 puntos (moderados 3368 → 2335; post-brote 3671 → 2576). El residuo de Ridge, LightGBM, XGBoost y red no mejora ese número, así que no entra al archivo. Santander queda en 100 y Bolívar en 222 en el segundo semestre de 2025. Los años de brote siguen altos.

## 2026-10-27 — Nivel sin copiar el brote

`ensayos/juan/2026-10-27-nivel-sin-brote/submission.csv`. Media 36,6. Primer envío público: 3754,91. Ya no es el archivo vigente.

El envío que copiaba 2024 (media 105) no está en la escala del tablero público (1.897 a 3.500). El clima de 2025 se parece a 2014–2021, no a 2024. En segundos semestres tranquilos el error queda en 702 (2017), 1.409 (2018) y 2.672 (2022). En años de brote sigue alto. El semestre 1, que define al ganador, no se ve hasta el cierre.

## 2026-10-27 — RNN de la etapa 1

`ensayos/juan/2026-10-27-rnn-etapa1/submission.csv`. Media 75,7. Línea de base del curso. No es el envío de la competencia.
