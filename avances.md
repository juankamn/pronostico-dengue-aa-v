# Avances

Resumen para no abrir cada ensayo. Lo nuevo se agrega arriba.

## 2026-11-04 — Cuaderno de entrega

`entrega/taller_final_dengue.ipynb` queda con el EDA (más los lotes de reporte y el cierre de 2024), las etapas 1 y 2 como estaban, la etapa 3 (bases por horizonte, que reproduce exacto el envío de 2048,50), la etapa 4 (MLP, RNN, LSTM, GRU y TCN de 26 semanas en walk-forward, con los experimentos de diseño) y una sección 6 nueva. Corre completo en 14 minutos, con 18 figuras, y genera `entrega/submission_mezcla_red.csv` y `entrega/submission_continuidad_red.csv`.

Son los mismos dos finalistas con red, salidos de otra corrida: difieren de `ensayos/juan/2026-11-04-finalista-red/` y `-continuidad-red/` en hasta 18 casos en el primer semestre y 0,23 en julio. Para que el cuaderno y Kaggle coincidan, estos son los que se suben y se eligen como selecciones finales.

Tablero público: mezcla con red 1971,91647; continuidad sobre la mezcla 1971,91564. Mejor que el 1981,99 esperado por el sensor, porque julio también llevaba el patrón de reporte. La cabecera del cuaderno ya tiene a los dos integrantes con su código.

## 2026-11-04 — Red de dinámica para el primer semestre

`ensayos/juan/2026-11-04-red-dinamica-h1/`. MLP, RNN, LSTM, GRU y TCN con ventanas rodantes de 52 semanas, 26 de horizonte y pérdida Poisson, evaluadas en walk-forward sobre el primer semestre, que es lo que mide el privado. Ninguna red sola le gana a B (3534 de media en 2014-2023; la mejor sola, MLP, 3595). Mezcladas sí: 25 % TCN + 25 % MLP + 50 % B da 3316 y gana 7 de 10 años; con otras tres semillas, 3328. En 2024 baja de 152.278 a 84.361.

Nuevas selecciones finales propuestas, pendientes de subir: `ensayos/juan/2026-11-04-finalista-red/` (la mezcla, media 67,3 en el primer semestre) y `ensayos/juan/2026-11-04-finalista-continuidad-red/` (la mezcla arrancando en 0,8 veces diciembre, media 80,4). Los dos llevan el patrón de reporte y el segundo semestre de B, así que el público debe quedar cerca de 1982. En 2014-2023 este par habría dado 3315 y el par B + C, 3490. Reemplazan a los finalistas de colapso y continuidad sin red.

## 2026-11-04 — Patrón de reporte y dos escenarios para el primer semestre

El privado mide enero a junio de 2025 y nadie lo ve hasta el cierre. Con los tres puntajes públicos anteriores, el segundo semestre real queda en 0,82 a 0,93 veces el envío de horizontes: B ya está en nivel. Su riesgo está en el primer semestre: abre 2025 en 78 casos por departamento y semana, cuando diciembre de 2024 sin picos estaba en 126. Ninguna corrección de esa apertura gana en el walk-forward 2014-2023, así que se cubren los dos escenarios con las dos selecciones finales.

`ensayos/juan/2026-11-04-sensor-picos/submission.csv`. Tablero público: 2016,28310. Desde septiembre de 2024 la semana del primer miércoles de cada mes carga un lote de reporte (Valle ×2,18; Sucre ×0,36). En agosto a diciembre de 2025 siguió con cerca del 29 % de esa fuerza. El mejor público posible en esa dirección es 1981,99.

`ensayos/juan/2026-11-04-h1-continuidad/submission.csv`. Tablero público: 2048,49606, igual a B, porque solo cambia el primer semestre. Confirma que el público son las semanas 27 a 52.

Selecciones finales propuestas, pendientes de subir: `ensayos/juan/2026-11-04-finalista-colapso/` (primer semestre de B, media 65,9) y `ensayos/juan/2026-11-04-finalista-continuidad/` (arranca en 0,8 veces diciembre, media 81,1). Los dos llevan el patrón de reporte con amplitud 0,291 y el mismo segundo semestre. Entre ellos hay 1359 de MSE en el primer semestre.

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
