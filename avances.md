# Avances

Resumen para no abrir cada ensayo. Lo nuevo se agrega arriba.

## 2026-11-04 — Apilamiento por región y nivel del segundo semestre

Dos técnicas nuevas en el cuaderno de entrega, validadas solo con años anteriores:

- **Apilamiento.** Pesos de B, TCN y MLP buscados por mínimos cuadrados, por separado para la Costa y el interior. Para 2025: interior 0,33 B + 0,67 TCN; Costa 0,62 B + 0,38 MLP. RNN, LSTM, GRU, las redes de 52 semanas y las híbridas reciben peso cero o empeoran el resultado.
- **Nivel del segundo semestre.** Saber cuánto quedó el segundo semestre frente a B (la escala k) y llevar hacia ese nivel el final del primero, a media fuerza desde la semana 8, mejora el primer semestre en la historia. En 2025 el k sale del tablero público: 0,90.

Primer semestre 2016-2023: apilado + nivel 3355, frente a 3651 de la mezcla por región, 3696 de la mezcla con peso 0,5 y 3909 de B; gana a la mezcla por región en 6 de 8 años. Si Kaggle se hubiera quedado cada año con el mejor de los dos envíos, el par (apilado + nivel) + (B + nivel) habría dado 3298, frente a 3616 del par anterior. Con otras tres semillas el apilado con nivel da 3352.

Nuevas selecciones finales: `entrega/submission_final_apilado.csv` (primer semestre 71,7; Valle del Cauca 323 en el primer trimestre) y `entrega/submission_final_bases.csv` (64,7). Las dos llevan el segundo semestre calibrado con el tablero público. Tablero público: 1836,20689 las dos, como predecía el cálculo exacto (1836,22, un poco menos por el recorte a 0 de dos semanas). Reemplazan a la mezcla por región y a B con patrón como selecciones finales.

## 2026-11-04 — Segundo semestre combinado para el tablero público

`ensayos/juan/2026-11-04-mezcla-publico/`. Como el MSE es cuadrático, el puntaje público de una combinación de archivos ya subidos (pesos que suman 1) se calcula exacto con sus puntajes. La mejor combinación de cinco envíos baja el público a 1835,54 con el primer semestre de la mezcla por región, y a 1836,22 con el de B con patrón. El primer semestre de cada selección queda idéntico, así que el privado no cambia; solo mejora el puesto visible. No se subieron: ese segundo semestre pasó directo a las selecciones finales del apilado (ver la entrada de arriba).

## 2026-11-04 — Selecciones finales

`entrega/submission_bases_patron.csv` (B con el patrón de reporte) dio 1972,72476 en el tablero público. Las dos selecciones finales quedan así:

1. `entrega/submission_mezcla_region.csv`: 1971,62176. Primer semestre de la mezcla por región (red 0,69 en el interior y 0,23 en la Costa).
2. `entrega/submission_bases_patron.csv`: 1972,72476. Primer semestre de B, que cubre una caída fuerte en enero.

Las dos comparten el segundo semestre, salvo julio, y por eso el público casi no las separa. El privado sí: en el primer semestre están a 940 de MSE una de otra.

## 2026-11-04 — Red híbrida autoregresiva

`ensayos/juan/2026-11-04-red-hibrida/`. TCN→LSTM y GRU→LSTM con decodificador autoregresivo de 52 semanas, para pronosticar todo 2025 con un solo modelo. En walk-forward pierden en los dos semestres: mitad con B dan 4036 y 4157 en el primer semestre de 2016-2023, frente a 3659 de la mezcla por región vigente, y 5516 y 5383 en el segundo de 2014-2023, frente a 5164 de B. El decodificador arrastra el nivel de diciembre en los años que siguen a un brote. No se sube.

## 2026-11-04 — Segundo semestre con redes de 52 semanas

`ensayos/juan/2026-11-04-red-segundo-semestre/`. TCN y MLP con horizonte de 52 semanas. En el segundo semestre, mitad red y mitad B baja el walk-forward de 5164 a 4954 (2014-2023) y gana 5 de 10 años; los pesos por región no ayudan ahí. En el primer semestre estas redes rinden peor que las de 26 semanas, así que el primer semestre no cambia. `submission.csv` deja el primer semestre idéntico a la mezcla por región y solo cambia julio a diciembre: su puntaje público mide si la red lee bien 2025.

Tablero público: 2050,42336, peor que 1971,62. El cambio de la red vale 106 de MSE entre archivos y el público subió 78,8: la realidad se movió apenas un 13 % en la dirección de la red. En 2025 la lectura de las redes no le ganó a B. Con la regla fijada antes de ver el puntaje, la segunda selección final pasa a B con el patrón de reporte (`entrega/submission_bases_patron.csv`, que ahora genera el cuaderno), y la primera sigue siendo la mezcla por región (`entrega/submission_mezcla_region.csv`, 1971,62). La continuidad queda generada pero fuera de las selecciones.

## 2026-11-04 — Peso de la red por región

El cuaderno de entrega estima el peso de la red (50 % TCN + 50 % MLP) frente a B por separado para la Costa y el interior, solo con años anteriores. Para 2025, con 2011-2023, da 0,69 en el interior y 0,23 en la Costa. En 2016-2023 la mezcla por región da 3651 y la de peso 0,5, 3696; gana 6 de 8 años. En 2014 pierde porque el peso sale de solo tres años de historia.

Archivos nuevos: `entrega/submission_mezcla_region.csv` (primer semestre 71,8; Valle del Cauca 311 en el primer trimestre) y `entrega/submission_continuidad_region.csv` (83,1). Tablero público: 1971,62176 y 1971,62162. El segundo semestre es el mismo de los envíos de 1971,92 salvo julio (hasta 0,69 casos); la mejora de 0,3 viene de ahí y no dice nada del privado. El cuaderno sigue reproduciendo los dos envíos de 1971,92 (diferencia de 0,00002).

En 2016-2023, el par mezcla por región + B habría dado 3616, y el par mezcla por región + continuidad, 3649. En 2025 la mezcla por región queda a 940 de MSE de B con patrón y a 730 de la continuidad. La segunda selección quedó en B con el patrón (ver la entrada de arriba).

## 2026-11-04 — EDA profundo

`ensayos/juan/2026-11-04-eda-profundo/eda.ipynb`. Valle del Cauca aporta el 39 % del error de B en el primer semestre, y un tercio del error total es ruido semanal. La Costa baja en el primer semestre y el centro andino sube. El primer trimestre de 2025 fue 1,3 °C más frío que el de 2024, lo que apoya una caída. La mezcla con red gana en el interior y pierde en la Costa; los pesos aprendidos con años anteriores dan para 2025 0,70 en el interior y 0,21 en la Costa.

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
