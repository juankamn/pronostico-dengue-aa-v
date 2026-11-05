# Corrección de residuos de B con clima

Fecha 2026-11-05. Modelo global `HistGradientBoostingRegressor`, entrenado para pronosticar los residuos `casos - B`, usando predicciones históricas walk-forward de B. Usa clima semanal de test, seno/coseno de semana y viento, identidad territorial, B y casos de la misma semana del año anterior. Se suma a B la mitad del residuo predicho; 100 iteraciones, tasa 0,04, máximo 7 hojas, mínimo 100 muestras por hoja, L2 30, semilla 42. No se usa el target del año de validación para entrenar ese fold. Clima de test se asume conocido por ser proporcionado; no implica disponibilidad operativa a diciembre.

MSE contra B en semanas 1–52:

| Año | H1 modelo / B | H2 modelo / B | Global modelo / B |
|---|---|---|---|
| 2022 | 929 / 1021 | 2462 / 2433 | 1696 / 1727 |
| 2023 | 4118 / 5109 | 16314 / 19464 | 10216 / 12286 |
| 2024 | 143612 / 152278 | 64941 / 65749 | 104276 / 109014 |

Solo tres años. Se compara contra B, no contra las selecciones finales. Las características e hiperparámetros fueron fijados para este ensayo, pero los datos históricos ya informaron decisiones anteriores; no es una prueba independiente final. Semana 53 de 2024 excluida porque B no la contiene.

`submission_modelo.csv` usa el modelo para las 52 semanas; no se propone reemplazar las selecciones con él. `submission_sonda_h2.csv` conserva H1 de la mezcla multimodelo pública de 1646,97677 y usa el modelo en H2. Su score público es desconocido: servirá para medir la dirección H2 y calcular una mezcla, manteniendo H1. No seleccionar automáticamente como finalista.

Reproducir desde la raíz: `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 .venv/bin/python ensayos/jairo/2026-11-05-residuos-clima/ejecutar.py`.

## Resultado público

Jairo reportó score 2174,34931 para `submission_sonda_h2.csv`. Frente a la mezcla pública anterior (1646,97677), el cambio H2 tiene distancia cuadrática media 555,71330. La combinación lineal óptima de ambas predicciones, calculada a partir de los scores, asigna 2,549944 % a esta sonda y estima 1646,61543. `mezclar.py` genera `submission_mezcla.csv`, H1 idéntico a selección 1. No se ha subido; la ganancia estimada es solo 0,36 puntos y no afecta el privado. Depende de que scores y archivos correspondan exactamente y de los decimales publicados. Ninguno de estos scores confirma que la corrección de H1 sea mejor que las selecciones finales.

## Comparación con aproximación al finalista

Se reconstruyó una aproximación histórica a selección 1: pesos regionales aprendidos con años anteriores y corrección de nivel regional. Las predicciones históricas de redes se tomaron del archivo `predicciones_final5.parquet`; no se ha demostrado que sean idénticas a las semillas o a todos los pasos de la selección final. El nivel regional del segundo semestre del año evaluado se calcula con sus casos reales: es una señal retrospectiva, usada para aproximar la información de sondas públicas de 2025, y no corresponde a un pronóstico operativo a inicio del año. No debe interpretarse como validación ciega ordinaria.

MSE H1 medio 2016–2023: aproximación del finalista 3355,6; residuos 3739,4; 75 % aproximación + 25 % residuos 3400,5; mezcla 50/50 3479,4. En 2022 la mezcla 25 % mejora de 863,1 a 833,6, pero en 2017 empeora de 713,5 a 945,0. El modelo se entrenó por año con datos anteriores; los diseños previos se exploraron sobre algunos años de evaluación, así que pueden existir sesgos de selección. No hay evidencia suficiente para sustituir H1 de selección 1. La comparación 2024 usa también el nivel real de su H2 y es especialmente retrospectiva; no extrapolarla como rendimiento independiente.

`comparar_finalista.py` genera `comparacion_aproximada_h1.csv`; requiere primero ejecutar `ejecutar.py`, que ahora guarda predicciones históricas 2014–2024. No modifica ningún finalista.
