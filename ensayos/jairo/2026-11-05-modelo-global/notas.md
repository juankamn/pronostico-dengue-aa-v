# Modelo global de árboles

Experimento 2026-11-05. HistGradientBoostingRegressor global, 150 iteraciones, tasa 0,05, 15 hojas, mínimo 40 muestras por hoja, L2 10, sin parada temprana y semilla 42. Entrenamiento directo para cada semana; no es red neuronal. Variables: clima de la semana objetivo disponible en test, seno/coseno de semana y viento, identidad del departamento (one-hot), casos de igual semana del año anterior y media, mediana y máximo de las últimas 8 semanas del año anterior. Semana 53 sin año anterior equivalente queda ausente; el modelo admite ausencias. No se usan casos del año evaluado como entrada.

Validación anual creciente: entrenamiento antes del año evaluado, predicciones 2022, 2023 y 2024. Comparación con B histórica en las mismas filas: B solo cubre semanas 1–52, por lo que se omite semana 53 de 2024. MSE H1/H2/global:

- 2022: modelo 3587/1894/2741; B 1021/2433/1727.
- 2023: modelo 4408/17670/11039; B 5109/19464/12286.
- 2024: modelo 129193/55500/92346; B 152278/65749/109014.

Es una primera prueba sin búsqueda de hiperparámetros. El modelo pierde en H1 2022, así que no hay evidencia suficiente para reemplazar la selección privada. B no equivale a las selecciones finales de Juan. El entrenamiento se hace con clima de la semana objetivo porque el test lo incluye; por tanto se trata de un ejercicio offline, no de un pronóstico operativo desde diciembre con clima desconocido. El uso de observaciones climáticas de 2025 refleja el formato de la competición. No se ha verificado que el clima de test tenga igual naturaleza que el histórico.

El archivo `submission.csv` pronostica 2025 completo, recortando negativos sin redondear y respetando el orden de `sample_submission.csv`; se subió a Kaggle y obtuvo 5773,16350 de score público. No reemplazar los finalistas automáticamente. Ejecutar desde raíz: `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 .venv/bin/python ensayos/jairo/2026-11-05-modelo-global/ejecutar.py`.

## Resultado público y mezcla calculada

Jairo reporta Kaggle público 5773,16350 para `submission.csv`, frente a 1689,03586 de selección 1. H2: media del modelo 81,17 frente a 53,92 y distancia cuadrática media entre ambos 4902,49571; no se conoce el valor real.

La identidad cuadrática del MSE permite calcular una mezcla sobre las mismas filas públicas: 91,653557 % de selección 1 y 8,346443 % del modelo, con MSE público estimado 1654,88355. `mezclar_publico.py` genera `submission_mezcla_publico.csv`: H1 idéntico a selección 1; H2 mezclado. La estimación depende de que ambos puntajes correspondan exactamente a los archivos locales y a la misma partición pública, y puede variar por redondeo. Es ajuste al tablero público, no evidencia de mejoría privada. Jairo la subió a Kaggle y confirmó score público 1654,88355.
