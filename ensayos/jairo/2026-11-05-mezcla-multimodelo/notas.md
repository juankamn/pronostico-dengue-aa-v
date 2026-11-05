# Mezcla multimodelo del tablero público

Fecha 2026-11-05. Se combinan siete archivos ya calificados por Kaggle; los pesos suman uno, están acotados entre -1 y 1,5 y las predicciones se restringen a no negativas. Para el segundo semestre, el MSE de la mezcla se calcula con la identidad cuadrática, los scores públicos informados y las predicciones de cada envío. Se mantiene exactamente el primer semestre de `submission_seleccion_1.csv`.

Predicción pública estimada: 1646,97677. Se mantienen 1.612 filas, IDs del ejemplo, valores finitos y no negativos. La mejora frente a 1654,88355 procede de optimizar sobre el mismo tablero y no indica mejora en el semestre privado. La cifra depende de que cada score corresponda al archivo local y al mismo subconjunto público; los scores tienen cinco decimales, por lo que puede haber pequeña diferencia. Algunos pesos son negativos y están acotados para reducir la sensibilidad; no tienen interpretación como proporciones probabilísticas. Jairo subió el envío y Kaggle confirmó score público 1646,97677. No consta que lo haya seleccionado como finalista.

Fuentes, scores y pesos: `pesos.csv`; código y comprobaciones: `ejecutar.py`. Reproducir desde la raíz con `OPENBLAS_NUM_THREADS=1 .venv/bin/python ensayos/jairo/2026-11-05-mezcla-multimodelo/ejecutar.py`.
