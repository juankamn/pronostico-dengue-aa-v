# Corrección histórica por departamento

Fecha: 2026-11-05. Experimento exploratorio, no seleccionado como finalista.

Hipótesis: corregir el sesgo medio de cada departamento a partir de errores de predicciones históricas fuera de muestra. Para cada año se usan solo años anteriores, con contracción equivalente a cuatro años sin corrección. Se aplica el 50 % de esa corrección, elegido por menor MSE promedio en 2016-2021 entre 0, 0,25, 0,5 y 1.

La validación usa como aproximación 50 % B, 25 % MLP y 25 % TCN de los archivos históricos de Juan, no la selección 1 exacta. MSE H1: 2022, 1053,8 → 1137,3; 2023, 4497,9 → 4229,3; 2024, 84360,9 → 82467,0. Mejora años de alta incidencia y perjudica años tranquilos. Las decisiones previas sobre modelos pueden haber usado esos mismos años, de modo que la evaluación no es completamente independiente. H2 no se evalúa: el candidato conserva H2 de selección 1.

El archivo aplica la corrección de la aproximación a selección 1, cuya receta difiere. Por ello la mejora medida no demuestra que el candidato mejore al finalista ni estima su error privado. H2 se conserva idéntico, por lo que se espera el mismo puntaje público (salvo detalles de precisión), y el tablero no sirve para medir esta modificación. No reemplazar las selecciones existentes sin más evidencia.

Ejecutar desde la raíz: `.venv/bin/python ensayos/jairo/2026-11-05-correccion-regional/ejecutar.py`. Entradas: `b_walkforward.parquet`, `predicciones_final5.parquet`, `submission_seleccion_1.csv` y `sample_submission.csv`. Salidas: `validacion.csv` y `submission.csv`.
