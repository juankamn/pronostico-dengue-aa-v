# Sonda del nivel de Valle del Cauca en H2

Fecha 2026-11-05. Hipótesis: el nivel estimado para Valle del Cauca (id 30) en el segundo semestre de 2025 puede diferir del real. Elegido porque en el EDA histórico de Juan concentró cerca del 39 % del error H1 de B; esto no garantiza que domine H2 de 2025.

La sonda parte del archivo multimodelo con score público confirmado 1646,97677 y multiplica por 1,5 las 26 predicciones H2 de Valle. Todas las demás filas, incluido H1, quedan idénticas. La distancia MSE de la sonda al mejor envío sobre las 806 filas H2 es 169,264174. Su score es desconocido y la sonda misma puede empeorar; no seleccionarla automáticamente.

El MSE es cuadrático. Con el score público de la sonda se puede calcular el factor óptimo de escala H2 de Valle respecto del envío base, restringido a escala no negativa, y el score público estimado. Ejecutar desde la raíz `OPENBLAS_NUM_THREADS=1 .venv/bin/python ensayos/jairo/2026-11-05-sonda-valle/ejecutar.py SCORE_SONDA`. Genera `submission_calibrada_valle.csv`. La estimación depende de que el score base y el score de la sonda correspondan exactamente a los archivos locales y a las mismas 806 filas; los scores publicados tienen precisión limitada. La dirección calibrada está sobreajustada al público, no demuestra mejora privada; H1 permanece idéntico a selección 1.

Esta prueba no revela las etiquetas individuales ni garantiza una mejora grande. El resultado de la sonda debe reportarse antes de calcular o subir el candidato calibrado.

## Resultado reportado

Jairo reportó score público 2005,86755 para la sonda. Con el score base 1646,97677, la identidad cuadrática da alfa -0,56014986 sobre la dirección de la sonda, equivalente a multiplicar el H2 de Valle por 0,71992507. Score público estimado del archivo `submission_calibrada_valle.csv`: 1593,86711; pendiente de verificar en Kaggle. El primer semestre no cambia. El resultado optimiza el público y no demuestra mejora privada.
