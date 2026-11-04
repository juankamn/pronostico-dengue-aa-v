# Red de dinámica para el primer semestre

El tablero privado mide enero a junio de 2025, pronosticado desde la última semana de 2024. Este ensayo entrena redes para esa tarea y las mide igual: walk-forward por año, pronosticando las semanas 1 a 26 de cada año Y desde la última semana de Y-1, con ventanas cuyo objetivo termina antes del 1 de enero de Y.

## Datos y forma del problema

Cada ejemplo es una ventana rodante: 52 semanas de entrada y 26 de objetivo, desde cualquier semana de origen. Son unas 26.700 ventanas para 2025, en lugar de las 31 transiciones diciembre-enero por año. La red ve:

- **Entrada (52 semanas, 10 variables):** casos en log relativos al nivel reciente (mediana de las últimas 8 semanas, que ignora los picos de reporte), seno y coseno de la semana, la marca de semana de lote (primer miércoles del mes), la marca de fin de año y cinco anomalías de clima (lluvia, temperatura media y mínima, punto de rocío y humedad del suelo) contra la climatología 2007-2013, que no incluye ningún año validado.
- **Futuro (26 semanas, 11 variables):** el mismo calendario, el clima de esas semanas (en 2025 viene en `test.parquet`), el perfil estacional del departamento relativo al nivel actual y el horizonte.
- **Estáticas (4):** nivel actual, nivel frente al normal de los 8 años previos, distancia al pico de las últimas 52 semanas y casos acumulados frente a lo normal.

La salida es una compuerta: para cada semana futura la red elige un punto entre la persistencia del nivel actual y el perfil estacional, más un residuo acotado de ±0,5 en log. La pérdida es Poisson, que apunta a la media condicional, la misma que premia el MSE.

Frente al código base del curso cambian la ventana (5 → 52 semanas), el horizonte (1 → 26), la pérdida (MSE → Poisson), el optimizador (Adam → AdamW con weight decay 1e-4), el lote (32 o 64 → 512) y las épocas (20 o 30 → 12). Por familia: MLP con 2 capas de 64 y dropout 0,2; RNN, LSTM y GRU con 1 capa de 32, dropout 0,1 y lr 2e-3; TCN con 4 niveles de 32 canales (antes [32, 32, 64, 64]), kernel 3, dilataciones 1 a 8 (campo receptivo de 61 semanas) y lr 1e-3. La configuración está en `config_final5.json`.

## Qué se probó con GRU (media 2014-2023 del MSE del primer semestre; B = 3534)

- Salida libre (hasta ×55 el nivel actual): 28.533 a 63.925. En 2016 la red extrapola el ascenso de diciembre de 2015 y llega a 599.004.
- Compuerta, 5, 12 y 25 épocas: 4646, 4466 y 4755. Sin explosiones.
- Perfil sin años de brote, residuo de 0,25, peso ×8 a los orígenes de fin de año y pérdida más cercana al MSE: entre 4418 y 4799. Ninguno mejora.

La red sola sube el primer semestre en los años que siguen a un pico: en 2014 Valle del Cauca pasó de 392 en diciembre a 193, y la red dijo 548.

## Cinco familias con la compuerta, 12 épocas y 2 semillas

| Familia | Sola | 50 % con B | Gana a B (de 10 años) |
| --- | --- | --- | --- |
| MLP | 3595 | 3353 | 4 |
| RNN | 4685 | 3565 | 2 |
| LSTM | 5224 | 3737 | 2 |
| GRU | 4601 | 3584 | 1 |
| TCN | 3796 | 3323 | 5 |
| Las cinco | 4157 | 3456 | 3 |

25 % TCN + 25 % MLP + 50 % B da 3316 y gana a B en 7 de 10 años (2014, 2016, 2017, 2018, 2020, 2021 y 2023); pierde por menos de 150 en 2015, 2019 y 2022. Con tres semillas nuevas da 3328 y también gana 7 de 10. En 2024, año de brote, baja de 152.278 a 84.361. Las redes sueltas no le ganan a B; mezcladas sí, porque se equivocan en otra dirección.

## Lectura de 2025

Con 5 semillas, la mezcla deja el primer semestre en 67,3 (B 65,8), pero lo reparte distinto: sube el interior que cerró 2024 alto (Valle del Cauca en el primer trimestre 179 → 262, Santander 94 → 124, Tolima 134 → 145) y baja la Costa y Meta (Bolívar 247 → 226, Atlántico 223 → 197, Meta 190 → 160). TCN sola apunta a la continuidad y MLP sola a B.

`pares.py` mide qué par de selecciones finales habría rendido mejor en 2014-2023, con Kaggle quedándose con el mejor de los dos: mezcla con red + continuidad sobre la mezcla, 3315; B + C, 3490.

## Archivos

`pipeline.py` entrena y evalúa (`--familias`, `--anios`, `--semillas`, `--epocas` y las opciones de los ensayos). `mezcla.py`, `resumen.py` y `pares.py` arman las tablas. `b_walkforward.parquet` es B en walk-forward 2011-2025. `predicciones_final5.parquet` y `predicciones_rep3.parquet` son el walk-forward de las redes y `predicciones_final2025.parquet` es el pronóstico de 2025 con 5 semillas, entrenado con todo hasta 2024.
