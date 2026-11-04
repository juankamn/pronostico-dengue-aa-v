# Segundo semestre con redes de 52 semanas

Las mejoras anteriores fueron en el primer semestre, que el tablero público no mide. El único dato real de 2025 que se puede ver es el segundo semestre. Este ensayo entrena TCN y MLP con horizonte de 52 semanas (el pipeline de `2026-11-04-red-dinamica-h1` con `HORIZONTE = 52`), 2 semillas por año y 5 para 2025, y compara su segundo semestre con B año por año. `correr.py` entrena; `evaluar.py` compara y arma el archivo.

## Walk-forward (MSE medio 2014-2023)

| | Primer semestre | Segundo semestre |
| --- | --- | --- |
| B | 3534 | 5164 |
| Red sola (50 % TCN + 50 % MLP) | 4179 | 5466 |
| Mitad red, mitad B | 3565 | 4954 |
| Pesos por región, con años anteriores | 3743 | 5283 |

En el segundo semestre, mitad red y mitad B le gana a B en 5 de 10 años y baja el promedio 4 % (2016-2023: 5753 frente a 5934). En 2024 baja de 65.749 a 31.827. Los pesos por región no ayudan en este semestre. En el primer semestre, las redes de 52 semanas rinden peor que las de 26, así que el primer semestre sigue saliendo de esas.

## Archivo de prueba

`submission.csv` tiene el primer semestre idéntico a `entrega/submission_mezcla_region.csv` (1971,62176 en el público) y cambia solo las semanas 27 a 52: mitad red y mitad B, con el mismo patrón de reporte semana a semana. Como el privado solo mira el primer semestre, este archivo tiene el mismo puntaje privado que la mezcla por región; el público mide solo la red.

Segundo semestre: media 55,7 frente a 57,9. Baja Meta (130 → 104), Sucre (99 → 84), Tolima, Antioquia, Huila y Cesar; sube Valle del Cauca (190 → 208) y Cundinamarca (46 → 54).

## Cómo leer el resultado

Regla fijada antes de ver el puntaje:

- Si baja de 1971,62, la red lee 2025 mejor que B. Se mantiene la mezcla por región como primera selección y este archivo la reemplaza en el público.
- Si sube, la red lee mal 2025. La segunda selección pasa a B con el patrón, y se revisa bajar el peso de la red en el interior.

## Resultado

Tablero público: 2050,42336, frente a 1971,62176 del mismo primer semestre con el segundo semestre de B. Como el MSE es cuadrático, con d = archivo nuevo - archivo anterior: cambio del puntaje = media(d²) + 2·media(d·(anterior - real)). Con media(d²) = 106,2 y un cambio de +78,8, la proyección de la realidad sobre d da 0,13: la realidad se movió apenas un 13 % en la dirección de la red. El mejor paso en esa dirección habría bajado el público en menos de 2 puntos.

En 2025 la lectura de las redes no le ganó a B en el segundo semestre, aunque en el walk-forward 2014-2023 sí le ganaba. Se aplica la regla: la segunda selección final pasa a B con el patrón (`entrega/submission_bases_patron.csv`). La primera sigue siendo la mezcla por región: su peso por región sale del walk-forward, y el par mezcla por región + B ya era el mejor en 2016-2023 (3616). No se bajó el peso de la red en el primer semestre: el par con B cubre el caso en que la red lea mal 2025.
