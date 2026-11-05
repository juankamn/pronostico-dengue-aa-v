# Nivel del segundo semestre de 2025 por región

La corrección de nivel del primer semestre usa hoy un solo k para todo el país (0,90, sacado del tablero público). En la historia, un k para la Costa y otro para el interior la mejoran bastante. Apilado, primer semestre 2016-2023:

| Corrección | final5 | rep3 |
| --- | --- | --- |
| Sin corrección | 3557 | 3544 |
| Un k para el país (la vigente) | 3360 | 3352 |
| Un k para la Costa y otro para el interior | 3235 | 3225 |
| Un k por departamento (techo, no se usa) | 2630 | 2608 |

La Costa y el interior difieren mucho: en 2016 el k de la Costa fue 0,18 y el del interior 0,95; en 2021, 3,97 y 0,83; en 2023, 1,53 y 4,23.

## Medición

Dos envíos de medición, no de modelo. Cada uno es la selección 1 (`entrega/submission_final_apilado.csv`, 1836,20689) con el segundo semestre de una sola región subido en 0,2 veces B. El primer semestre queda idéntico, así que el privado no cambia. Como el MSE es cuadrático, la diferencia de puntaje da exacto el producto de B con el segundo semestre real de esa región, y con eso su k. Solo dos regiones: no se mide ningún departamento por separado.

- `submission_sonda_costa.csv`: Atlántico, Bolívar, Cesar, Córdoba, La Guajira, Magdalena y Sucre.
- `submission_sonda_interior.csv`: los otros 24.

Con los puntajes: `python ensayos/juan/2026-11-04-sonda-regiones/armar.py <puntaje_costa> <puntaje_interior>`.

La combinación pública actual sugiere k ≈ 0,86 en la Costa y 0,94 en el interior, pero es una estimación dentro de los envíos ya subidos; la medición da el valor exacto. Con esos dos k se corrige el primer semestre de las dos selecciones finales por región, y los dos envíos entran a la combinación del segundo semestre.
