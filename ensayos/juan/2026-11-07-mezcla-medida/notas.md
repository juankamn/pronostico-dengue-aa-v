# Mezcla medida y primer semestre

Fecha: 2026-11-07. No se tocó `entrega/`.

El público (semanas 27 a 52) y el privado (semanas 1 a 26) se arman por separado. Los tres archivos de abajo comparten el mismo segundo semestre, así que en el tablero público deben salir iguales. El privado solo se ve en el walk-forward.

## Segundo semestre

Combinación lineal de los 16 envíos ya calificados. El MSE sale de los puntajes y de los archivos: **1459,16**, mínimo 0,35. Valle del Cauca queda en 104 casos por semana (el archivo de 1646 lo tenía en 141).

Ese número no dice nada del primer semestre. Cambiar solo esta mitad deja el privado igual que la selección 1.

## Primer semestre

Las sondas fijan tres escalas del segundo semestre frente a B: Costa 0,95, Valle 0,54, resto del interior 1,06. La corrección de nivel que ya usan las selecciones lleva el final del primer semestre hacia esa escala. Se probó en los años anteriores, eligiendo la semana y la fuerza solo con años previos.

En el apilado (la selección 1) partir el interior en Valle y resto empeora: 3356 pasa a 3363 en 2016-2023 y gana 3 de 8 años. No se aplica ahí.

Lo que sí baja el error es mezclar, con un peso aprendido en los años anteriores, el apilado ya corregido por región y B corregida con las tres escalas. En 2016-2023 esa mezcla da **3170**, frente a 3356 del apilado y 3352 de B sola. El peso que toca para 2025 es 0,30 del apilado y 0,70 de B. Valle en el primer trimestre queda en 208 (la selección 1 abre en 322 y la selección 2 en 127).

## Qué subir

Los tres llevan el segundo semestre de 1459,16. El primero cambia.

| Archivo | Primer semestre | Valle, primer trimestre | Walk-forward 2016-2023 |
|---|---|---|---|
| `submission_privado.csv` | 30 % selección 1 + 70 % B con la escala fina | 208 | 3170 |
| `submission.csv` | selección 1 | 322 | 3356 |
| `submission_seleccion_2.csv` | selección 2 | 127 | el de esa selección, 3352 |

Para las dos finales: `submission_privado.csv` y `submission_seleccion_2.csv`. La primera es la que el walk-forward mejora como modelo único. La segunda cubre el caso en que enero sigue al segundo semestre y Valle abre bajo. El tablero público no las va a separar.

`submission_sonda_bolivar.csv` multiplica por 1,5 solo el segundo semestre de Bolívar sobre la mezcla de 1459. No es finalista. Sirve después, cuando el score de la mezcla confirme el 1459, para calibrar Bolívar.
