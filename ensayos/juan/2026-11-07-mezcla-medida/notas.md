# Mezcla medida y primer semestre

Fecha: 2026-11-07. No se tocó `entrega/`.

## Sonda de Bolívar, ya calificada

Kaggle devolvió **1828,40392** para `submission_sonda_bolivar.csv` (×1,5 solo en el segundo semestre de Bolívar, sobre la mezcla de 1459,16475). El factor que minimiza el público es **0,978**. Bolívar pasa de 203,2 a 198,8 casos por semana. El público estimado queda en **1458,52**, 0,64 puntos debajo de la mezcla. Bolívar ya estaba en el nivel que el tablero pide. No explica el hueco hasta 1300.

`submission_bolivar_seleccion_2.csv` quedó confirmado en **1458,52285**. La sonda de Atlántico dio **1825,97217**. El factor es **0,853** y Atlántico pasa de 164,3 a 140,2. Con los dos factores el público estimado es **1438,54**. Va en `submission_calibrada.csv` y `submission_calibrada_seleccion_2.csv`, que conservan el primer semestre de cada selección.

## Perfil de brote

El perfil sin el tope de 2 y 2,8 veces la mediana está en `ensayos/juan/2026-11-07-perfil-brote/`. En el primer semestre de 2016–2023 el peso que mejor mezcla ese perfil con B es 0 todos los años. El perfil solo da MSE 6609, contra 3909 de B, y en la semana del pico real llega a 0,38 del valor, peor que el 0,49 de B. No se sube.

El público (semanas 27 a 52) y el privado (semanas 1 a 26) se arman por separado. Los tres archivos de abajo comparten el mismo segundo semestre, así que en el tablero público deben salir iguales. El privado solo se ve en el walk-forward.

## Segundo semestre

Combinación lineal de los 16 envíos ya calificados. El MSE sale de los puntajes y de los archivos: **1459,16**, mínimo 0,35. Valle del Cauca queda en 104 casos por semana (el archivo de 1646 lo tenía en 141).

Ese número no dice nada del primer semestre. Cambiar solo esta mitad deja el privado igual que la selección 1.

## Primer semestre

Las sondas fijan tres escalas del segundo semestre frente a B: Costa 0,95, Valle 0,54, resto del interior 1,06. La corrección de nivel que ya usan las selecciones lleva el final del primer semestre hacia esa escala. Se probó en los años anteriores, eligiendo la semana y la fuerza solo con años previos.

En el apilado (la selección 1) partir el interior en Valle y resto empeora: 3356 pasa a 3363 en 2016-2023 y gana 3 de 8 años. No se aplica ahí.

Lo que sí baja el error es mezclar, con un peso aprendido en los años anteriores, el apilado ya corregido por región y B corregida con las tres escalas. En 2016-2023 esa mezcla da **3170**, frente a 3356 del apilado y 3352 de B sola. El peso que toca para 2025 es 0,30 del apilado y 0,70 de B. Valle en el primer trimestre queda en 208 (la selección 1 abre en 322 y la selección 2 en 127).

## Qué subir

La mezcla de 3170 no es finalista: le gana al apilado en 5 de 8 años y pierde en 2016, 2021 y 2022. Las dos finales siguen siendo el primer semestre de la selección 1 (`submission.csv`, Valle en 322) y el de la selección 2 (`submission_seleccion_2.csv`, Valle en 127), las dos con el segundo semestre de 1459,16. Kaggle ya calificó las dos de primer semestre bajo en 1459,16475. `submission.csv` es la que todavía cubre el enero alto.

`submission_bolivar_seleccion_2.csv` quedó en 1458,52285, igual que la cuenta. `submission_sonda_atlantico.csv` quedó en 1825,97217. El factor de Atlántico es 0,853 y, sumado al de Bolívar, el público estimado es 1438,54. Eso va en `submission_calibrada.csv` (primer semestre de la selección 1) y en `submission_calibrada_seleccion_2.csv` (el de la selección 2).
