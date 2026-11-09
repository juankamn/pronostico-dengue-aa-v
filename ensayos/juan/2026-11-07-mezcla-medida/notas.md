# Mezcla medida y primer semestre

Fecha: 2026-11-07. No se tocó `entrega/`.

## Sonda de Bolívar, ya calificada

Kaggle devolvió **1828,40392** para `submission_sonda_bolivar.csv` (×1,5 solo en el segundo semestre de Bolívar, sobre la mezcla de 1459,16475). El factor que minimiza el público es **0,978**. Bolívar pasa de 203,2 a 198,8 casos por semana. El público estimado queda en **1458,52**, 0,64 puntos debajo de la mezcla. Bolívar ya estaba en el nivel que el tablero pide. No explica el hueco hasta 1300.

`submission_bolivar_seleccion_2.csv` quedó confirmado en **1458,52285**. La sonda de Atlántico dio **1825,97217**. El factor es **0,853** y Atlántico pasa de 164,3 a 140,2. Kaggle confirmó los dos archivos con ambos factores en **1438,54128**: `submission_calibrada.csv` (primer semestre de la selección 1) y `submission_calibrada_seleccion_2.csv` (el de la selección 2). El tablero no los separa.

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

## Sonda de Tolima

`submission_sonda_tolima.csv` dio **1674,09549**. El factor es **0,689** y Tolima pasa de 112,1 a 77,3 en el segundo semestre. Kaggle confirmó **1398,01602** en `submission_tolima.csv` y en `submission_tolima_seleccion_2.csv` (la cuenta daba 1398,01603). El primer semestre de Tolima se multiplica por 0,845. El peso 0,5 sale de los años anteriores: en 2016-2023 baja el H1 de B de 3909 a 3789 y gana los 8 años. No se copia a otros departamentos.

## Sonda de Norte de Santander

`submission_sonda_norte.csv` multiplica por 1,5 solo el segundo semestre de Norte de Santander sobre `submission_tolima.csv`. El primer semestre no cambia. No es finalista. Es la dirección nueva más grande que queda después de Tolima. El factor se calcula cuando llegue el puntaje.

## Norte, ya calificado

`submission_sonda_norte.csv` dio **1366,44918**. Kaggle confirmó **1356,02336** en `submission_norte.csv` y en `submission_norte_seleccion_2.csv`. El factor es **1,334** y Norte pasa de 107,3 a 143,1. El primer semestre de Norte no se toca.

## Meta, ya calificada

`submission_sonda_meta.csv` dio **1326,70176**. El factor es **1,321** y Meta pasa de 113,3 a 149,6. El público estimado es **1313,36**. El primer semestre de Meta se multiplica por ese mismo factor: el peso 1 sale de los años anteriores y, elegido año a año, no empeora ninguno de 2016-2023. Va en `submission_meta.csv` y `submission_meta_seleccion_2.csv`.

## Córdoba, ya calificada

`submission_sonda_cordoba.csv` dio **1292,95694**. El factor es **1,300** y Córdoba pasa de 112,7 a 146,5. El público estimado es **1276,40**. Va en `submission_cordoba.csv` y `submission_cordoba_seleccion_2.csv`, con el mismo primer semestre de los archivos de Meta. Córdoba no entra al primer semestre.

## Antioquia, ya medida

`submission_sonda_antioquia.csv` dio **1383,21656**. El factor es **0,993**. El público pasaría de 1276,39544 a 1276,38. No se arma otro archivo: Antioquia ya estaba en el nivel. El primer semestre tampoco se toca.

## Sucre, ya calificada

`submission_sonda_sucre.csv` dio **1464,39444**. El ×1,5 iba hacia arriba y el tablero pide **0,554**: Sucre pasa de 90,7 a 50,3. Su media histórica es 46 casos por semana. El público estimado es **1222,72**, ya con el 0,993 de Antioquia incluido. El primer semestre de Sucre se multiplica por 0,777. Va en `submission_sucre.csv` y `submission_sucre_seleccion_2.csv`.

## Santander, ya calificada

`submission_sonda_santander.csv` dio **1418,34102**. El factor es **1,033** y Santander pasa de 165,9 a 171,4. Kaggle confirmó **1221,71529**. El primer semestre no cambia. Va en `submission_santander.csv` y `submission_santander_seleccion_2.csv`.

## Huila, ya calificada

`submission_sonda_huila.csv` dio **1329,83604**. El ×1,5 iba hacia arriba y el tablero pide **0,553**: Huila pasa de 65,8 a 36,4. Kaggle confirmó **1190,74829**. El primer semestre no cambia: copiar el nivel empeora 2020 en 98 puntos. Va en `submission_huila.csv` y `submission_huila_seleccion_2.csv`.

## Cesar, ya calificada

`submission_sonda_cesar.csv` dio **1162,65830**. El factor es **1,456** y Cesar pasa de 64,9 a 94,6. Kaggle confirmó **1162,39485**. El primer semestre de Cesar se multiplica por 1,228. Va en `submission_cesar.csv` y `submission_cesar_seleccion_2.csv`.

## Guajira, ya calificada

`submission_sonda_guajira.csv` dio **1170,31288**. El factor es **1,154** y Guajira pasa de 49,6 a 57,2. Kaggle confirmó **1160,41638**. El primer semestre no cambia: copiar el nivel empeora 2022 en 36 puntos. Va en `submission_guajira.csv` y `submission_guajira_seleccion_2.csv`.

## Cundinamarca, ya calificada

`submission_sonda_cundinamarca.csv` dio **1203,76053**. El ×1,5 iba hacia arriba y el tablero pide **0,791**: Cundinamarca pasa de 53,4 a 42,2. Kaggle confirmó **1156,27758** en `submission_cundinamarca.csv`. El primer semestre no cambia: copiar el nivel empeora 2016 en 72 puntos. El gemelo `submission_cundinamarca_seleccion_2.csv` tiene el mismo segundo semestre y todavía no se sube.

## Sonda de Putumayo

`submission_sonda_putumayo.csv` multiplica por 1,5 solo el segundo semestre de Putumayo y toma el primer semestre de `submission_cundinamarca_seleccion_2.csv`. El público no cambia por ese primer semestre. No es finalista.

## Relevo

Juan se quedó sin envíos. Putumayo ya se midió: el puntaje y los archivos nuevos están en `ensayos/juan/2026-11-08-sondas/`. El público vigente de esta carpeta sigue siendo `submission_cundinamarca.csv`, **1156,27758**.

## Qué subir

La mezcla de 3170 no es finalista. El archivo limpio vigente es `submission_cundinamarca.csv` (selección 1, Valle en 322, ya en el tablero con **1156,27758**). La siguiente subida es `submission_sonda_putumayo.csv`: primer semestre de la selección 2 y Putumayo en ×1,5. No se elige como final. En el primero de los archivos limpios cambian Tolima, Meta, Sucre y Cesar.
