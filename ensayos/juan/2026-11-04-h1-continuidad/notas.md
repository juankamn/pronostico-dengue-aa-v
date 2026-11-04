# Primer semestre con continuidad desde diciembre

El envío de horizontes abre 2025 en 78 casos por departamento y semana. En diciembre de 2024, sin contar los picos de reporte, el nivel era 126. De octubre a diciembre la base nacional sin picos se quedó plana, cerca de 3.900 casos por semana. En 2010 y 2016 esa base ya venía cayendo en el último trimestre y enero siguió la caída sin salto. En 2024 no venía cayendo.

En 22 departamentos-año en que B abrió por debajo de 0,6 veces el nivel de diciembre, las semanas 1 a 4 reales quedaron en 0,66 veces de mediana; B había dicho 0,46. Nunca bajaron de 0,38. En 2025, B abre Valle del Cauca en 0,35, Santander en 0,29, Cundinamarca en 0,25 y Risaralda en 0,07.

Eso no basta para cambiar B. En el walk-forward 2014-2023 ninguna corrección de la apertura baja el MSE medio del primer semestre: B queda en 3534 y las correcciones parciales o de un solo lado, entre 3538 y 3738. 2017, el año posterior a un brote más parecido, empeora. Este archivo no compite con B: cubre el escenario contrario.

## Construcción

Donde B abre por debajo de 0,8 veces la mediana de las semanas 45 a 52 de 2024, el primer semestre arranca en 0,8 veces ese nivel y baja con una curva coseno hasta tocar a B en la semana 27. Solo sube, nunca baja. Cambian 19 departamentos. El segundo semestre es idéntico a B. `armar.py` lo reproduce.

Primer semestre: 65,8 → 81,0. Primer trimestre: 70,6 → 96,4. Valle del Cauca 179 → 357, Santander 94 → 225, Cundinamarca 52 → 137, Bolívar 247 → 289, Risaralda 7 → 62. Atlántico, Córdoba, Meta y Norte de Santander quedan iguales, porque B ya abre cerca o por encima de su nivel de diciembre. Entre B y este archivo hay 1329 de MSE en el primer semestre.

El 0,8 deja a este archivo más cerca de la realidad que B cuando la apertura real queda por encima de unas 0,55 veces diciembre; la mediana histórica está entre 0,58 y 0,66.

## Uso

En el tablero público tiene que salir exactamente 2048,49606, porque el segundo semestre es el de B. Si sale otro número, la partición pública no es la semana 27 a 52.

Va como segunda selección final junto con B. Kaggle se queda con el mejor de los dos en el privado: si 2025 colapsó en enero como en 2011 o 2017, gana B; si siguió desde diciembre como en 2020, gana este. Si el sensor de picos muestra que el patrón de reporte siguió en 2025, se rehacen los dos con el patrón en el primer semestre.

## Resultado

Tablero público: 2048,49606, igual a B. Confirma que el público solo mide las semanas 27 a 52. El sensor dio beta 0,29, así que este archivo queda reemplazado por `2026-11-04-finalista-continuidad`, que es este mismo escenario con el patrón de reporte.
