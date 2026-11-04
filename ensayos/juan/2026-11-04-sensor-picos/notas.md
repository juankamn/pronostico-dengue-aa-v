# Sensor del patrón mensual de reporte

De septiembre a diciembre de 2024, la semana que contiene el primer miércoles de cada mes carga un lote de reporte. En el interior sube: Valle del Cauca ×2,18, Huila ×2,48, Cauca ×2,05, Santander y Tolima ×1,71 frente a la media de su mes. En la Costa baja: Sucre ×0,36, Bolívar ×0,52, Atlántico ×0,61 y Córdoba ×0,67. El signo se mantiene en los cuatro meses. Hubo episodios parecidos en 2010, 2016, 2018, 2020 y 2021, a veces con otro calendario. En el segundo semestre de 2024, Valle sola aporta el 75 % del error que deja un pronóstico suave.

`submission.csv` es el envío de horizontes (2048,49606) con la mitad del patrón de 2024 en agosto a diciembre de 2025. Las semanas de lote son la 31, 35, 39, 44 y 48. Entran 13 departamentos con nivel de al menos 60 casos y signo estable; Antioquia, Meta y Quindío quedan por fuera. Los casos se mueven dentro de cada mes sin cambiar su total, así que el nivel de B no se toca. Julio y el primer semestre quedan iguales a B. `armar.py` lo reproduce.

## Cómo leer el puntaje público

Con beta como la fracción del patrón de 2024 que siguió en 2025:

`beta = (0,5 - (puntaje - 2048,49606) / 392,6) / 2`

- 2245: beta 0. El patrón no siguió o cambió de semana.
- 2048,5: beta 0,25.
- 1852: beta 0,5.
- 1460: beta 1. Con el patrón completo, en vez de la mitad, el público quedaría cerca de 1263.

Si beta sale en 0,25 o más, el patrón pasa al primer semestre de los dos candidatos finales con amplitud cercana a beta, en las semanas 5, 9, 13, 18, 22 y 26. Enero no tiene semana de lote dentro de 2025: su primer miércoles cae en la semana 53 de 2024, que salió baja. Si beta queda por debajo de 0,1, se descarta.

La lectura supone que el error de B dentro de cada mes no sigue el patrón. Como el interior y la Costa tienen signos opuestos, una tendencia común dentro del mes se cancela en parte.

## Resultado

Tablero público: 2016,28310, 32,2 puntos por debajo de B. Beta = 0,29: el patrón siguió en agosto a diciembre de 2025 con cerca del 29 % de su fuerza de 2024. El puntaje es cuadrático en la amplitud, así que en esta dirección el mejor público posible es 1981,99, con amplitud 0,291. El patrón no explica la distancia con el primero (1353,89); con beta 1 habría quedado cerca de 1263.

Pasa a los finalistas con amplitud 0,291: `2026-11-04-finalista-colapso` y `2026-11-04-finalista-continuidad`.
