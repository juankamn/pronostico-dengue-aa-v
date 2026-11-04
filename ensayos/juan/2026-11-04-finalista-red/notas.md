# Finalista con red

Primer semestre: 25 % TCN + 25 % MLP + 50 % B, con las redes de `2026-11-04-red-dinamica-h1` entrenadas con todo hasta 2024 (5 semillas). En el walk-forward 2014-2023 da 3316 de MSE en el primer semestre (3328 con otras tres semillas), frente a 3534 de B, y gana 7 de 10 años. TCN + B al 50 % queda igual de cerca (3323 y 3303); la mezcla de las dos redes reparte el riesgo entre una red que tiende a la continuidad (TCN) y otra que tiende a B (MLP). Segundo semestre: el de B. Encima va el patrón mensual de reporte con amplitud 0,291, de febrero a diciembre, igual que en los finalistas anteriores.

Primer semestre 2025 sin el patrón: media 67,3 y primer trimestre 73,0 (en el archivo, con el patrón, 67,4). B tenía 65,8 y 70,6. Valle del Cauca sube a 262 en el primer trimestre, Santander a 124 y Tolima a 145; Bolívar baja a 226, Atlántico a 197 y Meta a 160.

Tablero público esperado: cerca de 1982. El segundo semestre difiere del finalista de colapso en hasta 1,6 casos en julio, porque la media de julio incluye la semana 26.

Va como primera selección final, junto con `2026-11-04-finalista-continuidad-red`. Reemplaza a `2026-11-04-finalista-colapso`: en 2014-2023 el par con red habría dado 3315 y el par B + C, 3490.

`armar.py` construye este archivo y el de continuidad. Toma el patrón de `2026-11-04-finalista-colapso/armar.py` para que sea idéntico.

Estado: el cuaderno de entrega rehace este archivo con la misma receta en `entrega/submission_mezcla_red.csv` (hasta 18 casos de diferencia en el primer semestre, por la corrida de las redes). Ese es el que se sube, para que el cuaderno y Kaggle coincidan.
