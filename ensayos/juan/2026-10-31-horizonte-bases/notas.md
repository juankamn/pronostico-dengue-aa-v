# Horizontes con bases recientes

El envío de 3754,91 dejaba la Costa en el nivel de 2007 (Bolívar 26, Atlántico 50, Córdoba 32 en el segundo semestre) y subía Santander a 201 por doble conteo. Este ensayo separa semanas 1–13, 14–26 y 27–52.

La mezcla de bases, entrenada solo con el pasado y sin años de brote, baja el backtest del segundo semestre en más de 1000 puntos de MSE: años moderados de 3368 a 2335, y años posteriores a un brote de 3671 a 2576. El primer semestre, el del tablero privado, baja de 4715 a 2229 en los mismos años moderados.

Ridge, LightGBM, XGBoost y una red chica predicen el residuo ya normalizado. En el walk-forward no bajan el segundo semestre (2335 con la base sola, 2358 con un 15 % de residuo), así que el archivo no los suma.

`submission.csv` es el archivo vigente en Kaggle. Equipo Pandebono Predictors, tablero público: puesto 4, score 2048,49606. Media 61,9 (semestre 1: 65,8; semestre 2: 58,1). Segundo semestre: Santander 100, Valle del Cauca 193, Bolívar 222, Atlántico 165, Córdoba 122.

En los años moderados el error que queda en la Costa no es un nivel fijo: en 2017 y 2020 el signo cambia. Lo que pesa son los saltos de 2021 y 2022 (Atlántico cerca de 170 casos por debajo). Santander ya no queda 100 casos arriba todos los años. Los brotes de 2016, 2019, 2023 y 2024 siguen impredecibles. El público queda a unos 6 puntos del tercero y a unos 240 del primero.
