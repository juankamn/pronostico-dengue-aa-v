# Regimenes climaticos y pronostico multitarea

Fecha: 2026-11-07. No se toco `entrega/`.

## Que se probo

Un solo sistema, en `modelo.py` y `competencia_dengue.ipynb`, sin listas de departamentos.

- Cada ano se ajusta una mezcla gaussiana con el clima, la presion, la fase estacional, la escala y la autocorrelacion de los anos anteriores. El numero de grupos sale del BIC (2 a 4). El apilado usa las probabilidades, no una etiqueta fija.
- Familia A: mezcla convexa de las bases historicas por bloque de horizonte (semanas 1-13, 14-26, 27-52), mas un termino acotado del componente climatico nacional. Si el exceso de 104 semanas es alto y el cierre ya esta por debajo del pico, las bases recientes se acercan a la mediana. La fuerza de ese acercamiento se elige con los anos anteriores. Para 2025 salio 0,4.
- Familia B: LightGBM, XGBoost e HistGradientBoosting sobre el log-ratio contra la familia A. Entran las razones entre bases, el estado del ciclo, el clima de la semana y sus rezagos, el componente climatico nacional y las probabilidades de grupo. No entra el id del departamento.
- Familia C: MLP, TCN y GRU a 52 semanas, con compuerta entre el nivel reciente y el perfil estacional. La ancla de esa compuerta baja solo cuando el brote es largo y ya viene cayendo.
- Meta: mezcla convexa de las cinco salidas, con pesos distintos por grupo. Luego un empalme del primer semestre con el nivel que el propio modelo predice para el segundo, y un encogimiento de Bayes del efecto del primer miercoles del mes.

## Validacion

MSE medio. La referencia es la B de horizontes ya medida.

| Modelo | H1 2014-2023 | H1 2016-2023 | H2 2014-2023 | H2 2016-2023 |
| --- | ---: | ---: | ---: | ---: |
| B de horizontes | 3534 | 3909 | 5164 | 5934 |
| Familia A | 4272 | 4710 | 5349 | 6173 |
| Familia B | 4176 | 4609 | 5472 | 6221 |
| MLP | 4864 | 5581 | 6102 | 7199 |
| TCN | 4635 | 5094 | 6296 | 7168 |
| GRU | 4451 | 5000 | 5346 | 6188 |
| Meta | 4181 | 4595 | 5348 | 6172 |

Ninguna familia ni la meta bajan el error historico de B. La red de 52 semanas mezclada a medias con B, que ya estaba medida, sigue mejor en H2 (4954 / 5753).

## 2025, solo con las sondas ya calificadas

Medias del segundo semestre. El nivel implicado por las sondas es aproximado: Costa cerca de 0,95 veces B, Valle cerca de la mitad, el resto un poco por encima de B.

| Region | B | Meta | Meta / B | Lectura de las sondas |
| --- | ---: | ---: | ---: | --- |
| Costa | 105,7 | 119,7 | 1,13 | cerca de 100 |
| Valle | 192,9 | 211,7 | 1,10 | cerca de 101 |
| Resto | 37,7 | 42,9 | 1,14 | cerca de 40 |

La meta queda alrededor de un 10 % por encima de B en las tres regiones. No corrige el hueco de Valle y sube la Costa y el resto, que las sondas no piden subir tanto. El agotamiento aprendido (fuerza 0,4) evita el arrastre extremo de las redes, que sin ese ancla dejaban a Valle cerca de 400, pero no alcanza para llevarlo al nivel que sugieren las sondas. Una fuerza mayor empeoraba el walk-forward, asi que no se forzo.

`submission.csv` tiene 1612 filas, el orden de `sample_submission.csv`, sin negativos (minimo 0,98). Medias de 2025: H1 65,7 y H2 65,7.

## Decision

No reemplaza el envio de 1646,98. El archivo queda como salida del experimento, no como candidato a subir. El agrupamiento si separa solo, a partir del clima y la estacionalidad, un bloque de interior caliente (Valle, Antioquia, Santander, Tolima, Huila) de otro bloque donde caen varios departamentos de la Costa. Eso no hizo falta escribirlo a mano, pero los pesos de ese grupo siguen prefiriendo la persistencia que gano en 2024, y 2025 no se parece a 2024.
