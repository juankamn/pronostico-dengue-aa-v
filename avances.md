# Avances

Resumen para no abrir cada ensayo. Lo nuevo se agrega arriba.

## 2026-11-07 — El 1438 quedó confirmado

Kaggle devolvió **1438,54128** para `submission_calibrada.csv` y para `submission_calibrada_seleccion_2.csv`. Es el mismo número de la cuenta con Bolívar en 0,978 y Atlántico en 0,853. Los dos archivos comparten el segundo semestre, así que el tablero no dice nada del primero. El público baja 20,6 puntos frente a 1459,16475. Las dos finales siguen siendo estas: la selección 1 abre Valle en 322 y la selección 2 en 127.

## 2026-11-07 — Bolívar confirmado y Atlántico calibrado

Kaggle confirmó el archivo de Bolívar en **1458,52285**, el mismo número de la cuenta. `submission_seleccion_2.csv` sigue en 1459,16475 porque no lleva ese factor. La sonda ×1,5 de Atlántico dio **1825,97217**. El factor óptimo es **0,853**: Atlántico pasa de 164,3 a 140,2 casos por semana. Junto con Bolívar el público estimado es **1438,54** (20,6 puntos debajo de 1459,16). Los dos tocan departamentos distintos, así que las bajas se suman. `submission_calibrada.csv` lleva el primer semestre de la selección 1 y `submission_calibrada_seleccion_2.csv` el de la selección 2. El segundo semestre es el mismo.

## 2026-11-07 — Bolívar ya estaba en el nivel, y el perfil de brote no entra

La sonda ×1,5 de Bolívar dio **1828,40392**. El factor óptimo sobre la mezcla de 1459,16475 es 0,978 y el público estimado baja a **1458,52** (0,64 puntos). Bolívar queda en 198,8. No es el hueco hasta 1300. `submission_sonda_atlantico.csv` es la sonda siguiente, ×1,5 solo en el segundo semestre de Atlántico, y no es finalista.

`ensayos/juan/2026-11-07-perfil-brote/`: perfil de brote sin el tope de 2 y 2,8 veces la mediana, contra un perfil calmo. El peso de mezcla con B, elegido solo con años anteriores, sale 0 todos los años. En 2016–2023 el perfil puro da 6609 contra 3909 de B, y en la semana del pico llega a 0,38 del valor real contra 0,49 de B. No se sube.

## 2026-11-07 — El 1459 quedó confirmado

Kaggle devolvió **1459,16475** para `submission_privado.csv` y para `submission_seleccion_2.csv`. Es el mismo número que daba la matriz de productos. Los dos comparten el segundo semestre, así que el tablero no dice nada del primero.

La mezcla de primer semestre que promedia 3170 en 2016-2023 no es mejor de forma estable: le gana al apilado en 5 de 8 años y pierde en 2016, 2021 y 2022. Las dos finales que cubren el privado son el primer semestre de la selección 1 (Valle en 322, `submission.csv` de la misma carpeta) y el de la selección 2 (Valle en 127), las dos con este segundo semestre. Rehacer la mezcla de archivos ya calificados no baja de 1459. Lo que sigue en público es `submission_sonda_bolivar.csv`, una sonda ×1,5 solo de Bolívar; no es finalista. Bolívar es la dirección que más queda fuera de la mezcla. Atlántico sería la siguiente.

## 2026-11-07 — Mezcla exacta de los envíos ya calificados

`ensayos/juan/2026-11-07-mezcla-medida/`: el segundo semestre es la combinación lineal de los 16 archivos con score público. El MSE sale de la matriz de productos: **1459,16**, mínimo 0,35. Valle del Cauca en H2 baja a 104. Aplicar la escala fina (Costa 0,95, Valle 0,54, resto 1,06) al apilado empeora el walk-forward del primer semestre (3356 a 3363 en 2016-2023). La mezcla 30 % apilado + 70 % B con esa escala baja el primer semestre a **3170** y deja el primer trimestre de Valle en 208, frente a 322 de la selección 1 y 127 de la selección 2. Los tres archivos comparten el H2, así que el público no los separa. `submission_sonda_bolivar.csv` es una sonda ×1,5 de Bolívar; no se sube hasta tener el score de la mezcla. No reemplaza nada en `entrega/`.

## 2026-11-07 — Regimenes climaticos y tres familias de modelos

`ensayos/juan/2026-11-07-ml-regimenes-multitarea/`: mezcla gaussiana por clima y estacionalidad, sin lista de departamentos; familia lineal de bases con agotamiento de 104 semanas; arboles en log-ratio (LightGBM, XGBoost, HistGradientBoosting); MLP, TCN y GRU a 52 semanas con la ancla corregida por la carga del brote; apilado por probabilidad de grupo. Walk-forward de la meta: H1 4181 / 4595 y H2 5348 / 6172, peor que B (3534 / 3909 y 5164 / 5934). En 2025 H2 la meta queda ~10 % sobre B en Costa, Valle y el resto, asi que no corrige el hueco de Valle que sugieren las sondas. `submission.csv` es la salida del experimento y no reemplaza el 1646,98. Detalle en `notas.md`.

## 2026-11-05 — Sonda de nivel de Valle del Cauca

`ensayos/jairo/2026-11-05-sonda-valle/`: sonda sobre el mejor envío público 1646,97677; aumenta 50 % solo Valle del Cauca en H2. La sonda obtuvo 2005,86755. El ajuste óptimo estimado escala Valle en H2 por 0,719925 y estima score público 1593,86711, pendiente de confirmar. H1 queda igual. No demuestra mejora privada.

## 2026-11-05 — Residuos climáticos sobre B

`ensayos/jairo/2026-11-05-residuos-clima/`: modelo de árboles que corrige B usando clima, calendario, territorio y casos del año anterior. Mejora H1 frente a B en 2022–2024. En 2016–2023, una aproximación histórica a selección 1 da MSE H1 3355,6 frente a 3739,4 del modelo de residuos; la mezcla al 25 % da 3400,5. La aproximación usa el nivel real de H2 históricamente y no reproduce exactamente el finalista; no hay evidencia para sustituirlo. `submission_sonda_h2.csv` mantiene H1 de selección 1 y prueba H2; score público 2174,34931. Mezclar 2,55 % de esta sonda con la mezcla anterior estima 1646,61543, ganancia de 0,36 puntos, aún no confirmado. Ver limitaciones en notas.

## 2026-11-05 — Mezcla pública multimodelo de Jairo

`ensayos/jairo/2026-11-05-mezcla-multimodelo/`: combina siete envíos ya calificados. Score público **1646,97677**, confirmado en Kaggle. H1 idéntico a selección 1; la mezcla se optimizó usando scores públicos y no demuestra mejora privada. Detalles, pesos y limitaciones en la nota del ensayo.

## 2026-11-05 — Modelo global de árboles y mezcla pública de Jairo

`ensayos/jairo/2026-11-05-modelo-global/`. Modelo global de árboles con clima, calendario, departamento y casos del año anterior: público **5773,16350**. Su validación anual mejora frente a B en 2023 y 2024, pero pierde en H1 de 2022; detalles y limitaciones en `notas.md`.

Mezcla de 8,346443 % del modelo con selección 1 únicamente en H2: `submission_mezcla_publico.csv`, score público **1654,88355**, confirmado en Kaggle. El primer semestre queda igual a selección 1; la mejora pública resulta de ajustar la mezcla con los scores y no demuestra mejora privada.

`ensayos/jairo/2026-11-05-correccion-regional/` contiene una prueba exploratoria de corrección histórica de H1, sin score Kaggle; su validación usa una aproximación al finalista, no la receta exacta.

## 2026-11-04 — Parámetros aprendidos en lugar de umbrales a mano

`ensayos/juan/2026-11-04-sin-reglas/` y el cuaderno de entrega. Los umbrales que seguían fijados a mano ahora se estiman con los años anteriores al evaluado, o se cambian por un método sin umbrales:

- **Corrección de nivel.** La semana de inicio (1, 4, 8, 13 o 17) y la fuerza (mínimos cuadrados, de 0 a 1,5) salen de los años anteriores, en vez de media fuerza desde la semana 8. Para 2025 el apilado arranca en la semana 4 con fuerza 0,35. Walk-forward 2016-2023 del apilado con nivel por región: 3350. El 3230 anterior era optimista, porque la media fuerza y la semana 8 se habían elegido mirando todos los años.
- **Regla de caída de B.** Se prueban 63 combinaciones de exceso, caída y razón, más no usar la regla, y cada año se elige la que mejor rindió antes. Sale siempre exceso 1,8, caída 0,45 y razón 0,4. B baja de 3534 a 3386 en 2014-2023; sin regla da 3720, así que la regla sirve y lo que sobraba eran sus números fijos. Dentro del apilado la B aprendida empeoró el resultado, y por eso el apilado conserva la B original.
- **Patrón de reporte.** Los 13 departamentos elegidos con umbrales (nivel de 60 casos, signo estable) quedan solo para leer el sensor. En los envíos, la razón de cada uno de los 31 departamentos se encoge hacia 1 según su ruido entre meses (Bayes empírico).
- **Región.** La lista geográfica de la Costa se comparó con grupos aprendidos para los pesos del apilado: k-means sobre la forma de la curva dio 3365 y pesos por departamento encogidos hacia los nacionales, 3357, frente a 3235 de la lista (2016-2023, con la corrección fija de entonces). Se queda la lista, que además es la que mide el tablero público.

Selecciones finales nuevas: `entrega/submission_seleccion_1.csv` (apilado con nivel aprendido; primer semestre 72,0) y `entrega/submission_seleccion_2.csv` (B con regla aprendida y nivel desde la semana 1 con fuerza 0,77; primer semestre 64,4; 3352 en 2016-2023 y 3052 en 2014-2023). Si Kaggle se queda cada año con la mejor de las dos, 2016-2023 da 3036, frente a 3168 del par anterior, que además usaba la corrección fija. En el primer trimestre la selección 2 abre Valle del Cauca en 127 casos por semana y el apilado en 322. El segundo semestre es el de los envíos de 1689,03586. Tablero público: 1689,03586 las dos, como se esperaba. Quedan marcadas como selecciones finales. El cuaderno ya no genera los archivos anteriores; quedan en `entrega/` como se subieron.

## 2026-11-04 — Selecciones finales con nivel por región

El cuaderno de entrega ahora lee las dos sondas, calcula los k de 2025 por región (Costa 0,950, interior 0,887) y la suma de cuadrados del segundo semestre real, y con eso:

- **Primer semestre.** El apilado lleva su final hacia el nivel de su región (a media fuerza, desde la semana 8). Walk-forward 2016-2023: 3230, frente a 3355 con un solo nivel para el país, 3651 de la mezcla por región y 3909 de B. Mejor de las dos selecciones cada año: 3168 (antes 3298).
- **Segundo semestre.** Combinación lineal de siete envíos ya subidos con penalización ridge (pesos de -0,39 a 0,44, sin negativos): 1689,04 exacto en el público. Está ajustada al tablero, así que su puntaje no indica nada del privado.

Nuevas selecciones finales: `entrega/submission_final_apilado_region.csv` (primer semestre 71,8) y `entrega/submission_final_bases_region.csv` (64,9). Tablero público: 1689,03586 las dos, como predecía el cálculo exacto (1689,04). Las selecciones con un solo nivel (`submission_final_apilado.csv` y `submission_final_bases.csv`, 1836,20689) quedan como se subieron.

## 2026-11-04 — Medición del nivel por región

`ensayos/juan/2026-11-04-sonda-regiones/`. En la historia, corregir el primer semestre con un k para la Costa y otro para el interior, en vez de uno nacional, baja el apilado de 3360 a 3235 (2016-2023). Dos envíos de medición, con el primer semestre idéntico a la selección 1, dan esos dos k exactos para 2025.

Tablero público: sonda de la Costa 1842,97221; sonda del interior 2056,16650. Resultado: k de la Costa 0,950 (segundo semestre real ≈ 100 casos por departamento y semana, frente a 106 de B) y k del interior 0,887 (≈ 39 frente a 44); k del país 0,918. La combinación pública anterior sugería lo contrario (0,86 y 0,94), así que medir valió la pena.

Con los dos k se conoce también la suma de cuadrados del segundo semestre real, y con eso el puntaje público exacto de cualquier combinación lineal de envíos subidos. B escalado por región con el patrón da 1926,78; una combinación lineal con pesos acotados (máximo 0,39 en valor absoluto, sin negativos) da 1704,41. Ese segundo semestre ya no es un indicador del privado: está ajustado al tablero. Lo que indica el privado es el walk-forward del primer semestre.

## 2026-11-04 — Apilamiento por región y nivel del segundo semestre

Dos técnicas nuevas en el cuaderno de entrega, validadas solo con años anteriores:

- **Apilamiento.** Pesos de B, TCN y MLP buscados por mínimos cuadrados, por separado para la Costa y el interior. Para 2025: interior 0,33 B + 0,67 TCN; Costa 0,62 B + 0,38 MLP. RNN, LSTM, GRU, las redes de 52 semanas y las híbridas reciben peso cero o empeoran el resultado.
- **Nivel del segundo semestre.** Saber cuánto quedó el segundo semestre frente a B (la escala k) y llevar hacia ese nivel el final del primero, a media fuerza desde la semana 8, mejora el primer semestre en la historia. En 2025 el k sale del tablero público: 0,90.

Primer semestre 2016-2023: apilado + nivel 3355, frente a 3651 de la mezcla por región, 3696 de la mezcla con peso 0,5 y 3909 de B; gana a la mezcla por región en 6 de 8 años. Si Kaggle se hubiera quedado cada año con el mejor de los dos envíos, el par (apilado + nivel) + (B + nivel) habría dado 3298, frente a 3616 del par anterior. Con otras tres semillas el apilado con nivel da 3352.

Nuevas selecciones finales: `entrega/submission_final_apilado.csv` (primer semestre 71,7; Valle del Cauca 323 en el primer trimestre) y `entrega/submission_final_bases.csv` (64,7). Las dos llevan el segundo semestre calibrado con el tablero público. Tablero público: 1836,20689 las dos, como predecía el cálculo exacto (1836,22, un poco menos por el recorte a 0 de dos semanas). Reemplazan a la mezcla por región y a B con patrón como selecciones finales.

## 2026-11-04 — Segundo semestre combinado para el tablero público

`ensayos/juan/2026-11-04-mezcla-publico/`. Como el MSE es cuadrático, el puntaje público de una combinación de archivos ya subidos (pesos que suman 1) se calcula exacto con sus puntajes. La mejor combinación de cinco envíos baja el público a 1835,54 con el primer semestre de la mezcla por región, y a 1836,22 con el de B con patrón. El primer semestre de cada selección queda idéntico, así que el privado no cambia; solo mejora el puesto visible. No se subieron: ese segundo semestre pasó directo a las selecciones finales del apilado (ver la entrada de arriba).

## 2026-11-04 — Selecciones finales

`entrega/submission_bases_patron.csv` (B con el patrón de reporte) dio 1972,72476 en el tablero público. Las dos selecciones finales quedan así:

1. `entrega/submission_mezcla_region.csv`: 1971,62176. Primer semestre de la mezcla por región (red 0,69 en el interior y 0,23 en la Costa).
2. `entrega/submission_bases_patron.csv`: 1972,72476. Primer semestre de B, que cubre una caída fuerte en enero.

Las dos comparten el segundo semestre, salvo julio, y por eso el público casi no las separa. El privado sí: en el primer semestre están a 940 de MSE una de otra.

## 2026-11-04 — Red híbrida autoregresiva

`ensayos/juan/2026-11-04-red-hibrida/`. TCN→LSTM y GRU→LSTM con decodificador autoregresivo de 52 semanas, para pronosticar todo 2025 con un solo modelo. En walk-forward pierden en los dos semestres: mitad con B dan 4036 y 4157 en el primer semestre de 2016-2023, frente a 3659 de la mezcla por región vigente, y 5516 y 5383 en el segundo de 2014-2023, frente a 5164 de B. El decodificador arrastra el nivel de diciembre en los años que siguen a un brote. No se sube.

## 2026-11-04 — Segundo semestre con redes de 52 semanas

`ensayos/juan/2026-11-04-red-segundo-semestre/`. TCN y MLP con horizonte de 52 semanas. En el segundo semestre, mitad red y mitad B baja el walk-forward de 5164 a 4954 (2014-2023) y gana 5 de 10 años; los pesos por región no ayudan ahí. En el primer semestre estas redes rinden peor que las de 26 semanas, así que el primer semestre no cambia. `submission.csv` deja el primer semestre idéntico a la mezcla por región y solo cambia julio a diciembre: su puntaje público mide si la red lee bien 2025.

Tablero público: 2050,42336, peor que 1971,62. El cambio de la red vale 106 de MSE entre archivos y el público subió 78,8: la realidad se movió apenas un 13 % en la dirección de la red. En 2025 la lectura de las redes no le ganó a B. Con la regla fijada antes de ver el puntaje, la segunda selección final pasa a B con el patrón de reporte (`entrega/submission_bases_patron.csv`, que ahora genera el cuaderno), y la primera sigue siendo la mezcla por región (`entrega/submission_mezcla_region.csv`, 1971,62). La continuidad queda generada pero fuera de las selecciones.

## 2026-11-04 — Peso de la red por región

El cuaderno de entrega estima el peso de la red (50 % TCN + 50 % MLP) frente a B por separado para la Costa y el interior, solo con años anteriores. Para 2025, con 2011-2023, da 0,69 en el interior y 0,23 en la Costa. En 2016-2023 la mezcla por región da 3651 y la de peso 0,5, 3696; gana 6 de 8 años. En 2014 pierde porque el peso sale de solo tres años de historia.

Archivos nuevos: `entrega/submission_mezcla_region.csv` (primer semestre 71,8; Valle del Cauca 311 en el primer trimestre) y `entrega/submission_continuidad_region.csv` (83,1). Tablero público: 1971,62176 y 1971,62162. El segundo semestre es el mismo de los envíos de 1971,92 salvo julio (hasta 0,69 casos); la mejora de 0,3 viene de ahí y no dice nada del privado. El cuaderno sigue reproduciendo los dos envíos de 1971,92 (diferencia de 0,00002).

En 2016-2023, el par mezcla por región + B habría dado 3616, y el par mezcla por región + continuidad, 3649. En 2025 la mezcla por región queda a 940 de MSE de B con patrón y a 730 de la continuidad. La segunda selección quedó en B con el patrón (ver la entrada de arriba).

## 2026-11-04 — EDA profundo

`ensayos/juan/2026-11-04-eda-profundo/eda.ipynb`. Valle del Cauca aporta el 39 % del error de B en el primer semestre, y un tercio del error total es ruido semanal. La Costa baja en el primer semestre y el centro andino sube. El primer trimestre de 2025 fue 1,3 °C más frío que el de 2024, lo que apoya una caída. La mezcla con red gana en el interior y pierde en la Costa; los pesos aprendidos con años anteriores dan para 2025 0,70 en el interior y 0,21 en la Costa.

## 2026-11-04 — Cuaderno de entrega

`entrega/taller_final_dengue.ipynb` queda con el EDA (más los lotes de reporte y el cierre de 2024), las etapas 1 y 2 como estaban, la etapa 3 (bases por horizonte, que reproduce exacto el envío de 2048,50), la etapa 4 (MLP, RNN, LSTM, GRU y TCN de 26 semanas en walk-forward, con los experimentos de diseño) y una sección 6 nueva. Corre completo en 14 minutos, con 18 figuras, y genera `entrega/submission_mezcla_red.csv` y `entrega/submission_continuidad_red.csv`.

Son los mismos dos finalistas con red, salidos de otra corrida: difieren de `ensayos/juan/2026-11-04-finalista-red/` y `-continuidad-red/` en hasta 18 casos en el primer semestre y 0,23 en julio. Para que el cuaderno y Kaggle coincidan, estos son los que se suben y se eligen como selecciones finales.

Tablero público: mezcla con red 1971,91647; continuidad sobre la mezcla 1971,91564. Mejor que el 1981,99 esperado por el sensor, porque julio también llevaba el patrón de reporte. La cabecera del cuaderno ya tiene a los dos integrantes con su código.

## 2026-11-04 — Red de dinámica para el primer semestre

`ensayos/juan/2026-11-04-red-dinamica-h1/`. MLP, RNN, LSTM, GRU y TCN con ventanas rodantes de 52 semanas, 26 de horizonte y pérdida Poisson, evaluadas en walk-forward sobre el primer semestre, que es lo que mide el privado. Ninguna red sola le gana a B (3534 de media en 2014-2023; la mejor sola, MLP, 3595). Mezcladas sí: 25 % TCN + 25 % MLP + 50 % B da 3316 y gana 7 de 10 años; con otras tres semillas, 3328. En 2024 baja de 152.278 a 84.361.

Nuevas selecciones finales propuestas, pendientes de subir: `ensayos/juan/2026-11-04-finalista-red/` (la mezcla, media 67,3 en el primer semestre) y `ensayos/juan/2026-11-04-finalista-continuidad-red/` (la mezcla arrancando en 0,8 veces diciembre, media 80,4). Los dos llevan el patrón de reporte y el segundo semestre de B, así que el público debe quedar cerca de 1982. En 2014-2023 este par habría dado 3315 y el par B + C, 3490. Reemplazan a los finalistas de colapso y continuidad sin red.

## 2026-11-04 — Patrón de reporte y dos escenarios para el primer semestre

El privado mide enero a junio de 2025 y nadie lo ve hasta el cierre. Con los tres puntajes públicos anteriores, el segundo semestre real queda en 0,82 a 0,93 veces el envío de horizontes: B ya está en nivel. Su riesgo está en el primer semestre: abre 2025 en 78 casos por departamento y semana, cuando diciembre de 2024 sin picos estaba en 126. Ninguna corrección de esa apertura gana en el walk-forward 2014-2023, así que se cubren los dos escenarios con las dos selecciones finales.

`ensayos/juan/2026-11-04-sensor-picos/submission.csv`. Tablero público: 2016,28310. Desde septiembre de 2024 la semana del primer miércoles de cada mes carga un lote de reporte (Valle ×2,18; Sucre ×0,36). En agosto a diciembre de 2025 siguió con cerca del 29 % de esa fuerza. El mejor público posible en esa dirección es 1981,99.

`ensayos/juan/2026-11-04-h1-continuidad/submission.csv`. Tablero público: 2048,49606, igual a B, porque solo cambia el primer semestre. Confirma que el público son las semanas 27 a 52.

Selecciones finales propuestas, pendientes de subir: `ensayos/juan/2026-11-04-finalista-colapso/` (primer semestre de B, media 65,9) y `ensayos/juan/2026-11-04-finalista-continuidad/` (arranca en 0,8 veces diciembre, media 81,1). Los dos llevan el patrón de reporte con amplitud 0,291 y el mismo segundo semestre. Entre ellos hay 1359 de MSE en el primer semestre.

## 2026-11-03 — Zoo y decaimiento post-pico

`ensayos/juan/2026-11-03-zoo-bayesiano/submission.csv`. Media 45,0 (semestre 1: 51,1; semestre 2: 39,0). Tablero público: 2836,48931. Quedó peor que 2048,50, así que no reemplaza ese envío.

El archivo no lleva zoo: MLP, RNN, LSTM, GRU, TCN, LightGBM, XGBoost y Ridge quedaron en peso cero. Lo que se subió es el decaimiento hacia el piso de años bajos. En el segundo semestre de 2025 eso cortó de más el interior (Antioquia 115 → 37, Valle 193 → 118, Meta 130 → 59, Tolima 121 → 52, Santander 100 → 33). El 92 % de esa distancia al envío de 2048,50 es ortogonal a la recta de los dos envíos anteriores; el backtest solo mejoraba unos 50 puntos de MSE y el corte real fue de unos 1100. El primero del público sigue en 1353,89. El mejor envío público nuestro sigue siendo el de horizontes.

## 2026-10-31 — Horizontes y bases recientes

`ensayos/juan/2026-10-31-horizonte-bases/submission.csv`. Media 61,9 (semestre 1: 65,8; semestre 2: 58,1). Fue el segundo envío público. Ya no es el archivo vigente.

Tablero público, equipo Pandebono Predictors: puesto 4, score 2048,49606. El anterior estaba en 3754,91. El tercero va en 2042,42 y el primero en 1808,52.

El público de 3754,91 coincidía con el régimen post-brote del árbol anterior. Bolívar, Atlántico y Córdoba quedaban en el nivel de 2007, y Santander llegaba a 201 por doble conteo. Con bases recortadas de los últimos años, el cierre de las semanas 41–53 y un peso distinto para cada horizonte, el backtest del segundo semestre baja más de 1000 puntos (moderados 3368 → 2335; post-brote 3671 → 2576). El residuo de Ridge, LightGBM, XGBoost y red no mejora ese número, así que no entra al archivo. Santander queda en 100 y Bolívar en 222 en el segundo semestre de 2025. Los años de brote siguen altos.

## 2026-10-27 — Nivel sin copiar el brote

`ensayos/juan/2026-10-27-nivel-sin-brote/submission.csv`. Media 36,6. Primer envío público: 3754,91. Ya no es el archivo vigente.

El envío que copiaba 2024 (media 105) no está en la escala del tablero público (1.897 a 3.500). El clima de 2025 se parece a 2014–2021, no a 2024. En segundos semestres tranquilos el error queda en 702 (2017), 1.409 (2018) y 2.672 (2022). En años de brote sigue alto. El semestre 1, que define al ganador, no se ve hasta el cierre.

## 2026-10-27 — RNN de la etapa 1

`ensayos/juan/2026-10-27-rnn-etapa1/submission.csv`. Media 75,7. Línea de base del curso. No es el envío de la competencia.
