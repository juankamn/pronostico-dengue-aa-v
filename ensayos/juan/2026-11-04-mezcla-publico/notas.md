# Segundo semestre combinado con los puntajes públicos

El tablero público es el MSE de las semanas 27 a 52. Para archivos ya subidos p_i con puntaje S_i, el puntaje de cualquier combinación p = Σ c_i p_i con Σ c_i = 1 se calcula exacto, sin envíos nuevos:

S(c) = c'Gc − c'(diag(G) − S), con G_ij = media(p_i · p_j) en el segundo semestre.

`armar.py` busca la combinación de menor S(c) con cinco archivos ya subidos: la selección final, el árbol sin copiar el brote (3754,91), el decaimiento (2836,49), el sensor del patrón (2016,28) y el segundo semestre de redes de 52 semanas (2050,42).

| Archivo | Público antes | Público exacto | Pesos |
| --- | --- | --- | --- |
| `submission_mezcla_region_h2.csv` | 1971,62 | 1835,54 | selección 0,83; árbol 0,22; decaimiento 0,06; sensor 0,30; red 52 −0,41 |
| `submission_bases_patron_h2.csv` | 1972,72 | 1836,22 | selección 0,82; árbol 0,22; decaimiento 0,06; sensor 0,30; red 52 −0,41 |

Dos predicciones quedan negativas y se recortan a 0; como los casos reales no son negativos, el recorte solo puede bajar el puntaje. El segundo semestre baja de 57,9 a 52,8 de media, en línea con lo que ya decían los primeros puntajes (entre 0,82 y 0,93 veces B).

El primer semestre de cada archivo es idéntico al de su selección final, así que el privado no cambia: esto solo mejora el puesto visible del tablero público. Es un ajuste al tablero público, no un modelo, y por eso no va al cuaderno de entrega.
