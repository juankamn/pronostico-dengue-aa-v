# Perfil de brote contra perfil calmo

Fecha: 2026-11-07. No se sube.

La historia deja de recortarse a 2 y 2,8 veces la mediana. Cada departamento tiene un perfil de años de brote, uno de años calmos y uno de los últimos seis años, suavizados a cinco semanas. Se usa el de brote si el año anterior superó 1,8 veces su nivel y el cierre quedó por encima del pico. Se usa el calmo si ese año fue alto y el cierre quedó bajo 0,45 del pico. En los demás casos se usa el perfil reciente. Esos cortes salen de la regla de caída ya elegida y de la definición de cierre en subida. No se buscaron sobre los años de evaluación.

El único peso libre es la mezcla con B, en una rejilla de 0 a 1, ajustada solo con años anteriores. 2024 no entra en ese ajuste.

Resultado en el primer semestre: el peso sale 0 en todos los años de 2014 a 2024. En 2016–2023 el perfil puro tiene MSE 6609, contra 3909 de B. En la semana del pico real, el perfil llega a 0,38 del valor y B a 0,49. La mediana de los brotes viejos no alcanza el pico nuevo, y el error del resto del semestre es peor que el de B.

No reemplaza las selecciones. El segundo semestre público sigue en la mezcla de 1459,16.
