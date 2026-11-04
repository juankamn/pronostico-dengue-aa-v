# Decaimiento post-pico y zoo

El envío de 2048,50 trasladaba de más el cierre de 2024. La ventana 2019–2024 tiene dos olas, así que la mediana de cinco y seis años queda inflada en Valle, Santander, Tolima, Meta y Antioquia. Este ensayo no reemplaza esas bases: en un año que sigue a un pico nacional (2011, 2017, 2021 y 2025) las deja caer semana a semana.

El piso es el mínimo entre los dos años más bajos de los últimos cuatro y los tres más bajos de los últimos seis. Llegan ahí los departamentos que ya venían bajando después de dos años altos, o de un solo año extremo. La costa que creció de forma estructural (Bolívar, Atlántico y vecinos) no usa ese piso: en el segundo semestre se mezcla con su base reciente, para no arrastrar el diciembre de 2024.

Optuna, con TPE y semilla 42, sobre el walk-forward 2014–2024: tau 12,39, mezcla de costa 0,60, umbral de par 1,83, mega-brote 5,70, tope de cierre 1,08. Años moderados: H1 2159 y H2 2283 (antes 2229 y 2335). Años post-brote: H1 2758 y H2 2486 (antes 2880 y 2576). 2017 H2 baja de 968 a 617. 2021 H2 se queda en 3934: el salto de la costa de ese año no es un agotamiento.

El zoo son ocho modelos por horizonte. Cinco redes en PyTorch (MLP, RNN, LSTM, GRU, TCN) con una compuerta sobre las bases, para no salir de la escala. LightGBM, XGBoost (profundidad 3) y Ridge sin intercepto van aparte en el primer y en el segundo semestre. Una mezcla del 30 % con el decaimiento no mejora el segundo semestre. LightGBM baja un poco el primer semestre de los años moderados (2159 a 2131) pero sube el de los años post-pico, que son los que se parecen a 2025. Optuna repartió pesos en las semanas 1–13, 14–26 y 27–52, y en los tres tramos el peso quedó en el decaimiento.

`submission.csv` salió en el tablero público con 2836,48931. Es peor que 2048,50. 1612 filas, mismos ids que la muestra, sin nulos ni negativos. Media 45,0 (semestre 1: 51,1; semestre 2: 39,0). Segundo semestre: Bolívar 202, Atlántico 163, Valle del Cauca 118, Córdoba 100, Sucre 89, Meta 59, Tolima 52, Norte de Santander 50, Antioquia 37, Santander 33.

El zoo no iba en el archivo. El daño está en el decaimiento: Antioquia, Valle, Meta, Tolima y Santander concentran cerca del 77 % de la distancia al envío de 2048,50, y el 92 % de esa distancia es ortogonal a la recta entre los dos envíos ya calificados. Alpha 0,822 solo dice que no nos pasamos a lo largo de esa recta. El segundo semestre de 2025 no cayó hasta el piso de 2017. El mejor envío público sigue siendo el de horizontes.
