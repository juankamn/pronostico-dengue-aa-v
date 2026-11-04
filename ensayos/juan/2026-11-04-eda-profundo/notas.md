# EDA profundo para el primer semestre

`eda.ipynb` busca en los datos lo que puede bajar el error del tablero privado (semanas 1 a 26 de 2025). No entrena modelos: usa los datos, B en walk-forward y las predicciones de `2026-11-04-red-dinamica-h1`.

- **Dónde está el error.** Valle del Cauca aporta el 39 % del error de B en el primer semestre (2014-2023) y seis departamentos del interior el 84 %. El riesgo de escenario de 2025 está en Valle del Cauca, Santander y Cundinamarca.
- **Ruido.** Un tercio del error de B (1120 de 3534) es ruido semanal que ningún pronóstico suave captura.
- **Estacionalidad.** La Costa tiene el pico en noviembre-diciembre y el primer semestre en 0,5 a 0,8 veces diciembre. Tolima, Huila y Cundinamarca tienen el pico entre enero y mayo. Valle del Cauca, en la semana 52, con mínimo en la 17.
- **Después de un brote.** A mayor brote, más baja el arranque siguiente (r = -0,22, débil). Una regresión con forma estacional, tamaño del brote y nivel de diciembre no le gana a B en walk-forward.
- **Clima.** A escala nacional, los brotes coinciden con primeros trimestres cálidos (r = 0,69 con la incidencia del año). El primer trimestre de 2025 está en la media, 1,3 °C más frío que el de 2024, y es el más lluvioso del registro. La señal no baja al departamento (r cercano a 0,1).
- **Calendario.** Los lotes de reporte vienen por rachas; Semana Santa baja ~10 % con mucha dispersión; año nuevo no tiene rebote sistemático.
- **Errores.** La mezcla con red gana en el interior (Valle del Cauca -186, Tolima -45, Santander -28) y pierde en la Costa (Atlántico +51). Los pesos estimados solo con años anteriores se estabilizan desde 2017 en ~0,6 para la red en el interior y ~0,2 en la Costa; para 2025 dan 0,70 y 0,21.

Propuestas, en orden: peso de la red por región; validar el segundo semestre de las redes con el tablero público; revisar la amplitud del patrón de reporte en el primer semestre.
