# Sin reglas fijadas a mano

Pregunta: ¿qué pasa con B y con las selecciones finales si los umbrales puestos a mano se cambian por parámetros estimados con los años anteriores al evaluado?

## Regla de caída de B

`bases_variantes.py` corre B en walk-forward 2011-2025 con la regla de caída original (exceso 2,2, caída 0,65, razón 0,85) y sin ella, y guarda `b_walkforward_regla_si.parquet` y `b_walkforward_regla_no.parquet`.

| | Primer semestre 2014-2023 | Primer semestre 2016-2023 | Segundo semestre 2014-2023 |
| --- | --- | --- | --- |
| Con la regla original | 3534 | 3909 | 5164 |
| Sin la regla | 3720 | 4204 | 5193 |

Quitar la regla empeora. Lo que sobraba eran sus tres números fijos, así que la búsqueda pasó al cuaderno de entrega (etapa 3): 63 combinaciones, incluida no usar la regla, elegidas cada año con los años anteriores. Sale siempre exceso 1,8, caída 0,45 y razón 0,4, y B baja a 3386 en 2014-2023.

## En el cuaderno de entrega

- Corrección de nivel con semana de inicio y fuerza estimadas: apilado con nivel por región 3350 en 2016-2023 (para 2025, semana 4 y fuerza 0,35).
- B aprendida con nivel por región: 3352 en 2016-2023 y 3052 en 2014-2023 (semana 1 y fuerza 0,77). Es la selección 2.
- Dentro del apilado la B aprendida empeora, así que el apilado sigue con la B original.
- Patrón de reporte encogido hacia 1 en los 31 departamentos (Bayes empírico), en vez de los 13 elegidos con umbrales.
- La lista geográfica de la Costa le ganó a k-means (3235 frente a 3365) y a pesos por departamento (3357), con la corrección fija de entonces.
- Mejor de las dos selecciones cada año, 2016-2023: 3036.

## Reproducir

```bash
python ensayos/juan/2026-11-04-sin-reglas/bases_variantes.py
```
