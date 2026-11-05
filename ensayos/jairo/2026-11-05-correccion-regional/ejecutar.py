"""Evalúa una corrección histórica por departamento; no usa etiquetas del año evaluado."""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
DATA = ROOT / 'ensayos/juan/2026-11-04-red-dinamica-h1'
KEY = ['anio', 'id_departamento', 'semana']
b = pd.read_parquet(DATA / 'b_walkforward.parquet')
r = pd.read_parquet(DATA / 'predicciones_final5.parquet')
r = r[r.familia.isin(['mlp', 'tcn'])].pivot_table(index=KEY, columns='familia', values='pred').reset_index()
t = b.merge(r, on=KEY, validate='one_to_one')
t = t[t.semana.le(26) & t.y.notna()].copy()
t['base'] = .5*t.b + .25*t.mlp + .25*t.tcn
t['error'] = t.y - t.base
# Corrección media por departamento con contracción: 4 años equivalentes sin corrección.
def correccion(prev):
    e = prev.groupby(['id_departamento','anio']).error.mean().groupby('id_departamento').agg(['sum','count'])
    return e['sum'] / (e['count'] + 4)

filas=[]
for year in range(2016, 2025):
    pre=t[t.anio < year]; actual=t[t.anio == year].copy()
    if actual.empty: continue
    actual['ajuste'] = actual.id_departamento.map(correccion(pre)).fillna(0)
    for peso in (0, .25, .5, 1):
        pred=np.maximum(0,actual.base + peso*actual.ajuste)
        filas.append({'anio':year,'peso':peso,'mse_h1':float(np.mean((actual.y-pred)**2))})
res=pd.DataFrame(filas)
res.to_csv(HERE/'validacion.csv',index=False)
print(res.pivot(index='anio',columns='peso',values='mse_h1').round(1).to_string())
print('\nMSE promedio 2022-2024:')
print(res[res.anio.between(2022,2024)].groupby('peso').mse_h1.mean().round(1).to_string())
# Candidato exploratorio: traslada la corrección estimada al finalista actual.
# La validación anterior usa la mezcla proxy, no este finalista exacto.
sample = pd.read_csv(ROOT / 'data/sample_submission.csv')
final = pd.read_csv(ROOT / 'entrega/submission_seleccion_1.csv')
assert final.id.is_unique and sample.id.is_unique
out = sample[['id']].merge(final, on='id', validate='one_to_one')
assert len(out) == len(sample) and out.dengue.notna().all()
parts=out.id.str.split('_',expand=True).astype(int)
mask=(parts[1] == 2025) & (parts[2] <= 26)
shift=parts[0].map(correccion(t[t.anio < 2025])).fillna(0)
out.loc[mask,'dengue'] = np.maximum(0,out.loc[mask,'dengue'] + .5*shift[mask])
assert np.isfinite(out.dengue).all() and (out.dengue >= 0).all()
assert np.array_equal(out.loc[~mask,'dengue'].to_numpy(),final.set_index('id').loc[out.loc[~mask,'id'],'dengue'].to_numpy())
out.to_csv(HERE/'submission.csv',index=False)
print('\nSubmission exploratoria generada; H2 idéntico a selección 1.')
