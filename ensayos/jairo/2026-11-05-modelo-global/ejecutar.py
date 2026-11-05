"""Modelo global por árboles, con validación anual y pronóstico directo a 52 semanas."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
train = pd.read_parquet(ROOT / 'data/train.parquet')
test = pd.read_parquet(ROOT / 'data/test.parquet')
print('Columnas:', train.columns.tolist(), flush=True)
print('Datos:', train.shape, test.shape, flush=True)

CLIMA = ['precip_chirps_mm','temp_media_C','temp_max_C','temp_min_C','punto_rocio_C',
         'humedad_suelo_capa1','evapotranspiracion_mm','radiacion_solar_MJ_m2',
         'viento_vel_m_s','presion_superficial_hPa']
keys = ['id_departamento','anio','semana']
assert not train.duplicated(keys).any() and not test.duplicated(keys).any()
history=train[keys+['casos_dengue']].copy()
lag=history.copy(); lag['anio'] += 1; lag=lag.rename(columns={'casos_dengue':'casos_ano_anterior'})
last=history.sort_values(keys).groupby(['id_departamento','anio'],as_index=False).tail(8)
level=last.groupby(['id_departamento','anio']).casos_dengue.agg(['mean','median','max']).reset_index()
level['anio'] += 1
level=level.rename(columns={'mean':'cierre_media','median':'cierre_mediana','max':'cierre_maximo'})
frame=pd.concat([train,test],ignore_index=True).merge(lag,on=keys,how='left',validate='one_to_one').merge(level,on=['id_departamento','anio'],how='left',validate='many_to_one')
angle=2*np.pi*frame.semana/52
frame['semana_sin']=np.sin(angle); frame['semana_cos']=np.cos(angle)
angle=np.deg2rad(frame.viento_dir_grados)
frame['viento_sin']=np.sin(angle); frame['viento_cos']=np.cos(angle)
features=CLIMA+['casos_ano_anterior','cierre_media','cierre_mediana','cierre_maximo','semana_sin','semana_cos','viento_sin','viento_cos']
X=pd.concat([frame[features],pd.get_dummies(frame.id_departamento,prefix='dep',dtype=float)],axis=1)
base=pd.read_parquet(ROOT/'ensayos/juan/2026-11-04-red-dinamica-h1/b_walkforward.parquet')
rows=[]
for year in [2022,2023,2024,2025]:
    fit=(frame.anio<year)&frame.casos_dengue.notna()&frame.casos_ano_anterior.notna()
    valid=frame.anio.eq(year)
    model=HistGradientBoostingRegressor(loss='squared_error',max_iter=150,learning_rate=.05,max_leaf_nodes=15,min_samples_leaf=40,l2_regularization=10,early_stopping=False,random_state=42)
    print('Entrenando',year,fit.sum(),flush=True)
    model.fit(X.loc[fit],frame.loc[fit,'casos_dengue'])
    p=np.maximum(model.predict(X.loc[valid]),0)
    o=frame.loc[valid,keys+['casos_dengue']].copy();o['pred']=p
    if year<2025:
        o=o.merge(base,on=keys,how='left',validate='one_to_one')
        o=o[o.b.notna()].copy()  # B cubre 52 semanas; compara las mismas filas.
        for name,part in [('H1',o[o.semana<=26]),('H2',o[o.semana>26]),('global',o)]:
            rows.append({'anio':year,'periodo':name,'mse_modelo':mean_squared_error(part.casos_dengue,part.pred),'mse_B':mean_squared_error(part.casos_dengue,part.b)})
        print(rows[-3:],flush=True)
    else:
        submission=pd.DataFrame({'id':frame.loc[valid,'ID'].astype(str),'dengue':p})
        sample=pd.read_csv(ROOT/'data/sample_submission.csv')
        submission=sample[['id']].merge(submission,on='id',how='left',validate='one_to_one')
        assert len(submission)==len(sample) and submission.dengue.notna().all() and np.isfinite(submission.dengue).all()
        submission.to_csv(OUT/'submission.csv',index=False)
pd.DataFrame(rows).to_csv(OUT/'validacion.csv',index=False)
print('Finalizado',flush=True)
