"""Corrección de B con un modelo global de residuos y validación temporal anual."""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parent
key=['id_departamento','anio','semana']
train=pd.read_parquet(ROOT/'data/train.parquet'); test=pd.read_parquet(ROOT/'data/test.parquet')
b=pd.read_parquet(ROOT/'ensayos/juan/2026-11-04-red-dinamica-h1/b_walkforward.parquet')
assert not b.duplicated(key).any()
lag=train[key+['casos_dengue']].copy();lag['anio']+=1;lag=lag.rename(columns={'casos_dengue':'lag_anual'})
frame=pd.concat([train,test],ignore_index=True).merge(lag,on=key,how='left',validate='one_to_one').merge(b[key+['b']],on=key,how='inner',validate='one_to_one')
assert len(frame[frame.anio==2025])==1612
clima=['precip_chirps_mm','temp_media_C','temp_max_C','temp_min_C','punto_rocio_C','humedad_suelo_capa1','evapotranspiracion_mm','radiacion_solar_MJ_m2','viento_vel_m_s','presion_superficial_hPa']
a=2*np.pi*frame.semana/52; frame['sem_sin']=np.sin(a);frame['sem_cos']=np.cos(a)
a=np.deg2rad(frame.viento_dir_grados);frame['viento_sin']=np.sin(a);frame['viento_cos']=np.cos(a)
X=pd.concat([frame[clima+['b','lag_anual','sem_sin','sem_cos','viento_sin','viento_cos']],pd.get_dummies(frame.id_departamento,prefix='d',dtype=float)],axis=1)
rows=[]
historico=[]
for year in list(range(2014,2026)):
 fit=(frame.anio<year)&frame.casos_dengue.notna()&frame.lag_anual.notna(); valid=frame.anio==year
 model=HistGradientBoostingRegressor(max_iter=100,learning_rate=.04,max_leaf_nodes=7,min_samples_leaf=100,l2_regularization=30,early_stopping=False,random_state=42)
 print('Entrenando',year,flush=True);model.fit(X.loc[fit],(frame.loc[fit,'casos_dengue']-frame.loc[fit,'b']))
 pred=np.maximum(frame.loc[valid,'b'].to_numpy()+0.5*model.predict(X.loc[valid]),0)
 o=frame.loc[valid,key+['casos_dengue','b']].copy();o['pred']=pred
 if year<2025:
  historico.append(o.copy())
  for part,df in [('H1',o[o.semana<=26]),('H2',o[o.semana>26]),('global',o)]:
   rows.append({'anio':year,'periodo':part,'mse_residuo':np.mean((df.pred-df.casos_dengue)**2),'mse_B':np.mean((df.b-df.casos_dengue)**2)})
  print(rows[-3:],flush=True)
 else:
  out=o[key+['pred']].copy();out['id']=out.id_departamento.astype(str)+'_'+out.anio.astype(str)+'_'+out.semana.astype(str)
  sample=pd.read_csv(ROOT/'data/sample_submission.csv');out=sample[['id']].merge(out[['id','pred']],on='id',validate='one_to_one');assert out.pred.notna().all()
  out.rename(columns={'pred':'dengue'}).to_csv(OUT/'submission_modelo.csv',index=False)
  best=pd.read_csv(ROOT/'ensayos/jairo/2026-11-05-mezcla-multimodelo/submission.csv');assert sample.id.equals(best.id)
  h2=sample.id.str.rsplit('_',n=1).str[-1].astype(int)>=27
  best.loc[h2,'dengue']=out.loc[h2,'pred'];best.to_csv(OUT/'submission_sonda_h2.csv',index=False)
pd.DataFrame(rows).to_csv(OUT/'validacion.csv',index=False)
pd.concat(historico,ignore_index=True).to_parquet(OUT/'predicciones_historicas.parquet',index=False)
