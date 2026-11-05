"""Comparación aproximada a selección 1; recrea apilado regional y señal H2 retrospectiva."""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize
ROOT=Path(__file__).resolve().parents[3]; HERE=Path(__file__).resolve().parent
P=ROOT/'ensayos/juan/2026-11-04-red-dinamica-h1'
key=['anio','id_departamento','semana']; coast=[4,5,11,13,16,19,28]
b=pd.read_parquet(P/'b_walkforward.parquet');r=pd.read_parquet(P/'predicciones_final5.parquet');r=r[r.familia.isin(['mlp','tcn'])].pivot_table(index=key,columns='familia',values='pred').reset_index()
t=r.merge(b,on=key,validate='one_to_one');t['costa']=t.id_departamento.isin(coast);t['apilado']=np.nan
for yr in range(2012,2025):
 for c in [False,True]:
  prev=t[(t.anio<yr)&(t.costa==c)&t.y.notna()]; idx=(t.anio==yr)&(t.costa==c)
  X=prev[['b','tcn','mlp']].to_numpy();y=prev.y.to_numpy()
  res=minimize(lambda w:np.mean((X@w-y)**2),np.ones(3)/3,method='SLSQP',bounds=[(0,1)]*3,constraints=[{'type':'eq','fun':lambda w:w.sum()-1}],options={'maxiter':1000})
  if not res.success:raise RuntimeError(res.message)
  t.loc[idx,'apilado']=t.loc[idx,['b','tcn','mlp']].to_numpy()@res.x
h2=b[b.semana.ge(27)&b.y.notna()].copy();h2['costa']=h2.id_departamento.isin(coast)
k=h2.groupby(['anio','costa']).apply(lambda z:float(z.b.to_numpy()@z.y.to_numpy()/(z.b.to_numpy()@z.b.to_numpy())),include_groups=False)
t['k']=[k.get((yr,c),np.nan) for yr,c in zip(t.anio,t.costa)]
t['finalista_aprox']=np.nan
for yr in range(2014,2025):
 prev=t[(t.anio>=2012)&(t.anio<yr)].dropna(subset=['apilado','k','y']);cur=t.anio==yr
 best=None
 for start in [1,4,8,13,17]:
  ramp=np.clip((prev.semana.to_numpy()-start)/(26-start),0,1);x=prev.apilado.to_numpy()*(prev.k.to_numpy()-1)*ramp;e=(prev.y-prev.apilado).to_numpy();f=np.clip(x@e/(x@x),0,1.5) if x@x else 0;loss=np.mean((prev.apilado.to_numpy()+f*x-prev.y.to_numpy())**2)
  if best is None or loss<best[0]:best=(loss,start,f)
 _,start,f=best;z=t.loc[cur];t.loc[cur,'finalista_aprox']=z.apilado*(1+f*(z.k-1)*np.clip((z.semana-start)/(26-start),0,1))
hist=pd.read_parquet(HERE/'predicciones_historicas.parquet');t=t.merge(hist[key+['pred']],on=key,validate='one_to_one')
rows=[]
for yr,g in t.groupby('anio'):
 if yr<2014:continue
 g=g.copy(); g['mezcla_25']=.75*g.finalista_aprox+.25*g.pred; g['mezcla_50']=.5*g.finalista_aprox+.5*g.pred
 for model in ['b','pred','finalista_aprox','mezcla_25','mezcla_50']:
  rows.append({'anio':yr,'modelo':model,'mse_h1':float(np.mean((g[model]-g.y)**2))})
out=pd.DataFrame(rows);out.to_csv(HERE/'comparacion_aproximada_h1.csv',index=False)
print(out.pivot(index='anio',columns='modelo',values='mse_h1').round(1).to_string())
print('\nPromedios 2016–2023:',out[out.anio.between(2016,2023)].groupby('modelo').mse_h1.mean().round(1).to_dict())
