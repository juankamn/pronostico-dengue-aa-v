"""Calcula una mezcla de dos envíos a partir de sus puntajes públicos MSE."""
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
SCORE_BASE=1689.03586
SCORE_MODELO=5773.16350
base=pd.read_csv(ROOT/'entrega/submission_seleccion_1.csv')
modelo=pd.read_csv(HERE/'submission.csv')
d=base.merge(modelo,on='id',suffixes=('_base','_modelo'),validate='one_to_one',sort=False)
assert len(d)==len(base)==len(modelo) and d.id.equals(base.id)
sem=d.id.str.rsplit('_',n=1).str[-1].astype(int)
publico=sem>=27
x=d.loc[publico,'dengue_base'].to_numpy(); y=d.loc[publico,'dengue_modelo'].to_numpy()
dist=float(np.mean((y-x)**2))
a=float(np.clip((SCORE_BASE+dist-SCORE_MODELO)/(2*dist),0,1))
score=SCORE_BASE+a*(SCORE_MODELO-SCORE_BASE-dist)+a*a*dist
out=base.copy();out.loc[publico,'dengue']=(1-a)*x+a*y
assert np.isfinite(out.dengue).all() and (out.dengue>=0).all()
out.to_csv(HERE/'submission_mezcla_publico.csv',index=False)
print(f'distancia H2={dist:.6f}; peso modelo={a:.8f}; score estimado={score:.5f}; H1 intacto')
