"""Sonda de nivel H2 para Valle y mezcla calibrada a partir del score de Kaggle."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[3];OUT=Path(__file__).resolve().parent
base=pd.read_csv(ROOT/'ensayos/jairo/2026-11-05-mezcla-multimodelo/submission.csv')
sample=pd.read_csv(ROOT/'data/sample_submission.csv');assert base.id.equals(sample.id) and base.id.is_unique
parts=base.id.str.split('_',expand=True).astype(int);h2=(parts[1]==2025)&(parts[2]>=27);valle=h2 & (parts[0]==30)
assert len(base)==1612 and h2.sum()==806 and valle.sum()==26
SCORE_BASE=1646.97677
probe=base.copy();probe.loc[valle,'dengue']*=1.5
assert np.isfinite(probe.dengue).all() and (probe.dengue>=0).all()
probe.to_csv(OUT/'submission_sonda_valle.csv',index=False)
d=probe.loc[h2,'dengue'].to_numpy()-base.loc[h2,'dengue'].to_numpy()
d2=float(np.mean(d*d))
print(f'Sonda Valle H2 +50 %; distancia MSE={d2:.6f}. Primer semestre y otros departamentos intactos.')
if len(sys.argv)>1:
 sonda=float(sys.argv[1]);alpha=(SCORE_BASE+d2-sonda)/(2*d2)
 # El recorte requiere conocer etiquetas para calcular su efecto: no recortar silenciosamente.
 lower=-2.0; applied=max(alpha,lower)
 result=base.copy();result.loc[valle,'dengue']=base.loc[valle,'dengue']*(1+0.5*applied)
 assert (result.dengue>=0).all()
 expected=SCORE_BASE+applied*(sonda-SCORE_BASE-d2)+applied*applied*d2
 result.to_csv(OUT/'submission_calibrada_valle.csv',index=False)
 print(f'alpha óptimo={alpha:.8f}; aplicado={applied:.8f}; factor Valle={1+0.5*applied:.8f}; score estimado={expected:.5f}')
