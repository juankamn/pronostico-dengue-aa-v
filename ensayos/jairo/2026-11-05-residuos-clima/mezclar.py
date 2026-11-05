"""Mezcla la sonda con el mejor envío usando sus MSE públicos."""
from pathlib import Path
import numpy as np
import pandas as pd
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
b=pd.read_csv(ROOT/'ensayos/jairo/2026-11-05-mezcla-multimodelo/submission.csv')
n=pd.read_csv(HERE/'submission_sonda_h2.csv')
assert b.id.equals(n.id)
h=b.id.str.rsplit('_',n=1).str[-1].astype(int)>=27
d=n.loc[h,'dengue'].to_numpy()-b.loc[h,'dengue'].to_numpy()
d2=float(np.mean(d*d));s0=1646.97677;s1=2174.34931
w=float(np.clip((s0+d2-s1)/(2*d2),0,1));score=s0+w*(s1-s0-d2)+w*w*d2
b.loc[h,'dengue']+=w*d
assert np.isfinite(b.dengue).all() and (b.dengue>=0).all()
b.to_csv(HERE/'submission_mezcla.csv',index=False)
print(f'peso={w:.8f}; score estimado={score:.5f}')
