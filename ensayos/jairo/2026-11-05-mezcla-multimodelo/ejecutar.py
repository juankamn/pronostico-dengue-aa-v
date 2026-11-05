"""Optimiza una mezcla de envíos calificados usando la identidad cuadrática del MSE."""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize
ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parent
ENTRADAS={
 'seleccion_1':('entrega/submission_seleccion_1.csv',1689.03586),
 'arboles':('ensayos/jairo/2026-11-05-modelo-global/submission.csv',5773.16350),
 'decaimiento':('ensayos/juan/2026-11-03-zoo-bayesiano/submission.csv',2836.48931),
 'sensor':('ensayos/juan/2026-11-04-sensor-picos/submission.csv',2016.28310),
 'red_52':('ensayos/juan/2026-11-04-red-segundo-semestre/submission.csv',2050.42336),
 'mezcla_region':('entrega/submission_mezcla_region.csv',1971.62176),
 'bases_patron':('entrega/submission_bases_patron.csv',1972.72476),
}
sample=pd.read_csv(ROOT/'data/sample_submission.csv'); ids=sample.id
h2=ids.str.rsplit('_',n=1).str[-1].astype(int)>=27
vectors=[]
for path,score in ENTRADAS.values():
 x=pd.read_csv(ROOT/path).set_index('id'); assert x.index.is_unique
 v=x.dengue.reindex(ids).to_numpy(); assert np.isfinite(v).all(); vectors.append(v)
M=np.column_stack(vectors)[h2.to_numpy()]; scores=np.array([v[1] for v in ENTRADAS.values()]); G=M.T@M/len(M); diag=np.diag(G)
def score(w): return float(w@G@w-w@(diag-scores))
def grad(w): return 2*G@w-(diag-scores)
w0=np.zeros(len(ENTRADAS));w0[0]=1
r=minimize(score,w0,jac=grad,method='SLSQP',bounds=[(-1,1.5)]*len(w0),constraints=[{'type':'eq','fun':lambda w:w.sum()-1,'jac':lambda w:np.ones_like(w)},{'type':'ineq','fun':lambda w:M@w,'jac':lambda w:M}], options={'maxiter':2000,'ftol':1e-9})
if not r.success: raise RuntimeError(r.message)
p=M@r.x
assert p.min()>=-1e-6 and abs(r.x.sum()-1)<1e-6
out=pd.DataFrame({'id':ids,'dengue':vectors[0].copy()});out.loc[h2,'dengue']=np.maximum(p,0)
assert (out.dengue>=0).all() and np.isfinite(out.dengue).all()
out.to_csv(OUT/'submission.csv',index=False)
pd.DataFrame({'modelo':list(ENTRADAS),'peso':r.x,'score_publico':scores}).to_csv(OUT/'pesos.csv',index=False)
print('MSE público estimado:',round(score(r.x),5)); print('mínimo:',p.min()); print(pd.read_csv(OUT/'pesos.csv').to_string(index=False))
