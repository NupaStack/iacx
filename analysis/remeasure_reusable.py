import sys, os, importlib.util, pandas as pd, numpy as np
s=importlib.util.spec_from_file_location('v1','iacx/iacx.py'); v1=importlib.util.module_from_spec(s); s.loader.exec_module(v1)
sys.argv=['x','data/modules_corrected.csv']
src=open('analysis/analyse3.py').read(); exec(src.split("ALL = prep(full)")[0])
F=full.copy(); F['role']=F.module.map(role)
R=F[F.role.isin(('root','submodule','other'))].drop_duplicates(['repo','module'])
rows=[]
for r in R.itertuples():
    d=os.path.join('mining/clones',r.repo.replace('/','__'),'' if r.module=='(root)' else r.module)
    a=v1.analyse(d,recursive=False) if os.path.isdir(d) else None
    if a is None: rows.append(dict(repo=r.repo,module=r.module,v1_ok=0)); continue
    rows.append(dict(repo=r.repo,module=r.module,v1_ok=1,v1_cyclic=int(a['cyclic']),v1_blocks=a['n_blocks'],v1_nodes=a['nodes'],v1_edges=a['edges'],
        v1_two=a['two_paths'],v1_D=a['D_depth'],v1_P=a['P_propagation'],v1_X=a['X_expression'],v1_M=a['M_modularity'],v1_K=a['K_crossing'],v1_S=a['IaCX'],v1_loc=a['loc']))
x=pd.DataFrame(rows); x.to_csv('data/reusable_v1_measures.csv',index=False)
print('reusable rows',len(R),'measured',int(x.v1_ok.sum()),'cyclic',int(x.get('v1_cyclic',pd.Series()).fillna(0).sum()))
m=R.merge(x,on=['repo','module'])
ok=m[m.v1_ok==1]
print('loc identical %.1f%% | blocks identical %.1f%% | edges identical %.1f%% | X identical %.1f%%'%(
  100*(ok.loc_==ok.v1_loc).mean() if 'loc_' in ok else 100*(ok['loc']==ok.v1_loc).mean(), 100*(ok.n_blocks==ok.v1_blocks).mean(),100*(ok.edges==ok.v1_edges).mean(),100*(ok.X_expression==ok.v1_X).mean()))
