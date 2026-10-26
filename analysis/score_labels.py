import json, glob, os, sys, numpy as np, collections
sys.argv=['x','data/modules_corrected.csv']
src=open('analysis/analyse3.py').read(); ns={}
exec(src[src.index('def role'):src.index('def prep')],ns); role=ns['role']
K=json.load(open('audits/label_sample_key.json'))
S={c['id']:c for c in K['sample']}
def load(r):
    out={}
    for f in glob.glob(f'audits/ratings/label/{r}/items/*.json'):
        d=json.load(open(f)); body=d.get('data',d); out[os.path.basename(f)[:-5]]=body
    return out
A,B=load('A'),load('B')
ids=sorted(S)
va=[(A.get(i) or {}).get('v') for i in ids]; vb=[(B.get(i) or {}).get('v') for i in ids]
print('rated A',sum(v is not None for v in va),'B',sum(v is not None for v in vb))
cats=['Yes','No','Unsure']
def kappa(x,y,cats):
    n=len(x); po=sum(a==b for a,b in zip(x,y))/n
    pe=sum((x.count(c)/n)*(y.count(c)/n) for c in cats); return po,(po-pe)/(1-pe)
pairs=[(a,b) for a,b in zip(va,vb) if a and b]
x=[p[0] for p in pairs]; y=[p[1] for p in pairs]
po3,k3=kappa(x,y,cats)
xb=['Y' if v=='Yes' else 'N' for v in x]; yb=['Y' if v=='Yes' else 'N' for v in y]
po2,k2=kappa(xb,yb,['Y','N'])
tab=collections.Counter(zip(x,y))
print('3-cat agreement %.3f kappa %.3f | binary agreement %.3f kappa %.3f'%(po3,k3,po2,k2))
print('confusion A x B:',{f'{a}/{b}':n for (a,b),n in sorted(tab.items())})
print('marginals A',collections.Counter(x),' B',collections.Counter(y))
# consensus labels
R=[]
for i,a,b in zip(ids,va,vb):
    c=S[i]; reus=any(role(m if m else '(root)') in ('root','submodule','other') for m in c['modules'])
    R.append(dict(id=i,repo=c['repo'],flag=c['flag'],reus=reus,a=a,b=b,
        strict=int(a=='Yes' and b=='Yes'), lenient=int(a=='Yes' or b=='Yes'),
        agreed=(a==b and a in ('Yes','No'))))
pool_f,pool_u=K['pool_flagged'],K['pool_unflagged']; pf=pool_f/(pool_f+pool_u)
rng=np.random.default_rng(20261026)
def est(rows,key):
    F=[r for r in rows if r['flag']]; U=[r for r in rows if not r['flag']]
    prec=np.mean([r[key] for r in F]); miss=np.mean([r[key] for r in U])
    rec=pf*prec/(pf*prec+(1-pf)*miss) if (pf*prec+(1-pf)*miss)>0 else np.nan
    return prec,miss,rec,len(F),len(U)
def boot(rows,key,B=2000):
    repos=sorted({r['repo'] for r in rows}); by={p:[r for r in rows if r['repo']==p] for p in repos}; out=[]
    for _ in range(B):
        rs=[r for p in rng.choice(repos,len(repos)) for r in by[p]]
        F=[r for r in rs if r['flag']]; U=[r for r in rs if not r['flag']]
        if not F or not U: continue
        out.append(est(rs,key)[:3])
    o=np.array(out); return np.nanpercentile(o,[2.5,97.5],axis=0)
res={}
for lab,rows in [('all',R),('reusable',[r for r in R if r['reus']]),('fixtures',[r for r in R if not r['reus']])]:
    for key in ('strict','lenient'):
        p,m,rc,nf,nu=est(rows,key); ci=boot(rows,key)
        res[f'{lab}/{key}']=dict(n_flagged=nf,n_unflagged=nu,precision=round(p,3),precision_ci=[round(ci[0][0],3),round(ci[1][0],3)],
            missed_rate=round(m,3),missed_ci=[round(ci[0][1],3),round(ci[1][1],3)],recall_est=round(rc,3),recall_ci=[round(ci[0][2],3),round(ci[1][2],3)])
        r=res[f'{lab}/{key}']
        print('%-18s n=%3d/%3d  precision %.3f [%.3f,%.3f]  unflagged-repair %.3f [%.3f,%.3f]  recall %.3f [%.3f,%.3f]'%(
            lab+'/'+key,nf,nu,r['precision'],*r['precision_ci'],r['missed_rate'],*r['missed_ci'],r['recall_est'],*r['recall_ci']))
out=dict(n=len(ids),pool_flagged=pool_f,pool_unflagged=pool_u,agreement_3cat=round(po3,3),kappa_3cat=round(k3,3),
         agreement_binary=round(po2,3),kappa_binary=round(k2,3),confusion={f'{a}/{b}':n for (a,b),n in tab.items()},
         disagreements=sum(1 for r in R if r['a']!=r['b']),estimates=res,
         n_repos=len({r['repo'] for r in R}),n_reusable=sum(r['reus'] for r in R))
json.dump(out,open('results/label_audit_results.json','w'),indent=1)
json.dump(R,open('results/label_audit_rows.json','w'),indent=1)
