import os, re, subprocess, random, json, sys
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
sys.path.insert(0,'mining')
from mine import FIX_POS, FIX_NEG, FEAT
CL='mining/audit_clones'
d=pd.read_csv('data/modules_corrected.csv')
p=d[(d.n_blocks>=5)&(d.commits>=2)&(d.edges>=1)]
mods={}
for r,m in zip(p.repo,p.module): mods.setdefault(r,set()).add('' if m=='(root)' else m)
repos=sorted(mods); rng=random.Random(20261018); rng.shuffle(repos)
target=repos[:110]
def sh(a,cwd=None,t=180): return subprocess.run(a,cwd=cwd,capture_output=True,text=True,timeout=t,errors='replace')
def work(repo):
    dest=os.path.join(CL,repo.replace('/','__'))
    try:
        if not os.path.isdir(dest+'/.git'):
            r=sh(['git','clone','--filter=blob:none','--no-checkout','--single-branch','--quiet',f'https://github.com/{repo}.git',dest],t=150)
            if r.returncode: return repo,[],'clone-fail'
        r=sh(['git','log','--no-merges','--pretty=format:@@@%H|%at|%an|%s','--name-only','--','*.tf'],cwd=dest,t=150)
        out=[];cur=None
        for line in r.stdout.splitlines():
            if line.startswith('@@@'):
                h,ts,au,su=line[3:].split('|',3); cur=dict(repo=repo,hash=h,ts=int(ts),author=au,subject=su,dirs=set())
                out.append(cur)
            elif line.strip() and cur and line.endswith('.tf'):
                cur['dirs'].add(os.path.dirname(line))
        keep=[]
        for c in out:
            hit=c['dirs'] & mods[repo]
            if not hit: continue
            s=c['subject']
            c['flag']=int(bool(FIX_POS.search(s)) and not FIX_NEG.search(s) and not FEAT.match(s))
            c['modules']=sorted(hit); del c['dirs']; keep.append(c)
        return repo,keep,'ok'
    except Exception as e:
        return repo,[],'err:'+type(e).__name__
allc=[];status={}
with ThreadPoolExecutor(10) as ex:
    for f in as_completed([ex.submit(work,r) for r in target]):
        repo,c,s=f.result(); status[repo]=s; allc+=c
print('repos tried',len(target),'ok',sum(v=='ok' for v in status.values()),'commits',len(allc),
      'flagged',sum(c['flag'] for c in allc))
json.dump(allc,open('audits/commits_pool.json','w'))
json.dump(status,open('audits/fetch_status.json','w'),indent=1)
