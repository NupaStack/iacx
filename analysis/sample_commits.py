import json, random, subprocess, os, collections
allc=json.load(open('audits/commits_pool.json'))
rng=random.Random(20261018)
def strat(pool,n,cap=3):
    rng.shuffle(pool); per=collections.Counter(); out=[]
    for c in pool:
        if per[c['repo']]>=cap: continue
        out.append(c); per[c['repo']]+=1
        if len(out)==n: break
    return out
pos=[c for c in allc if c['flag']]; neg=[c for c in allc if not c['flag']]
S=strat(pos,100)+strat(neg,100)
CL='mining/audit_clones'
for c in S:
    dest=os.path.join(CL,c['repo'].replace('/','__'))
    r=subprocess.run(['git','show','-s','--format=%B',c['hash']],cwd=dest,capture_output=True,text=True,errors='replace',timeout=60)
    body=r.stdout.strip()
    subj=c['subject']; c['body']=body[len(subj):].strip() if body.startswith(subj) else body
rng.shuffle(S)
for i,c in enumerate(S,1): c['id']='C%03d'%i
json.dump(dict(sample=S,pool_flagged=len(pos),pool_unflagged=len(neg),
               repos=len({c['repo'] for c in allc}),seed=20261018),open('audits/label_sample_key.json','w'),indent=1)
print('sampled',len(S),'| flagged',sum(c['flag'] for c in S),'| repos in sample',len({c['repo'] for c in S}))
print('pool flagged/unflagged',len(pos),len(neg))
