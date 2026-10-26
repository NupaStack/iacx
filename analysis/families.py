"""Related-module families across repositories for the V4 primary population (A6).
Two modules in different repositories are related when their normalised *.tf source is identical
or when the Jaccard similarity of their declaration-address sets is at least 0.8.
Repositories linked by any related pair form one family (union-find)."""
import os, re, json, hashlib, importlib.util, itertools, pandas as pd
s = importlib.util.spec_from_file_location('v1', 'iacx/iacx.py'); v1 = importlib.util.module_from_spec(s); s.loader.exec_module(v1)
P = pd.read_csv('results/v4/primary_dataset_v4.csv')
rows = []
for r in P.itertuples():
    d = os.path.join('mining/clones', r.repo.replace('/', '__'), '' if r.module == '(root)' else r.module)
    src = []
    for fn in sorted(os.listdir(d)):
        if fn.endswith('.tf') and os.path.isfile(os.path.join(d, fn)):
            t = open(os.path.join(d, fn), errors='replace').read()
            t = re.sub(r'#[^\n]*|//[^\n]*', '', t); t = re.sub(r'\s+', ' ', t).strip(); src.append(t)
    h = hashlib.sha1('\n'.join(src).encode()).hexdigest()
    nodes, _ = v1.parse_directory(d, recursive=False)
    rows.append(dict(repo=r.repo, module=r.module, h=h, addrs=frozenset(nodes)))
par = {}
def find(x):
    par.setdefault(x, x)
    while par[x] != x: par[x] = par[par[x]]; x = par[x]
    return x
def union(a, b): par[find(a)] = find(b)
pairs_exact = pairs_near = 0; dup_of = {}
for a, b in itertools.combinations(rows, 2):
    if a['repo'] == b['repo']: continue
    if a['h'] == b['h']:
        pairs_exact += 1; union(a['repo'], b['repo']); dup_of.setdefault((b['repo'], b['module']), (a['repo'], a['module'])); continue
    A, Bs = a['addrs'], b['addrs']
    if len(A) >= 5 and len(Bs) >= 5 and len(A & Bs) / len(A | Bs) >= 0.8:
        pairs_near += 1; union(a['repo'], b['repo'])
fam = {r: find(r) for r in P.repo.unique()}
out = pd.DataFrame({'repo': list(fam), 'family': [fam[r] for r in fam]})
out.to_csv('data/repo_families.csv', index=False)
pd.DataFrame([dict(repo=k[0], module=k[1], dup_repo=v[0], dup_module=v[1]) for k, v in dup_of.items()]).to_csv('data/exact_duplicates.csv', index=False)
sizes = out.family.value_counts()
S = dict(modules=len(rows), repos=int(P.repo.nunique()), exact_pairs=pairs_exact, near_pairs=pairs_near,
         families=int(len(sizes)), multi_repo_families=int((sizes > 1).sum()), repos_in_multi=int(sizes[sizes > 1].sum()),
         largest_family=int(sizes.iloc[0]), exact_duplicate_modules=len(dup_of))
json.dump(S, open('results/families/summary.json', 'w'), indent=1); print(S)
