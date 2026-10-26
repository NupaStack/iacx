"""Main comparisons with family-level grouping and with exact duplicates removed (A6)."""
import sys, json, pandas as pd
sys.argv = ['x', 'data/modules_corrected.csv', 'results/families']
src = open('analysis/analyse3.py').read()
exec(src.split("REUSE = ")[0]); exec(src[src.index('REUSE = '):src.index('ALL5 = prep(pop_rule(full, 5, 2, True))')])
P = pd.read_csv('results/v4/primary_dataset_v4.csv')
extra = full[['repo', 'module', 'age_days']].drop_duplicates(['repo', 'module'])
P = P.merge(extra, on=['repo', 'module'], how='left')
fam = pd.read_csv('data/repo_families.csv'); dup = pd.read_csv('data/exact_duplicates.csv')
def go(df, name):
    Q = Pop(prep(df.drop(columns=['nonfix', 'y'])), name)
    for k in ('base', 'base+sum', 'base+five', 'base+edges'): Q.heldout(MAIN[k], k)
    return dict(n=int(len(Q.d)), groups=int(Q.d.repo.nunique()), base_auc=round(Q.metrics(Q.pred['base'])['auc'], 4),
                d_sum=Q.diff('base+sum', 'base'), d_five=Q.diff('base+five', 'base'), d_edges=Q.diff('base+edges', 'base'))
F = P.merge(fam, on='repo'); F['module'] = F['repo'] + ':' + F['module']; F['repo'] = F['family']
R = {'family_grouped': go(F.drop(columns=['family']), 'family'),
     'exact_duplicates_removed': go(P.merge(dup[['repo', 'module']].assign(_d=1), on=['repo', 'module'], how='left').query('_d != 1').drop(columns=['_d']), 'dedup')}
json.dump(R, open('results/families/rerun.json', 'w'), indent=1); print(json.dumps(R))
