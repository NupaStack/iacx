"""
analyse4: primary analysis of the V4 revision.

Population: reusable modules (root, submodule, other directory roles), at least five declared blocks,
at least two .tf file-change commits, at least one resolved edge, acyclic under tool v1.0.0.
Measures: recomputed with tool v1.0.0 (literal-safe reference and construct scanning) at the
reconstructed snapshot of each repository.
Same models, folds, bootstrap and metrics as analyse3.
"""
import sys, json, os, numpy as np, pandas as pd, hashlib
sys.argv = ['x', 'data/modules_corrected.csv', 'results/v4']
src = open('analysis/analyse3.py').read()
exec(src.split("REUSE = ")[0])
exec(src[src.index('REUSE = '):src.index('ALL5 = prep(pop_rule(full, 5, 2, True))')])
OUT = 'results/v4'; os.makedirs(OUT, exist_ok=True)
R = {'seed': SEED, 'bootstrap_resamples': B, 'tool': '1.0.0'}

v1 = pd.read_csv('data/reusable_v1_measures.csv')
base = full.copy(); base['role'] = base.module.map(role)
base = base[base.role.isin(('root', 'submodule', 'other'))].drop_duplicates(['repo', 'module'])
base = base.merge(v1, on=['repo', 'module'], how='inner')
base = base[base.v1_ok == 1].copy()
R['reusable_all'] = int(len(base)); R['reusable_cyclic'] = int(base.v1_cyclic.sum())
for a, b in [('n_blocks', 'v1_blocks'), ('nodes', 'v1_nodes'), ('edges', 'v1_edges'), ('two_paths', 'v1_two'),
             ('D_depth', 'v1_D'), ('P_propagation', 'v1_P'), ('X_expression', 'v1_X'),
             ('M_modularity', 'v1_M'), ('K_crossing', 'v1_K'), ('IaCX', 'v1_S')]:
    base[a] = base[b]
acyc = base[base.v1_cyclic == 0].copy()
# remine extras: distinct commits, bot commits, first commit time, snapshot time
dd = pd.read_csv('data/remeasured_directories.csv')
acyc = acyc.merge(dd[['repo', 'module', 'snapshot_ct', 'first_at', 'dcommits', 'dfix', 'dfix_nobot', 'bot']],
                  on=['repo', 'module'], how='left')

def sel(df, blk, cmt, edge):
    return df[(df.n_blocks >= blk) & (df.commits >= cmt) & ((df.edges >= 1) if edge else True)]

def clean(df):
    return df.drop(columns=[c for c in df.columns if c.startswith('l_') or c in ('y', 'nonfix', 'has_edge') or c.startswith('role_')], errors='ignore')

P0 = sel(acyc, 5, 2, True)
cyc_excluded = int(len(sel(base, 5, 2, True)) - len(P0))

prim, R['primary'] = run_pop(prep(clean(P0)), 'v4-primary', full_report=True)
R['primary']['cyclic_excluded'] = cyc_excluded
R['primary']['spearman'] = spearman_table(prim)
R['primary']['describe'] = describe(prim)
R['primary']['coefficients'] = {k: prim.coefs(MAIN[k]) for k in ['base', 'base+sum', 'base+five', 'base+edges', 'edgebase+D+twostep_term'] + ['base+' + lab for _, _, lab in FACT]}
d = prim.d
R['primary']['rho_D_P'] = round(float(stats.spearmanr(d.D_depth, d.P_propagation)[0]), 3)
R['primary']['rho_P_loc'] = round(float(stats.spearmanr(d.P_propagation, d.sloc)[0]), 3)
R['primary']['rho_M_K'] = round(float(stats.spearmanr(d.M_modularity, d.K_crossing)[0]), 3)
R['primary']['rho_X_loc'] = round(float(stats.spearmanr(d.X_expression, d.sloc)[0]), 3)
R['primary']['rho_edges_loc'] = round(float(stats.spearmanr(d.edges, d.sloc)[0]), 3)
R['primary']['role_counts'] = d.role.value_counts().to_dict()
R['primary']['loo_vs_five'] = {lab: prim.diff('LOO-' + lab, 'base+five') for _, _, lab in FACT}

# leakage on identical folds
lk = prep(clean(P0), activity='commits')
pl = Pop(lk, 'v4-leaky', folds=prim.folds)
for k in ('base',): pl.heldout(MAIN[k], k)
R['primary']['leakage'] = dict(base_total_commits=round(pl.metrics(pl.pred['base'])['auc'], 4),
                               base_nonfix=round(R['primary']['models']['base']['auc'], 4))

# repeated grouped CV
rng = np.random.default_rng(SEED); reps = []
for r in range(20):
    perm = rng.permutation(prim.U); fo = {repo: i % 10 for i, repo in enumerate(perm)}
    f = np.array([fo[x] for x in prim.g]); a = {}
    for k in ('base', 'base+sum', 'base+five', 'base+edges'): a[k] = roc_auc_score(prim.y, prim.heldout(MAIN[k], None, folds=f))
    reps.append({k: a[k] - a['base'] for k in ('base+sum', 'base+five', 'base+edges')})
R['primary']['repeated_cv'] = {k: dict(mean=round(float(np.mean([x[k] for x in reps])), 4), sd=round(float(np.std([x[k] for x in reps], ddof=1)), 4),
    min=round(float(min(x[k] for x in reps)), 4), max=round(float(max(x[k] for x in reps)), 4), n_positive=int(sum(x[k] > 0 for x in reps)))
    for k in ('base+sum', 'base+five', 'base+edges')}

# exposure age (snapshot minus first commit) and the span variable it replaces
pe = P0[P0.snapshot_ct.notna() & P0.first_at.notna()].copy()
pe['exposure_days'] = (pe.snapshot_ct - pe.first_at) / 86400.0
Pe = Pop(prep(clean(pe)).assign(l_expo=lambda z: np.log1p(((z.snapshot_ct - z.first_at) / 86400.0).clip(lower=0))), 'v4-exposure')
mods = {'base': BASE, 'base+expo': BASE + ['l_expo'], 'base+expo+sum': BASE + ['l_expo', 'l_sum'],
        'base+expo+five': BASE + ['l_expo'] + FIVE, 'base+expo+edges': BASE + ['l_expo', 'l_edg'], 'base+span': BASE + ['l_span']}
for k, c in mods.items(): Pe.heldout(c, k)
R['exposure'] = dict(n=int(len(Pe.d)), repos=int(Pe.d.repo.nunique()), median_days=round(float(pe.exposure_days.median()), 1),
    models={k: round(Pe.metrics(Pe.pred[k])['auc'], 4) for k in mods},
    diffs={'expo | base': Pe.diff('base+expo', 'base'), 'sum | base+expo': Pe.diff('base+expo+sum', 'base+expo'),
           'five | base+expo': Pe.diff('base+expo+five', 'base+expo'), 'edges | base+expo': Pe.diff('base+expo+edges', 'base+expo')})

# distinct-commit counting and bot exclusion
def rerun(df, label, fixcol):
    x = df[df.dcommits.notna()].copy(); x['fix_commits'] = x[fixcol]; x['commits'] = x['dcommits']
    x = x[(x.commits >= 2)]
    Q = Pop(prep(clean(x)), label)
    for k in ('base', 'base+sum', 'base+five', 'base+edges'): Q.heldout(MAIN[k], k)
    return dict(n=int(len(Q.d)), repos=int(Q.d.repo.nunique()), events=int(Q.y.sum()), base_auc=round(Q.metrics(Q.pred['base'])['auc'], 4),
                d_sum=Q.diff('base+sum', 'base'), d_five=Q.diff('base+five', 'base'), d_edges=Q.diff('base+edges', 'base'))
cand = sel(acyc, 5, 0, True)
R['distinct_commits'] = rerun(cand, 'v4-distinct', 'dfix')
R['distinct_commits_nobot'] = rerun(cand, 'v4-distinct-nobot', 'dfix_nobot')
R['bot_share'] = round(float(P0.bot.sum() / max(1, P0.dcommits.sum())), 4)
R['filechange_per_commit'] = round(float(P0.commits.sum() / max(1, P0.dcommits.sum())), 3)

# sensitivity to inclusion rules within reusable modules
R['sensitivity'] = []
for lab, blk, cmt, edge in [('>=3 blocks, >=2 commits', 3, 2, True), ('>=3 blocks, >=5 commits', 3, 5, True),
                            ('>=5 blocks, >=2 commits', 5, 2, True), ('>=5 blocks, >=5 commits', 5, 5, True),
                            ('>=10 blocks, >=2 commits', 10, 2, True), ('>=10 blocks, >=5 commits', 10, 5, True),
                            ('no resolved-edge criterion', 5, 2, False), ('all reusable, no thresholds', 0, 0, False)]:
    Q, res = run_pop(prep(clean(sel(acyc, blk, cmt, edge))), 'v4 ' + lab)
    res['rule'] = lab
    res['rho'] = {c: round(float(stats.spearmanr(Q.d[c], Q.d.fix_commits)[0]), 3) for c in ('sloc', 'IaCX', 'D_depth')}
    res.pop('models', None)
    R['sensitivity'].append(res)
# cycles retained with the pre-1.0 approximate depth, as a sensitivity
cy = prep(clean(sel(base, 5, 2, True))); cy = cy[cy.D_depth.notna()]
R['notes'] = prim.notes
pd.DataFrame({'repo': prim.d.repo, 'module': prim.d.module, 'role': prim.d.role, 'fold': prim.folds, 'y': prim.y,
              **{f'pred_{k}': prim.pred[k] for k in ('size', 'activity', 'base', 'base+blocks', 'base+edges', 'base+sum', 'base+five')}}
             ).to_csv(f'{OUT}/heldout_predictions_primary.csv', index=False)
pd.DataFrame({'repo': list(prim.U), 'fold': [int(prim.folds[prim.idx[r][0]]) for r in prim.U]}).to_csv(f'{OUT}/fold_assignments_primary.csv', index=False)
prim.d[['repo', 'module', 'role', 'sloc', 'n_blocks', 'edges', 'two_paths', 'D_depth', 'P_propagation', 'X_expression', 'M_modularity',
        'K_crossing', 'IaCX', 'commits', 'fix_commits', 'nonfix', 'y']].to_csv(f'{OUT}/primary_dataset_v4.csv', index=False)
json.dump(R, open(f'{OUT}/results4.json', 'w'), indent=1, default=str)
print('done', {k: R['primary'][k] for k in ('n', 'repos', 'events', 'pct_events', 'cyclic_excluded')})
