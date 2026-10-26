"""
analyse5: all-role secondary analyses of the V4 revision with tool v1.0.0 measures.
All 19,365 mined directories were re-measured at the reconstructed snapshot (litfix/remeasure_all.py).
Directories that fail to parse or whose graph is cyclic have no defined D and are excluded and counted.
"""
import sys, json, os, numpy as np, pandas as pd
sys.argv = ['x', 'data/modules_corrected.csv', 'results/v5']
src = open('analysis/analyse3.py').read()
exec(src.split("REUSE = ")[0])
exec(src[src.index('REUSE = '):src.index('ALL5 = prep(pop_rule(full, 5, 2, True))')])
OUT = 'results/v5'; os.makedirs(OUT, exist_ok=True)
R = {'seed': SEED, 'bootstrap_resamples': B, 'tool': '1.0.0'}
v1 = pd.read_csv('data/all_v1_measures.csv')
b = full.drop_duplicates(['repo', 'module']).merge(v1, on=['repo', 'module'], how='left')
R['mined'] = int(len(b)); R['parse_failed'] = int((b.v1_ok != 1).sum()); R['cyclic'] = int((b.v1_cyclic == 1).sum())
b = b[(b.v1_ok == 1) & (b.v1_cyclic == 0)].copy()
for a, c in [('n_blocks', 'v1_blocks'), ('nodes', 'v1_nodes'), ('edges', 'v1_edges'), ('two_paths', 'v1_two'),
             ('D_depth', 'v1_D'), ('P_propagation', 'v1_P'), ('X_expression', 'v1_X'),
             ('M_modularity', 'v1_M'), ('K_crossing', 'v1_K'), ('IaCX', 'v1_S')]:
    b[a] = b[c]
b = b[[c for c in full.columns]]
R['analysable'] = int(len(b)); R['analysable_repos'] = int(b.repo.nunique())
A5 = prep(pop_rule(b, 5, 2, True))
R['role_counts_all5'] = A5.role.value_counts().to_dict()
top = A5.repo.value_counts()
R['concentration_all5'] = dict(top_repo=top.index[0], top_n=int(top.iloc[0]), top_pct=round(100 * top.iloc[0] / len(A5), 1),
                               top10_pct=round(100 * top.iloc[:10].sum() / len(A5), 1))
R['pre_edge_all'] = int(len(pop_rule(b, 5, 2, False)))
_, R['all_roles'] = run_pop(A5, 'v5-all-roles')
_, R['stratum_tests'] = run_pop(A5[A5.role == 'test'], 'v5-tests')
_, R['stratum_examples'] = run_pop(A5[A5.role == 'example'], 'v5-examples')
U, R['unfiltered'] = run_pop(prep(b), 'v5-unfiltered')
ROLE = ['role_test', 'role_example', 'role_submodule', 'role_other']
ctrl = {'c_he': BASE + ['has_edge'], 'c_he5': BASE + ['has_edge'] + FIVE, 'c_ro': BASE + ROLE, 'c_ro5': BASE + ROLE + FIVE,
        'c_hero': BASE + ['has_edge'] + ROLE, 'c_hero5': BASE + ['has_edge'] + ROLE + FIVE}
for k, c in ctrl.items(): U.heldout(c, k)
R['unfiltered_controls'] = {'five | base': U.diff('base+five', 'base'), 'has_edge | base': U.diff('c_he', 'base'),
    'five | base+has_edge': U.diff('c_he5', 'c_he'), 'role | base': U.diff('c_ro', 'base'),
    'five | base+role': U.diff('c_ro5', 'c_ro'), 'five | base+has_edge+role': U.diff('c_hero5', 'c_hero'),
    'pct_zero_edges': round(100 * float((U.d.edges == 0).mean()), 1), 'role_counts': U.d.role.value_counts().to_dict()}
R['unfiltered_rho'] = {c: round(float(stats.spearmanr(U.d[c], U.d.fix_commits)[0]), 3) for c in ('sloc', 'IaCX', 'D_depth')}
R['all5_rho'] = {c: round(float(stats.spearmanr(A5[c], A5.fix_commits)[0]), 3) for c in ('sloc', 'IaCX', 'D_depth')}
R['notes'] = U.notes
json.dump(R, open(f'{OUT}/results5.json', 'w'), indent=1, default=str)
print(json.dumps({k: R[k] for k in ('mined', 'parse_failed', 'cyclic', 'analysable', 'role_counts_all5', 'concentration_all5', 'pre_edge_all')}))
for k in ('all_roles', 'stratum_tests', 'stratum_examples', 'unfiltered'):
    print(k, R[k]['n'], R[k]['repos'], R[k]['events'], R[k]['models']['base']['auc'], json.dumps(R[k]['diffs']))
print(json.dumps(R['unfiltered_controls'])); print(R['unfiltered_rho'], R['all5_rho'])
