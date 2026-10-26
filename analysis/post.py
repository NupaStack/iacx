"""
post.py: A1 manifest sequence, A3 same-revision before/after comparison, exposure-age control,
distinct-commit and bot-exclusion sensitivity, all from remine3 output.
"""
import json, glob, os, sys, numpy as np, pandas as pd, hashlib
sys.argv = ['x', 'data/modules_corrected.csv', 'results/post']
src = open('analysis/analyse3.py').read()
exec(src.split("REUSE = ")[0])
OUT = 'results/post'; os.makedirs(OUT, exist_ok=True)
P = {}

# ------------------------------------------------------------------ manifest
recs = [json.load(open(f)) for f in glob.glob('mining/out/*.json')]
man = pd.DataFrame([{k: r.get(k) for k in ('repo', 'status', 'branch', 'head', 'snapshot', 'snapshot_ct', 'matched',
                                             'mismatch_dirs', 'rec_dirs', 'dropped', 'in_dataset', 'detail')} for r in recs])
man['snapshot_date'] = pd.to_datetime(man.snapshot_ct, unit='s', errors='coerce').dt.strftime('%Y-%m-%d')
man.sort_values('repo').to_csv('data/manifest.csv', index=False)
cand = json.load(open('mining/candidate_repositories.json'))
P['candidates'] = len(cand)
P['processed'] = len(man)
P['status_counts'] = man.status.value_counts().to_dict()
ok = man[man.status == 'ok']
P['ok_in_dataset'] = int(ok.in_dataset.sum())
P['ok_exact_match'] = int(ok.matched.sum())
P['ok_not_in_dataset'] = int((ok.in_dataset == 0).sum())
P['dataset_repos_total'] = int(full.repo.nunique())
P['dataset_repos_processed_ok'] = int(ok[ok.in_dataset == 1].repo.nunique())
P['clone_failed_repos'] = man[man.status == 'clone_failed'].repo.tolist()

# ------------------------------------------------------------------ per directory
rows = []
for r in recs:
    if r.get('status') != 'ok': continue
    for m in r['modules']:
        f, o = m['fixed'], m['orig']
        rows.append(dict(repo=r['repo'], module=m['module'], snapshot_ct=r.get('snapshot_ct'),
                         tf_json=m['tf_json'], override=m['override'], tf_files=m['tf_files'],
                         file_changes=m['file_changes'], fix_file_changes=m['fix_file_changes'],
                         dcommits=m['commits'], dfix=m['fix_commits'], dfix_nobot=m['fix_commits_nobot'],
                         bot=m['bot_commits'], first_at=m['first_at'],
                         f_ok=f.get('ok'), f_parse_err=f.get('parse_errors'), f_cyc=f.get('cyclic'),
                         f_blocks=f.get('n_blocks'), f_edges=f.get('edges'), f_D=f.get('D'), f_P=f.get('P'),
                         f_K=f.get('K'), f_nodes=f.get('nodes'),
                         o_ok=o.get('ok'), o_blocks=o.get('n_blocks'), o_edges=o.get('edges'), o_D=o.get('D'),
                         o_P=o.get('P'), o_K=o.get('K'), o_nodes=o.get('nodes')))
dd = pd.DataFrame(rows)
dd.to_csv('data/remeasured_directories.csv', index=False)
P['directories_remeasured'] = len(dd)
P['dir_parse_fail_total'] = int((dd.f_ok == 0).sum())
P['dir_with_file_parse_errors'] = int((dd.f_parse_err.fillna(0) > 0).sum())
P['dir_cyclic'] = int((dd.f_cyc == 1).sum())
P['dir_with_tf_json'] = int((dd.tf_json > 0).sum())
P['dir_with_override'] = int((dd.override > 0).sum())

# reproduction against the analysed dataset (same repo/module, same snapshot)
j = dd.merge(full[['repo', 'module', 'n_blocks', 'edges', 'commits', 'fix_commits']], on=['repo', 'module'], how='inner')
P['dataset_rows_matched_by_path'] = len(j)
P['identical_blocks_edges'] = int(((j.f_blocks == j.n_blocks) & (j.f_edges == j.edges)).sum())
P['identical_file_change_counts'] = int((j.file_changes == j.commits).sum())

# ------------------------------------------------------------------ A3 same-revision comparison
s = dd[(dd.f_ok == 1) & (dd.o_ok == 1)]
P['a3_cohort_dirs'] = len(s); P['a3_cohort_repos'] = int(s.repo.nunique())
def tot(c): return int(pd.to_numeric(s[c], errors='coerce').fillna(0).sum())
P['a3'] = {k: dict(before=tot('o_' + k), after=tot('f_' + k)) for k in ('edges', 'D', 'P', 'K', 'blocks', 'nodes')}
for k, v in P['a3'].items(): v['ratio'] = round(v['after'] / v['before'], 2) if v['before'] else None
elig = s[(s.f_blocks >= 5) & (s.file_changes >= 2)]
P['a3_eligible_same_cohort'] = len(elig)
P['a3_pct_with_edge'] = dict(before=round(100 * float((elig.o_edges >= 1).mean()), 1),
                             after=round(100 * float((elig.f_edges >= 1).mean()), 1))
P['a3_blocks_identical_pct'] = round(100 * float((s.f_blocks == s.o_blocks).mean()), 2)
P['a3_nodes_diff_mean'] = round(float((s.o_nodes - s.f_nodes).mean()), 3)

# ------------------------------------------------------------------ models on the primary population
ALL5 = prep(pop_rule(full, 5, 2, True))
prim_df = ALL5[ALL5.role.isin(('root', 'submodule', 'other'))].copy()
prim_df = prim_df.merge(dd[['repo', 'module', 'snapshot_ct', 'first_at', 'dcommits', 'dfix', 'dfix_nobot', 'bot']],
                        on=['repo', 'module'], how='left')
P['primary_n'] = len(prim_df); P['primary_with_remine'] = int(prim_df.snapshot_ct.notna().sum())
pm = prim_df[prim_df.snapshot_ct.notna() & prim_df.first_at.notna()].reset_index(drop=True)
pm['exposure_days'] = (pm.snapshot_ct - pm.first_at) / 86400.0
pm['l_expo'] = np.log1p(pm.exposure_days.clip(lower=0))
P['exposure'] = dict(n=len(pm), median_days=round(float(pm.exposure_days.median()), 1),
                     rho_span_exposure=round(float(stats.spearmanr(pm.age_days, pm.exposure_days)[0]), 3))
Pp = Pop(pm, 'primary-exposure')
mods = {'base': BASE, 'base+expo': BASE + ['l_expo'], 'base+expo+sum': BASE + ['l_expo', 'l_sum'],
        'base+expo+five': BASE + ['l_expo'] + FIVE, 'base+expo+edges': BASE + ['l_expo', 'l_edg'],
        'base+span': BASE + ['l_span']}
for k, c in mods.items(): Pp.heldout(c, k)
P['exposure_models'] = {k: round(Pp.metrics(Pp.pred[k])['auc'], 4) for k in mods}
P['exposure_diffs'] = {'expo | base': Pp.diff('base+expo', 'base'), 'sum | base+expo': Pp.diff('base+expo+sum', 'base+expo'),
                       'five | base+expo': Pp.diff('base+expo+five', 'base+expo'),
                       'edges | base+expo': Pp.diff('base+expo+edges', 'base+expo')}

# distinct-commit definitions and bot exclusion
def rerun(df, label, fixcol, comcol):
    x = df.copy()
    x['fix_commits'] = x[fixcol]; x['commits'] = x[comcol]
    x = x[(x.commits >= 2)]
    x = prep(x.drop(columns=[c for c in x.columns if c.startswith('l_') or c in ('y', 'nonfix', 'has_edge', 'role') or c.startswith('role_')]))
    Q = Pop(x, label)
    for k in ('base', 'base+sum', 'base+five', 'base+edges'): Q.heldout(MAIN[k], k)
    return dict(n=len(x), repos=int(x.repo.nunique()), events=int(Q.y.sum()), base_auc=round(Q.metrics(Q.pred['base'])['auc'], 4),
                d_sum=Q.diff('base+sum', 'base'), d_five=Q.diff('base+five', 'base'), d_edges=Q.diff('base+edges', 'base'),
                rho_sum=round(float(stats.spearmanr(x.IaCX, x.fix_commits)[0]), 3), rho_loc=round(float(stats.spearmanr(x.sloc, x.fix_commits)[0]), 3))
pc = prim_df[prim_df.dcommits.notna()]
P['distinct_commits'] = rerun(pc, 'distinct', 'dfix', 'dcommits')
P['distinct_commits_nobot'] = rerun(pc, 'distinct-nobot', 'dfix_nobot', 'dcommits')
P['bot_commit_share_primary'] = round(float(pc.bot.sum() / max(1, pc.dcommits.sum())), 4)
P['file_change_vs_commit_ratio_primary'] = round(float(pc.commits.sum() / max(1, pc.dcommits.sum())), 3)
json.dump(P, open(f'{OUT}/post_results.json', 'w'), indent=1, default=str)
print(json.dumps({k: v for k, v in P.items() if k not in ('clone_failed_repos',)}, indent=1, default=str)[:6000])
