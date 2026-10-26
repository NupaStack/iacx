# -*- coding: utf-8 -*-
"""
analyse3: single pipeline for every empirical table in the V3 revision.
Addresses professor items A3 (leakage on identical folds), A4 (provenance),
A5 (paired intervals for every sensitivity row, edge-first baselines,
zero-edge and directory-role controls), A7 (bootstrap definition, repeated
grouped CV, exported folds and predictions).

Model:      statsmodels Logit, unregularised maximum likelihood, Newton, maxiter 400,
            intercept included, predictors log(1+x).
Validation: 10-fold GroupKFold by repository (deterministic, no shuffling) for all
            point estimates; predictions pooled across folds, one AUC per model.
Intervals:  repository bootstrap, 2,000 resamples, percentile; paired, i.e. both models'
            fixed held-out predictions are evaluated on the same resampled repositories.
            Conditional on the fitted folds (no refitting inside the bootstrap).
Retraining variability: 20 repeats of random repository-to-fold assignment.
PR-AUC:     average precision (sklearn average_precision_score), step-wise, not trapezoidal.
Ties:       ROC AUC by sklearn roc_auc_score (ties receive half credit).
"""
import json, os, sys, hashlib, numpy as np, pandas as pd, warnings
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor as VIF
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.model_selection import GroupKFold
from scipy import stats
warnings.filterwarnings('ignore')
SEED = 20261018; B = 2000
OUT = sys.argv[2] if len(sys.argv) > 2 else 'out3'; os.makedirs(OUT, exist_ok=True)
full = pd.read_csv(sys.argv[1]).rename(columns={'loc': 'sloc'})
R = {'seed': SEED, 'bootstrap_resamples': B, 'input_sha256':
     hashlib.sha256(open(sys.argv[1], 'rb').read()).hexdigest(), 'log': []}
L = np.log1p
FACT = [('D_depth','l_D','Depth'),('P_propagation','l_P','Two-step'),('X_expression','l_X','Expression'),
        ('M_modularity','l_M','Module calls'),('K_crossing','l_K','Interface coupling')]
FIVE = [n for _, n, _ in FACT]

def role(m):
    """Directory role from the module path. Rule fixed before outcomes were examined on it.
    test: any path component containing 'test' or 'fixture' (testdata, integrationtest, acctest, ...)
    example: any component containing 'example', or a documentation/demo component
             (samples, demo, quickstarts, docs, guides, blog, codelabs, gists, sandbox, tmp, templates, evals)
    root: the repository root; submodule: a path under a 'modules' component; other: anything else."""
    m = str(m).lower()
    if m == '(root)': return 'root'
    parts = m.split('/')
    if any(('test' in p) or ('fixture' in p) or p == 'integration' for p in parts): return 'test'
    DOCS = {'sample','samples','demo','demos','quickstart','quickstarts','docs','doc','guides','blog',
            'codelabs','gists','sandbox','playground','tmp','templates','evals','evidence'}
    if any(('example' in p) or p in DOCS for p in parts): return 'example'
    if 'modules' in parts: return 'submodule'
    return 'other'

def prep(df, activity='nonfix'):
    df = df.drop_duplicates(['repo','module']).copy()
    df['nonfix'] = df.commits - df.fix_commits
    df['y'] = (df.fix_commits > 0).astype(int)
    df['l_loc'] = L(df.sloc); df['l_act'] = L(df[activity]); df['l_sum'] = L(df.IaCX)
    df['l_blk'] = L(df.n_blocks); df['l_edg'] = L(df.edges); df['l_two'] = L(df.two_paths)
    df['has_edge'] = (df.edges > 0).astype(int); df['l_span'] = L(df.age_days)
    for c, n, _ in FACT: df[n] = L(df[c])
    df['role'] = df.module.map(role)
    for r in ('test','example','submodule','other'): df['role_'+r] = (df.role == r).astype(int)
    return df.reset_index(drop=True)

class Pop:
    def __init__(self, df, name, folds=None):
        self.d, self.name = df, name
        self.y = df.y.values; self.g = df.repo.values
        self.U = np.unique(self.g); self.idx = {r: np.flatnonzero(self.g == r) for r in self.U}
        self.folds = folds if folds is not None else self._gkf()
        self.pred = {}; self.notes = []
    def _gkf(self):
        f = np.empty(len(self.d), int)
        for k, (_, te) in enumerate(GroupKFold(n_splits=10).split(self.d, self.y, self.g)): f[te] = k
        return f
    def heldout(self, cols, key=None, folds=None):
        folds = self.folds if folds is None else folds
        X = sm.add_constant(self.d[cols].values.astype(float), has_constant='add')
        p = np.full(len(self.d), np.nan); skipped = nonconv = 0
        for k in range(10):
            te = folds == k; tr = ~te
            if len(np.unique(self.y[tr])) < 2 or len(np.unique(self.y[te])) < 2: skipped += 1; continue
            try:
                m = sm.Logit(self.y[tr], X[tr]).fit(disp=0, maxiter=400)
                if not m.mle_retvals.get('converged', True): nonconv += 1
                p[te] = m.predict(X[te])
            except Exception: skipped += 1
        if key: self.pred[key] = p
        if skipped or nonconv: self.notes.append(f'{self.name}/{key}: skipped folds {skipped}, non-converged {nonconv}')
        return p
    def metrics(self, p):
        ok = ~np.isnan(p); y = self.y[ok]; q = p[ok]
        k = max(1, int(round(.2 * len(q)))); top = np.argsort(-q, kind='stable')[:k]
        return dict(auc=float(roc_auc_score(y, q)), pr_auc=float(average_precision_score(y, q)),
                    brier=float(brier_score_loss(y, q)), recall20=float(y[top].sum() / y.sum()),
                    n_scored=int(ok.sum()))
    def _prep_auc(self, p, ok):
        sc = p[ok]; uniq, inv = np.unique(sc, return_inverse=True)
        return inv, len(uniq)
    @staticmethod
    def _wauc(inv, G, y, w):
        pos = np.bincount(inv, weights=w * y, minlength=G); neg = np.bincount(inv, weights=w * (1 - y), minlength=G)
        P, N = pos.sum(), neg.sum()
        if P == 0 or N == 0: return np.nan
        below = np.cumsum(neg) - neg
        return float((pos * (below + 0.5 * neg)).sum() / (P * N))
    def diff(self, a, b):
        pa, pb = self.pred[a], self.pred[b]; ok = ~np.isnan(pa) & ~np.isnan(pb)
        y = self.y[ok].astype(float); g = self.g[ok]
        ia, Ga = self._prep_auc(pa, ok); ib, Gb = self._prep_auc(pb, ok)
        one = np.ones(len(y))
        est = self._wauc(ia, Ga, y, one) - self._wauc(ib, Gb, y, one)
        chk = roc_auc_score(y, pa[ok]) - roc_auc_score(y, pb[ok])
        assert abs(est - chk) < 1e-9, (est, chk)
        ridx = np.searchsorted(self.U, g)
        seed = int(hashlib.md5(f'{self.name}|{a}|{b}'.encode()).hexdigest()[:8], 16)
        rng = np.random.default_rng(seed); out = []
        for _ in range(B):
            cnt = rng.multinomial(len(self.U), np.full(len(self.U), 1 / len(self.U)))
            w = cnt[ridx].astype(float)
            v = self._wauc(ia, Ga, y, w) - self._wauc(ib, Gb, y, w)
            if np.isfinite(v): out.append(v)
        return dict(diff=round(float(est), 4), ci=[round(float(x), 4) for x in np.percentile(out, [2.5, 97.5])])
    def coefs(self, cols):
        X = sm.add_constant(self.d[cols].values.astype(float), has_constant='add')
        rob = sm.Logit(self.y, X).fit(cov_type='cluster', cov_kwds={'groups': pd.factorize(self.g)[0]}, disp=0, maxiter=400)
        ci = np.asarray(rob.conf_int()); names = ['const'] + cols
        out = {n: dict(b=round(float(rob.params[i]), 4), se=round(float(rob.bse[i]), 4),
                       or_=round(float(np.exp(rob.params[i])), 3),
                       or_ci=[round(float(np.exp(ci[i][0])), 3), round(float(np.exp(ci[i][1])), 3)],
                       p=float(rob.pvalues[i])) for i, n in enumerate(names)}
        vif = {c: round(float(VIF(self.d[cols].values.astype(float), i)), 2) for i, c in enumerate(cols)} if len(cols) > 1 else {}
        return dict(coef=out, vif=vif, converged=bool(rob.mle_retvals.get('converged', True)),
                    pseudo_r2=round(float(rob.prsquared), 4))

BASE = ['l_loc','l_act']
MAIN = {'size': ['l_loc'], 'activity': ['l_act'], 'base': BASE,
        'base+blocks': BASE+['l_blk'], 'base+edges': BASE+['l_edg'], 'base+sum': BASE+['l_sum'],
        'base+five': BASE+FIVE}
for _, n, lab in FACT: MAIN['base+'+lab] = BASE+[n]
for _, n, lab in FACT: MAIN['LOO-'+lab] = BASE+[m for m in FIVE if m != n]
EB = BASE+['l_edg']
MAIN.update({'edgebase+D': EB+['l_D'], 'edgebase+twostep_term': EB+['l_two'],
             'edgebase+D+twostep_term': EB+['l_D','l_two'], 'edgebase+five': EB+FIVE})

def pop_rule(df, blk, cmt, edge):
    return df[(df.n_blocks >= blk) & (df.commits >= cmt) & ((df.edges >= 1) if edge else True)]


REUSE = ('root','submodule','other')
def run_pop(df, name, full_report=False):
    P = Pop(df.reset_index(drop=True), name)
    keys = list(MAIN) if full_report else ['size','base','base+edges','base+sum','base+five','edgebase+five']
    res = dict(n=int(len(P.d)), repos=int(P.d.repo.nunique()), events=int(P.y.sum()),
               pct_events=round(100*float(P.y.mean()),1), models={}, diffs={})
    for k in keys: res['models'][k] = dict(cols=MAIN[k], **P.metrics(P.heldout(MAIN[k], k)))
    for k in keys:
        if k.startswith('base+') or k.startswith('LOO-'): res['diffs'][k+' vs base'] = P.diff(k, 'base')
        if k.startswith('edgebase+'): res['diffs'][k+' vs base+edges'] = P.diff(k, 'base+edges')
    res['notes'] = P.notes
    return P, res

def spearman_table(P):
    out = {}
    for c in ['sloc','n_blocks','edges','IaCX','D_depth','P_propagation','X_expression','M_modularity','K_crossing']:
        x = P.d[c].values; z = P.d.fix_commits.values; rho = stats.spearmanr(x, z)[0]
        rng = np.random.default_rng(int(hashlib.md5((P.name+c).encode()).hexdigest()[:8],16)); bs=[]
        for _ in range(B):
            rows = np.concatenate([P.idx[r] for r in rng.choice(P.U, len(P.U), replace=True)])
            v = stats.spearmanr(x[rows], z[rows])[0]
            if np.isfinite(v): bs.append(v)
        out[c] = dict(rho=round(float(rho),3), ci=[round(float(v),3) for v in np.percentile(bs,[2.5,97.5])])
    return out
def describe(P):
    return {c: dict(median=float(P.d[c].median()), p90=float(P.d[c].quantile(.9)), max=float(P.d[c].max()),
                    pct_zero=round(100*float((P.d[c]==0).mean()),1))
            for c in ['sloc','n_blocks','edges','IaCX','D_depth','P_propagation','X_expression','M_modularity','K_crossing','commits','nonfix','fix_commits']}

ALL5 = prep(pop_rule(full, 5, 2, True))
R['role_counts_5800'] = ALL5.role.value_counts().to_dict()
top = ALL5.groupby('repo').size().sort_values(ascending=False)
R['concentration_5800'] = dict(top_repo=top.index[0], top_n=int(top.iloc[0]), top_pct=round(100*top.iloc[0]/len(ALL5),1),
                                top10_pct=round(100*top.head(10).sum()/len(ALL5),1))

# ---------------- PRIMARY: reusable modules
prim, R['primary'] = run_pop(ALL5[ALL5.role.isin(REUSE)], 'primary-reusable', full_report=True)
R['primary']['spearman'] = spearman_table(prim); R['primary']['describe'] = describe(prim)
R['primary']['coefficients'] = {k: prim.coefs(MAIN[k]) for k in ['base','base+sum','base+five','base+edges','edgebase+D+twostep_term'] + ['base+'+lab for _,_,lab in FACT]}
R['primary']['rho_D_P'] = round(float(stats.spearmanr(prim.d.D_depth, prim.d.P_propagation)[0]),3)
R['primary']['rho_P_loc'] = round(float(stats.spearmanr(prim.d.P_propagation, prim.d.sloc)[0]),3)
R['primary']['rho_M_K'] = round(float(stats.spearmanr(prim.d.M_modularity, prim.d.K_crossing)[0]),3)
R['primary']['role_counts'] = prim.d.role.value_counts().to_dict()

# leakage on identical folds (A3)
leak = ALL5[ALL5.role.isin(REUSE)].copy(); leak = prep(leak.drop(columns=[c for c in leak.columns if c.startswith('l_') or c in ('y','nonfix','has_edge','role') or c.startswith('role_')]), activity='commits')
pl = Pop(leak.reset_index(drop=True), 'primary-leaky', folds=prim.folds)
for k in ('base','base+sum'): pl.heldout(MAIN[k], k)
R['leakage_primary'] = dict(base_total_commits=round(pl.metrics(pl.pred['base'])['auc'],4),
                            base_nonfix=round(R['primary']['models']['base']['auc'],4), same_folds=True)

# repeated grouped CV (A7)
rng = np.random.default_rng(SEED); reps=[]
for r in range(20):
    perm = rng.permutation(prim.U); fo = {repo: i % 10 for i, repo in enumerate(perm)}
    f = np.array([fo[x] for x in prim.g]); a = {}
    for k in ('base','base+sum','base+five','base+edges'): a[k] = roc_auc_score(prim.y, prim.heldout(MAIN[k], None, folds=f))
    reps.append({k: a[k]-a['base'] for k in ('base+sum','base+five','base+edges')})
R['primary']['repeated_cv'] = {k: dict(mean=round(float(np.mean([x[k] for x in reps])),4), sd=round(float(np.std([x[k] for x in reps],ddof=1)),4),
    min=round(float(min(x[k] for x in reps)),4), max=round(float(max(x[k] for x in reps)),4), n_positive=int(sum(x[k]>0 for x in reps)))
    for k in ('base+sum','base+five','base+edges')}

# ---------------- SECONDARY and strata
_, R['secondary_all_roles_5800'] = run_pop(ALL5, 'all-roles-5800')
_, R['stratum_tests'] = run_pop(ALL5[ALL5.role=='test'], 'tests')
_, R['stratum_examples'] = run_pop(ALL5[ALL5.role=='example'], 'examples')
TOOL = r'terraform-provider-|^hashicorp/terraform$|tflint|driftctl|terratest|kics|checkov|infracost|terrascan|tfsec|terraform-exec|terraform-ls|tfsort|tfgen|/fogg$|redc$|pike$|blast-radius|atlantis$|atmos$'
_, R['reusable_excluding_tool_repos'] = run_pop(ALL5[ALL5.role.isin(REUSE) & ~ALL5.repo.str.contains(TOOL, regex=True)], 'reusable-no-tools')

# ---------------- sensitivity within reusable modules (A5)
R['sensitivity_reusable'] = []
for lab, blk, cmt, edge in [('>=3 blocks, >=2 commits',3,2,True),('>=3 blocks, >=5 commits',3,5,True),('>=5 blocks, >=2 commits',5,2,True),
                            ('>=5 blocks, >=5 commits',5,5,True),('>=10 blocks, >=2 commits',10,2,True),('>=10 blocks, >=5 commits',10,5,True),
                            ('no resolved-edge criterion',5,2,False),('all reusable, no thresholds',0,0,False)]:
    df = prep(pop_rule(full, blk, cmt, edge)); df = df[df.role.isin(REUSE)]
    P, res = run_pop(df, 'reuse '+lab)
    res['rule'] = lab
    res['rho'] = {c: round(float(stats.spearmanr(P.d[c], P.d.fix_commits)[0]),3) for c in ('sloc','IaCX','D_depth')}
    R['sensitivity_reusable'].append(res)

# ---------------- unfiltered all-roles corpus and controls (A5)
U, R['unfiltered_all_roles'] = run_pop(prep(full), 'unfiltered')
ROLE = ['role_test','role_example','role_submodule','role_other']
ctrl = {'c_he': BASE+['has_edge'], 'c_he5': BASE+['has_edge']+FIVE, 'c_ro': BASE+ROLE, 'c_ro5': BASE+ROLE+FIVE,
        'c_hero': BASE+['has_edge']+ROLE, 'c_hero5': BASE+['has_edge']+ROLE+FIVE}
for k, c in ctrl.items(): U.heldout(c, k)
R['unfiltered_controls'] = {'five | base': U.diff('base+five','base'), 'has_edge | base': U.diff('c_he','base'),
    'five | base+has_edge': U.diff('c_he5','c_he'), 'role | base': U.diff('c_ro','base'),
    'five | base+role': U.diff('c_ro5','c_ro'), 'five | base+has_edge+role': U.diff('c_hero5','c_hero'),
    'pct_zero_edges': round(100*float((U.d.edges==0).mean()),1), 'role_counts': U.d.role.value_counts().to_dict()}

# ---------------- span variable
R['span_note'] = 'age_days = (last - first commit touching the module)/86400; repair commits help set it; not a clean exposure measure.'
R['primary']['models']['base+span'] = dict(cols=BASE+['l_span'], **prim.metrics(prim.heldout(BASE+['l_span'], 'base+span')))
R['notes'] = prim.notes + U.notes

pd.DataFrame({'repo': prim.d.repo, 'module': prim.d.module, 'role': prim.d.role, 'fold': prim.folds, 'y': prim.y,
              **{f'pred_{k}': prim.pred[k] for k in ('size','activity','base','base+blocks','base+edges','base+sum','base+five')}}
             ).to_csv(f'{OUT}/heldout_predictions_primary.csv', index=False)
pd.DataFrame({'repo': list(prim.U), 'fold': [int(prim.folds[prim.idx[r][0]]) for r in prim.U]}).to_csv(f'{OUT}/fold_assignments_primary.csv', index=False)
pd.DataFrame({'repo': ALL5.repo, 'module': ALL5.module, 'role': ALL5.role}).to_csv(f'{OUT}/directory_roles_5800.csv', index=False)
json.dump(R, open(f'{OUT}/results3.json','w'), indent=1, default=str)
print('done', {k: R['primary'][k] for k in ('n','repos','events','pct_events')})
