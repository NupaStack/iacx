"""
remine3: reconstruct the snapshot of every candidate repository, then re-measure every
module directory at that snapshot with both the corrected and the original extractor.

Per repository:
  1. blobless single-branch clone (status recorded on failure)
  2. git log of *.tf file-changes, newest first (same command and counting as mine.py)
  3. snapshot reconstruction: drop newest commits until the per-directory counts of
     .tf file-changes equal the counts in modules_corrected.csv for that repository
  4. sparse checkout of *.tf at the snapshot commit
  5. every directory containing .tf: measure with corrected and original extractor;
     record parse failures, cycles, unsupported inputs, true distinct-commit counts,
     first-commit time (exposure origin)
"""
import os, sys, json, subprocess, time, traceback, importlib.util, re
import pandas as pd, networkx as nx
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
OUTD = f'{BASE}/mining/out'
CL = f'{BASE}/mining/clones'
os.makedirs(OUTD, exist_ok=True)
os.makedirs(CL, exist_ok=True)
sys.path.insert(0, os.path.join(BASE, 'mining'))
from mine import FIX_POS, FIX_NEG, FEAT


def load(n, p):
    s = importlib.util.spec_from_file_location(n, p)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


FIXED = load('fixed', f'{BASE}/iacx/iacx_label_corrected_pre1.py')  # extractor used for the Table 5 comparison; v1.0.0 measures come from analysis/remeasure_*.py
ORIG = load('orig', f'{BASE}/iacx/iacx_original.py')
REPOS = json.load(open(f'{BASE}/mining/candidate_repositories.json'))
DS = pd.read_csv(f'{BASE}/data/modules_corrected.csv')
REC = {r: {('' if m == '(root)' else m): int(c) for m, c in zip(g.module, g.commits)}
       for r, g in DS.groupby('repo')}
BOT = re.compile(r'\[bot\]|dependabot|renovate|github-actions|semantic-release|pre-commit-ci', re.I)


def sh(a, cwd=None, t=1800):
    return subprocess.run(a, cwd=cwd, capture_output=True, text=True, errors='replace', timeout=t)


def isfix(s):
    return bool(FIX_POS.search(s)) and not FIX_NEG.search(s) and not FEAT.match(s)


def measure(mod, d):
    try:
        n, _ = mod.parse_directory(d, recursive=False)
        E = mod.build_graph(n)
        G = nx.DiGraph(); G.add_nodes_from(n); G.add_edges_from(E)
        a = mod.analyse(d, recursive=False)
        return dict(ok=1, nodes=len(n), edges=len(E), cyclic=int(not nx.is_directed_acyclic_graph(G)),
                    D=a.get('D_depth'), P=a.get('P_propagation'), X=a.get('X_expression'),
                    M=a.get('M_modularity'), K=a.get('K_crossing'), IaCX=a.get('IaCX'),
                    n_blocks=a.get('n_blocks'), loc=a.get('loc'), parse_errors=a.get('parse_errors'), two_paths=a.get('two_paths'), n_resources=a.get('n_resources'))
    except Exception as e:
        return dict(ok=0, err=type(e).__name__ + ': ' + str(e)[:120])


def module_dirs(path):
    dirs = []
    for dp, dn, fn in os.walk(path):
        dn[:] = [x for x in dn if x not in ('.git', '.terraform', 'node_modules', 'vendor')]
        if any(f.endswith('.tf') for f in fn):
            dirs.append(dp)
    return dirs


def work(repo):
    res = dict(repo=repo, t0=time.time())
    dest = os.path.join(CL, repo.replace('/', '__'))
    try:
        if not (os.path.isdir(dest + '/.git') and sh(['git', 'rev-parse', 'HEAD'], dest, 60).returncode == 0):
            sh(['rm', '-rf', dest])
            r = sh(['git', 'clone', '--filter=blob:none', '--no-checkout', '--single-branch', '--quiet',
                    f'https://github.com/{repo}.git', dest], t=1500)
            if r.returncode:
                res.update(status='clone_failed', detail=r.stderr.strip()[-200:])
                return res
        res['branch'] = sh(['git', 'rev-parse', '--abbrev-ref', 'HEAD'], dest, 60).stdout.strip()
        res['head'] = sh(['git', 'rev-parse', 'HEAD'], dest, 60).stdout.strip()
        lg = sh(['git', 'log', '--no-merges', '--date=unix', '--pretty=format:@@@%H|%at|%ct|%an|%s',
                 '--name-only', '--', '*.tf'], dest, 1500).stdout
        commits = []
        cur = None
        for line in lg.splitlines():
            if line.startswith('@@@'):
                try:
                    h, at, ct, an, su = line[3:].split('|', 4)
                except ValueError:
                    cur = None
                    continue
                cur = dict(h=h, at=int(at), ct=int(ct), an=an, fix=isfix(su), bot=bool(BOT.search(an)), dirs=[])
                commits.append(cur)
            elif line.strip() and cur is not None and line.endswith('.tf'):
                cur['dirs'].append(os.path.dirname(line))
        rec = REC.get(repo, {})
        # snapshot reconstruction on .tf file-change counts, exactly as mine.py counted them
        cnt = {}
        for c in commits:
            for d in c['dirs']:
                cnt[d] = cnt.get(d, 0) + 1
        diff = {d: cnt.get(d, 0) - v for d, v in rec.items()}
        bad = sum(1 for v in diff.values() if v != 0)
        best = (bad, 0)
        k = 0
        if rec:
            while bad and k < len(commits):
                for d in commits[k]['dirs']:
                    if d in diff:
                        before = diff[d] != 0
                        diff[d] -= 1
                        after = diff[d] != 0
                        bad += int(after and not before) - int(before and not after)
                k += 1
                if bad < best[0]:
                    best = (bad, k)
            if bad:
                k = best[1]
        res.update(n_tf_commits=len(commits), dropped=k, mismatch_dirs=int(best[0]) if rec else -1,
                   rec_dirs=len(rec), matched=int(bool(rec) and best[0] == 0), in_dataset=int(bool(rec)))
        hist = commits[k:]
        if not hist:
            res.update(status='no_tf_history')
            return res
        snap = hist[0]
        res.update(snapshot=snap['h'], snapshot_ct=snap['ct'])
        sh(['git', 'sparse-checkout', 'set', '--no-cone', '*.tf'], dest, 300)
        r = sh(['git', 'checkout', '-q', '--force', snap['h']], dest, 1500)
        if r.returncode:
            res.update(status='checkout_failed', detail=r.stderr.strip()[-200:])
            return res
        H = {}
        for c in hist:
            for d in c['dirs']:
                h = H.setdefault(d, dict(fc=0, ffc=0, commits=set(), fix=set(), bot=set(), first=c['at']))
                h['fc'] += 1
                h['ffc'] += int(c['fix'])
                h['first'] = min(h['first'], c['at'])
                h['commits'].add(c['h'])
                if c['fix']:
                    h['fix'].add(c['h'])
                if c['bot']:
                    h['bot'].add(c['h'])
        rows = []
        for dp in module_dirs(dest):
            rel = os.path.relpath(dp, dest)
            rel = '' if rel == '.' else rel
            files = os.listdir(dp)
            h = H.get(rel)
            row = dict(module=rel or '(root)', tf_files=sum(f.endswith('.tf') for f in files),
                       tf_json=sum(f.endswith('.tf.json') for f in files),
                       override=sum(f == 'override.tf' or f.endswith('_override.tf') for f in files),
                       file_changes=h['fc'] if h else 0, fix_file_changes=h['ffc'] if h else 0,
                       commits=len(h['commits']) if h else 0, fix_commits=len(h['fix']) if h else 0,
                       bot_commits=len(h['bot']) if h else 0,
                       fix_commits_nobot=len(h['fix'] - h['bot']) if h else 0,
                       first_at=h['first'] if h else None)
            row['fixed'] = measure(FIXED, dp)
            row['orig'] = measure(ORIG, dp)
            rows.append(row)
        res.update(status='ok', modules=rows)
    except Exception:
        res.update(status='error', detail=traceback.format_exc()[-400:])
    finally:
        res['secs'] = round(time.time() - res['t0'], 1)
    return res


if __name__ == '__main__':
    todo = [r for r in REPOS if not os.path.exists(os.path.join(OUTD, r.replace('/', '__') + '.json'))]
    big = {'hashicorp/terraform-provider-aws', 'hashicorp/terraform-provider-azurerm',
           'aliyun/terraform-provider-alicloud', 'IBM-Cloud/terraform-provider-ibm',
           'hashicorp/terraform-provider-google', 'hashicorp/terraform', 'Azure/terraform-provider-azapi'}
    todo.sort(key=lambda r: (r in big, r))
    print('todo', len(todo), flush=True)
    with ThreadPoolExecutor(int(sys.argv[1]) if len(sys.argv) > 1 else 8) as ex:
        futs = {ex.submit(work, r): r for r in todo}
        for i, f in enumerate(as_completed(futs), 1):
            res = f.result()
            r = futs[f]
            json.dump(res, open(os.path.join(OUTD, r.replace('/', '__') + '.json'), 'w'))
            print(f"[{i}/{len(todo)}] {r} {res.get('status')} matched={res.get('matched')} "
                  f"dropped={res.get('dropped')} {res.get('secs')}s", flush=True)
    print('ALL DONE', flush=True)
