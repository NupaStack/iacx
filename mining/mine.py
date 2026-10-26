import os, sys, json, csv, subprocess, shutil, re, time, traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tool', 'iacx'))
import iacx

BASE   = os.path.dirname(os.path.abspath(__file__))
CLONES = os.path.join(BASE, 'clones')
OUT    = os.path.join(BASE, 'out')
os.makedirs(CLONES, exist_ok=True); os.makedirs(OUT, exist_ok=True)

# Defect-fix commit classification. Positive keywords follow the convention
# used in defect-prediction work on IaC (Dalla Palma et al.); the negative
# list removes release automation and feature commits that merely contain
# the word "fix" in a changelog body.
FIX_POS = re.compile(r'\b(fix(e[sd])?|bug(fix)?|defect|fault|error|issue|'
                     r'incorrect|wrong|broken|break(s|age)?|crash|fail(s|ed|ure|ing)?|'
                     r'regress(ion)?|patch|resolve[sd]?|correct(ed|ion)?|'
                     r'workaround|mistake|typo|invalid|missing|revert)\b', re.I)
FIX_NEG = re.compile(r'^\s*(chore\(release\)|release v?\d|bump version|'
                     r'\[skip ci\]|merge (pull request|branch)|initial commit)', re.I)
FEAT    = re.compile(r'^\s*(feat|feature)[:(]', re.I)

def sh(args, cwd=None, timeout=300):
    return subprocess.run(args, cwd=cwd, capture_output=True, text=True,
                          timeout=timeout, errors='replace')

def clone(repo):
    dest = os.path.join(CLONES, repo.replace('/', '__'))
    if os.path.isdir(os.path.join(dest, '.git')):
        return dest
    r = sh(['git', 'clone', '--filter=blob:none', '--single-branch', '--quiet',
            f'https://github.com/{repo}.git', dest], timeout=300)
    return dest if r.returncode == 0 else None

def mine_history(path):
    """dir -> {commits, fix_commits, authors, first_ts, last_ts, churn_files}"""
    r = sh(['git', 'log', '--no-merges', '--date=unix',
            '--pretty=format:@@@%H|%at|%an|%s', '--name-only', '--', '*.tf'],
           cwd=path, timeout=420)
    if r.returncode != 0:
        return {}
    agg = {}
    cur_fix = cur_ts = None; cur_auth = None
    for line in r.stdout.splitlines():
        if line.startswith('@@@'):
            try:
                _h, ts, auth, subj = line[3:].split('|', 3)
            except ValueError:
                cur_fix = None; continue
            cur_ts = int(ts); cur_auth = auth
            cur_fix = bool(FIX_POS.search(subj)) and not FIX_NEG.search(subj) and not FEAT.match(subj)
        elif line.strip() and cur_fix is not None:
            if not line.endswith('.tf'):
                continue
            d = os.path.dirname(line)
            a = agg.setdefault(d, {'commits': 0, 'fix_commits': 0, 'authors': set(),
                                   'first_ts': cur_ts, 'last_ts': cur_ts})
            a['commits'] += 1
            if cur_fix: a['fix_commits'] += 1
            a['authors'].add(cur_auth)
            a['first_ts'] = min(a['first_ts'], cur_ts); a['last_ts'] = max(a['last_ts'], cur_ts)
    for d, a in agg.items():
        a['authors'] = len(a['authors'])
    return agg

def module_dirs(path):
    dirs = []
    for dp, dn, fn in os.walk(path):
        dn[:] = [d for d in dn if d not in ('.git', '.terraform', 'node_modules', 'vendor')]
        if any(f.endswith('.tf') for f in fn):
            dirs.append(dp)
    return dirs

def process(repo):
    rows = []
    dest = None
    try:
        dest = clone(repo)
        if not dest: return rows
        hist = mine_history(dest)
        for d in module_dirs(dest):
            rel = os.path.relpath(d, dest)
            rel = '' if rel == '.' else rel
            m = iacx.analyse(d, recursive=False)
            if not m or m['n_blocks'] == 0: continue
            h = hist.get(rel, {'commits': 0, 'fix_commits': 0, 'authors': 0,
                               'first_ts': 0, 'last_ts': 0})
            age_days = ((h['last_ts'] - h['first_ts']) / 86400.0) if h['first_ts'] else 0.0
            rows.append({
                'repo': repo, 'module': rel or '(root)',
                'IaCX': m['IaCX'], 'D_depth': m['D_depth'], 'P_propagation': m['P_propagation'],
                'X_expression': m['X_expression'], 'M_modularity': m['M_modularity'],
                'K_crossing': m['K_crossing'],
                'nodes': m['nodes'], 'edges': m['edges'], 'two_paths': m['two_paths'],
                'density': m['density'], 'max_fan_in': m['max_fan_in'], 'max_fan_out': m['max_fan_out'],
                'loc': m['loc'], 'tf_files': m['tf_files'], 'n_blocks': m['n_blocks'],
                'n_resources': m['n_resources'], 'n_modules': m['n_modules'],
                'n_variables': m['n_variables'], 'n_outputs': m['n_outputs'], 'n_data': m['n_data'],
                'commits': h['commits'], 'fix_commits': h['fix_commits'],
                'authors': h['authors'], 'age_days': round(age_days, 1),
            })
    except Exception:
        traceback.print_exc(file=sys.stderr)
    finally:
        if dest and os.path.isdir(dest) and os.environ.get('KEEP') != '1':
            shutil.rmtree(dest, ignore_errors=True)
    return rows

if __name__ == '__main__':
    repos = json.load(open(os.path.join(BASE, 'repos.json')))
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else len(repos)
    repos = repos[:limit]
    fields = ['repo','module','IaCX','D_depth','P_propagation','X_expression','M_modularity',
              'K_crossing','nodes','edges','two_paths','density','max_fan_in','max_fan_out',
              'loc','tf_files','n_blocks','n_resources','n_modules','n_variables','n_outputs',
              'n_data','commits','fix_commits','authors','age_days']
    out = open(os.path.join(OUT, 'modules.csv'), 'w', newline='')
    w = csv.DictWriter(out, fieldnames=fields); w.writeheader()
    done = 0; t0 = time.time(); nrows = 0
    with ThreadPoolExecutor(max_workers=12) as ex:
        futs = {ex.submit(process, r): r for r in repos}
        for f in as_completed(futs):
            done += 1
            for row in f.result():
                w.writerow(row); nrows += 1
            if done % 10 == 0:
                out.flush()
                print(f'[{done}/{len(repos)}] rows={nrows} elapsed={time.time()-t0:.0f}s',
                      flush=True)
    out.close()
    print(f'DONE repos={done} rows={nrows} elapsed={time.time()-t0:.0f}s')
