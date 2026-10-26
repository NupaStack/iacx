"""
iacx - Infrastructure-as-Code compleXity
=========================================
Computes the IaCX complexity metric for declarative Terraform (HCL2)
configurations from the resource dependency graph, together with the
size-based baseline metrics used for comparison.

Metric definition (see paper, Section IV):

    IaCX(C) = wD*D(C) + wC*P(C) + wX*X(C) + wM*M(C) + wK*K(C)

where, for the dependency graph G = (V, E) induced by configuration C:

    D(C)  depth      - length of the longest directed dependency chain
    P(C)  propagation- |E| + sum_v fanin(v)*fanout(v)   (edges + 2-paths)
    X(C)  expression - weighted count of deferred-decision constructs
                       (count, for_each, dynamic, conditionals, try/can, splat)
    M(C)  modularity - sum of module instantiation nesting depths
    K(C)  crossing   - dependency edges that cross a module boundary

Unit weights are the base model; calibrated weights are fitted empirically.
"""
from __future__ import annotations
import json, os, re, sys, math
from collections import defaultdict

try:
    import hcl2
except ImportError:
    hcl2 = None

# --------------------------------------------------------------------------
# Reference extraction
# --------------------------------------------------------------------------

# Terraform reference forms we resolve to graph nodes.
RE_VAR      = re.compile(r'\bvar\.([A-Za-z_][A-Za-z0-9_-]*)')
RE_LOCAL    = re.compile(r'\blocal\.([A-Za-z_][A-Za-z0-9_-]*)')
RE_MODULE   = re.compile(r'\bmodule\.([A-Za-z_][A-Za-z0-9_-]*)')
RE_DATA     = re.compile(r'\bdata\.([A-Za-z_][A-Za-z0-9_-]*)\.([A-Za-z_][A-Za-z0-9_-]*)')
RE_RESOURCE = re.compile(r'\b([a-z][a-z0-9]*(?:_[a-z0-9]+)+)\.([A-Za-z_][A-Za-z0-9_-]*)')
RE_EACHCNT  = re.compile(r'\b(each|count)\.')

# Deferred-decision constructs. Weights are the base (unit-family) model:
# a construct that introduces one deferred decision counts 1.
EXPR_WEIGHTS = {
    'count':       1.0,   # instance multiplicity decided at plan time
    'for_each':    1.5,   # multiplicity + key space
    'dynamic':     1.5,   # block structure itself is generated
    'conditional': 1.0,   # ternary
    'for_expr':    1.0,   # comprehension
    'try_can':     0.5,   # error-tolerant evaluation
    'splat':       0.5,   # implicit iteration
    'depends_on':  1.0,   # explicit ordering constraint
    'lifecycle':   0.5,   # engine behaviour override
    'provisioner': 1.5,   # escape hatch into imperative execution
}

RE_TERNARY  = re.compile(r'\?[^?:]{0,200}:')
RE_FOR_EXPR = re.compile(r'\bfor\s+[A-Za-z_]')
RE_TRY_CAN  = re.compile(r'\b(?:try|can)\s*\(')
RE_SPLAT    = re.compile(r'\[\s*\*\s*\]|\.\*\.')

BLOCK_KINDS = ('resource', 'data', 'module', 'variable', 'local', 'locals',
               'output', 'provider', 'terraform')


def _stringify(obj) -> str:
    """Flatten an arbitrary parsed HCL value into searchable text."""
    if obj is None:
        return ''
    if isinstance(obj, str):
        return obj
    if isinstance(obj, (int, float, bool)):
        return str(obj)
    if isinstance(obj, dict):
        return ' '.join(_stringify(k) + ' ' + _stringify(v) for k, v in obj.items())
    if isinstance(obj, (list, tuple)):
        return ' '.join(_stringify(x) for x in obj)
    return str(obj)


class Node:
    __slots__ = ('nid', 'kind', 'rtype', 'name', 'body', 'text', 'file', 'module_path')

    def __init__(self, nid, kind, rtype, name, body, file, module_path):
        self.nid = nid
        self.kind = kind
        self.rtype = rtype
        self.name = name
        self.body = body
        self.text = _stringify(body)
        self.file = file
        self.module_path = module_path


def parse_directory(root: str, max_files: int = 4000, recursive: bool = True):
    """Parse every .tf file under root into nodes. Returns (nodes, stats)."""
    nodes = {}
    stats = {'tf_files': 0, 'loc': 0, 'parse_errors': 0, 'blocks': 0}
    tf_files = []
    if recursive:
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames
                           if d not in ('.git', '.terraform', 'node_modules', 'vendor')]
            for fn in filenames:
                if fn.endswith('.tf'):
                    tf_files.append(os.path.join(dirpath, fn))
    else:
        try:
            for fn in sorted(os.listdir(root)):
                if fn.endswith('.tf') and os.path.isfile(os.path.join(root, fn)):
                    tf_files.append(os.path.join(root, fn))
        except OSError:
            pass
    tf_files = tf_files[:max_files]

    for path in tf_files:
        rel = os.path.relpath(path, root)
        module_path = os.path.dirname(rel)
        try:
            with open(path, 'r', encoding='utf-8', errors='replace') as fh:
                raw = fh.read()
        except OSError:
            continue
        stats['tf_files'] += 1
        stats['loc'] += sum(1 for ln in raw.splitlines()
                            if ln.strip() and not ln.strip().startswith('#')
                            and not ln.strip().startswith('//'))
        parsed = None
        if hcl2 is not None:
            try:
                parsed = hcl2.loads(raw)
            except Exception:
                stats['parse_errors'] += 1
                parsed = None
        if parsed is None:
            continue

        for kind, entries in parsed.items():
            if not isinstance(entries, list):
                entries = [entries]
            for entry in entries:
                if not isinstance(entry, dict):
                    continue
                if kind == 'resource':
                    for rtype, insts in entry.items():
                        if not isinstance(insts, dict):
                            continue
                        for name, body in insts.items():
                            nid = f'{rtype}.{name}'
                            nodes[nid] = Node(nid, 'resource', rtype, name, body, rel, module_path)
                            stats['blocks'] += 1
                elif kind == 'data':
                    for rtype, insts in entry.items():
                        if not isinstance(insts, dict):
                            continue
                        for name, body in insts.items():
                            nid = f'data.{rtype}.{name}'
                            nodes[nid] = Node(nid, 'data', rtype, name, body, rel, module_path)
                            stats['blocks'] += 1
                elif kind == 'module':
                    for name, body in entry.items():
                        nid = f'module.{name}'
                        nodes[nid] = Node(nid, 'module', None, name, body, rel, module_path)
                        stats['blocks'] += 1
                elif kind == 'variable':
                    for name, body in entry.items():
                        nid = f'var.{name}'
                        nodes[nid] = Node(nid, 'variable', None, name, body, rel, module_path)
                        stats['blocks'] += 1
                elif kind == 'output':
                    for name, body in entry.items():
                        nid = f'output.{name}'
                        nodes[nid] = Node(nid, 'output', None, name, body, rel, module_path)
                        stats['blocks'] += 1
                elif kind in ('locals', 'local'):
                    for name, body in entry.items():
                        nid = f'local.{name}'
                        nodes[nid] = Node(nid, 'local', None, name, body, rel, module_path)
                        stats['blocks'] += 1
    return nodes, stats


def build_graph(nodes):
    """Edges point from dependency -> dependent (b depends on a  =>  a -> b)."""
    edges = set()
    for nid, node in nodes.items():
        text = node.text
        refs = set()
        for m in RE_VAR.finditer(text):
            refs.add('var.' + m.group(1))
        for m in RE_LOCAL.finditer(text):
            refs.add('local.' + m.group(1))
        for m in RE_MODULE.finditer(text):
            refs.add('module.' + m.group(1))
        for m in RE_DATA.finditer(text):
            refs.add(f'data.{m.group(1)}.{m.group(2)}')
        for m in RE_RESOURCE.finditer(text):
            cand = f'{m.group(1)}.{m.group(2)}'
            if cand in nodes:
                refs.add(cand)
        for r in refs:
            if r in nodes and r != nid:
                edges.add((r, nid))
    return edges


def longest_path_len(nodes, edges):
    """Longest directed path (in edges) via DAG longest-path with cycle guard."""
    adj = defaultdict(list)
    indeg = defaultdict(int)
    for a, b in edges:
        adj[a].append(b)
        indeg[b] += 1
    order, queue = [], [n for n in nodes if indeg[n] == 0]
    indeg_local = dict(indeg)
    while queue:
        n = queue.pop()
        order.append(n)
        for m in adj[n]:
            indeg_local[m] -= 1
            if indeg_local[m] == 0:
                queue.append(m)
    if len(order) < len(nodes):          # cycle present (rare, malformed config)
        return _longest_path_dfs(nodes, adj)
    dist = {n: 0 for n in nodes}
    for n in order:
        for m in adj[n]:
            if dist[n] + 1 > dist[m]:
                dist[m] = dist[n] + 1
    return max(dist.values()) if dist else 0


def _longest_path_dfs(nodes, adj, cap=20000):
    best, seen = 0, {}
    def dfs(n, stack):
        nonlocal best
        if n in seen:
            return seen[n]
        seen[n] = 0
        d = 0
        for m in adj[n]:
            if m in stack:
                continue
            stack.add(m)
            d = max(d, 1 + dfs(m, stack))
            stack.discard(m)
        seen[n] = d
        best = max(best, d)
        return d
    sys.setrecursionlimit(cap)
    for n in list(nodes)[:cap]:
        try:
            dfs(n, {n})
        except RecursionError:
            break
    return best


def expression_factor(nodes):
    """X(C): weighted count of deferred-decision constructs."""
    counts = defaultdict(float)
    for node in nodes.values():
        body, text = node.body, node.text
        if isinstance(body, dict):
            if 'count' in body:      counts['count'] += 1
            if 'for_each' in body:   counts['for_each'] += 1
            if 'depends_on' in body: counts['depends_on'] += 1
            if 'lifecycle' in body:  counts['lifecycle'] += 1
            if 'dynamic' in body:
                dyn = body['dynamic']
                counts['dynamic'] += len(dyn) if isinstance(dyn, (list, dict)) else 1
            if 'provisioner' in body:
                pv = body['provisioner']
                counts['provisioner'] += len(pv) if isinstance(pv, (list, dict)) else 1
        counts['conditional'] += len(RE_TERNARY.findall(text))
        counts['for_expr']    += len(RE_FOR_EXPR.findall(text))
        counts['try_can']     += len(RE_TRY_CAN.findall(text))
        counts['splat']       += len(RE_SPLAT.findall(text))
    total = sum(EXPR_WEIGHTS.get(k, 1.0) * v for k, v in counts.items())
    return total, dict(counts)


def analyse(root: str, weights=None, recursive: bool = True):
    w = {'D': 1.0, 'C': 1.0, 'X': 1.0, 'M': 1.0, 'K': 1.0}
    if weights:
        w.update(weights)

    nodes, stats = parse_directory(root, recursive=recursive)
    if not nodes:
        return None
    edges = build_graph(nodes)

    fan_in, fan_out = defaultdict(int), defaultdict(int)
    for a, b in edges:
        fan_out[a] += 1
        fan_in[b] += 1

    D = longest_path_len(nodes, edges)
    two_paths = sum(fan_in[n] * fan_out[n] for n in nodes)
    P = len(edges) + two_paths
    X, xdetail = expression_factor(nodes)

    # M(C) composition: each module instantiation contributes its nesting
    # position. A locally sourced module contributes one per path component
    # of its source; a registry or remote module contributes one.
    M = 0
    for n in nodes.values():
        if n.kind != 'module':
            continue
        src = ''
        if isinstance(n.body, dict):
            src = _stringify(n.body.get('source', ''))
        if src.startswith('./') or src.startswith('../'):
            M += 1 + sum(1 for p in src.split('/')[1:] if p and p != '..')
        else:
            M += 1

    # K(C) interface coupling: dependency edges incident to a module
    # instantiation, i.e. references that cross a declared module boundary.
    mod_ids = {nid for nid, n in nodes.items() if n.kind == 'module'}
    K = sum(1 for a, b in edges if a in mod_ids or b in mod_ids)

    iacx = w['D'] * D + w['C'] * P + w['X'] * X + w['M'] * M + w['K'] * K

    n_res = sum(1 for n in nodes.values() if n.kind == 'resource')
    n_mod = sum(1 for n in nodes.values() if n.kind == 'module')
    n_var = sum(1 for n in nodes.values() if n.kind == 'variable')
    n_out = sum(1 for n in nodes.values() if n.kind == 'output')
    n_dat = sum(1 for n in nodes.values() if n.kind == 'data')

    return {
        'path': root,
        # --- IaCX and its factors ---
        'IaCX': round(iacx, 3),
        'D_depth': D,
        'P_propagation': P,
        'X_expression': round(X, 3),
        'M_modularity': M,
        'K_crossing': K,
        # --- graph shape ---
        'nodes': len(nodes),
        'edges': len(edges),
        'two_paths': two_paths,
        'density': round(len(edges) / max(1, len(nodes) * (len(nodes) - 1)), 6),
        'max_fan_in': max(fan_in.values()) if fan_in else 0,
        'max_fan_out': max(fan_out.values()) if fan_out else 0,
        # --- size baselines (Dalla Palma et al. style) ---
        'loc': stats['loc'],
        'tf_files': stats['tf_files'],
        'n_blocks': stats['blocks'],
        'n_resources': n_res,
        'n_modules': n_mod,
        'n_variables': n_var,
        'n_outputs': n_out,
        'n_data': n_dat,
        'parse_errors': stats['parse_errors'],
        'expr_detail': xdetail,
    }


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(description='IaCX - complexity metric for declarative IaC')
    ap.add_argument('path')
    ap.add_argument('--json', action='store_true')
    a = ap.parse_args()
    r = analyse(a.path)
    if r is None:
        print('no Terraform blocks found', file=sys.stderr); sys.exit(1)
    if a.json:
        print(json.dumps(r, indent=2))
    else:
        for k in ('IaCX','D_depth','P_propagation','X_expression','M_modularity','K_crossing',
                  'nodes','edges','loc','n_resources'):
            print(f'{k:>16}: {r[k]}')
