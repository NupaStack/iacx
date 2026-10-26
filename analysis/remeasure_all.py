"""Re-measure every mined directory (19,365) with tool v1.0.0 at the reconstructed snapshot."""
import sys, os, importlib.util, pandas as pd
from multiprocessing import Pool
s = importlib.util.spec_from_file_location('v1', 'iacx/iacx.py'); v1 = importlib.util.module_from_spec(s); s.loader.exec_module(v1)
F = pd.read_csv('data/modules_corrected.csv')[['repo', 'module']].drop_duplicates()
def one(t):
    repo, module = t
    d = os.path.join('mining/clones', repo.replace('/', '__'), '' if module == '(root)' else module)
    try:
        a = v1.analyse(d, recursive=False) if os.path.isdir(d) else None
    except Exception as e:
        a = None
    if a is None: return dict(repo=repo, module=module, v1_ok=0)
    return dict(repo=repo, module=module, v1_ok=1, v1_cyclic=int(a['cyclic']), v1_blocks=a['n_blocks'], v1_nodes=a['nodes'], v1_edges=a['edges'],
                v1_two=a['two_paths'], v1_D=a['D_depth'], v1_P=a['P_propagation'], v1_X=a['X_expression'], v1_M=a['M_modularity'],
                v1_K=a['K_crossing'], v1_S=a['IaCX'], v1_loc=a['loc'])
if __name__ == '__main__':
    with Pool(2) as p: rows = p.map(one, list(zip(F.repo, F.module)), chunksize=50)
    x = pd.DataFrame(rows); x.to_csv('data/all_v1_measures.csv', index=False)
    print('rows', len(x), 'ok', int(x.v1_ok.sum()), 'cyclic', int(x.v1_cyclic.fillna(0).sum()))
