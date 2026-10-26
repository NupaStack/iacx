"""Randomised counterexample search for the Briand classifications of Table 3,
run against the CORRECTED extractor."""
import sys, os, tempfile, random, itertools
sys.path.insert(0,'tool/iacx'); import iacx
import networkx as nx

rng = random.Random(20261007)

def emit(nodes, edges, extras=None):
    """Write a Terraform module: nodes are resource names, edges (u,v) means v references u."""
    inb = {}
    for u,v in edges: inb.setdefault(v,[]).append(u)
    L=[]
    for n in nodes:
        refs = inb.get(n,[])
        body = '\n'.join('  arg%d = aws_thing.%s.id'%(i,r) for i,r in enumerate(refs))
        ex = (extras or {}).get(n,'')
        L.append('resource "aws_thing" "%s" {\n%s\n%s}\n'%(n,body,ex))
    return '\n'.join(L)

def meas(text):
    d=tempfile.mkdtemp(); open(os.path.join(d,'main.tf'),'w').write(text)
    n,_=iacx.parse_directory(d,recursive=False); E=iacx.build_graph(n)
    G=nx.DiGraph(); G.add_nodes_from(n); G.add_edges_from(E)
    tp=sum(G.in_degree(v)*G.out_degree(v) for v in G)
    m=iacx.analyse(d,recursive=False)
    D = nx.dag_longest_path_length(G) if nx.is_directed_acyclic_graph(G) else None
    return dict(D=D, P=G.number_of_edges()+tp, V=G.number_of_nodes(), E=G.number_of_edges(),
                X=m.get('X_expression',0.0), M=m.get('M_modularity',0), K=m.get('K_crossing',0))

def randdag(k, p, tag):
    ns=['%s%d'%(tag,i) for i in range(k)]
    es=[(ns[i],ns[j]) for i in range(k) for j in range(i+1,k) if rng.random()<p]
    return ns, es

FAIL=[]
N=0
for trial in range(400):
    k1=rng.randint(2,6); k2=rng.randint(2,6)
    n1,e1=randdag(k1,rng.choice([.2,.35,.5]),'a')
    n2,e2=randdag(k2,rng.choice([.2,.35,.5]),'b')
    t1,t2 = emit(n1,e1), emit(n2,e2)
    tu = t1+'\n'+t2                       # disjoint union: no edge can span
    m1,m2,mu = meas(t1),meas(t2),meas(tu)
    N+=1
    # P: additive over disjoint modules
    if mu['P'] != m1['P']+m2['P']:
        FAIL.append(('P disjoint additivity',m1['P'],m2['P'],mu['P'],t1,t2))
    # P: symmetry under full edge reversal
    rev = emit(n1,[(v,u) for u,v in e1])
    if meas(rev)['P'] != m1['P']:
        FAIL.append(('P symmetry',m1['P'],meas(rev)['P'],'',t1,''))
    # D: disjoint modules take the MAXIMUM (length property)
    if mu['D'] != max(m1['D'],m2['D']):
        FAIL.append(('D disjoint maximum',m1['D'],m2['D'],mu['D'],t1,t2))
    # D: must NOT be additive (that is the length/size distinction)
    # X, M additive over disjoint union
    if abs(mu['X']-(m1['X']+m2['X']))>1e-9:
        FAIL.append(('X disjoint additivity',m1['X'],m2['X'],mu['X'],t1,t2))
    if mu['M'] != m1['M']+m2['M']:
        FAIL.append(('M disjoint additivity',m1['M'],m2['M'],mu['M'],t1,t2))
    if mu['K'] != m1['K']+m2['K']:
        FAIL.append(('K disjoint additivity',m1['K'],m2['K'],mu['K'],t1,t2))

# null-value checks: empty relationship set
nullX = meas('resource "aws_thing" "a" {\n  count = 2\n}\n')
nullM = meas('module "m" {\n  source = "./x"\n}\n')
print('trials: %d disjoint-union comparisons'%N)
print('null-value probe  X-case: E=%d X=%.1f  (X>0 with no edges => violates complexity null value: %s)'
      %(nullX['E'],nullX['X'], nullX['X']>0))
print('null-value probe  M-case: E=%d M=%d  (M>0 with no edges => violates: %s)'
      %(nullM['E'],nullM['M'], nullM['M']>0))
print('counterexamples found: %d'%len(FAIL))
seen=set()
for f in FAIL:
    if f[0] in seen: continue
    seen.add(f[0]); print('  !!',f[0],'parts',f[1],f[2],'union',f[3])
