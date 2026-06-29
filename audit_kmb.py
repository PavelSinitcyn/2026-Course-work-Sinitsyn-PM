"""Сравнение SA и PI-GNN с тривиальной эвристикой Коу-Марковски-Бермана (KMB)."""
import pandas as pd, numpy as np, pickle
import networkx as nx
from networkx.algorithms.approximation import steiner_tree
from experiments import inst_graph
from steiner_qubo import reduce_steiner, exact_steiner_ilp

df = pd.read_csv('results_eth.csv'); f = lambda c: pd.to_numeric(df[c], errors='coerce')
ev = df[(df['ilp_status'] == 'Optimal') & (f('sa_ratio').notna())].copy()
I = pickle.load(open('eth_instances.pkl', 'rb'))
rows = []
for _, row in ev.iterrows():
    g = I[int(row['idx'])]
    G = inst_graph(g); T = g['terminals']
    R, TR = reduce_steiner(G, T)
    ilp = float(row['ilp_cost'])
    kmb = steiner_tree(R, TR, weight='cost')
    kc = kmb.size(weight='cost')
    rows.append(dict(idx=int(row['idx']), red_nodes=int(row['red_nodes']),
                     ilp=ilp, kmb_ratio=kc / ilp,
                     sa_ratio=float(row['sa_ratio']), gnn_ratio=float(row['gnn_ratio'])))
a = pd.DataFrame(rows)
print(f"=== AUDIT: KMB vs SA vs GNN на {len(a)} инстансах (ratio к ILP-оптимуму) ===")
print(f"  KMB ratio: mean={a['kmb_ratio'].mean():.3f} median={a['kmb_ratio'].median():.3f} opt={(a['kmb_ratio']<=1.0001).mean()*100:.0f}%")
print(f"  SA  ratio: mean={a['sa_ratio'].mean():.3f} median={a['sa_ratio'].median():.3f} opt={(a['sa_ratio']<=1.0001).mean()*100:.0f}%")
print(f"  GNN ratio: mean={a['gnn_ratio'].mean():.3f} median={a['gnn_ratio'].median():.3f} opt={(a['gnn_ratio']<=1.0001).mean()*100:.0f}%")
print(f"  KMB не хуже GNN (kmb<=gnn): {(a['kmb_ratio']<=a['gnn_ratio']+1e-9).mean()*100:.0f}%")
print(f"  KMB не хуже SA  (kmb<=sa):  {(a['kmb_ratio']<=a['sa_ratio']+1e-9).mean()*100:.0f}%")
print(f"  GNN строго лучше KMB: {(a['gnn_ratio']<a['kmb_ratio']-1e-9).mean()*100:.0f}%")
print(f"  |GNN-KMB|<0.01 (фактически совпали): {(abs(a['gnn_ratio']-a['kmb_ratio'])<0.01).mean()*100:.0f}%")
a.to_csv('audit_kmb_results.csv', index=False)
