"""Подсчёт альтернативных путей и обходимых узлов Штейнера в инстансах."""
import pickle, os, sys
import numpy as np
import networkx as nx
import pulp
from steiner_qubo import reduce_steiner

REL_TOL  = 1e-6     # относительная толерантность к ничьим по стоимости
CAP_TREES= 64       # потолок на счётчик оптимальных деревьев (>=cap => «много»)
TLIM     = 30       # сек на один ILP-solve

def _build_model(G, T, weight='cost'):
    """Много-коммодитная flow-модель minimum Steiner tree; возвращает (problem, y, edges)."""
    T = list(T); r = T[0]; sinks = [t for t in T if t != r]
    nodes = list(G.nodes)
    arcs = [(u, v) for a, b in G.edges() for (u, v) in ((a, b), (b, a))]
    edges = [tuple(e) for e in G.edges()]
    inn = {v: [] for v in nodes}; out = {v: [] for v in nodes}
    for a in arcs:
        out[a[0]].append(a); inn[a[1]].append(a)
    p = pulp.LpProblem("steiner", pulp.LpMinimize)
    y = {e: pulp.LpVariable(f"y_{i}", cat="Binary") for i, e in enumerate(edges)}
    fk = {(k, a): pulp.LpVariable(f"f_{ki}_{ai}", lowBound=0, upBound=1)
          for ki, k in enumerate(sinks) for ai, a in enumerate(arcs)}
    p += pulp.lpSum(G[u][v][weight] * y[(u, v)] for (u, v) in edges)
    ec = lambda a: y[a] if a in y else y[(a[1], a[0])]
    for a in arcs:
        for k in sinks:
            p += fk[(k, a)] <= ec(a)
    for k in sinks:
        for v in nodes:
            b = (-1 if v == r else (1 if v == k else 0))
            p += pulp.lpSum(fk[(k, a)] for a in inn[v]) - pulp.lpSum(fk[(k, a)] for a in out[v]) == b
    return p, y, edges

def count_min_trees(R, T):
    """(opt, n_alt, exact?) — число РАЗЛИЧНЫХ минимальных Steiner-деревьев через
    no-good cuts: решаем ILP, запрещаем найденное дерево, пока стоимость == opt."""
    T = [t for t in T if t in R]
    if len(T) < 2:
        return 0.0, 1, True
    try:
        p, y, edges = _build_model(R, T)
        cmd = pulp.PULP_CBC_CMD(msg=False, timeLimit=TLIM)
        opt = None; n = 0
        while n < CAP_TREES:
            p.solve(cmd)
            if pulp.LpStatus[p.status] != 'Optimal':
                return (opt if opt is not None else float('nan')), (n if opt is not None else -1), False
            cost = sum(R[u][v]['cost'] for e in edges if y[e].value() and y[e].value() > 0.5
                       for u, v in [e])
            if opt is None:
                opt = cost
            elif cost > opt * (1 + REL_TOL) + 1e-12:
                break                              # следующий оптимум дороже — все собраны
            sel = [e for e in edges if y[e].value() and y[e].value() > 0.5]
            n += 1
            # no-good cut: запретить ровно это множество рёбер
            p += (pulp.lpSum((1 - y[e]) for e in sel) + pulp.lpSum(y[e] for e in edges if e not in set(sel))) >= 1
        return opt, n, n < CAP_TREES
    except Exception:
        return float('nan'), -1, False

def bucketize(n):
    if n <= 40:  return 'малые (≤40)'
    if n <= 100: return 'средние (41–100)'
    return 'крупные (>100)'

def analyze(path, label):
    if not os.path.exists(path):
        print(f"[{label}] нет файла {path}"); return
    INST = pickle.load(open(path, 'rb'))
    print(f"\n===== {label}: {len(INST)} инстансов =====")
    rows = []
    for i, g in enumerate(INST):
        G = nx.Graph()
        for e in g['edges']:
            G.add_edge(e['u'], e['v'], cost=e['cost'])
        T = g['terminals']
        R, TR = reduce_steiner(G, T)
        opt, n_alt, exact = count_min_trees(R, TR)
        rows.append({'size': g['metrics']['n_nodes'], 'n_alt': n_alt, 'exact': exact,
                     'bucket': bucketize(g['metrics']['n_nodes'])})
        if (i + 1) % 25 == 0:
            print(f"  ...{i+1}/{len(INST)}", flush=True)

    def altclass(r):
        if not r['exact'] or r['n_alt'] < 0: return '≥потолок/не точно'
        if r['n_alt'] == 1: return '1 (единственное)'
        if r['n_alt'] == 2: return '2'
        if r['n_alt'] <= 5: return '3–5'
        if r['n_alt'] <= 20: return '6–20'
        return '>20'
    classes = ['1 (единственное)', '2', '3–5', '6–20', '>20', '≥потолок/не точно']
    buckets = ['малые (≤40)', 'средние (41–100)', 'крупные (>100)']
    table = {b: {c: 0 for c in classes} for b in buckets}
    for r in rows:
        table[r['bucket']][altclass(r)] += 1
    # печать
    label_col = 'размер / альт.путей'
    hdr = f"{label_col:22}" + "".join(f"{c:>20}" for c in classes) + f"{'итого':>8}"
    print(hdr)
    for b in buckets:
        tot = sum(table[b].values())
        line = f"{b:22}" + "".join(f"{table[b][c]:>20}" for c in classes) + f"{tot:>8}"
        print(line)
    allc = {c: sum(table[b][c] for b in buckets) for c in classes}
    print(f"{'ВСЕГО':22}" + "".join(f"{allc[c]:>20}" for c in classes) + f"{len(rows):>8}")

if __name__ == '__main__':
    analyze('instances.pkl', 'Elliptic++ (Bitcoin)')
    analyze('eth_instances.pkl', 'Ethereum (MulDiGraph)')
