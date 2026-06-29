"""Пересчёт весов рёбер (объёмная и рисковая компоненты) для готовых инстансов без переразбора графа."""
import pickle, os, shutil, time
import numpy as np
import networkx as nx
from networkx.algorithms.approximation import steiner_tree

SRC = 'eth_instances.pkl'
RAW = 'Ethereum Phishing Transaction Network/MulDiGraph.pkl'

def rank01(x):
    x = np.asarray(x, float); n = len(x)
    if n == 1: return np.array([0.5])
    order = np.argsort(x, kind='mergesort'); sx = x[order]
    r = np.empty(n); i = 0
    while i < n:
        j = i
        while j + 1 < n and sx[j + 1] == sx[i]: j += 1
        r[order[i:j + 1]] = (i + j) / 2.0
        i = j + 1
    return r / (n - 1)

def main():
    INST = pickle.load(open(SRC, 'rb'))
    print(f'инстансов: {len(INST)}', flush=True)

    # 1) собрать все уникальные неориентированные пары
    pairs = set()
    for g in INST:
        for e in g['edges']:
            pairs.add(frozenset((e['u'], e['v'])))
    print(f'уникальных пар (u,v) для извлечения: {len(pairs)}', flush=True)

    # 2) загрузить MulDiGraph и извлечь признаки пар (memory-frugal: только нужные lookups)
    print('загружаю MulDiGraph (~2 мин, swap)...', flush=True)
    t0 = time.time()
    G = pickle.load(open(RAW, 'rb'))
    print(f'  загружен за {time.time()-t0:.0f}s: {G.number_of_nodes()} узлов', flush=True)

    def pair_feats(u, v):
        amts, tss, net = [], [], 0.0
        for a, b, sign in ((u, v, 1.0), (v, u, -1.0)):
            d = G._succ.get(a, {}).get(b)
            if d:
                for k in d.values():
                    am = float(k.get('amount', 0.0)); ts = float(k.get('timestamp', 0.0))
                    amts.append(am); tss.append(ts); net += sign * am
        n = len(amts)
        if n == 0:
            return dict(n_tx=0, sum_amt=0.0, max_amt=0.0, net_amt=0.0,
                        t_first=0.0, t_last=0.0, t_span=0.0, mean_dt=0.0)
        ts_s = sorted(tss); dts = [ts_s[i+1]-ts_s[i] for i in range(n-1)]
        return dict(n_tx=n, sum_amt=float(sum(amts)), max_amt=float(max(amts)),
                    net_amt=float(abs(net)), t_first=ts_s[0], t_last=ts_s[-1],
                    t_span=float(ts_s[-1]-ts_s[0]), mean_dt=float(np.mean(dts)) if dts else 0.0)

    feats = {}
    for i, p in enumerate(pairs):
        a, b = tuple(p) if len(p) == 2 else (next(iter(p)), next(iter(p)))
        feats[p] = pair_feats(a, b)
        if (i + 1) % 20000 == 0:
            print(f'  пары {i+1}/{len(pairs)}', flush=True)
    del G  # освободить ОЗУ
    print('признаки пар извлечены, граф выгружен', flush=True)

    # 3) пересчитать веса по инстансам + метрики
    def assess(Gw, phish, terms):
        tree = steiner_tree(Gw, terms, weight='cost')
        steiner = [n for n in tree.nodes if n not in phish]
        chords = Gw.subgraph(tree.nodes).number_of_edges() - tree.number_of_edges()
        byp = 0
        for v in steiner:
            Hv = Gw.subgraph([x for x in Gw if x != v])
            if all(t in Hv for t in terms) and all(nx.has_path(Hv, terms[0], t) for t in terms):
                byp += 1
        reason = 'trivially_connected' if not steiner else ('unique_path' if byp == 0 else 'ok')
        return len(steiner), byp, chords, reason

    changed = 0
    for g in INST:
        phish = set(g['terminals'])
        E = g['edges']
        sa = np.array([feats[frozenset((e['u'], e['v']))]['sum_amt'] for e in E], float)
        cvol = 1.0 - rank01(np.log1p(sa))
        Gw = nx.Graph(); newempty = []
        for e, cv in zip(E, cvol):
            u, v = e['u'], e['v']; f = feats[frozenset((u, v))]
            cr = 0.5 * ((1 if u not in phish else 0) + (1 if v not in phish else 0))
            cost = 1e-3 + 0.5 * (float(cv) + cr)
            rec = dict(u=u, v=v, cost=cost, c_vol=float(cv), c_risk=cr, **f)
            newempty.append(rec); Gw.add_edge(u, v, cost=cost)
        g['edges'] = newempty
        nst, byp, chords, reason = assess(Gw, phish, list(phish))   # терминалы = все phishing
        old = g['metrics'].get('reason')
        g['metrics'].update(n_steiner=nst, bypassable=byp, chords=chords,
                            interesting=(reason == 'ok'), reason=reason)
        g['weights'] = 'eth_v2_rank_vol_risk_no_abs_time'
        if old != reason: changed += 1

    # 4) бэкап и запись
    for f in (SRC, 'eth_instances_summary.csv'):
        if os.path.exists(f): shutil.copy(f, f.replace('.', '_oldweights.', 1))
    pickle.dump(INST, open(SRC, 'wb'))
    import csv
    with open('eth_instances_summary.csv', 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['region_id','n_nodes','n_edges','n_terminals','n_steiner','bypassable','chords','interesting','reason'])
        for g in INST:
            m = g['metrics']
            w.writerow([g.get('region_id',''), m['n_nodes'], m['n_edges'], m['n_terminals'],
                        m['n_steiner'], m['bypassable'], m['chords'], m['interesting'], m['reason']])

    # отчёт
    nz = [1 for g in INST for e in g['edges'] if e['sum_amt'] == 0]
    tot = sum(len(g['edges']) for g in INST)
    multi = sum(1 for g in INST for e in g['edges'] if e['n_tx'] > 1)
    reasons = {}
    for g in INST: reasons[g['metrics']['reason']] = reasons.get(g['metrics']['reason'],0)+1
    print(f"\nГОТОВО. рёбер всего={tot} | нулевой Σamount: {sum(nz)} ({sum(nz)/tot:.1%}) | "
          f"пар с >1 транзакцией: {multi} ({multi/tot:.1%})")
    print(f"интересность под новым cost: {reasons} | сменили категорию: {changed}/{len(INST)}")
    print("бэкап старых весов -> eth_instances_oldweights.pkl / eth_instances_summary_oldweights.csv")

if __name__ == '__main__':
    main()
