"""Проверка устойчивости выводов к неориентированности графа: направленная арборесценция против неориентированного дерева (L2)."""
import pickle, csv, time
import numpy as np
import networkx as nx
import pulp
from rq_common import inst_graph_mode, reduce_with_trace, expand_tree, solve_tree

DATA = "eth_instances.pkl"
RAW = "Ethereum Phishing Transaction Network/MulDiGraph.pkl"
MAX_NODES = 120          # подвыборка: малые/средние интересные инстансы (directed ILP должен тянуть)
MAX_INSTANCES = 40


def directed_steiner_ilp(nodes, arcs_cost, terminals, root, time_limit=60):
    """Направленный Steiner arborescence: min-cost подмножество дуг, дающее направленные пути
       root→каждый терминал. arcs_cost: dict (u,v)->cost (только реальные дуги).
       Возвращает (cost, selected_arcs) или (None, None) если инфизибл."""
    sinks = [t for t in terminals if t != root]
    arcs = list(arcs_cost.keys())
    by_in = {v: [] for v in nodes}; by_out = {v: [] for v in nodes}
    for (u, v) in arcs:
        by_out[u].append((u, v)); by_in[v].append((u, v))
    p = pulp.LpProblem("dirsteiner", pulp.LpMinimize)
    y = {a: pulp.LpVariable(f"y_{i}", cat="Binary") for i, a in enumerate(arcs)}
    fk = {(k, a): pulp.LpVariable(f"f_{ki}_{ai}", lowBound=0, upBound=1)
          for ki, k in enumerate(sinks) for ai, a in enumerate(arcs)}
    p += pulp.lpSum(arcs_cost[a] * y[a] for a in arcs)
    for a in arcs:
        for k in sinks:
            p += fk[(k, a)] <= y[a]
    for k in sinks:
        for v in nodes:
            inf = pulp.lpSum(fk[(k, a)] for a in by_in[v])
            out = pulp.lpSum(fk[(k, a)] for a in by_out[v])
            b = (-1 if v == root else (1 if v == k else 0))
            p += inf - out == b
    p.solve(pulp.PULP_CBC_CMD(msg=False, timeLimit=time_limit))
    if pulp.LpStatus[p.status] not in ("Optimal",):
        return None, None
    sel = [a for a in arcs if y[a].value() and y[a].value() > 0.5]
    cost = sum(arcs_cost[a] for a in sel)
    return cost, sel


def main():
    I = pickle.load(open(DATA, "rb"))
    pool = [g for g in I if g["metrics"].get("interesting", False)
            and g["metrics"]["n_nodes"] <= MAX_NODES and g["metrics"]["n_terminals"] >= 3]
    pool = sorted(pool, key=lambda g: g["metrics"]["n_nodes"])[:MAX_INSTANCES]
    print(f"подвыборка для directed-robustness: {len(pool)} инстансов (≤{MAX_NODES} узлов)")

    pairs = set()
    for g in pool:
        for e in g["edges"]:
            if e["u"] != e["v"]:                  # пропускаем self-loop (в дерево не входит)
                pairs.add(frozenset((e["u"], e["v"])))
    print(f"уникальных пар для извлечения направлений: {len(pairs)}")
    print("загружаю MulDiGraph (~3 мин, swap)...", flush=True)
    t0 = time.time(); G = pickle.load(open(RAW, "rb"))
    print(f"  загружен за {time.time()-t0:.0f}s", flush=True)

    dirsum = {}        # frozenset -> (uv_sum, vu_sum) для конкретной упорядоченной пары (a,b)
    pair_order = {}
    for p in pairs:
        a, b = tuple(p)
        uv = sum(float(k.get("amount", 0.0)) for k in G._succ.get(a, {}).get(b, {}).values())
        vu = sum(float(k.get("amount", 0.0)) for k in G._succ.get(b, {}).get(a, {}).values())
        # exist flags: была ли ХОТЬ ОДНА транзакция (даже нулевого ETH) в направлении
        ex_uv = b in G._succ.get(a, {})
        ex_vu = a in G._succ.get(b, {})
        dirsum[p] = (uv, vu, ex_uv, ex_vu); pair_order[p] = (a, b)
    del G
    print("направления извлечены, граф выгружен", flush=True)

    rows = []
    for g in pool:
        Gv = inst_graph_mode(g, "vol")
        T = g["terminals"]
        H, TR = reduce_with_trace(Gv, T)
        _, te, _ = solve_tree(H, TR)
        unodes, uedges = expand_tree(H, te)
        unodes |= set(TR)
        undir_cost = sum(Gv[u][v]["cost"] for e in uedges for (u, v) in [tuple(e)])

        # (1) directional consistency: root undirected tree, check parent->child arc exists
        Tree = nx.Graph([tuple(e) for e in uedges])
        if Tree.number_of_nodes() == 0:
            continue
        root = TR[0]
        if root not in Tree:
            root = list(Tree.nodes)[0]
        cons, tot = 0, 0
        for parent, child in nx.bfs_edges(Tree, root):
            if parent == child:
                continue
            p = frozenset((parent, child))
            if p not in pair_order:
                continue
            a, b = pair_order[p]
            ex_uv, ex_vu = dirsum[p][2], dirsum[p][3]
            # направление parent->child существует?
            fwd = (ex_uv if (parent, child) == (a, b) else ex_vu)
            cons += int(fwd); tot += 1
        consist = cons / tot if tot else float("nan")

        # (2) directed arborescence на ИСХОДНЫХ узлах инстанса (не редуцируем — направления нужны на дугах)
        # cost дуги = vol-cost соответствующего неориентированного ребра
        nodes = list(Gv.nodes)
        arcs_cost = {}
        for u, v, d in Gv.edges(data=True):
            if u == v:
                continue
            p = frozenset((u, v))
            if p not in pair_order:
                continue
            a, b = pair_order[p]
            ex_uv, ex_vu = dirsum[p][2], dirsum[p][3]
            if (u, v) == (a, b):
                if ex_uv: arcs_cost[(u, v)] = d["cost"]
                if ex_vu: arcs_cost[(v, u)] = d["cost"]
            else:
                if ex_uv: arcs_cost[(v, u)] = d["cost"]  # (a,b)=(v,u)
                if ex_vu: arcs_cost[(u, v)] = d["cost"]
        best = None
        for r in T:
            c, sel = directed_steiner_ilp(nodes, arcs_cost, T, r, time_limit=30)
            if c is not None and (best is None or c < best[0]):
                best = (c, sel, r)
        if best is None:
            rows.append(dict(region_id=g.get("region_id"), n_nodes=g["metrics"]["n_nodes"],
                             n_terminals=len(T), undir_cost=round(undir_cost, 4),
                             consistency=round(consist, 3), dir_feasible=0,
                             dir_cost="", cost_ratio="", node_jaccard=""))
        else:
            dcost, dsel, dr = best
            dnodes = set([x for a in dsel for x in a]) | set(T)
            jn = len(dnodes & unodes) / len(dnodes | unodes)
            rows.append(dict(region_id=g.get("region_id"), n_nodes=g["metrics"]["n_nodes"],
                             n_terminals=len(T), undir_cost=round(undir_cost, 4),
                             consistency=round(consist, 3), dir_feasible=1,
                             dir_cost=round(dcost, 4), cost_ratio=round(dcost / undir_cost, 3),
                             node_jaccard=round(jn, 3)))
        print(f"  region {g.get('region_id')}: consist={consist:.2f} "
              f"dir_feas={rows[-1]['dir_feasible']} ratio={rows[-1]['cost_ratio']}", flush=True)

    with open("directed_robustness_results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    cons = np.array([r["consistency"] for r in rows if isinstance(r["consistency"], float)])
    feas = np.mean([r["dir_feasible"] for r in rows])
    ratios = [r["cost_ratio"] for r in rows if r["cost_ratio"] != ""]
    jns = [r["node_jaccard"] for r in rows if r["node_jaccard"] != ""]
    print(f"\n=== L2 directed-robustness ({len(rows)} инстансов) ===")
    print(f"  directional consistency undirected-дерева: mean={cons.mean():.3f} median={np.median(cons):.3f}")
    print(f"  доля инстансов с ВЫПОЛНИМОЙ directed-арборесценцией: {feas*100:.0f}%")
    if ratios:
        print(f"  directed/undirected стоимость: mean={np.mean(ratios):.3f} median={np.median(ratios):.3f}")
        print(f"  Jaccard узлов (directed vs undirected): mean={np.mean(jns):.3f} median={np.median(jns):.3f}")
    print("  интерпретация: высокая consistency и Jaccard → undirected-ядро устойчиво к L2;")
    print("                 низкие → направление существенно меняет путь (L2 — реальная угроза).")


if __name__ == "__main__":
    main()
