"""Общие утилиты для RQ5-RQ8: lossless-редукция дерева Штейнера с трассировкой происхождения и разворот найденного дерева обратно в исходные узлы/рёбра."""
import networkx as nx
from networkx.algorithms.approximation import steiner_tree
from steiner_qubo import exact_steiner_ilp


def inst_graph_mode(g, mode="hybrid"):
    """nx.Graph инстанса с cost по режиму:
       'hybrid' = e['cost'] (1e-3+0.5(c_vol+c_risk));  'vol' = 1e-3+c_vol (leakage-free);
       'risk' = 1e-3+c_risk."""
    G = nx.Graph()
    for e in g["edges"]:
        if mode == "hybrid":
            c = e["cost"]
        elif mode == "vol":
            c = 1e-3 + e["c_vol"]
        elif mode == "risk":
            c = 1e-3 + e["c_risk"]
        else:
            raise ValueError(mode)
        # параллельных рёбер в инстансе нет (агрегированы), но на всякий случай — min
        if G.has_edge(e["u"], e["v"]):
            if c < G[e["u"]][e["v"]]["cost"]:
                G[e["u"]][e["v"]]["cost"] = c
        else:
            G.add_edge(e["u"], e["v"], cost=c)
    return G


def reduce_with_trace(G, terminals, weight="cost"):
    """Как reduce_steiner, но каждое редуцированное ребро несёт провенанс:
       orig_edges (set frozenset исходных рёбер) и orig_nodes (set стянутых промежуточных узлов).
       Возвращает (H, terminals_in_H)."""
    H = nx.Graph()
    for u, v, d in G.edges(data=True):
        H.add_edge(u, v, **{weight: d[weight],
                            "oe": {frozenset((u, v))}, "on": set()})
    H.add_nodes_from(G.nodes)
    T = set(terminals)
    changed = True
    while changed:
        changed = False
        for v in list(H.nodes):
            if v in T:
                continue
            deg = H.degree(v)
            if deg <= 1:
                H.remove_node(v); changed = True
            elif deg == 2:
                nbrs = list(H.neighbors(v))
                a, b = nbrs[0], nbrs[1]
                if a == b:
                    H.remove_node(v); changed = True; continue
                da, db = H[v][a], H[v][b]
                newc = da[weight] + db[weight]
                new_oe = da["oe"] | db["oe"]
                new_on = da["on"] | db["on"] | {v}
                if H.has_edge(a, b):
                    if newc < H[a][b][weight]:
                        H[a][b][weight] = newc; H[a][b]["oe"] = new_oe; H[a][b]["on"] = new_on
                else:
                    H.add_edge(a, b, **{weight: newc, "oe": new_oe, "on": new_on})
                H.remove_node(v); changed = True
    return H, [t for t in terminals if t in H]


def expand_tree(H, tree_edges):
    """Разворачивает дерево (список/множество (u,v) на редуцированном H) в исходные
       узлы и рёбра. Возвращает (orig_nodes:set, orig_edges:set(frozenset))."""
    nodes, edges = set(), set()
    for u, v in tree_edges:
        nodes.add(u); nodes.add(v)
        d = H[u][v]
        nodes |= d["on"]
        edges |= d["oe"]
    return nodes, edges


def solve_tree(H, T, weight="cost", model_cap=3000, time_limit=60):
    """Решить Steiner tree на H: точный ILP если модель невелика, иначе KMB-приближение.
       Возвращает (cost, tree_edges:list[(u,v)], method)."""
    model_size = len(T) * 2 * H.number_of_edges()
    if len(T) <= 1:
        return 0.0, [], "trivial"
    if model_size <= model_cap:
        cost, sel = exact_steiner_ilp(H, T, weight=weight, time_limit=time_limit)
        edges = [tuple(e) for e in sel]
        # страховка: если ILP вернул несвязное (редко при time_limit) — fallback KMB
        Htree = nx.Graph(edges)
        if not (set(T).issubset(Htree.nodes) and nx.is_connected(Htree) and Htree.number_of_edges() == Htree.number_of_nodes() - 1):
            kmb = steiner_tree(H, T, weight=weight)
            return kmb.size(weight=weight), list(kmb.edges()), "KMB_fallback"
        return cost, edges, "ILP"
    kmb = steiner_tree(H, T, weight=weight)
    return kmb.size(weight=weight), list(kmb.edges()), "KMB"


def jaccard(a, b):
    a, b = set(a), set(b)
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def node_labels(g):
    """node -> 1 (phishing) | 3 (прочее) из инстанса."""
    return dict(g["node_class"])
