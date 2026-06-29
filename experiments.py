"""Вспомогательные функции: построение графа инстанса, проверка валидности дерева Штейнера и достраивание (repair)."""
import pickle, time
import networkx as nx
from networkx.algorithms.approximation import steiner_tree
from steiner_qubo import (build_qubo_steiner, exact_steiner_ilp, solve_qubo_sa,
                          decode_solution, reduce_steiner)


def center_terminal(tree_edges, terminals):
    """Терминал с минимальным эксцентриситетом в дереве -> меньше уровней L."""
    H = nx.Graph([tuple(e) for e in tree_edges])
    cand = [t for t in terminals if t in H]
    return min(cand, key=lambda t: nx.eccentricity(H, t)) if cand else terminals[0]


def inst_graph(g):
    G = nx.Graph()
    for e in g["edges"]:
        G.add_edge(e["u"], e["v"], cost=e["cost"])
    return G


def rooted_depth(tree_edges, root):
    """Глубина дерева (множество frozenset рёбер) от корня."""
    H = nx.Graph([tuple(e) for e in tree_edges])
    if root not in H:
        return 1
    return max(nx.single_source_shortest_path_length(H, root).values())


def tree_cost(G, edges):
    return sum(G[u][v]["cost"] for e in edges for u, v in [tuple(e)])


def is_valid_steiner(G, edges, terminals):
    """edges образуют дерево, связывающее все терминалы."""
    if not edges:
        return False
    H = nx.Graph([tuple(e) for e in edges])
    nodes = set(H.nodes)
    return (set(terminals).issubset(nodes) and nx.is_connected(H)
            and H.number_of_edges() == H.number_of_nodes() - 1)


def repair(G, decoded_nodes, terminals):
    """Достроить валидное дерево: MST на индуцированном (терминалы ∪ decoded), иначе KMB на всём G."""
    cand = set(terminals) | set(decoded_nodes)
    H = G.subgraph(cand)
    if set(terminals).issubset(H.nodes) and nx.is_connected(H):
        mst = nx.minimum_spanning_tree(H, weight="cost")
        return {frozenset(e) for e in mst.edges}
    kmb = steiner_tree(G, terminals, weight="cost")        # гарантированно валидно
    return {frozenset(e) for e in kmb.edges}


def run(n_instances=4, L_slack=0, num_reads=500, num_sweeps=5000, seed=0):
    """RQ3 baseline (консолидированный): lossless Steiner-редукция + center-rooting + SA на QUBO,
    сверка с точным ILP. Печатаются все инстансы без отбора."""
    I = pickle.load(open("instances.pkl", "rb"))
    I = sorted(I, key=lambda g: g["metrics"]["n_nodes"])[:n_instances]
    print(f"{'N/E':>9} {'->':>2} {'redN/E':>8} {'vars':>5} {'L':>3} | "
          f"{'ILP':>7} {'SA':>7} {'feas':>5} {'ratio':>5} {'t_SA':>6}")
    for g in I:
        G = inst_graph(g); T = g["terminals"]
        R, TR = reduce_steiner(G, T)                         # lossless: оптимум сохраняется
        ilp_cost, ilp_edges = exact_steiner_ilp(R, TR)
        r = center_terminal(ilp_edges, TR)
        L = rooted_depth(ilp_edges, r) + L_slack
        qubo, idx, dec = build_qubo_steiner(R, TR, L=L, root=r)
        nvars = len(idx)
        t0 = time.time()
        _, x = solve_qubo_sa(qubo, nvars, num_reads=num_reads, num_sweeps=num_sweeps, seed=seed)
        t_sa = time.time() - t0
        edges, nodes = decode_solution(x, dec)
        feas = is_valid_steiner(R, edges, TR)
        cost = tree_cost(R, edges) if feas else tree_cost(R, repair(R, nodes, TR))
        ratio = cost / ilp_cost if ilp_cost > 0 else float("nan")
        print(f"{G.number_of_nodes():4d}/{G.number_of_edges():<4d} -> "
              f"{R.number_of_nodes():3d}/{R.number_of_edges():<4d} {nvars:5d} {L:3d} | "
              f"{ilp_cost:7.3f} {cost:7.3f} {str(feas):>5} {ratio:5.2f} {t_sa:6.1f}")


if __name__ == "__main__":
    run()
