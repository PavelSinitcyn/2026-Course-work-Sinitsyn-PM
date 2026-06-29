"""Проверка корректности QUBO: кодирование точного оптимума в переменные и сверка энергий (оптимум = глобальный минимум)."""
import pickle, numpy as np, networkx as nx
from experiments import inst_graph, tree_cost, is_valid_steiner, repair, center_terminal, rooted_depth
from steiner_qubo import (reduce_steiner, exact_steiner_ilp, build_qubo_steiner,
                          solve_qubo_sa, decode_solution, qubo_energy)


def encode_tree(tree_edges, root, idx, L):
    """Кодирует дерево (рёбра) в присвоение переменных QUBO. None если глубина > L."""
    H = nx.Graph([tuple(e) for e in tree_edges])
    if root not in H:
        return None
    lev = nx.single_source_shortest_path_length(H, root)
    if max(lev.values()) > L:
        return None
    x = {i: 0 for i in idx.values()}
    par = {}
    for child, lc in lev.items():
        if child == root:
            continue
        for nb in H.neighbors(child):
            if lev.get(nb) == lc - 1:
                par[child] = nb; break
        if child not in par:
            return None
    for v, lv in lev.items():
        if v == root:
            continue
        if ("nd", v, lv) in idx:
            x[idx[("nd", v, lv)]] = 1
    for child, lc in lev.items():
        if child == root:
            continue
        p = par[child]
        if ("pe", p, child, lc) in idx:
            x[idx[("pe", p, child, lc)]] = 1
        else:
            return None      # ребро/ориентация не в модели -> не кодируется
    return x


def main():
    I = pickle.load(open("eth_instances.pkl", "rb"))
    # инстансы разного размера, включая «худший SA» idx142
    targets = [142, 18, 8, 146, 215]
    print(f"{'idx':>4} {'redN/E':>8} {'L':>2} {'ILP':>7} | "
          f"{'E(opt)':>8} {'feasΔ':>7} {'E(SA)':>9} {'cost(SA)':>8} {'вердикт QUBO'}")
    for ti in targets:
        g = I[ti]; G = inst_graph(g); T = g["terminals"]
        R, TR = reduce_steiner(G, T)
        ilp, ie = exact_steiner_ilp(R, TR)
        r = center_terminal(ie, TR); L = rooted_depth(ie, r)
        qubo, idx, dec = build_qubo_steiner(R, TR, L=L, root=r)
        # энергия закодированного оптимума
        xopt = encode_tree(ie, r, idx, L)
        if xopt is None:
            print(f"{ti:>4} encode_tree FAILED (L={L})"); continue
        Eopt = qubo_energy(xopt, qubo)
        feas_gap = Eopt - ilp              # должно быть ~0 (offset включён в energy)
        # SA
        _, xsa = solve_qubo_sa(qubo, len(idx), num_reads=500, num_sweeps=5000)
        Esa = qubo_energy(xsa, qubo)
        e, nd = decode_solution(xsa, dec); feas = is_valid_steiner(R, e, TR)
        csa = tree_cost(R, e) if feas else tree_cost(R, repair(R, nd, TR))
        # вердикт: оптимум должен быть penalty-feasible (gap~0) и НЕ выше SA по энергии
        ok_feas = abs(feas_gap) < 1e-6
        ok_min = Eopt <= Esa + 1e-9
        verdict = ("OK: оптимум feasible и E(opt)<=E(SA) -> QUBO верна, SA не дошёл"
                   if ok_feas and ok_min else
                   ("BUG: оптимум penalty-INfeasible (gap!=0)" if not ok_feas else
                    "BUG: E(SA)<E(opt) при cost(SA)>ILP -> штрафы мискалиброваны"))
        print(f"{ti:>4} {R.number_of_nodes():3d}/{R.number_of_edges():<4d} {L:>2} {ilp:7.3f} | "
              f"{Eopt:8.3f} {feas_gap:+7.1e} {Esa:9.3f} {csa:8.2f} {verdict}")


if __name__ == "__main__":
    main()
