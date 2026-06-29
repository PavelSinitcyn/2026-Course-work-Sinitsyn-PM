"""QUBO-формулировка минимального дерева Штейнера (глубинное кодирование Лукаса) с точным ILP, перебором, SA и lossless-редукцией. Содержит валидацию корректности формулировки (RQ1)."""
from __future__ import annotations
from collections import defaultdict
from itertools import combinations
import numpy as np
import networkx as nx


# =============================================================== независимый эталон
def exact_steiner_bruteforce(G, terminals, weight="cost"):
    """Точный minimum Steiner tree перебором подмножеств нетерминальных узлов + MST.
    Годен для малых графов (|V\\T| ≲ 18). Возвращает (cost, set(frozenset рёбер))."""
    T = list(terminals)
    others = [v for v in G.nodes if v not in set(T)]
    best_cost, best_edges = float("inf"), None
    for k in range(len(others) + 1):
        for S in combinations(others, k):
            nodes = set(T) | set(S)
            H = G.subgraph(nodes)
            if not nx.is_connected(H):
                continue
            mst = nx.minimum_spanning_tree(H, weight=weight)
            c = mst.size(weight=weight)
            if c < best_cost - 1e-12:
                best_cost = c
                best_edges = {frozenset(e) for e in mst.edges}
        if best_edges is not None and k >= 1:
            # маленькая оптимизация: если уже нашли дерево без Steiner-узлов и оно дешевле любого
            # расширения — всё равно продолжаем (стоимости произвольны), поэтому не выходим рано
            pass
    return best_cost, best_edges


# ================================================= точный ILP (ground truth, масштаб)
def exact_steiner_ilp(G, terminals, weight="cost", msg=False, time_limit=None, return_status=False):
    """Точный minimum Steiner tree через rooted MULTI-commodity flow ILP (PuLP/CBC).
    Много-коммодитная формулировка имеет ПЛОТНУЮ LP-релаксацию (= cut-bound) -> CBC сходится
    быстро даже при многих терминалах (одно-коммодитная взрывалась). На каждый терминал k≠r —
    1 единица потока из корня. time_limit (сек) — страховка от зависания.
    Возвращает (cost, edges[, status])."""
    import pulp
    T = list(terminals); r = T[0]
    sinks = [t for t in T if t != r]
    nodes = list(G.nodes)
    arcs = [(u, v) for a, b in G.edges() for (u, v) in ((a, b), (b, a))]
    edges = [tuple(e) for e in G.edges()]
    by_node_in = {v: [] for v in nodes}; by_node_out = {v: [] for v in nodes}
    for a in arcs:
        by_node_out[a[0]].append(a); by_node_in[a[1]].append(a)

    p = pulp.LpProblem("steiner", pulp.LpMinimize)
    y = {e: pulp.LpVariable(f"y_{i}", cat="Binary") for i, e in enumerate(edges)}
    fk = {(k, a): pulp.LpVariable(f"f_{ki}_{ai}", lowBound=0, upBound=1)
          for ki, k in enumerate(sinks) for ai, a in enumerate(arcs)}
    p += pulp.lpSum(G[u][v][weight] * y[(u, v)] for (u, v) in edges)
    ec = lambda a: y[a] if a in y else y[(a[1], a[0])]
    for a in arcs:                                            # связь потока с выбором ребра
        for k in sinks:
            p += fk[(k, a)] <= ec(a)
    for k in sinks:                                           # сохранение для коммодити k
        for v in nodes:
            inf = pulp.lpSum(fk[(k, a)] for a in by_node_in[v])
            out = pulp.lpSum(fk[(k, a)] for a in by_node_out[v])
            b = (-1 if v == r else (1 if v == k else 0))
            p += inf - out == b
    cmd = pulp.PULP_CBC_CMD(msg=msg, timeLimit=time_limit) if time_limit else pulp.PULP_CBC_CMD(msg=msg)
    p.solve(cmd)
    status = pulp.LpStatus[p.status]
    sel = {frozenset(e) for e in edges if y[e].value() and y[e].value() > 0.5}
    cost = sum(G[u][v][weight] for e in sel for u, v in [tuple(e)])
    return (cost, sel, status) if return_status else (cost, sel)


# ============================================ редукция графа (lossless preprocessing)
def reduce_steiner(G, terminals, weight="cost"):
    """Стандартные lossless-редукции Steiner: удаление нетерминальных листьев (deg 0/1) и
    стягивание нетерминальных вершин степени 2 (два ребра -> одно, стоимости суммируются).
    Сохраняет оптимальную стоимость Steiner tree. Возвращает (G', terminals)."""
    H = nx.Graph()
    for u, v, d in G.edges(data=True):
        H.add_edge(u, v, **{weight: d[weight]})
    H.add_nodes_from(G.nodes)
    T = set(terminals)
    changed = True
    while changed:
        changed = False
        for v in list(H.nodes):
            if v in T:
                continue
            deg = H.degree(v)
            if deg <= 1:                                  # нетерминальный лист/изолят -> убрать
                H.remove_node(v); changed = True
            elif deg == 2:                                # стянуть цепочку
                (a, da), (b, db) = ((nb, H[v][nb][weight]) for nb in H.neighbors(v))
                if a == b:
                    H.remove_node(v); changed = True; continue
                newc = da + db
                if H.has_edge(a, b):
                    if newc < H[a][b][weight]:
                        H[a][b][weight] = newc
                else:
                    H.add_edge(a, b, **{weight: newc})
                H.remove_node(v); changed = True
    return H, [t for t in terminals if t in H]


# =================================================================== построение QUBO
class QUBO:
    """Накопитель QUBO: Q[(i,j)] (i<=j) + offset. Поддерживает добавление квадрата лин.выражения."""
    def __init__(self):
        self.Q = defaultdict(float)
        self.offset = 0.0

    def lin(self, i, w):
        self.Q[(i, i)] += w

    def quad(self, i, j, w):
        if i == j:
            self.Q[(i, i)] += w
        else:
            self.Q[(min(i, j), max(i, j))] += w

    def add_square(self, expr, const, weight):
        """weight * (Σ c_k x_k + const)^2,  expr = {var:coeff}.  x_k^2 = x_k."""
        items = list(expr.items())
        for k, ck in items:
            self.lin(k, weight * (ck * ck + 2 * const * ck))
        for a in range(len(items)):
            for b in range(a + 1, len(items)):
                self.quad(items[a][0], items[b][0], weight * 2 * items[a][1] * items[b][1])
        self.offset += weight * const * const


def build_qubo_steiner(G, terminals, L=None, P=None, weight="cost", root=None):
    """Строит QUBO. Возвращает (qubo, var_index, decode_info)."""
    V = list(G.nodes)
    T = set(terminals)
    r = root if root is not None else next(iter(terminals))
    nonroot = [v for v in V if v != r]
    if L is None:
        L = len(nonroot)                      # верхняя граница глубины дерева
    if P is None:
        P = sum(d[weight] for *_e, d in G.edges(data=True)) + 1.0   # доминирует над целью

    idx = {}                                  # имя переменной -> int
    def vid(name):
        if name not in idx:
            idx[name] = len(idx)
        return idx[name]

    # регистрируем переменные
    for v in nonroot:
        for l in range(1, L + 1):
            vid(("nd", v, l))
    cost_of = {}
    for u, v, d in G.edges(data=True):
        for (a, b) in ((u, v), (v, u)):       # обе ориентации (родитель a -> ребёнок b)
            if b == r:                         # у корня нет родителя
                continue
            for l in range(1, L + 1):
                vid(("pe", a, b, l))
            cost_of[(a, b)] = d[weight]

    qubo = QUBO()
    def inT(v, l):
        """Выражение 'вершина v в дереве на уровне l': (vars dict, const)."""
        if l == 0:
            return ({}, 1.0 if v == r else 0.0)
        if v == r:
            return ({}, 0.0)                   # корень только на уровне 0
        return ({vid(("nd", v, l)): 1.0}, 0.0)

    # P1: <=1 уровень на вершину  -> 2P * Σ_{l<l'} nd_l nd_l'
    for v in nonroot:
        ls = [vid(("nd", v, l)) for l in range(1, L + 1)]
        for a in range(len(ls)):
            for b in range(a + 1, len(ls)):
                qubo.quad(ls[a], ls[b], 2 * P)

    # P2: терминал (≠r) в дереве: P*(1 - Σ_l nd)^2
    for t in T:
        if t == r:
            continue
        expr = {vid(("nd", t, l)): 1.0 for l in range(1, L + 1)}
        qubo.add_square(expr, const=-1.0, weight=P)

    # P3a: ровно один родитель у in-tree вершины: P*(Σ_l nd_v - Σ_{u,l} pe[u,v,l])^2
    parents_of = defaultdict(list)            # v -> list of (u,l) pe-vars
    for (a, b) in cost_of:                    # (parent a, child b)
        for l in range(1, L + 1):
            parents_of[b].append(vid(("pe", a, b, l)))
    for v in nonroot:
        expr = {vid(("nd", v, l)): 1.0 for l in range(1, L + 1)}
        for pv in parents_of[v]:
            expr[pv] = expr.get(pv, 0.0) - 1.0
        qubo.add_square(expr, const=0.0, weight=P)

    # P3b: согласованность уровней: P*Σ pe*( (1-inT(u,l-1)) + (1-inT(v,l)) )
    for (a, b) in cost_of:
        for l in range(1, L + 1):
            p = vid(("pe", a, b, l))
            qubo.lin(p, 2 * P)                # из (1-..)+(1-..) = 2 - inTu - inTv
            for (vexpr, vconst) in (inT(a, l - 1), inT(b, l)):
                # вычитаем P * pe * inT
                qubo.lin(p, -P * vconst)
                for w, coef in vexpr.items():
                    qubo.quad(p, w, -P * coef)

    # Цель: Σ c * pe
    for (a, b), c in cost_of.items():
        for l in range(1, L + 1):
            qubo.lin(vid(("pe", a, b, l)), c)

    decode = {"idx": idx, "root": r, "L": L, "cost_of": cost_of, "weight": weight, "P": P}
    return qubo, idx, decode


# ============================================================ декод и проверка решения
def decode_solution(x, decode):
    """x: dict var_int->0/1. Возвращает (edges:set(frozenset), nodes:set, feasible_flags)."""
    idx, r = decode["idx"], decode["root"]
    inv = {i: name for name, i in idx.items()}
    nodes = {r}
    for i, val in x.items():
        if val and inv[i][0] == "nd":
            nodes.add(inv[i][1])
    edges = set()
    for i, val in x.items():
        if val and inv[i][0] == "pe":
            _, a, b, l = inv[i]
            edges.add(frozenset((a, b)))
    return edges, nodes


def qubo_energy(x, qubo):
    e = qubo.offset
    for (i, j), w in qubo.Q.items():
        if i == j:
            e += w * x.get(i, 0)
        else:
            e += w * x.get(i, 0) * x.get(j, 0)
    return e


# ============================================================ перебор QUBO (для крошечных)
def solve_qubo_bruteforce(qubo, n):
    """Глобальный минимум QUBO полным перебором 2^n (n ≲ 24)."""
    Q = np.zeros((n, n))
    for (i, j), w in qubo.Q.items():
        Q[i, j] += w
    best_e, best_x = float("inf"), None
    for mask in range(1 << n):
        x = np.fromiter(((mask >> k) & 1 for k in range(n)), dtype=np.float64, count=n)
        e = qubo.offset + x @ Q @ x
        if e < best_e:
            best_e, best_x = e, x.copy()
    return best_e, {i: int(best_x[i]) for i in range(n)}


def solve_qubo_sa(qubo, n, num_reads=400, num_sweeps=1000, seed=0):
    """Минимум QUBO симуляцией отжига (neal). Не доказывает глобальность, но при многих reads
    надёжно находит оптимум на малых/средних инстансах. Также служит baseline (RQ3)."""
    import dimod, neal
    Qd = {(i, j): w for (i, j), w in qubo.Q.items()}
    bqm = dimod.BinaryQuadraticModel.from_qubo(Qd, offset=qubo.offset)
    sampler = neal.SimulatedAnnealingSampler()
    res = sampler.sample(bqm, num_reads=num_reads, num_sweeps=num_sweeps, seed=seed)
    best = res.first
    x = {i: int(best.sample.get(i, 0)) for i in range(n)}
    return best.energy, x


# ============================================================================ валидация
def _toy_graphs():
    """Набор крошечных графов с заведомым оптимумом."""
    cases = []

    # (name, G, terminals, L) — L ограничивает глубину дерева (меньше переменных)
    # (1) путь a-b-c, терминалы {a,c}: оптимум включает Steiner-узел b
    G = nx.Graph(); G.add_edge("a", "b", cost=1.0); G.add_edge("b", "c", cost=1.0)
    cases.append(("path_a_b_c", G, ["a", "c"], 2))

    # (2) выбор: терминалы a,c; путь a-b-c (1+1) против прямого a-c (3)
    G = nx.Graph(); G.add_edge("a", "b", cost=1.0); G.add_edge("b", "c", cost=1.0); G.add_edge("a", "c", cost=3.0)
    cases.append(("choice_cheap_via_b", G, ["a", "c"], 2))

    # (3) выбор-2: тот же граф, но прямое a-c=1.5 дешевле пути 1+1 -> оптимум без Steiner-узла
    G = nx.Graph(); G.add_edge("a", "b", cost=1.0); G.add_edge("b", "c", cost=1.0); G.add_edge("a", "c", cost=1.5)
    cases.append(("choice_direct_ac", G, ["a", "c"], 2))

    # (4) звезда Штейнера: 3 терминала вокруг центра s (рёбра 1) против попарных 1.9 -> через центр
    G = nx.Graph()
    for t in ("t1", "t2", "t3"):
        G.add_edge("s", t, cost=1.0)
    G.add_edge("t1", "t2", cost=1.9); G.add_edge("t2", "t3", cost=1.9); G.add_edge("t1", "t3", cost=1.9)
    cases.append(("steiner_star_3term", G, ["t1", "t2", "t3"], 2))

    # (5) звезда без выгоды: попарные рёбра дёшевы (0.9) -> центр не нужен
    G = nx.Graph()
    for t in ("t1", "t2", "t3"):
        G.add_edge("s", t, cost=1.0)
    G.add_edge("t1", "t2", cost=0.9); G.add_edge("t2", "t3", cost=0.9); G.add_edge("t1", "t3", cost=0.9)
    cases.append(("star_no_steiner_needed", G, ["t1", "t2", "t3"], 2))

    return cases


def validate(verbose=True, bf_limit=22):
    ok_all = True
    for name, G, T, L in _toy_graphs():
        exact_cost, exact_edges = exact_steiner_bruteforce(G, T)
        qubo, idx, dec = build_qubo_steiner(G, T, L=L)
        n = len(idx)
        method = "brute" if n <= bf_limit else "SA"
        qcost, x = (solve_qubo_bruteforce(qubo, n) if method == "brute"
                    else solve_qubo_sa(qubo, n))
        edges, nodes = decode_solution(x, dec)
        dec_cost = sum(G[u][v]["cost"] for e in edges for u, v in [tuple(e)])
        spans = set(T).issubset(nodes)
        is_tree = (len(edges) == len(nodes) - 1) and spans
        cost_match = abs(dec_cost - exact_cost) < 1e-6
        ok = spans and is_tree and cost_match
        ok_all &= ok
        if verbose:
            flags = ("" if cost_match else "[COST MISMATCH]") + ("" if is_tree else "[NOT TREE]")
            print(f"{'✅' if ok else '❌'} {name:24s} vars={n:3d} {method:5s} "
                  f"exact={exact_cost:.2f} qubo={dec_cost:.2f} "
                  f"edges={sorted(tuple(sorted(e)) for e in edges)} {flags}")
    print(f"\nRQ1 QUBO correctness: {'ALL PASSED' if ok_all else 'FAILURES'}")
    return ok_all


def validate_ilp(verbose=True):
    """ILP (PuLP/CBC) обязан совпадать с независимым перебором на крошечных графах."""
    ok = True
    for name, G, T, L in _toy_graphs():
        bf, _ = exact_steiner_bruteforce(G, T)
        il, _ = exact_steiner_ilp(G, T)
        match = abs(bf - il) < 1e-6
        ok &= match
        if verbose:
            print(f"{'✅' if match else '❌'} {name:24s} bruteforce={bf:.2f} ilp={il:.2f}")
    print(f"ILP vs bruteforce: {'ALL PASSED' if ok else 'FAILURES'}")
    return ok


if __name__ == "__main__":
    print("=== RQ1: QUBO == точный Steiner (крошечные графы) ===")
    validate()
    print("\n=== ILP ground truth == перебор ===")
    validate_ilp()
