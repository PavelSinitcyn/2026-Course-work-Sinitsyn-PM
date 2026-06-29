"""Проверка достижимости и feasibility нарезанных регионов."""
import sys, os, pickle, random
import numpy as np, networkx as nx
from networkx.algorithms.approximation import steiner_tree

PATH = sys.argv[1] if len(sys.argv) > 1 else "Ethereum Phishing Transaction Network/MulDiGraph.pkl"
HUB = 200            # узлы степени выше не разворачиваем (хабы-биржи), если не phishing

def main():
    print("загружаю", PATH, "...")
    G = pickle.load(open(PATH, "rb"))
    isp = nx.get_node_attributes(G, "isp")
    phish = [n for n, v in isp.items() if v == 1]
    pset = set(phish)
    print("phishing-узлов:", len(phish))

    def uneigh(n):
        return set(G._succ.get(n, ())) | set(G._pred.get(n, ()))
    def udeg(n):
        return len(uneigh(n))

    def ball(seed, R, cap=1500):
        seen = {seed: 0}; frontier = [seed]
        for r in range(1, R + 1):
            nf = []
            for u in frontier:
                if u != seed and udeg(u) > HUB and u not in pset:
                    continue            # не разворачиваем нелицитный хаб
                for v in uneigh(u):
                    if v not in seen:
                        seen[v] = r; nf.append(v)
                if len(seen) > cap:
                    return seen
            frontier = nf
        return seen

    random.seed(0); samp = random.sample(phish, min(300, len(phish)))

    # ---- (A) достижимость phishing на радиусах 2..5 ----
    print("\n=== (A) сколько ДРУГИХ phishing достижимо от phishing-узла ===")
    for R in (2, 3, 4):
        cnts = []
        for s in samp[:150]:
            b = ball(s, R)
            cnts.append(sum(1 for n in b if n in pset and n != s))
        cnts = np.array(cnts)
        print(f"  радиус {R}: др.phishing в регионе — median={int(np.median(cnts))} "
              f"mean={cnts.mean():.1f} | доля регионов с >=1: {100*(cnts>=1).mean():.0f}% | >=3: {100*(cnts>=3).mean():.0f}%")

    # ---- (B) пробная сборка инстансов (region-from-paths) ----
    print("\n=== (B) пробные инстансы (region = пути между phishing в радиусе 4, +1hop) ===")
    print(f"{'nodes':>6} {'term':>5} {'steiner':>7} {'chords':>6} {'cycles':>6}")
    made = 0; stats = []
    for s in samp:
        if made >= 15:
            break
        dist = ball(s, 4)
        terms = [n for n in dist if n in pset]
        if len(terms) < 3:
            continue
        # локальный undirected граф на шаре (для путей/Steiner), cost = 1 (структурный прогон)
        nodes = set(dist)
        H = nx.Graph()
        for u in nodes:
            for v in uneigh(u):
                if v in nodes:
                    H.add_edge(u, v, cost=1.0)
        terms = [t for t in terms if t in H]
        # компонента с макс. числом терминалов
        comp = max(nx.connected_components(H), key=lambda c: sum(1 for t in terms if t in c))
        H = H.subgraph(comp).copy(); terms = [t for t in terms if t in H]
        if len(terms) < 3:
            continue
        # region = объединение кратчайших путей между терминалами + 1 hop
        core = set()
        for i in range(len(terms)):
            for j in range(i + 1, len(terms)):
                if nx.has_path(H, terms[i], terms[j]):
                    core.update(nx.shortest_path(H, terms[i], terms[j]))
        region = set(core)
        for u in core:
            region |= set(H.neighbors(u))
        if len(region) > 600:
            region = set(core)
        R = H.subgraph(region)
        if R.number_of_nodes() < 4 or not nx.is_connected(R):
            continue
        tree = steiner_tree(R, terms, weight="cost")
        nsteiner = sum(1 for n in tree.nodes if n not in pset)
        chords = R.subgraph(tree.nodes).number_of_edges() - tree.number_of_edges()
        cycles = R.number_of_edges() - R.number_of_nodes() + 1
        print(f"{R.number_of_nodes():6d} {len(terms):5d} {nsteiner:7d} {chords:6d} {cycles:6d}")
        stats.append((R.number_of_nodes(), len(terms), nsteiner, chords)); made += 1

    if stats:
        a = np.array(stats)
        print(f"\nИТОГ по {len(stats)} пробным: nodes med={int(np.median(a[:,0]))} | "
              f"терминалов med={int(np.median(a[:,1]))} | Steiner-узлов med={int(np.median(a[:,2]))} | "
              f"хорд med={int(np.median(a[:,3]))}")
        print("сравни с Elliptic: там Steiner-узлов med=2, хорд med=3, терминалов med=18")
    else:
        print("не удалось собрать инстансы с >=3 терминалами в радиусе 4 — phishing слишком разрознены")


if __name__ == "__main__":
    main()
