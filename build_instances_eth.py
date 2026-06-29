"""Нарезка независимых плотных индуцированных инстансов вокруг фишинговых узлов из исходного MulDiGraph."""
import sys, pickle, random, itertools
from collections import Counter
import numpy as np, networkx as nx, pandas as pd
from networkx.algorithms.approximation import steiner_tree

PATH = "Ethereum Phishing Transaction Network/MulDiGraph.pkl"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 300
HUB = 200; MIN_TERM = 3; MAX_NODES = 400
OUT_PKL, OUT_CSV = "eth_instances.pkl", "eth_instances_summary.csv"
rng = random.Random(0)


def main():
    print("загружаю", PATH, "...", flush=True)
    G = pickle.load(open(PATH, "rb"))
    isp = nx.get_node_attributes(G, "isp")
    phish = set(n for n, v in isp.items() if v == 1)
    print("phishing:", len(phish), flush=True)

    def uneigh(n):
        return set(G._succ.get(n, ())) | set(G._pred.get(n, ()))
    def udeg(n):
        return len(uneigh(n))
    def pair_amount(u, v):
        best = None
        for a, b in ((u, v), (v, u)):
            d = G._succ.get(a, {}).get(b)
            if d:
                for k in d.values():
                    am = k.get("amount", 0.0)
                    if best is None or am < best[0]:
                        best = (am, k.get("timestamp", 0.0))
        return best or (0.0, 0.0)

    def induced_region(seed):
        """ИНДУЦИРОВАННЫЙ шар (радиус 2, при нехватке терминалов — 3) + обрезка нелицитных листьев."""
        for R in (2, 3):
            seen = {seed}; frontier = [seed]
            for _ in range(R):
                nf = []
                for u in frontier:
                    if u != seed and udeg(u) > HUB and u not in phish:
                        continue
                    for v in uneigh(u):
                        if v not in seen:
                            seen.add(v); nf.append(v)
                    if len(seen) > 1500:
                        break
                frontier = nf
            H = nx.Graph()
            for u in seen:
                for v in uneigh(u):
                    if v in seen:
                        H.add_edge(u, v)
            # обрезаем нелицитные листья (2-core по non-phishing), phishing не трогаем
            changed = True
            while changed:
                changed = False
                for n in [x for x in H if x not in phish and H.degree(x) <= 1]:
                    H.remove_node(n); changed = True
            if H.number_of_nodes() == 0:
                continue
            comp = max(nx.connected_components(H), key=lambda c: sum(1 for t in c if t in phish))
            H = H.subgraph(comp).copy()
            pin = [t for t in H if t in phish]
            if len(pin) >= MIN_TERM and 6 <= H.number_of_nodes() <= MAX_NODES:
                return H, pin
        return None, None

    def weighted(H):
        edges, ams, tss = [], [], []
        for u, v in H.edges():
            am, ts = pair_amount(u, v); edges.append((u, v, am, ts)); ams.append(am); tss.append(ts)
        ams = np.array(ams, float); tss = np.array(tss, float)
        mm = lambda x: (x - x.min()) / ((x.max() - x.min()) or 1.0)
        nv, nt = mm(ams), mm(tss)
        Rw = nx.Graph(); eout = []
        for (u, v, am, _), vol, tm in zip(edges, nv, nt):
            risk = 1.0 - 0.5 * ((1 if u in phish else 0) + (1 if v in phish else 0))
            cost = 1e-3 + (vol + tm + risk) / 3.0
            eout.append({"u": u, "v": v, "cost": float(cost), "vol": float(vol),
                         "time": float(tm), "risk": float(risk), "amount": float(am)})
            Rw.add_edge(u, v, cost=cost)
        return Rw, eout

    def assess(Rw, terms):
        tree = steiner_tree(Rw, terms, weight="cost")
        steiner = [n for n in tree.nodes if n not in phish]
        chords = Rw.subgraph(tree.nodes).number_of_edges() - tree.number_of_edges()
        byp = 0
        for v in steiner:
            Hv = Rw.subgraph([x for x in Rw if x != v])
            if all(t in Hv for t in terms) and all(nx.has_path(Hv, terms[0], t) for t in terms):
                byp += 1
        reason = "trivially_connected" if not steiner else ("unique_path" if byp == 0 else "ok")
        return len(steiner), byp, chords, reason

    seeds = list(phish); rng.shuffle(seeds)
    instances, rows, region_keys = [], [], set()
    rid = 0
    for s in seeds:
        if len(instances) >= N:
            break
        H, pin = induced_region(s)
        if H is None:
            continue
        rkey = frozenset(H.nodes)
        if rkey in region_keys:
            continue
        region_keys.add(rkey); rid += 1
        Rw, eout = weighted(H)
        node_class = {n: (1 if n in phish else 3) for n in H.nodes}
        # НЕЗАВИСИМЫЙ набор: 1 инстанс на регион, терминалы = ВСЕ phishing (без подвыборки)
        terms = list(pin)
        nst, byp, chords, reason = assess(Rw, terms)
        m = {"region_id": rid, "n_nodes": H.number_of_nodes(), "n_edges": H.number_of_edges(),
             "n_terminals": len(terms), "n_steiner": nst, "bypassable": byp, "chords": chords,
             "interesting": reason == "ok", "reason": reason}
        rows.append(m)
        if reason == "ok":
            instances.append({"edges": eout, "node_class": node_class, "terminals": terms,
                              "metrics": m, "weights": "eth", "coverage": 1.0, "region_id": rid})
            if len(instances) % 25 == 0:
                print(f"  [{len(instances)}] (регионов просмотрено {rid})", flush=True)

    pickle.dump(instances, open(OUT_PKL, "wb"))
    df = pd.DataFrame(rows); df.to_csv(OUT_CSV, index=False)
    print("\n===== ИТОГ =====")
    print(f"уникальных регионов: {rid} | кандидатов: {len(rows)} | интересных собрано: {len(instances)}")
    if len(df):
        print("причины:", dict(df.reason.value_counts()))
    di = pd.DataFrame([g["metrics"] for g in instances])
    if len(di):
        for k in ["n_nodes", "n_edges", "n_terminals", "n_steiner", "chords"]:
            print("  %-12s med=%.0f [%d..%d]" % (k, di[k].median(), di[k].min(), di[k].max()))
        print("  edge/node    med=%.2f" % (di["n_edges"] / di["n_nodes"]).median())
        print("  уник.регионов среди собранных:", di["region_id"].nunique(),
              "| инстансов на регион med=%.1f" % di.groupby("region_id").size().median())
    print("сохранено ->", OUT_PKL, OUT_CSV)


if __name__ == "__main__":
    main()
