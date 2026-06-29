"""RQ8: структурное обогащение посредников (степень, betweenness) против случайной нуль-модели."""
import pickle, csv
import numpy as np
import networkx as nx
from rq_common import (inst_graph_mode, reduce_with_trace, expand_tree, solve_tree, node_labels)

DATA = "eth_instances.pkl"
SEED = 0
NULL_DRAWS = 1000


def main():
    I = pickle.load(open(DATA, "rb"))
    pool = [g for g in I if g["metrics"].get("interesting", False)]
    rng = np.random.default_rng(SEED)
    rows = []
    for gi, g in enumerate(pool):
        G = inst_graph_mode(g, "vol")
        T = g["terminals"]
        H, TR = reduce_with_trace(G, T)
        _, te, _ = solve_tree(H, TR)
        nodes, _ = expand_tree(H, te)
        steiner = [n for n in (nodes - set(T)) if n in G]
        if not steiner:
            continue
        deg = dict(G.degree())
        btw = nx.betweenness_centrality(G)         # инстансы малы (≤~400 узлов)
        nonterm = [n for n in G.nodes if n not in set(T)]
        k = len(steiner)
        if len(nonterm) <= k:
            continue
        obs_deg = np.mean([deg[n] for n in steiner])
        obs_btw = np.mean([btw[n] for n in steiner])
        # degree-preserving null: случайные k нетерминальных узлов
        nd, nb = [], []
        idxarr = np.array(nonterm, dtype=object)
        for _ in range(NULL_DRAWS):
            samp = rng.choice(len(idxarr), k, replace=False)
            S = idxarr[samp]
            nd.append(np.mean([deg[n] for n in S]))
            nb.append(np.mean([btw[n] for n in S]))
        nd, nb = np.array(nd), np.array(nb)
        z_deg = (obs_deg - nd.mean()) / (nd.std() + 1e-12)
        z_btw = (obs_btw - nb.mean()) / (nb.std() + 1e-12)
        pct_deg = (nd < obs_deg).mean()
        pct_btw = (nb < obs_btw).mean()
        rows.append(dict(region_id=g.get("region_id"), n_nodes=g["metrics"]["n_nodes"],
                         k_steiner=k, obs_deg=round(obs_deg, 3), null_deg=round(nd.mean(), 3),
                         z_deg=round(z_deg, 3), pct_deg=round(pct_deg, 3),
                         obs_btw=round(obs_btw, 5), null_btw=round(nb.mean(), 5),
                         z_btw=round(z_btw, 3), pct_btw=round(pct_btw, 3)))
        if (gi + 1) % 25 == 0:
            print(f"  ...{gi+1}/{len(pool)}", flush=True)

    with open("rq8_results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    zdeg = np.array([r["z_deg"] for r in rows])
    zbtw = np.array([r["z_btw"] for r in rows])
    print(f"\n=== RQ8: структурное обогащение посредников (vol-only, {len(rows)} инстансов) ===")
    print("  (label-обогащение phishing вырождено: все phishing — терминалы; см. RQ7 для held-out)")
    print(f"  СТЕПЕНЬ посредников vs degree-preserving null:")
    print(f"    z-score: mean={zdeg.mean():.3f} median={np.median(zdeg):.3f} | доля z>0: {(zdeg>0).mean()*100:.0f}%")
    print(f"    инстансов с посредником в топ-полупроцентиле (pct_deg>0.5): {np.mean([r['pct_deg']>0.5 for r in rows])*100:.0f}%")
    print(f"  BETWEENNESS посредников vs null:")
    print(f"    z-score: mean={zbtw.mean():.3f} median={np.median(zbtw):.3f} | доля z>0: {(zbtw>0).mean()*100:.0f}%")
    print(f"    доля pct_btw>0.5: {np.mean([r['pct_btw']>0.5 for r in rows])*100:.0f}%")
    # комбинированный знак-тест по betweenness
    pos = (zbtw > 0).sum(); n = len(zbtw)
    print(f"  знак-тест betweenness: {pos}/{n} инстансов с z>0 "
          f"({'обогащение' if pos > n*0.5 else 'нет обогащения'})")


if __name__ == "__main__":
    main()
