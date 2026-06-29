"""RQ8: честная configuration-model нуль-модель (rewiring с сохранением степеней) для проверки обогащения посредников."""
import pickle, csv, time
import numpy as np
import networkx as nx
from networkx.algorithms.approximation import steiner_tree
from rq_common import inst_graph_mode

SEED = 0
K = 40                 # число rewiring на инстанс
MAX_NODES = 220        # ограничение по размеру (rewiring дорог на крупных)
rng = np.random.default_rng(SEED)


def kmb_steiner_nodes(G, T):
    tree = steiner_tree(G, T, weight="cost")
    return set(tree.nodes) - set(T)


def main():
    I = pickle.load(open("eth_instances.pkl", "rb"))
    pool = [g for g in I if g["metrics"].get("interesting")]
    rows = []
    t0 = time.time()
    for gi, g in enumerate(pool):
        G = inst_graph_mode(g, "vol")
        if not nx.is_connected(G) or G.number_of_nodes() > MAX_NODES or G.number_of_nodes() < 5:
            continue
        T = g["terminals"]
        costs = [d["cost"] for _, _, d in G.edges(data=True)]
        # наблюдаемое
        obs_st = kmb_steiner_nodes(G, T)
        if not obs_st:
            continue
        btw = nx.betweenness_centrality(G)
        obs_b = np.mean([btw[n] for n in obs_st])
        # null через configuration model
        nb = []
        nedges = G.number_of_edges()
        nswap = min(3 * nedges, 1200)
        for _ in range(K):
            H = G.copy()
            try:
                nx.connected_double_edge_swap(H, nswap=nswap, seed=int(rng.integers(1e9)))
            except Exception:
                continue
            # переразметка стоимостей (та же мультимножество, случайно по рёбрам)
            cc = costs.copy(); rng.shuffle(cc)
            for (u, v), c in zip(list(H.edges()), cc):
                H[u][v]["cost"] = c
            try:
                nst = kmb_steiner_nodes(H, T)
            except Exception:
                continue
            if not nst:
                continue
            bH = nx.betweenness_centrality(H)
            nb.append(np.mean([bH[n] for n in nst]))
        if len(nb) < 10:
            continue
        nb = np.array(nb)
        z = (obs_b - nb.mean()) / (nb.std() + 1e-12)
        pct = (nb < obs_b).mean()
        rows.append(dict(region_id=g.get("region_id"), n_nodes=G.number_of_nodes(),
                         k_obs_steiner=len(obs_st), obs_btw=round(obs_b, 5),
                         null_btw_mean=round(nb.mean(), 5), null_btw_std=round(nb.std(), 5),
                         z_cm=round(z, 3), pct_cm=round(pct, 3), n_null=len(nb)))
        if (gi + 1) % 25 == 0:
            print(f"  ...{gi+1}/{len(pool)} ({len(rows)} done, {time.time()-t0:.0f}s)", flush=True)

    with open("rq8_configmodel_results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    z = np.array([r["z_cm"] for r in rows])
    pct = np.array([r["pct_cm"] for r in rows])
    print(f"\n=== RQ8 configuration-model null ({len(rows)} инстансов, K≈{K} rewirings) ===")
    print("  СРАВНЕНИЕ: betweenness наблюдаемых посредников vs null-посредников (тоже связников!)")
    print(f"  z_cm: mean={z.mean():.3f} median={np.median(z):.3f} | доля z>0: {(z>0).mean()*100:.0f}%")
    print(f"  pct (наблюдаемое выше null): mean={pct.mean():.3f} | доля инстансов pct>0.5: {(pct>0.5).mean()*100:.0f}%")
    pos = (z > 0).sum(); n = len(z)
    rng2 = np.random.default_rng(SEED)
    boot = [rng2.choice(z, n, replace=True).mean() for _ in range(5000)]
    lo, hi = np.percentile(boot, [2.5, 97.5])
    print(f"  знак-тест: {pos}/{n} z>0 | mean z 95% CI [{lo:.3f}, {hi:.3f}]")
    verdict = ("ОБОГАЩЕНИЕ ВЫЖИВАЕТ честный null (не тавтология)" if lo > 0.3
               else ("слабое/незначимое — обогащение было в основном тавтологичным" if hi < 0.5
                     else "пограничное"))
    print(f"  ВЕРДИКТ: {verdict}")


if __name__ == "__main__":
    main()
