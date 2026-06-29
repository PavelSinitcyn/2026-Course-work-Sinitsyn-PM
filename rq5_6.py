"""RQ5/RQ6: чувствительность найденного пути к определению веса ребра (объём / риск / гибрид)."""
import pickle, csv, statistics as st
import numpy as np
from rq_common import (inst_graph_mode, reduce_with_trace, expand_tree, solve_tree,
                       jaccard, node_labels)

DATA = "eth_instances.pkl"
MODES = ["hybrid", "vol", "risk"]


def solve_mode(g, mode):
    G = inst_graph_mode(g, mode)
    H, T = reduce_with_trace(G, g["terminals"])
    cost, te, method = solve_tree(H, T)
    nodes, edges = expand_tree(H, te)
    nodes |= set(T)                      # терминалы всегда в дереве
    steiner = nodes - set(g["terminals"])
    return dict(cost=cost, method=method, nodes=nodes, edges=edges, steiner=steiner)


def phish_adjacency(g, edges):
    """доля рёбер дерева, инцидентных phishing-узлу (прокси правдоподобности hop'а)."""
    lab = node_labels(g)
    if not edges:
        return float("nan")
    adj = sum(1 for e in edges for (u, v) in [tuple(e)]
              if lab.get(u) == 1 or lab.get(v) == 1)
    return adj / len(edges)


def main():
    I = pickle.load(open(DATA, "rb"))
    pool = [g for g in I if g["metrics"].get("interesting", False)]
    rows = []
    for gi, g in enumerate(pool):
        try:
            sol = {m: solve_mode(g, m) for m in MODES}
        except Exception as ex:
            print(f"  inst region={g.get('region_id')} ERROR {type(ex).__name__}: {ex}")
            continue
        r = {"region_id": g.get("region_id"), "n_nodes": g["metrics"]["n_nodes"],
             "n_terminals": g["metrics"]["n_terminals"], "method": sol["hybrid"]["method"]}
        for a, b in [("hybrid", "vol"), ("hybrid", "risk"), ("vol", "risk")]:
            r[f"jE_{a}_{b}"] = jaccard(sol[a]["edges"], sol[b]["edges"])
            r[f"jN_{a}_{b}"] = jaccard(sol[a]["nodes"], sol[b]["nodes"])
            r[f"jS_{a}_{b}"] = jaccard(sol[a]["steiner"], sol[b]["steiner"])
        # RQ6: hybrid отличается от ОБОИХ single-param?
        r["hybrid_distinct"] = int(sol["hybrid"]["edges"] != sol["vol"]["edges"]
                                   and sol["hybrid"]["edges"] != sol["risk"]["edges"])
        for m in MODES:
            r[f"phadj_{m}"] = phish_adjacency(g, sol[m]["edges"])
            r[f"nsteiner_{m}"] = len(sol[m]["steiner"])
        rows.append(r)
        if (gi + 1) % 25 == 0:
            print(f"  ...{gi+1}/{len(pool)}", flush=True)

    keys = list(rows[0].keys())
    with open("rq5_6_results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)

    def col(name):
        return [r[name] for r in rows if not (isinstance(r[name], float) and np.isnan(r[name]))]

    def desc(name):
        v = col(name)
        return f"mean={np.mean(v):.3f} median={np.median(v):.3f} (n={len(v)})"

    print(f"\n=== RQ5/RQ6 на {len(rows)} интересных инстансах ===")
    print(f"метод решения: ILP={sum(1 for r in rows if r['method']=='ILP')}, "
          f"KMB={sum(1 for r in rows if r['method'] in ('KMB','KMB_fallback'))}")
    print("\n--- RQ5: чувствительность пути к cost (Jaccard; 1.0 = путь не меняется) ---")
    for a, b in [("hybrid", "vol"), ("hybrid", "risk"), ("vol", "risk")]:
        print(f"  {a:6s} vs {b:6s}: рёбра {desc(f'jE_{a}_{b}')}")
        print(f"  {'':6s}    {'':6s}  Steiner-узлы {desc(f'jS_{a}_{b}')}")
    inv = sum(1 for r in rows if r["jE_hybrid_vol"] > 0.999 and r["jE_hybrid_risk"] > 0.999)
    print(f"  путь ИНВАРИАНТЕН к cost (оба Jaccard=1): {inv}/{len(rows)} ({inv/len(rows)*100:.0f}%)")
    print("\n--- RQ6: гибрид vs single-param ---")
    hd = sum(r["hybrid_distinct"] for r in rows)
    print(f"  hybrid-дерево ОТЛИЧАЕТСЯ от обоих single-param: {hd}/{len(rows)} ({hd/len(rows)*100:.0f}%)")
    print(f"  phishing-смежность рёбер: hybrid {desc('phadj_hybrid')}")
    print(f"                            vol    {desc('phadj_vol')}")
    print(f"                            risk   {desc('phadj_risk')}")
    print("  (risk использует phishing-метки напрямую -> высокая phadj ОЖИДАЕМА = утечка, не качество)")
    print("  # Steiner-узлов: hybrid", desc("nsteiner_hybrid"),
          "| vol", desc("nsteiner_vol"), "| risk", desc("nsteiner_risk"))


if __name__ == "__main__":
    main()
