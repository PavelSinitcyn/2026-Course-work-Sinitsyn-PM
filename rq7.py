"""RQ7: held-out скрытые фишинговые узлы — recall попадания в дерево против degree-matched нуль-модели."""
import pickle, csv, random
import numpy as np
from rq_common import (inst_graph_mode, reduce_with_trace, expand_tree, solve_tree, node_labels)

DATA = "eth_instances.pkl"
SEED = 0
NULL_PER_TRIAL = 20          # сколько случайных degree-matched узлов на trial
random.seed(SEED); np.random.seed(SEED)


def solve_hit(g, visible_terms):
    """Решает Steiner tree (vol-only) на visible_terms, возвращает множество узлов дерева."""
    G = inst_graph_mode(g, "vol")
    H, T = reduce_with_trace(G, visible_terms)
    if len(T) < 2:
        return None, G
    _, te, _ = solve_tree(H, T)
    nodes, _ = expand_tree(H, te)
    nodes |= set(T)
    return nodes, G


def main():
    I = pickle.load(open(DATA, "rb"))
    pool = [g for g in I if g["metrics"].get("interesting", False)
            and g["metrics"]["n_terminals"] >= 3]
    print(f"инстансов с ≥3 терминалами: {len(pool)}")
    rows = []
    trials_true, trials_null = [], []     # параллельные списки: hit∈{0,1}, null_rate∈[0,1]
    for gi, g in enumerate(pool):
        phish = list(g["terminals"])
        lab = node_labels(g)
        Gfull = inst_graph_mode(g, "vol")
        deg = dict(Gfull.degree())
        # пул для null: не-phishing, не-терминал
        nonph = [n for n in Gfull.nodes if lab.get(n) != 1]
        for h in phish:
            visible = [t for t in phish if t != h]
            if len(visible) < 2:
                continue
            S, _ = solve_hit(g, visible)
            if S is None:
                continue
            true_hit = int(h in S)
            # degree-matched null
            dh = deg.get(h, 0)
            cand = [n for n in nonph if 0.5 * dh <= deg.get(n, 0) <= 2 * dh + 1 and n not in visible]
            if cand:
                samp = random.sample(cand, min(NULL_PER_TRIAL, len(cand)))
                null_rate = np.mean([1.0 if n in S else 0.0 for n in samp])
            else:
                null_rate = float("nan")
            trials_true.append(true_hit)
            if not np.isnan(null_rate):
                trials_null.append((true_hit, null_rate))
            rows.append(dict(region_id=g.get("region_id"), n_nodes=g["metrics"]["n_nodes"],
                             n_terminals=g["metrics"]["n_terminals"], hidden=h, deg_hidden=dh,
                             true_hit=true_hit, null_rate=round(null_rate, 4) if not np.isnan(null_rate) else "",
                             tree_size=len(S)))
        if (gi + 1) % 25 == 0:
            print(f"  ...{gi+1}/{len(pool)} ({len(rows)} trials)", flush=True)

    with open("rq7_results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    true_recall = np.mean(trials_true)
    paired = np.array(trials_null)               # (true_hit, null_rate)
    null_recall = paired[:, 1].mean()
    diff = paired[:, 0] - paired[:, 1]           # per-trial (hit - null_expectation)
    # bootstrap CI на средней разнице
    rng = np.random.default_rng(SEED)
    boot = [rng.choice(diff, len(diff), replace=True).mean() for _ in range(5000)]
    lo, hi = np.percentile(boot, [2.5, 97.5])

    print(f"\n=== RQ7: скрытые phishing-терминалы (leave-one-out, vol-only, {len(trials_true)} trials) ===")
    print(f"  RECALL по скрытым phishing (доля попавших в дерево): {true_recall:.3f}")
    print(f"  NULL recall (degree-matched случайные не-phishing): {null_recall:.3f}")
    print(f"  разница (true - null): mean={diff.mean():.3f}  95% bootstrap CI [{lo:.3f}, {hi:.3f}]")
    sig = "ЗНАЧИМО (>0)" if lo > 0 else ("ЗНАЧИМО (<0)" if hi < 0 else "НЕ значимо (CI включает 0)")
    print(f"  вывод: дерево проходит через скрытые phishing чаще случайного — {sig}")
    # доп: доля инстансов, где хоть один скрытый узел пойман
    byinst = {}
    for r in rows:
        byinst.setdefault(r["region_id"], []).append(r["true_hit"])
    any_hit = np.mean([1 if any(v) else 0 for v in byinst.values()])
    print(f"  инстансов, где пойман ≥1 скрытый phishing: {any_hit*100:.0f}% из {len(byinst)}")


if __name__ == "__main__":
    main()
