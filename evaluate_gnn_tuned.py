"""Перемеривание настроенного PI-GNN против ILP/KMB/SA на тех же инстансах (results_gnn_tuned.csv)."""
import os, csv, time, pickle
import pandas as pd, networkx as nx
from networkx.algorithms.approximation import steiner_tree
from experiments import inst_graph, center_terminal, rooted_depth, tree_cost
from steiner_qubo import reduce_steiner, exact_steiner_ilp
from steiner_gnn_tuned import solve_instance_gnn_tuned

RES = "results_gnn_tuned.csv"
FIELDS = ["idx", "red_nodes", "red_edges", "vars_est", "L", "ilp_cost",
          "kmb_ratio", "sa_ratio", "gnn_old_ratio",
          "gnn_tuned_cost", "gnn_tuned_ratio", "gnn_tuned_feas_raw", "t"]


def main():
    df = pd.read_csv("results_eth.csv")
    f = lambda c: pd.to_numeric(df[c], errors="coerce")
    todo = df[(df["ilp_status"] == "Optimal") & f("sa_ratio").notna()].copy()
    idxs = todo["idx"].astype(int).tolist()
    I = pickle.load(open("eth_instances.pkl", "rb"))
    done = set()
    if os.path.exists(RES):
        done = set(pd.read_csv(RES)["idx"].astype(int).tolist())
    new = not os.path.exists(RES)
    print(f"инстансов для перемеривания: {len(idxs)}, уже сделано: {len(done)}", flush=True)
    with open(RES, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        if new:
            w.writeheader()
        for pos, gi in enumerate(idxs):
            if gi in done:
                continue
            row0 = todo[todo["idx"] == gi].iloc[0]
            g = I[gi]; G = inst_graph(g); T = g["terminals"]
            R, TR = reduce_steiner(G, T)
            ilp = float(row0["ilp_cost"])
            kmb = steiner_tree(R, TR, weight="cost"); kmb_r = kmb.size(weight="cost") / ilp
            r = center_terminal(list({frozenset(e) for e in exact_steiner_ilp(R, TR)[1]}), TR) \
                if False else None
            # корень/L как в основном прогоне (из ILP-дерева)
            _, ie = exact_steiner_ilp(R, TR)
            r = center_terminal(ie, TR); L = rooted_depth(ie, r)
            nvars = int(row0["vars"]) if not pd.isna(row0["vars"]) else 0
            restarts = 4 if nvars <= 800 else 2
            t0 = time.time()
            try:
                e, nd, feas_any, cost = solve_instance_gnn_tuned(R, TR, L, r,
                                                                 epochs=1000, restarts=restarts)
                tt = time.time() - t0
                row = dict(idx=gi, red_nodes=R.number_of_nodes(), red_edges=R.number_of_edges(),
                           vars_est=nvars, L=L, ilp_cost=round(ilp, 4),
                           kmb_ratio=round(kmb_r, 4),
                           sa_ratio=round(float(row0["sa_ratio"]), 4),
                           gnn_old_ratio=round(float(row0["gnn_ratio"]), 4),
                           gnn_tuned_cost=round(cost, 4), gnn_tuned_ratio=round(cost / ilp, 4),
                           gnn_tuned_feas_raw=feas_any, t=round(tt, 1))
            except Exception as ex:
                row = {k: "" for k in FIELDS}; row.update(idx=gi, L=f"ERR:{type(ex).__name__}")
            w.writerow(row); fh.flush()
            print(f"[{pos+1}/{len(idxs)}] idx{gi} vars={nvars} ILP={round(ilp,3)} "
                  f"KMB={row.get('kmb_ratio')} SA={row.get('sa_ratio')} "
                  f"GNNold={row.get('gnn_old_ratio')} GNNtuned={row.get('gnn_tuned_ratio')} "
                  f"feas={row.get('gnn_tuned_feas_raw')} {row.get('t')}s", flush=True)


if __name__ == "__main__":
    main()
