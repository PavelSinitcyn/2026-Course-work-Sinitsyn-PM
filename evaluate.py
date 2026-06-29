"""Массовый прогон решателей (точный ILP, SA, PI-GNN) по всем инстансам с записью результатов в results_eth.csv."""
import os, csv, time, pickle, signal
import networkx as nx
from networkx.algorithms.approximation import steiner_tree
from steiner_qubo import (build_qubo_steiner, exact_steiner_ilp, solve_qubo_sa,
                          decode_solution, reduce_steiner)
from experiments import (inst_graph, center_terminal, rooted_depth, tree_cost,
                         is_valid_steiner, repair)
from steiner_gnn import solve_instance_gnn

DATA = os.path.join(os.path.dirname(__file__), "eth_instances.pkl")   # Ethereum-only (v2-веса)
RESULTS = os.path.join(os.path.dirname(__file__), "results_eth.csv")  # отдельно от старого v1 Elliptic
BENCH_ONLY_INTERESTING = True   # основной бенчмарк = 200 'ok' (interesting==True); 35 unique_path вне RQ2-RQ4
VAR_CAP = 1500            # выше — SA/GNN пропускаем (слишком дорого / не сходится)
SA_READS, SA_SWEEPS = 500, 5000
GNN_EPOCHS = 1500

PER_INSTANCE_LIMIT = 120 # жёсткий wall-clock на инстанс (любая стадия) -> иначе 'timeout'
ILP_TIME_LIMIT = 60      # сек на solve ILP (build не лимитируется -> отдельный предиктор ниже)
ILP_MODEL_CAP = 3000     # |T|*2*|E| выше -> модель строится/решается долго: ILP пропускаем
                         # (KMB ref + skip SA/GNN) = данные RQ4 «точный/QUBO не тянет на масштабе»

FIELDS = ["idx", "n_nodes", "n_edges", "n_terminals", "red_nodes", "red_edges", "vars", "L",
          "ilp_cost", "ilp_status", "ilp_t",
          "sa_cost", "sa_ratio", "sa_feas_raw", "sa_t",
          "gnn_cost", "gnn_ratio", "gnn_feas_raw", "gnn_t", "note"]


def eval_instance(gi, g):
    G = inst_graph(g); T = g["terminals"]
    row = {k: "" for k in FIELDS}
    row.update(idx=gi, n_nodes=G.number_of_nodes(), n_edges=G.number_of_edges(), n_terminals=len(T))
    R, TR = reduce_steiner(G, T)
    row.update(red_nodes=R.number_of_nodes(), red_edges=R.number_of_edges())

    # предиктор тяжести ILP: размер многокоммодитной модели = |T| * 2|E|
    model_size = len(TR) * 2 * R.number_of_edges()
    if model_size > ILP_MODEL_CAP:                    # ILP не тянет на масштабе (данные RQ4)
        kmb = steiner_tree(R, TR, weight="cost")
        row["ilp_cost"] = round(kmb.size(weight="cost"), 4); row["ilp_status"] = "skipped_KMBref"
        ie = list(kmb.edges())
        row["note"] = "ilp_skipped_large"
        r = center_terminal([frozenset(e) for e in ie], TR); row["L"] = rooted_depth(ie, r)
        row["vars"] = (R.number_of_nodes() - 1 + 2 * R.number_of_edges()) * max(1, row["L"])
        return row

    t0 = time.time()
    ilp_cost, ie, st = exact_steiner_ilp(R, TR, time_limit=ILP_TIME_LIMIT, return_status=True)
    row["ilp_t"] = round(time.time() - t0, 2); row["ilp_status"] = st
    row["ilp_cost"] = round(ilp_cost, 4)
    r = center_terminal(ie, TR); L = rooted_depth(ie, r); row["L"] = L

    # оценка числа переменных ДО построения (само построение QUBO дорого при большом L)
    n_dir = sum(1 for a, b in [(u, v) for x, y in R.edges() for (u, v) in ((x, y), (y, x))] if b != r)
    est_vars = (R.number_of_nodes() - 1) * L + n_dir * L
    row["vars"] = est_vars
    if est_vars > VAR_CAP:
        row["note"] = "too_large_skip_SA_GNN"
        return row

    qubo, idx, dec = build_qubo_steiner(R, TR, L=L, root=r)
    row["vars"] = len(idx)
    nvars = len(idx)

    # SA baseline
    t0 = time.time(); _, x = solve_qubo_sa(qubo, nvars, num_reads=SA_READS, num_sweeps=SA_SWEEPS); sa_t = time.time() - t0
    e, nd = decode_solution(x, dec); sa_feas = is_valid_steiner(R, e, TR)
    sa_cost = tree_cost(R, e) if sa_feas else tree_cost(R, repair(R, nd, TR))
    row.update(sa_cost=round(sa_cost, 4), sa_ratio=round(sa_cost / ilp_cost, 4),
               sa_feas_raw=sa_feas, sa_t=round(sa_t, 2))

    # PI-GNN
    t0 = time.time()
    ge, gn, _, _ = solve_instance_gnn(R, TR, L, r, epochs=GNN_EPOCHS)
    gnn_t = time.time() - t0
    gnn_feas = is_valid_steiner(R, ge, TR)
    gnn_cost = tree_cost(R, ge) if gnn_feas else tree_cost(R, repair(R, gn, TR))
    row.update(gnn_cost=round(gnn_cost, 4), gnn_ratio=round(gnn_cost / ilp_cost, 4),
               gnn_feas_raw=gnn_feas, gnn_t=round(gnn_t, 2))
    return row


def main():
    I = pickle.load(open(DATA, "rb"))
    pool = [i for i in range(len(I))
            if (not BENCH_ONLY_INTERESTING) or I[i]["metrics"].get("interesting", True)]
    order = sorted(pool, key=lambda i: I[i]["metrics"]["n_nodes"])  # малые сначала
    done = set()
    if os.path.exists(RESULTS):                       # resume: пропустить уже посчитанные
        import pandas as pd
        done = set(pd.read_csv(RESULTS)["idx"].astype(int).tolist())
    print(f"benchmark: {len(order)}/{len(I)} instances (interesting-only={BENCH_ONLY_INTERESTING}, "
          f"small-first), already done: {len(done)} -> {RESULTS}")
    new = not os.path.exists(RESULTS)
    with open(RESULTS, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        signal.signal(signal.SIGALRM, lambda s, f: (_ for _ in ()).throw(TimeoutError()))
        for pos, gi in enumerate(order):
            if gi in done:
                continue
            try:
                signal.alarm(PER_INSTANCE_LIMIT)
                row = eval_instance(gi, I[gi])      # idx = оригинальный индекс инстанса
            except TimeoutError:
                os.system("pkill -9 -f cbc >/dev/null 2>&1")   # убрать осиротевший CBC
                row = {k: "" for k in FIELDS}; row.update(idx=gi, note="timeout")
            except Exception as ex:
                row = {k: "" for k in FIELDS}; row.update(idx=gi, note=f"ERROR:{type(ex).__name__}")
            finally:
                signal.alarm(0)
            w.writerow(row); f.flush()
            print(f"[{pos+1}/{len(order)}] inst{gi} redN={row['red_nodes']} vars={row['vars']} "
                  f"ILP={row['ilp_cost']}({row['ilp_status']}) SA_r={row['sa_ratio']} "
                  f"GNN_r={row['gnn_ratio']} {row['note']}", flush=True)


if __name__ == "__main__":
    main()
