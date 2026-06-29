"""Агрегированная статистика по results_eth.csv (вопросы RQ2-RQ4)."""
import os
import numpy as np
import pandas as pd

RES = os.path.join(os.path.dirname(__file__), "results_eth.csv")


def f(x):
    return pd.to_numeric(x, errors="coerce")


def main():
    df = pd.read_csv(RES)
    n = len(df)
    solved = df[f(df["sa_ratio"]).notna() | f(df["gnn_ratio"]).notna()].copy()
    too_large = (df["note"] == "too_large_skip_SA_GNN").sum()
    errors = df["note"].astype(str).str.startswith("ERROR").sum()
    print(f"=== БЕНЧМАРК: {n} инстансов ===")
    print(f"  оценено SA/GNN: {len(solved)} | too_large (RQ4): {too_large} | errors: {errors}")
    print(f"  размеры: nodes med={f(df['n_nodes']).median():.0f} "
          f"[{f(df['n_nodes']).min():.0f}-{f(df['n_nodes']).max():.0f}], "
          f"reduced med={f(df['red_nodes']).median():.0f}, vars med={f(df['vars']).median():.0f}")

    sa_r, gnn_r = f(solved["sa_ratio"]), f(solved["gnn_ratio"])
    print("\n=== RQ2 (PI-GNN качество) ===")
    print(f"  GNN ratio: mean={gnn_r.mean():.3f} median={gnn_r.median():.3f} "
          f"| optimal(ratio=1.0): {(gnn_r<=1.0001).mean()*100:.0f}%")
    print(f"  GNN сырой feasible (до repair): {(solved['gnn_feas_raw']==True).mean()*100:.0f}%")

    print("\n=== RQ3 (SA baseline vs GNN на той же QUBO) ===")
    print(f"  SA ratio: mean={sa_r.mean():.3f} median={sa_r.median():.3f} "
          f"| optimal: {(sa_r<=1.0001).mean()*100:.0f}%")
    print(f"  SA сырой feasible: {(solved['sa_feas_raw']==True).mean()*100:.0f}%")
    both = solved.dropna(subset=[]).copy()
    both["saR"], both["gnnR"] = f(both["sa_ratio"]), f(both["gnn_ratio"])
    both = both.dropna(subset=["saR", "gnnR"])
    if len(both):
        print(f"  SA не хуже GNN (saR<=gnnR): {(both['saR']<=both['gnnR']+1e-9).mean()*100:.0f}% из {len(both)}")
        print(f"  GNN строго лучше SA: {(both['gnnR']<both['saR']-1e-9).mean()*100:.0f}%")

    print("\n=== RQ4 (масштабируемость) ===")
    print(f"  ILP время: mean={f(df['ilp_t']).mean():.2f}s max={f(df['ilp_t']).max():.2f}s")
    sv = solved.copy(); sv["nb"] = pd.cut(f(sv["red_nodes"]), [0, 20, 40, 80, 1e9],
                                          labels=["<=20", "21-40", "41-80", ">80"])
    g = sv.groupby("nb", observed=True).agg(
        n=("idx", "size"), sa=("sa_ratio", lambda s: f(s).mean()),
        gnn=("gnn_ratio", lambda s: f(s).mean()),
        gnn_feas=("gnn_feas_raw", lambda s: (s == True).mean()))
    print("  по размеру (reduced nodes): mean ratio + GNN raw-feas")
    for nb, r in g.iterrows():
        print(f"    {str(nb):>6}: n={int(r['n']):3d}  SA={r['sa']:.3f}  GNN={r['gnn']:.3f}  GNNfeas={r['gnn_feas']*100:.0f}%")


if __name__ == "__main__":
    main()
