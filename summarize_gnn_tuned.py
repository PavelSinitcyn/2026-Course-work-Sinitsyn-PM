"""Сводное сравнение настроенного PI-GNN с остальными решателями."""
import pandas as pd, numpy as np
d = pd.read_csv("results_gnn_tuned.csv")
d = d[pd.to_numeric(d["gnn_tuned_ratio"], errors="coerce").notna()].copy()
for c in ["kmb_ratio", "sa_ratio", "gnn_old_ratio", "gnn_tuned_ratio"]:
    d[c] = pd.to_numeric(d[c], errors="coerce")
n = len(d)
print(f"=== Тюнингованный PI-GNN vs остальные ({n} инстансов, ratio к ILP-оптимуму) ===")
def line(name, col, feas=None):
    s = d[col]
    extra = ""
    if feas is not None:
        extra = f" | raw-feasible: {(d[feas]==True).mean()*100:.0f}%"
    print(f"  {name:16s}: mean={s.mean():.3f} median={s.median():.3f} "
          f"opt(<=1.0001)={ (s<=1.0001).mean()*100:.0f}%{extra}")
line("KMB", "kmb_ratio")
line("SA (старый)", "sa_ratio")
line("PI-GNN старый", "gnn_old_ratio")
line("PI-GNN ТЮНИНГ", "gnn_tuned_ratio", feas="gnn_tuned_feas_raw")
print("\n--- парные сравнения тюнингованного GNN ---")
print(f"  GNNtuned лучше старого GNN: {(d['gnn_tuned_ratio']<d['gnn_old_ratio']-1e-9).mean()*100:.0f}% | "
      f"хуже: {(d['gnn_tuned_ratio']>d['gnn_old_ratio']+1e-9).mean()*100:.0f}%")
print(f"  GNNtuned не хуже SA:  {(d['gnn_tuned_ratio']<=d['sa_ratio']+1e-9).mean()*100:.0f}%")
print(f"  GNNtuned не хуже KMB: {(d['gnn_tuned_ratio']<=d['kmb_ratio']+1e-9).mean()*100:.0f}%")
print(f"  GNNtuned строго лучше KMB: {(d['gnn_tuned_ratio']<d['kmb_ratio']-1e-9).mean()*100:.0f}%")
print(f"  GNNtuned == оптимум (1.0): {(d['gnn_tuned_ratio']<=1.0001).mean()*100:.0f}% "
      f"(было у старого: {(d['gnn_old_ratio']<=1.0001).mean()*100:.0f}%)")
# по размеру
print("\n--- по размеру (reduced nodes) ---")
d["nb"] = pd.cut(pd.to_numeric(d["red_nodes"]), [0, 20, 40, 80, 1e9], labels=["<=20", "21-40", "41-80", ">80"])
for nb, gr in d.groupby("nb", observed=True):
    print(f"  {str(nb):>6}: n={len(gr):2d}  KMB={gr['kmb_ratio'].mean():.2f}  SA={gr['sa_ratio'].mean():.2f}  "
          f"GNNold={gr['gnn_old_ratio'].mean():.2f}  GNNtuned={gr['gnn_tuned_ratio'].mean():.2f}")
