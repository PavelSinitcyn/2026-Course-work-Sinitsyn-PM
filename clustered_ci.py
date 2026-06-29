"""Region-clustered доверительные интервалы с учётом перекрытия инстансов по узлам (ограничение L4)."""
import pickle
import numpy as np
import networkx as nx
import pandas as pd
from itertools import combinations
from collections import defaultdict

SEED = 0
rng = np.random.default_rng(SEED)
I = pickle.load(open("eth_instances.pkl", "rb"))
pool = [(i, g) for i, g in enumerate(I) if g["metrics"].get("interesting")]
rid = {g["region_id"]: i for i, g in pool}        # region_id -> idx (предполагаем уникальность)
assert len(rid) == len(pool), "region_id не уникальны!"
nodesets = {g["region_id"]: set(n for e in g["edges"] for n in (e["u"], e["v"])) for _, g in pool}
regions = list(nodesets)


def build_clusters(min_jac=None, min_shared=None):
    OV = nx.Graph(); OV.add_nodes_from(regions)
    for a, b in combinations(regions, 2):
        inter = len(nodesets[a] & nodesets[b])
        if inter == 0:
            continue
        if min_shared is not None and inter >= min_shared:
            OV.add_edge(a, b)
        if min_jac is not None and inter / len(nodesets[a] | nodesets[b]) >= min_jac:
            OV.add_edge(a, b)
    lab = {}
    for k, comp in enumerate(nx.connected_components(OV)):
        for r in comp:
            lab[r] = k
    return lab


def cluster_bootstrap_mean(values_by_cluster, n_boot=5000):
    """Кластерный bootstrap: ресемпл кластеров с возвращением, среднее по всем значениям внутри."""
    clusters = list(values_by_cluster)
    out = []
    for _ in range(n_boot):
        pick = rng.choice(len(clusters), len(clusters), replace=True)
        vals = np.concatenate([values_by_cluster[clusters[i]] for i in pick])
        out.append(vals.mean())
    return np.percentile(out, [2.5, 97.5])


def naive_bootstrap_mean(vals, n_boot=5000):
    vals = np.asarray(vals)
    out = [rng.choice(vals, len(vals), replace=True).mean() for _ in range(n_boot)]
    return np.percentile(out, [2.5, 97.5])


# ===================== RQ7: разница (true_hit - null_rate), кластерно =====================
print("=" * 70)
print("RQ7 — region-clustered CI на разнице recall (true - null)")
r7 = pd.read_csv("rq7_results.csv")
r7 = r7[pd.to_numeric(r7["null_rate"], errors="coerce").notna()].copy()
r7["null_rate"] = pd.to_numeric(r7["null_rate"], errors="coerce")
r7["diff"] = r7["true_hit"] - r7["null_rate"]
print(f"  trials с null: {len(r7)}  средняя разница: {r7['diff'].mean():+.4f}")
lo, hi = naive_bootstrap_mean(r7["diff"].values)
print(f"  NAIVE bootstrap (по trials):           95% CI [{lo:+.4f}, {hi:+.4f}]  {'значимо' if lo>0 else 'НЕ значимо'}")
for label, kw in [("Jaccard>=0.20 (158 кл.)", dict(min_jac=0.20)),
                  ("общих>=25 (51 кл.)", dict(min_shared=25)),
                  ("любой общий узел (2 кл., худший случай)", dict(min_shared=1))]:
    lab = build_clusters(**kw)
    vbc = defaultdict(list)
    for _, row in r7.iterrows():
        vbc[lab[row["region_id"]]].append(row["diff"])
    vbc = {k: np.array(v) for k, v in vbc.items()}
    lo, hi = cluster_bootstrap_mean(vbc)
    print(f"  CLUSTERED [{label:38s}]: 95% CI [{lo:+.4f}, {hi:+.4f}]  {'значимо' if lo>0 else 'НЕ значимо'}  (кластеров={len(vbc)})")

# ===================== RQ8: betweenness z-score, кластерно =====================
print("=" * 70)
print("RQ8 — region-clustered значимость структурного обогащения (z_btw)")
r8 = pd.read_csv("rq8_results.csv")
print(f"  инстансов: {len(r8)}  доля z_btw>0 (naive): {(r8['z_btw']>0).mean()*100:.0f}%  mean z={r8['z_btw'].mean():.2f}")
lo, hi = naive_bootstrap_mean(r8["z_btw"].values)
print(f"  NAIVE bootstrap mean z_btw:            95% CI [{lo:+.3f}, {hi:+.3f}]")
for label, kw in [("Jaccard>=0.20", dict(min_jac=0.20)),
                  ("общих>=25", dict(min_shared=25)),
                  ("любой общий узел (худший)", dict(min_shared=1))]:
    lab = build_clusters(**kw)
    vbc = defaultdict(list)
    for _, row in r8.iterrows():
        vbc[lab[row["region_id"]]].append(row["z_btw"])
    # кластер-уровень: среднее z по кластеру, затем доля кластеров с z>0 + bootstrap среднего
    cl_means = np.array([np.mean(v) for v in vbc.values()])
    vbc = {k: np.array(v) for k, v in vbc.items()}
    lo, hi = cluster_bootstrap_mean(vbc)
    print(f"  CLUSTERED [{label:26s}]: mean z 95% CI [{lo:+.3f}, {hi:+.3f}] | "
          f"кластеров с mean z>0: {int((cl_means>0).sum())}/{len(cl_means)}")
print("\nВывод: чем грубее (честнее) кластеризация, тем шире CI. Смотрим, выживает ли значимость.")
