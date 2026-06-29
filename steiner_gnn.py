"""Базовый physics-inspired GNN (вариант C) для depth-QUBO дерева Штейнера: SAGEConv + узловая и рёберная головы, лосс = релаксированная энергия p^T M p."""
from __future__ import annotations
import numpy as np
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn.pytorch import SAGEConv

from steiner_qubo import build_qubo_steiner, reduce_steiner, exact_steiner_ilp, decode_solution
from experiments import inst_graph, center_terminal, rooted_depth, tree_cost, is_valid_steiner, repair

torch.manual_seed(0)


class PIGNNSteiner(nn.Module):
    def __init__(self, N, L, emb=20, hid=64):
        super().__init__()
        self.emb = nn.Embedding(N, emb)
        self.c1 = SAGEConv(emb, hid, "mean")
        self.c2 = SAGEConv(hid, hid, "mean")
        self.nd_head = nn.Linear(hid, L)
        self.pe_head = nn.Sequential(nn.Linear(2 * hid, hid), nn.ReLU(), nn.Linear(hid, L))

    def embed(self, g, ids):
        h = F.relu(self.c1(g, self.emb(ids)))
        return self.c2(g, h)


def _prepare(G, terminals, L, root):
    """Строит QUBO, dgl-граф и индексные тензоры для сборки p-вектора из выходов GNN."""
    qubo, idx, dec = build_qubo_steiner(G, terminals, L=L, root=root)
    n = len(idx)
    M = torch.zeros(n, n)
    for (i, j), w in qubo.Q.items():
        M[i, j] += w                                   # верхнетреугольная (i<=j)

    nodes = list(G.nodes); nmap = {v: k for k, v in enumerate(nodes)}; N = len(nodes)
    # dgl-граф: обе ориентации рёбер для message passing
    src, dst = [], []
    for u, v in G.edges():
        src += [nmap[u], nmap[v]]; dst += [nmap[v], nmap[u]]
    g = dgl.graph((torch.tensor(src), torch.tensor(dst)), num_nodes=N)
    ids = torch.arange(N)

    # направленные рёбра pe (parent a -> child b) из dec['cost_of']
    dir_edges = list(dec["cost_of"].keys())
    de_index = {ab: k for k, ab in enumerate(dir_edges)}
    pe_src = torch.tensor([nmap[a] for a, b in dir_edges])
    pe_dst = torch.tensor([nmap[b] for a, b in dir_edges])

    # позиции для scatter: nd и pe
    nd_pos, nd_node, nd_lvl = [], [], []
    pe_pos, pe_edge, pe_lvl = [], [], []
    for name, pos in idx.items():
        if name[0] == "nd":
            _, v, l = name
            nd_pos.append(pos); nd_node.append(nmap[v]); nd_lvl.append(l - 1)
        else:
            _, a, b, l = name
            pe_pos.append(pos); pe_edge.append(de_index[(a, b)]); pe_lvl.append(l - 1)
    T = lambda x: torch.tensor(x, dtype=torch.long)
    meta = dict(n=n, N=N, L=L, g=g, ids=ids, M=M, dec=dec,
                pe_src=pe_src, pe_dst=pe_dst,
                nd_pos=T(nd_pos), nd_node=T(nd_node), nd_lvl=T(nd_lvl),
                pe_pos=T(pe_pos), pe_edge=T(pe_edge), pe_lvl=T(pe_lvl))
    return meta


def _assemble_p(model, m):
    h = model.embed(m["g"], m["ids"])                  # N x hid
    p_nd = torch.sigmoid(model.nd_head(h))             # N x L
    he = torch.cat([h[m["pe_src"]], h[m["pe_dst"]]], dim=1)
    p_pe = torch.sigmoid(model.pe_head(he))            # E_dir x L
    p = torch.zeros(m["n"])
    p = p.scatter(0, m["nd_pos"], p_nd[m["nd_node"], m["nd_lvl"]])
    p = p.scatter(0, m["pe_pos"], p_pe[m["pe_edge"], m["pe_lvl"]])
    return p


def solve_instance_gnn(G, terminals, L, root, epochs=3000, lr=1e-2, emb=20, hid=64,
                       patience=300, tol=1e-4, verbose=False):
    m = _prepare(G, terminals, L, root)
    model = PIGNNSteiner(m["N"], L, emb, hid)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best, best_loss, wait = None, float("inf"), 0
    for ep in range(epochs):
        opt.zero_grad()
        p = _assemble_p(model, m)
        loss = p @ m["M"] @ p
        loss.backward(); opt.step()
        lv = loss.item()
        if lv < best_loss - tol:
            best_loss, wait = lv, 0
            best = {i: int(v > 0.5) for i, v in enumerate(p.detach())}
        else:
            wait += 1
            if wait >= patience:
                break
        if verbose and ep % 500 == 0:
            print(f"  ep {ep:4d} loss {lv:.3f}")
    edges, nodes = decode_solution(best, m["dec"])
    return edges, nodes, best_loss, ep + 1


def run(n_instances=4, seed=0):
    import pickle
    torch.manual_seed(seed)
    I = sorted(pickle.load(open("instances.pkl", "rb")), key=lambda g: g["metrics"]["n_nodes"])[:n_instances]
    print(f"{'redN/E':>8} {'L':>2} | {'ILP':>7} {'GNN':>7} {'feas':>5} {'ratio':>5} {'epochs':>6}")
    for g in I:
        G = inst_graph(g); T = g["terminals"]
        R, TR = reduce_steiner(G, T)
        ilp, ie = exact_steiner_ilp(R, TR)
        r = center_terminal(ie, TR); L = rooted_depth(ie, r)
        edges, nodes, _, eps = solve_instance_gnn(R, TR, L, r)
        feas = is_valid_steiner(R, edges, TR)
        cost = tree_cost(R, edges) if feas else tree_cost(R, repair(R, nodes, TR))
        print(f"{R.number_of_nodes():3d}/{R.number_of_edges():<4d} {L:2d} | "
              f"{ilp:7.3f} {cost:7.3f} {str(feas):>5} {cost/ilp:5.2f} {eps:6d}")


if __name__ == "__main__":
    run()
