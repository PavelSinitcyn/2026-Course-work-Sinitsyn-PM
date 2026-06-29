"""Настроенный PI-GNN: отжиг штрафа (разделённые матрицы цели и штрафа), несколько перезапусков и перебор порога округления."""
from __future__ import annotations
import numpy as np, networkx as nx, torch
import torch.nn as nn, torch.nn.functional as F
import dgl
from steiner_qubo import build_qubo_steiner, decode_solution
from steiner_gnn import PIGNNSteiner, _assemble_p
from experiments import tree_cost, is_valid_steiner, repair


def _prepare_tuned(G, terminals, L, root):
    """Готовит M_obj, M_pen (P=1) и dgl/scatter-метаданные."""
    Q1, idx, dec = build_qubo_steiner(G, terminals, L=L, P=1.0, root=root)  # штраф с весом 1
    n = len(idx)
    M1 = torch.zeros(n, n)
    for (i, j), w in Q1.Q.items():
        M1[i, j] += w
    M_obj = torch.zeros(n, n)
    for (a, b), c in dec["cost_of"].items():
        for l in range(1, L + 1):
            i = idx[("pe", a, b, l)]
            M_obj[i, i] += c
    M_pen = M1 - M_obj
    P_full = sum(c for c in dec["cost_of"].values()) + 1.0     # корректный финальный штраф

    nodes = list(G.nodes); nmap = {v: k for k, v in enumerate(nodes)}; N = len(nodes)
    src, dst = [], []
    for u, v in G.edges():
        src += [nmap[u], nmap[v]]; dst += [nmap[v], nmap[u]]
    g = dgl.graph((torch.tensor(src), torch.tensor(dst)), num_nodes=N)
    ids = torch.arange(N)
    dir_edges = list(dec["cost_of"].keys()); de_index = {ab: k for k, ab in enumerate(dir_edges)}
    pe_src = torch.tensor([nmap[a] for a, b in dir_edges])
    pe_dst = torch.tensor([nmap[b] for a, b in dir_edges])
    nd_pos, nd_node, nd_lvl, pe_pos, pe_edge, pe_lvl = [], [], [], [], [], []
    for name, pos in idx.items():
        if name[0] == "nd":
            _, v, l = name; nd_pos.append(pos); nd_node.append(nmap[v]); nd_lvl.append(l - 1)
        else:
            _, a, b, l = name; pe_pos.append(pos); pe_edge.append(de_index[(a, b)]); pe_lvl.append(l - 1)
    T = lambda x: torch.tensor(x, dtype=torch.long)
    meta = dict(n=n, N=N, L=L, g=g, ids=ids, dec=dec, idx=idx,
                M_obj=M_obj, M_pen=M_pen, P_full=P_full,
                pe_src=pe_src, pe_dst=pe_dst,
                nd_pos=T(nd_pos), nd_node=T(nd_node), nd_lvl=T(nd_lvl),
                pe_pos=T(pe_pos), pe_edge=T(pe_edge), pe_lvl=T(pe_lvl))
    return meta


def _decode_best(p, m, R, TR):
    """Threshold-sweep + top-k: лучшее ВАЛИДНОЕ дерево (min cost), иначе (None, лучший repair-набор)."""
    pv = p.detach()
    best_feas = None      # (cost, edges)
    feas_any = False
    cand_thresholds = [0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7]
    decodes = []
    for thr in cand_thresholds:
        decodes.append({i: int(v > thr) for i, v in enumerate(pv)})
    # top-k по вероятности (k = число переменных оптимального масштаба): добавим жадный вариант
    for x in decodes:
        e, nd = decode_solution(x, m["dec"])
        e = {fe for fe in e if len(fe) == 2}      # отбрасываем self-loop (в дерево не входит)
        if is_valid_steiner(R, e, TR):
            feas_any = True
            c = tree_cost(R, e)
            if best_feas is None or c < best_feas[0]:
                best_feas = (c, e, nd)
    if best_feas is not None:
        return best_feas[1], best_feas[2], True
    # ни один порог не дал валидное -> берём набор узлов с макс. суммарной вероятностью для repair
    # (порог 0.5 как опорный)
    x = {i: int(v > 0.5) for i, v in enumerate(pv)}
    e, nd = decode_solution(x, m["dec"])
    e = {fe for fe in e if len(fe) == 2}
    return e, nd, False


def solve_instance_gnn_tuned(R, TR, L, root, epochs=1200, restarts=4, lr=1e-2,
                             emb=20, hid=64, lam0=1.0, patience=250, tol=1e-5):
    m = _prepare_tuned(R, TR, L, root)
    P_full = m["P_full"]
    best_overall = None     # (cost, edges, nodes, feas)
    feas_any_global = False
    for rs in range(restarts):
        torch.manual_seed(rs)
        model = PIGNNSteiner(m["N"], L, emb, hid)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        best_loss, wait = float("inf"), 0
        best_p = None
        for ep in range(epochs):
            lam = lam0 * (P_full / lam0) ** (ep / max(1, epochs - 1))   # геом. annealing штрафа
            opt.zero_grad()
            p = _assemble_p(model, m)
            loss = p @ m["M_obj"] @ p + lam * (p @ m["M_pen"] @ p)
            loss.backward(); opt.step()
            lv = loss.item()
            if lv < best_loss - tol:
                best_loss, wait = lv, 0; best_p = p.detach().clone()
            else:
                wait += 1
                if wait >= patience and ep > epochs // 2:    # annealing должен дойти до большого λ
                    break
        # финальное p (после полного annealing) — декодируем И best_p
        pf = _assemble_p(model, m).detach()
        for cand in (pf, best_p if best_p is not None else pf):
            e, nd, feas = _decode_best(cand, m, R, TR)
            feas_any_global |= feas
            cost = tree_cost(R, e) if feas else tree_cost(R, repair(R, nd, TR))
            if best_overall is None or cost < best_overall[0]:
                best_overall = (cost, e, nd, feas)
    cost, e, nd, feas = best_overall
    return e, nd, feas_any_global, cost
