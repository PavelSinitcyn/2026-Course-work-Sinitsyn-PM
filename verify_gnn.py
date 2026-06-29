"""Проверка корректности применения GNN: падение лосса и сравнение энергий GNN/SA/оптимума/случайного решения."""
import pickle, numpy as np, torch
from experiments import inst_graph, tree_cost, is_valid_steiner, repair, center_terminal, rooted_depth
from steiner_qubo import (reduce_steiner, exact_steiner_ilp, build_qubo_steiner,
                          solve_qubo_bruteforce, solve_qubo_sa, decode_solution, qubo_energy)
from steiner_gnn import _prepare, _assemble_p, PIGNNSteiner
from verify_foundations import encode_tree

I = pickle.load(open("eth_instances.pkl", "rb"))

# ---------- (1) глобальность: перебор на самом малом инстансе ----------
print("=== (1) ГЛОБАЛЬНОСТЬ минимума QUBO (перебор) ===")
small = min(range(len(I)), key=lambda i: (not I[i]['metrics'].get('interesting', False),
                                          I[i]['metrics']['n_nodes']))
g = I[small]; G = inst_graph(g); T = g["terminals"]
R, TR = reduce_steiner(G, T)
ilp, ie = exact_steiner_ilp(R, TR)
r = center_terminal(ie, TR); L = rooted_depth(ie, r)
qubo, idx, dec = build_qubo_steiner(R, TR, L=L, root=r)
n = len(idx)
print(f"  инстанс idx{small}: reduced {R.number_of_nodes()}N/{R.number_of_edges()}E, L={L}, vars={n}, ILP={ilp:.3f}")
if n <= 24:
    Eg, xg = solve_qubo_bruteforce(qubo, n)
    e, nd = decode_solution(xg, dec); feas = is_valid_steiner(R, e, TR)
    c = tree_cost(R, e) if feas else float('nan')
    print(f"  ГЛОБАЛЬНЫЙ минимум (перебор 2^{n}): E={Eg:.4f} -> дерево feasible={feas} cost={c:.3f}")
    print(f"  совпадает с ILP-оптимумом: {'ДА' if abs(c-ilp)<1e-6 else 'НЕТ — БАГ'}")
else:
    print(f"  vars={n} > 24, перебор пропущен")

# ---------- (2) честность GNN ----------
print("\n=== (2) КОРРЕКТНОСТЬ ПРИМЕНЕНИЯ GNN (energy vs SA vs random vs optimum) ===")
for ti in [18, 8, 142]:
    g = I[ti]; G = inst_graph(g); T = g["terminals"]
    R, TR = reduce_steiner(G, T)
    ilp, ie = exact_steiner_ilp(R, TR)
    r = center_terminal(ie, TR); L = rooted_depth(ie, r)
    qubo, idx, dec = build_qubo_steiner(R, TR, L=L, root=r)
    off = qubo.offset
    xopt = encode_tree(ie, r, idx, L); Eopt = qubo_energy(xopt, qubo)
    # random rounding baseline (энергия случайных битов, среднее по 200)
    rngE = []
    for s in range(200):
        rng = np.random.default_rng(s)
        xr = {i: int(rng.random() < 0.5) for i in range(len(idx))}
        rngE.append(qubo_energy(xr, qubo))
    Erand = np.mean(rngE)
    # GNN: следим за лоссом
    torch.manual_seed(0)
    m = _prepare(R, TR, L, r); model = PIGNNSteiner(m["N"], L, 20, 64)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    losses = []
    for ep in range(1500):
        opt.zero_grad(); p = _assemble_p(model, m); loss = p @ m["M"] @ p
        loss.backward(); opt.step(); losses.append(loss.item())
    p = _assemble_p(model, m); xg = {i: int(v > 0.5) for i, v in enumerate(p.detach())}
    Egnn = qubo_energy(xg, qubo)
    # SA энергия
    _, xsa = solve_qubo_sa(qubo, len(idx), num_reads=500, num_sweeps=5000); Esa = qubo_energy(xsa, qubo)
    print(f"  idx{ti}: vars={len(idx)} | E(opt)={Eopt:.3f}  E(GNN)={Egnn:.3f}  E(SA)={Esa:.3f}  "
          f"E(random)≈{Erand:.1f}")
    print(f"         лосс GNN: старт={losses[0]:.2f} -> финал={losses[-1]:.3f}  "
          f"(снижение {losses[0]-losses[-1]:.2f}); GNN бьёт random: {'ДА' if Egnn<Erand else 'НЕТ'}; "
          f"GNN раунд относит. опт.: {Egnn/Eopt:.2f}x энергии")


if __name__ == "__main__":
    pass
