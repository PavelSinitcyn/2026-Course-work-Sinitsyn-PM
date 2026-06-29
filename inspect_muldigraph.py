"""Проверка схемы и плотности исходного графа MulDiGraph."""
import sys, os, pickle, random
from collections import Counter
import numpy as np
import networkx as nx

PATH = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "..", "data", "MulDiGraph.pkl")


def main():
    if not os.path.exists(PATH):
        print(f"НЕ НАЙДЕН: {PATH}\nСкачай MulDiGraph.pkl (Kaggle/xblock) и положи туда, либо передай путь аргументом.")
        return
    print(f"загружаю {PATH} ...")
    G = pickle.load(open(PATH, "rb"))

    # ---------- (1) СХЕМА ----------
    print("\n===== (1) СХЕМА =====")
    print("тип объекта:", type(G).__name__)
    print("directed:", G.is_directed(), "| multigraph:", G.is_multigraph())
    print("узлов:", G.number_of_nodes(), "| рёбер:", G.number_of_edges())
    # пример узла
    n0 = next(iter(G.nodes))
    print("пример узла:", repr(n0)[:60], "| атрибуты:", dict(G.nodes[n0]))
    # пример ребра (с учётом multigraph)
    e0 = next(iter(G.edges(keys=True, data=True))) if G.is_multigraph() else next(iter(G.edges(data=True)))
    print("пример ребра:", e0)
    # какие вообще ключи атрибутов встречаются
    natt, eatt = set(), set()
    for i, (_, d) in enumerate(G.nodes(data=True)):
        natt |= set(d);
        if i > 2000: break
    for i, e in enumerate(G.edges(data=True)):
        eatt |= set(e[-1]);
        if i > 2000: break
    print("атрибуты узлов (выборка):", natt)
    print("атрибуты рёбер (выборка):", eatt)

    # детект ключей
    isp_key = next((k for k in natt if k.lower() in ("isp", "phishing", "label", "is_phishing")), None)
    amt_key = next((k for k in eatt if k.lower() in ("amount", "value", "amt")), None)
    ts_key  = next((k for k in eatt if k.lower() in ("timestamp", "time", "block_timestamp")), None)
    print(f"\nдетект ключей -> phishing: {isp_key} | amount: {amt_key} | timestamp: {ts_key}")
    if isp_key is None:
        print("⚠️ не нашёл метку phishing на узле — проверь атрибуты выше вручную.");

    # ---------- (2) ПЛОТНОСТЬ вокруг phishing (memory-frugal: без копии графа) ----------
    if isp_key is None:
        return
    directed = G.is_directed()
    def uneigh(n):                               # неориентированные соседи (succ ∪ pred), без копии графа
        if directed:
            return set(G._succ.get(n, ())) | set(G._pred.get(n, ()))
        return set(G._adj.get(n, ()))

    phish = [n for n, d in G.nodes(data=True) if d.get(isp_key) in (1, "1", True, "phishing")]
    pset = set(phish)
    print(f"\n===== (2) ТОПОЛОГИЯ вокруг phishing ({len(phish)} узлов) =====")
    degs = np.array([len(uneigh(n)) for n in phish])
    print("степень phishing-узлов: median=%d mean=%.1f | листья(deg==1)=%.0f%% | deg<=2=%.0f%% | deg>=5=%.0f%%"
          % (int(np.median(degs)), degs.mean(), 100*(degs==1).mean(), 100*(degs<=2).mean(), 100*(degs>=5).mean()))

    random.seed(0); samp = random.sample(phish, min(2000, len(phish)))
    ccs, homo = [], []
    for n in samp:
        nb = list(uneigh(n)); d = len(nb)
        if d:
            homo.append(sum(1 for x in nb if x in pset)/d)
        if 2 <= d <= 150:
            nbset = set(nb); links = pairs = 0
            for i in range(len(nb)):
                ni = uneigh(nb[i])
                for j in range(i+1, len(nb)):
                    pairs += 1
                    if nb[j] in ni: links += 1
            if pairs: ccs.append(links/pairs)
    ccmean = np.mean(ccs) if ccs else 0.0
    print("локальный clustering вокруг phishing: %.3f  (Elliptic=0.003; >0.05 уже заметно плотнее)" % ccmean)
    print("homophily (доля соседей-phishing): %.0f%%  (Elliptic illicit=66%%)" % (100*np.mean(homo)))

    # превью плотности ИНСТАНСА: 2-hop шар вокруг phishing, цикломатика
    print("\n превью плотности инстансов (2-hop шар, без хабов deg>200):")
    crs, sizes = [], []
    for s in samp[:150]:
        ball = {s}; frontier = {s}
        for _ in range(2):
            nf = set()
            for u in frontier:
                un = uneigh(u)
                if len(un) > 200: continue
                nf |= un
            nf -= ball; ball |= nf; frontier = nf
            if len(ball) > 1500: break
        # рёбра в индуцированном подграфе (неориентированные уникальные пары)
        E = 0
        for u in ball:
            E += sum(1 for v in uneigh(u) if v in ball and (u < v))
        cr = E - len(ball) + 1
        crs.append(max(cr, 0)); sizes.append(len(ball))
    print("  размер шара: median=%d узлов | цикломатика(циклов): median=%d mean=%.0f"
          % (int(np.median(sizes)), int(np.median(crs)), np.mean(crs)))

    print("\n===== ВЕРДИКТ =====")
    better = ccmean > 0.02 or np.median(degs) >= 3
    print("плотнее Elliptic?", "ВЕРОЯТНО ДА — строим адаптер и бенчмарк" if better
          else "НЕТ/под вопросом — структура тоже hub-spoke, честнее смотреть в сторону AMLSim")


if __name__ == "__main__":
    main()
