"""Сборка ноутбука с визуализацией результатов (Plotly + matplotlib) по примерам инстансов и каждому RQ."""
import nbformat as nbf
from nbformat.v4 import new_notebook, new_code_cell, new_markdown_cell
from nbclient import NotebookClient

C = []
def md(s): C.append(new_markdown_cell(s))
def code(s): C.append(new_code_cell(s))

# ---------------------------------------------------------------- заголовок
md("""# Визуализация результатов: Steiner tree на Ethereum-графе (QUBO + PI-GNN)

Наглядные графики по примерам инстансов и по каждому исследовательскому вопросу (RQ1–RQ8 + L2).
Все цифры — из реальных прогонов (`results_eth.csv`, `results_gnn_tuned.csv`, `rq*_results.csv`,
`directed_robustness_results.csv`). Графики — Plotly (интерактивные, с подписями осей и легендами).

**Краткий итог:** QUBO-формулировка корректна (RQ1), но QUBO+PI-GNN не бьёт тривиальный KMB и не
масштабируется (RQ2–RQ4); из доменных гипотез устойчив лишь RQ5 (cost управляет путём), а RQ6/RQ7/RQ8
отрицательны, L2 — реальное ограничение.""")

# ---------------------------------------------------------------- setup
code("""import pickle, numpy as np, pandas as pd
import networkx as nx
from networkx.algorithms.approximation import steiner_tree
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.io as pio
pio.renderers.default = 'notebook'        # встраиваем plotly.js (офлайн-просмотр)

PALETTE = dict(term='#d62728', steiner='#ff7f0e', other='#b0b0b0',
               tree='#d62728', bg='rgba(120,120,120,0.18)',
               kmb='#2ca02c', sa='#1f77b4', gnn='#9467bd', gnnt='#e377c2', ilp='#000000')

INST = pickle.load(open('eth_instances.pkl','rb'))
interesting = [g for g in INST if g['metrics'].get('interesting')]
print('интересных инстансов:', len(interesting))

def load(name):
    return pd.read_csv(name)
res   = load('results_eth.csv')
rest  = load('results_gnn_tuned.csv')
r56   = load('rq5_6_results.csv')
r7    = load('rq7_results.csv')
r8    = load('rq8_results.csv')
r8cm  = load('rq8_configmodel_results.csv')
rdir  = load('directed_robustness_results.csv')
num = lambda s: pd.to_numeric(s, errors='coerce')""")

# ---------------------------------------------------------------- примеры инстансов
md("""## 1. Примеры инстансов разного размера

Узлы = Ethereum-адреса, рёбра = транзакции. <span style="color:#d62728">**Красные звёзды**</span> —
phishing-терминалы (их связывает дерево), <span style="color:#ff7f0e">**оранжевые ромбы**</span> —
промежуточные (Steiner) узлы оптимального дерева, серые точки — остальные узлы региона.
<span style="color:#d62728">Жирные красные рёбра</span> — минимальное Steiner-дерево (hybrid-cost,
KMB), тонкие серые — прочие рёбра региона (показывают альтернативность путей).""")

code("""def draw_instance(g, title, seed=2):
    G = nx.Graph()
    for e in g['edges']:
        if e['u'] != e['v']:
            G.add_edge(e['u'], e['v'], cost=e['cost'])
    T = [t for t in g['terminals'] if t in G]
    tree = steiner_tree(G, T, weight='cost')
    tedges = set(frozenset(e) for e in tree.edges())
    steiner = set(tree.nodes()) - set(T)
    pos = nx.spring_layout(G, seed=seed, k=1.6/np.sqrt(G.number_of_nodes()), iterations=80)

    def edge_xy(pred):
        xs, ys = [], []
        for u, v in G.edges():
            if pred(frozenset((u, v))):
                xs += [pos[u][0], pos[v][0], None]; ys += [pos[u][1], pos[v][1], None]
        return xs, ys
    bx, by = edge_xy(lambda fe: fe not in tedges)
    tx, ty = edge_xy(lambda fe: fe in tedges)

    def node_trace(nodes, name, color, symbol, size):
        xs = [pos[n][0] for n in nodes]; ys = [pos[n][1] for n in nodes]
        deg = [G.degree(n) for n in nodes]
        return go.Scatter(x=xs, y=ys, mode='markers', name=f'{name} ({len(nodes)})',
                          marker=dict(color=color, symbol=symbol, size=size,
                                      line=dict(width=1, color='white')),
                          hovertext=[f'{n[:10]}… deg={d}' for n, d in zip(nodes, deg)],
                          hoverinfo='text')
    others = [n for n in G.nodes() if n not in set(T) and n not in steiner]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=bx, y=by, mode='lines', name='прочие рёбра',
                             line=dict(color=PALETTE['bg'], width=0.8), hoverinfo='skip'))
    fig.add_trace(go.Scatter(x=tx, y=ty, mode='lines', name='Steiner-дерево',
                             line=dict(color=PALETTE['tree'], width=3), hoverinfo='skip'))
    fig.add_trace(node_trace(others, 'прочие узлы', PALETTE['other'], 'circle', 6))
    fig.add_trace(node_trace(sorted(steiner), 'Steiner-узлы', PALETTE['steiner'], 'diamond', 12))
    fig.add_trace(node_trace(T, 'phishing-терминалы', PALETTE['term'], 'star', 15))
    m = g['metrics']
    fig.update_layout(title=f'{title}<br><sub>узлов={m[\"n_nodes\"]}, рёбер={m[\"n_edges\"]}, '
                            f'терминалов={len(T)}, Steiner-узлов={len(steiner)}, '
                            f'μ(цикл.)={g[\"metrics\"].get(\"chords\",\"?\")} хорд</sub>',
                      showlegend=True, width=820, height=560,
                      xaxis=dict(visible=False), yaxis=dict(visible=False),
                      plot_bgcolor='white', legend=dict(orientation='h', y=-0.05))
    return fig

# выбираем малый / средний / крупный интересные инстансы
by_size = sorted(interesting, key=lambda g: g['metrics']['n_nodes'])
small  = by_size[len(by_size)//12]
medium = min(interesting, key=lambda g: abs(g['metrics']['n_nodes']-80))
large  = by_size[-1]
print('малый:', small['metrics']['n_nodes'], '| средний:', medium['metrics']['n_nodes'],
      '| крупный:', large['metrics']['n_nodes'], 'узлов')""")

code("draw_instance(small,  'МАЛЫЙ инстанс')")
code("draw_instance(medium, 'СРЕДНИЙ инстанс')")
code("draw_instance(large,  'КРУПНЫЙ инстанс (высокая альтернативность путей)')")

# ---------------------------------------------------------------- RQ1
md("""## RQ1 — Корректность QUBO-формулировки

Глобальный минимум depth-QUBO совпадает с независимым точным Steiner (перебор) и с ILP на 5
контрольных графах. Дополнительно проверено на реальных инстансах: закодированный ILP-оптимум
penalty-feasible с энергией = стоимости (до 1e-14).""")
code("""cases = ['path_a_b_c','choice_cheap_via_b','choice_direct_ac','steiner_star_3term','star_no_steiner']
exact = [2.00,2.00,1.50,3.00,1.80]; qubo = [2.00,2.00,1.50,3.00,1.80]; ilp = [2.00,2.00,1.50,3.00,1.80]
fig = go.Figure()
fig.add_bar(name='Точный перебор', x=cases, y=exact, marker_color='#1f77b4')
fig.add_bar(name='QUBO-минимум', x=cases, y=qubo, marker_color='#ff7f0e')
fig.add_bar(name='ILP', x=cases, y=ilp, marker_color='#2ca02c')
fig.update_layout(title='RQ1: QUBO-минимум = точный Steiner = ILP (полное совпадение, 5/5)',
                  xaxis_title='контрольный граф', yaxis_title='стоимость дерева',
                  barmode='group', width=860, height=440, plot_bgcolor='white')
fig""")

# ---------------------------------------------------------------- RQ2/3/4
md("""## RQ2 / RQ3 / RQ4 — PI-GNN vs SA vs тривиальный KMB vs точный ILP

Approximation ratio к истинному ILP-оптимуму (81 инстанс, где ILP точен). **Тривиальный KMB
доминирует**; PI-GNN после честного тюнинга (penalty/restart/threshold annealing) заметно лучше
наивного и бьёт SA, но **строго лучше KMB в 0%** случаев.""")
code("""d = rest.copy()
for c in ['kmb_ratio','sa_ratio','gnn_old_ratio','gnn_tuned_ratio']:
    d[c] = num(d[c])
d = d.dropna(subset=['gnn_tuned_ratio'])
methods = [('KMB (тривиальный)','kmb_ratio',PALETTE['kmb']),
           ('SA на QUBO','sa_ratio',PALETTE['sa']),
           ('PI-GNN наивный','gnn_old_ratio',PALETTE['gnn']),
           ('PI-GNN тюнинг','gnn_tuned_ratio',PALETTE['gnnt'])]
fig = go.Figure()
for name,col,color in methods:
    fig.add_trace(go.Box(y=d[col], name=name, marker_color=color, boxmean=True, boxpoints='outliers'))
fig.add_hline(y=1.0, line_dash='dash', line_color='black',
              annotation_text='оптимум (ratio=1)', annotation_position='top left')
fig.update_layout(title='RQ2/RQ3: распределение approximation ratio к ILP-оптимуму (81 инстанс)',
                  yaxis_title='ratio = стоимость / ILP-оптимум (лог-шкала)', yaxis_type='log',
                  xaxis_title='метод', width=880, height=500, plot_bgcolor='white', showlegend=False)
fig""")

code("""# сводная таблица «медиана / оптимум / raw-feasible»
m = res[res['idx'].isin(d['idx'])]
feas = {'KMB':100.0,
        'SA':(m['sa_feas_raw']==True).mean()*100,
        'PI-GNN наивный':(m['gnn_feas_raw']==True).mean()*100,
        'PI-GNN тюнинг':(rest['gnn_tuned_feas_raw']==True).mean()*100}
summ = pd.DataFrame({
  'метод':['KMB','SA','PI-GNN наивный','PI-GNN тюнинг'],
  'mean ratio':[d['kmb_ratio'].mean(),d['sa_ratio'].mean(),d['gnn_old_ratio'].mean(),d['gnn_tuned_ratio'].mean()],
  'median ratio':[d['kmb_ratio'].median(),d['sa_ratio'].median(),d['gnn_old_ratio'].median(),d['gnn_tuned_ratio'].median()],
  '% оптимум':[(d['kmb_ratio']<=1.0001).mean()*100,(d['sa_ratio']<=1.0001).mean()*100,
               (d['gnn_old_ratio']<=1.0001).mean()*100,(d['gnn_tuned_ratio']<=1.0001).mean()*100],
}).round(3)
summ['% raw-feasible'] = [round(feas[x],0) for x in summ['метод']]
print('RQ2/RQ3 сводка: KMB (median 1.0) доминирует над SA/GNN')
summ.style.hide(axis='index').format({'mean ratio':'{:.3f}','median ratio':'{:.3f}',
                                      '% оптимум':'{:.0f}%','% raw-feasible':'{:.0f}%'})\\
    .set_properties(**{'text-align':'left'})""")

md("""### RQ4 — масштабируемость: качество падает с размером""")
code("""g = d.copy(); g['bin'] = pd.cut(num(g['red_nodes']),[0,20,40,80,1e9],labels=['≤20','21–40','41–80','>80'])
agg = g.groupby('bin', observed=True).agg(SA=('sa_ratio','mean'),
        GNN_наив=('gnn_old_ratio','mean'), GNN_тюнинг=('gnn_tuned_ratio','mean'),
        KMB=('kmb_ratio','mean')).reset_index()
fig = go.Figure()
for col,color in [('KMB',PALETTE['kmb']),('SA',PALETTE['sa']),('GNN_наив',PALETTE['gnn']),('GNN_тюнинг',PALETTE['gnnt'])]:
    fig.add_trace(go.Scatter(x=agg['bin'], y=agg[col], mode='lines+markers', name=col,
                             line=dict(color=color, width=3), marker=dict(size=10)))
fig.add_hline(y=1.0, line_dash='dash', line_color='gray')
fig.update_layout(title='RQ4: средний ratio по размеру инстанса — SA деградирует, KMB стабилен ~1.0',
                  xaxis_title='размер редуцированного инстанса (узлов)',
                  yaxis_title='средний ratio к ILP-оптимуму', width=860, height=470,
                  plot_bgcolor='white', legend=dict(orientation='h', y=-0.18))
fig""")
code("""# ILP время vs размер модели (где ILP точен)
e = res.copy(); e['ilp_t']=num(e['ilp_t']); e['model']=num(e['n_terminals'])*2*num(e['red_edges'])
e = e[(e['ilp_status']=='Optimal') & e['ilp_t'].notna()]
fig = px.scatter(e, x='model', y='ilp_t', hover_data=['idx'],
                 labels={'model':'размер ILP-модели = |T|·2|E|','ilp_t':'время ILP, c'},
                 title='RQ4: точный ILP дёшев до своего порога (mean 0.29 c, max 3.7 c)')
fig.update_traces(marker=dict(size=8, color=PALETTE['ilp'], opacity=0.6))
fig.update_layout(width=820, height=440, plot_bgcolor='white')
fig""")

# ---------------------------------------------------------------- RQ5
md("""## RQ5 — Чувствительность пути к определению стоимости

Один инстанс, решённый при cost ∈ {hybrid, vol-only, risk-only}, даёт РАЗНЫЕ деревья. Jaccard
рёбер < 1 ⇒ путь зависит от cost. Показаны достоверные точные ILP-инстансы.""")
code("""il = r56[r56['method']=='ILP']
pairs = [('hybrid vs vol','jE_hybrid_vol'),('hybrid vs risk','jE_hybrid_risk'),('vol vs risk','jE_vol_risk')]
fig = go.Figure()
for name,col in pairs:
    fig.add_trace(go.Violin(y=il[col], name=name, box_visible=True, meanline_visible=True, points=False))
fig.add_hline(y=1.0, line_dash='dash', line_color='green',
              annotation_text='путь не меняется (Jaccard=1)')
fig.update_layout(title='RQ5: Jaccard рёбер дерева между cost-режимами (точные ILP, n=%d)'%len(il),
                  yaxis_title='Jaccard рёбер (1=идентичны, 0=не пересекаются)',
                  xaxis_title='пара cost-режимов', width=860, height=470, plot_bgcolor='white',
                  showlegend=False)
fig""")

# ---------------------------------------------------------------- RQ6
md("""## RQ6 — Гибридный вес: пользы не доказано

Прокси «правдоподобности» (phishing-смежность рёбер) **confounded**: сильно зависит от размера
дерева (vol даёт бóльшие деревья → ниже phadj) и течёт через risk-метки. Не качество.""")
code("""fig = make_subplots(rows=1, cols=2, subplot_titles=(
        'phishing-смежность рёбер по режимам', 'phadj анти-коррелирует с размером дерева'))
modes=['hybrid','vol','risk']; cols=['#9467bd','#1f77b4','#d62728']
for m,c in zip(modes,cols):
    fig.add_trace(go.Box(y=r56['phadj_'+m], name=m, marker_color=c, boxmean=True), row=1, col=1)
fig.add_trace(go.Scatter(x=r56['nsteiner_vol'], y=r56['phadj_vol'], mode='markers',
              marker=dict(color='#1f77b4', size=6, opacity=0.5), name='vol-режим',
              showlegend=False), row=1, col=2)
corr = r56['phadj_vol'].corr(r56['nsteiner_vol'])
fig.update_yaxes(title_text='phishing-смежность рёбер', row=1, col=1)
fig.update_xaxes(title_text='число Steiner-узлов (vol)', row=1, col=2)
fig.update_yaxes(title_text='phadj (vol)', row=1, col=2)
fig.update_layout(title='RQ6: phadj — confounded прокси (corr с размером = %.2f)'%corr,
                  width=950, height=450, plot_bgcolor='white', showlegend=False)
fig""")

# ---------------------------------------------------------------- RQ7
md("""## RQ7 — Скрытые phishing-узлы: отрицательный результат

Leave-one-out: прячем phishing-терминал, смотрим, попадёт ли он в дерево (vol-only, leakage-free).
Recall скрытых **не отличается значимо** от degree-matched null — и это устойчиво к кластеризации
перекрывающихся инстансов (L4).""")
code("""rr = r7.copy(); rr['null_rate']=num(rr['null_rate'])
recall = rr['true_hit'].mean(); nullr = rr['null_rate'].mean()
fig = go.Figure()
fig.add_trace(go.Bar(x=['Скрытые phishing<br>(recall)','Degree-matched<br>null'],
                     y=[recall, nullr], marker_color=['#d62728','#b0b0b0'], width=0.5,
                     text=[f'{recall:.3f}',f'{nullr:.3f}'], textposition='outside',
                     textfont=dict(size=14)))
fig.update_layout(title='RQ7: recall по скрытым phishing ≈ случайному<br><sub>разница незначима, %d trials</sub>'%len(rr),
                  yaxis_title='доля попавших в Steiner-дерево',
                  yaxis_range=[0, max(recall,nullr)*1.35],
                  width=640, height=460, plot_bgcolor='white', bargap=0.4,
                  margin=dict(t=80,b=60,l=70,r=40))
fig""")
code("""# forest plot: кластерные CI разницы (true-null) при разных уровнях огрубления
levels = ['naive (trials)','Jaccard≥0.20','общих≥25','любой общий узел']
lo = [-0.0032,-0.0094,-0.0373,-0.1806]; hi=[0.0347,0.0446,0.0345,0.0164]; mid=[(a+b)/2 for a,b in zip(lo,hi)]
fig = go.Figure()
fig.add_trace(go.Scatter(x=mid, y=levels, mode='markers',
    error_x=dict(type='data', symmetric=False,
                 array=[h-m for h,m in zip(hi,mid)], arrayminus=[m-l for m,l in zip(mid,lo)]),
    marker=dict(size=10, color='#d62728')))
fig.add_vline(x=0.0, line_dash='dash', line_color='black')
fig.update_layout(title='RQ7: 95% CI разницы (recall − null) — на ВСЕХ уровнях включает 0',
                  xaxis_title='разница recall − null (>0 = сигнал)', yaxis_title='уровень кластеризации (L4)',
                  width=820, height=380, plot_bgcolor='white')
fig""")

# ---------------------------------------------------------------- RQ8
md("""## RQ8 — Обогащение посредников: исчезает под честным null

Против **случайных узлов** Steiner-посредники кажутся центральными (z≈3.3) — но это тавтология
(связники обязаны быть центральными). Против **configuration-model null** (degree-preserving
rewiring, где null-узлы — тоже связники дерева) обогащение **исчезает** (z≈0.2, CI включает 0).""")
code("""fig = go.Figure()
fig.add_trace(go.Histogram(x=r8['z_btw'], name='vs случайные узлы (слабый null)',
                           marker_color='#ff7f0e', opacity=0.7, nbinsx=40))
fig.add_trace(go.Histogram(x=r8cm['z_cm'], name='vs config-model (честный null)',
                           marker_color='#1f77b4', opacity=0.7, nbinsx=40))
fig.add_vline(x=0.0, line_dash='dash', line_color='black', annotation_text='нет обогащения (z=0)')
fig.update_layout(title='RQ8: z-score betweenness посредников — честный null обнуляет «обогащение»',
                  xaxis_title='z-score betweenness (>0 = посредники центральнее null)',
                  yaxis_title='число инстансов', barmode='overlay', width=900, height=470,
                  plot_bgcolor='white', legend=dict(orientation='h', y=-0.2))
fig""")
code("""comp = pd.DataFrame({'null':['Случайные узлы\\n(слабый)','Config-model\\n(честный)'],
                     'mean z':[r8['z_btw'].mean(), r8cm['z_cm'].mean()],
                     'доля z>0, %':[(r8['z_btw']>0).mean()*100,(r8cm['z_cm']>0).mean()*100]})
fig = make_subplots(rows=1, cols=2, subplot_titles=('средний z-score','доля инстансов z>0, %'))
fig.add_bar(x=comp['null'], y=comp['mean z'], marker_color=['#ff7f0e','#1f77b4'], row=1, col=1)
fig.add_bar(x=comp['null'], y=comp['доля z>0, %'], marker_color=['#ff7f0e','#1f77b4'], row=1, col=2)
fig.add_hline(y=0, line_dash='dash', line_color='black', row=1, col=1)
fig.add_hline(y=50, line_dash='dash', line_color='black', row=1, col=2)
fig.update_layout(title='RQ8: слабый null показывает «обогащение», честный — нет (≈шанс)',
                  width=900, height=420, plot_bgcolor='white', showlegend=False)
fig""")

# ---------------------------------------------------------------- L2
md("""## L2 — Directed-robustness: неориентированность как ограничение

Лишь ~44% рёбер undirected-дерева согласованы с реальным направлением транзакций, и single-source
directed-арборесценция выполнима только у 20% инстансов — directed flow-интерпретацию делать нельзя.""")
code("""fig = make_subplots(rows=1, cols=2, column_widths=[0.55,0.45],
        specs=[[{'type':'xy'},{'type':'domain'}]], subplot_titles=(
        'Directional consistency undirected-дерева', 'Выполнима ли directed-арборесценция?'))
fig.add_trace(go.Histogram(x=rdir['consistency'], nbinsx=20, marker_color='#1f77b4',
                           name='consistency'), row=1, col=1)
fig.add_vline(x=rdir['consistency'].mean(), line_dash='dash', line_color='red',
              annotation_text='mean %.2f'%rdir['consistency'].mean(), row=1, col=1)
feas = rdir['dir_feasible'].mean()*100
fig.add_trace(go.Pie(labels=['выполнима','НЕ выполнима'], values=[feas,100-feas],
                     marker_colors=['#2ca02c','#d62728'], hole=0.4, sort=False), row=1, col=2)
fig.update_xaxes(title_text='доля рёбер, согласованных с направлением потока', row=1, col=1)
fig.update_yaxes(title_text='число инстансов', row=1, col=1)
fig.update_layout(title='L2: undirected-дерево слабо реализуемо как направленный поток (%d инстансов)'%len(rdir),
                  width=950, height=440, plot_bgcolor='white', showlegend=True)
fig""")

# ============================================================ ВЕСА / ДЕРЕВО / GNN-ВЕРОЯТНОСТИ
md("""## Дополнительно: веса рёбер, Steiner-дерево и вероятности GNN (инстансы разного размера)

Для малого / среднего / крупного инстанса на ОДНОЙ раскладке узлов. **Сами рёбра (линии)**
раскрашены непрерывной шкалой (colormap viridis: жёлтый = низкое значение, фиолетовый = высокое):

1. **Веса рёбер по компонентам** (3 панели — ровно те веса, что входят в совокупный):
   - **Объём транзакций** — компонента из суммы перенесённого ETH между адресами (крупный объём = дёшево);
   - **Риск** — phishing-смежность концов ребра;
   - **Итоговый гибрид** `cost = 1e-3 + 0.5·(объём + риск)`.
   Чёрной обводкой выделено **минимальное Steiner-дерево** — видно, что дерево идёт по дешёвым
   (жёлтым) рёбрам, и что объём и риск подсвечивают РАЗНЫЕ рёбра (наглядная иллюстрация RQ5).

2. **Дерево vs soft-вероятности PI-GNN** `P(ребро войдёт в дерево)` — рёбра раскрашены по вероятности
   (max по ориентациям/уровням depth-QUBO, спроецировано на исходные рёбра через провенанс редукции).
   GNN даёт диффузные вероятности и сам не выделяет чёткое дерево (отсюда 0% raw-feasibility).""")

code("""import torch
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from networkx.algorithms.approximation import steiner_tree
from experiments import center_terminal, rooted_depth
from rq_common import reduce_with_trace
from steiner_gnn_tuned import _prepare_tuned
from steiner_gnn import _assemble_p, PIGNNSteiner

def inst_full_graph(g):
    G = nx.Graph()
    for e in g['edges']:
        if e['u'] != e['v']:
            G.add_edge(e['u'], e['v'], cost=e['cost'], c_vol=e['c_vol'], c_risk=e['c_risk'])
    return G

def steiner_on(G, T):
    tr = steiner_tree(G, T, weight='cost')
    return set(frozenset(e) for e in tr.edges()), set(tr.nodes())

def gnn_edge_probs(g, epochs=400):
    '''Обучает PI-GNN (penalty-annealing) на редуцированном инстансе, возвращает
       {frozenset(исходное ребро) -> P(в дереве)} через провенанс редукции.'''
    G = inst_full_graph(g); T = [t for t in g['terminals'] if t in G]
    R, TR = reduce_with_trace(G, T)
    if R.number_of_edges() == 0 or len(TR) < 2:
        return {}
    tr = steiner_tree(R, TR, weight='cost'); ie = list(tr.edges())
    if not ie:
        return {}
    r = center_terminal(ie, TR); L = rooted_depth(ie, r)
    m = _prepare_tuned(R, TR, L, r)
    torch.manual_seed(0); model = PIGNNSteiner(m['N'], L, 20, 64)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2); P_full = m['P_full']
    for ep in range(epochs):
        lam = 1.0 * (P_full / 1.0) ** (ep / max(1, epochs - 1))
        opt.zero_grad(); p = _assemble_p(model, m)
        loss = p @ m['M_obj'] @ p + lam * (p @ m['M_pen'] @ p)
        loss.backward(); opt.step()
    p = _assemble_p(model, m).detach(); idx = m['idx']
    redp = {}
    for (a, b) in R.edges():
        vals = []
        for (x, y) in [(a, b), (b, a)]:
            for l in range(1, L + 1):
                k = ('pe', x, y, l)
                if k in idx:
                    vals.append(float(p[idx[k]]))
        redp[frozenset((a, b))] = max(vals) if vals else 0.0
    op = {}
    for (a, b) in R.edges():
        pr = redp[frozenset((a, b))]
        for oe in R[a][b]['oe']:
            op[oe] = max(op.get(oe, 0.0), pr)
    return op

def pick(target, vmax=1600):
    for g in sorted(interesting, key=lambda g: abs(g['metrics']['n_nodes'] - target)):
        G = inst_full_graph(g); T = [t for t in g['terminals'] if t in G]
        R, TR = reduce_with_trace(G, T)
        if R.number_of_edges() == 0 or len(TR) < 2:
            continue
        tr = steiner_tree(R, TR, weight='cost'); ie = list(tr.edges())
        if not ie:
            continue
        r = center_terminal(ie, TR); L = rooted_depth(ie, r)
        if (R.number_of_nodes() - 1) * L + 2 * R.number_of_edges() * L <= vmax:
            return g
    return sorted(interesting, key=lambda g: abs(g['metrics']['n_nodes'] - target))[0]

EXAMPLES = [('МАЛЫЙ', pick(25)), ('СРЕДНИЙ', pick(70)), ('КРУПНЫЙ', pick(140))]
EX = {}
for label, g in EXAMPLES:
    G = inst_full_graph(g); T = [t for t in g['terminals'] if t in G]
    pos = nx.spring_layout(G, seed=3, k=1.7/np.sqrt(G.number_of_nodes()), iterations=90)
    tedges, tnodes = steiner_on(G, T)
    EX[label] = dict(g=g, G=G, T=T, pos=pos, tedges=tedges, snodes=tnodes - set(T),
                     op=gnn_edge_probs(g, epochs=400))
    print(f'{label}: {g[\"metrics\"][\"n_nodes\"]} узлов, GNN-probs на {len(EX[label][\"op\"])} рёбрах')""")

code("""%matplotlib inline
from matplotlib.lines import Line2D
plt.rcParams['figure.dpi'] = 115
CMAP = 'viridis'

def _lw(G):
    e = G.number_of_edges()
    return 2.2 if e < 80 else (1.3 if e < 220 else 0.7)

def _edges_colored(ax, G, pos, vf, lw, vmin=0.0, vmax=1.0, lw_by_val=False):
    '''Раскрашивает САМИ линии рёбер непрерывной шкалой (LineCollection + colorbar).
       lw_by_val=True: толщина растёт со значением (для слабого GNN-сигнала — высокие толще).'''
    edges = list(G.edges())
    segs = [[pos[u], pos[v]] for u, v in edges]
    vals = np.array([min(max(vf(u, v), vmin), vmax) for u, v in edges])
    if lw_by_val and vmax > vmin:
        frac = (vals - vmin) / (vmax - vmin)
        lws = lw * (0.4 + 3.2 * frac)              # высокое значение -> толще
    else:
        lws = lw
    lc = LineCollection(segs, cmap=CMAP, norm=mcolors.Normalize(vmin, vmax), linewidths=lws, zorder=1)
    lc.set_array(vals); ax.add_collection(lc)
    return lc

def _overlay_tree(ax, G, pos, tedges, lw, color='black', alpha=0.6, ls='-'):
    segs = [[pos[u], pos[v]] for u, v in G.edges() if frozenset((u, v)) in tedges]
    ax.add_collection(LineCollection(segs, colors=color, linewidths=lw, alpha=alpha, linestyles=ls, zorder=2))

def _nodes(ax, G, pos, T, snodes=None):
    others = [n for n in G if n not in set(T) and (snodes is None or n not in snodes)]
    ax.scatter([pos[n][0] for n in others], [pos[n][1] for n in others], s=16, c='lightgray',
               edgecolors='gray', linewidths=0.4, zorder=3)
    if snodes:
        ax.scatter([pos[n][0] for n in snodes], [pos[n][1] for n in snodes], s=75, marker='D',
                   c='#ff7f0e', edgecolors='white', linewidths=0.7, zorder=4)
    ax.scatter([pos[t][0] for t in T], [pos[t][1] for t in T], s=120, c='red',
               edgecolors='black', linewidths=0.7, zorder=5)

def _clean(ax):
    ax.set_aspect('equal'); ax.axis('off'); ax.margins(0.05); ax.autoscale()

def draw_weights(label):
    d = EX[label]; G, pos, tedges, T = d['G'], d['pos'], d['tedges'], d['T']
    lw = _lw(G)
    specs = [('Объём транзакций (Σ ETH)', lambda u, v: G[u][v]['c_vol']),
             ('Риск (phishing-смежность)', lambda u, v: G[u][v]['c_risk']),
             ('Итоговый гибрид cost',      lambda u, v: G[u][v]['cost'])]
    fig, axes = plt.subplots(1, 3, figsize=(16, 7.4))
    lc = None
    for ax, (ttl, vf) in zip(axes, specs):
        lc = _edges_colored(ax, G, pos, vf, lw)
        _overlay_tree(ax, G, pos, tedges, lw + 1.6)
        _nodes(ax, G, pos, T)
        ax.set_title(ttl, fontsize=12); _clean(ax)
    cb = fig.colorbar(lc, ax=axes, fraction=0.022, pad=0.015)
    cb.set_label('значение веса ребра  (0 = дёшево / правдоподобно  →  1 = дорого)', fontsize=10, labelpad=10)
    nn = d['g']['metrics']['n_nodes']
    fig.suptitle(f'{label} инстанс ({nn} узлов): цвет ребра = вес  ·  чёрная обводка = minimum Steiner tree  '
                 f'·  красные = phishing-терминалы', fontsize=13, y=0.97)
    plt.show()

draw_weights('МАЛЫЙ')""")
code("draw_weights('СРЕДНИЙ')")
code("draw_weights('КРУПНЫЙ')")

code("""def draw_tree_gnn(label):
    d = EX[label]; G, pos, tedges, snodes, T, op = d['G'], d['pos'], d['tedges'], d['snodes'], d['T'], d['op']
    lw = _lw(G)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6.3))
    segs_all = [[pos[u], pos[v]] for u, v in G.edges()]
    # --- слева: дерево ---
    ax1.add_collection(LineCollection(segs_all, colors='lightgray', linewidths=max(0.5, lw*0.6), zorder=1))
    _overlay_tree(ax1, G, pos, tedges, lw + 1.2, color='#d62728', alpha=1.0)
    _nodes(ax1, G, pos, T, snodes=snodes)
    ax1.set_title('Minimum Steiner tree (hybrid cost)', fontsize=12); _clean(ax1)
    ax1.legend(handles=[
        Line2D([0],[0],color='#d62728',lw=2.5,label='Steiner-дерево'),
        Line2D([0],[0],marker='D',color='w',markerfacecolor='#ff7f0e',markersize=9,label='Steiner-узлы'),
        Line2D([0],[0],marker='o',color='w',markerfacecolor='red',markeredgecolor='black',markersize=11,label='phishing-терминалы'),
        Line2D([0],[0],marker='o',color='w',markerfacecolor='lightgray',markeredgecolor='gray',markersize=8,label='прочие узлы')],
        loc='lower left', fontsize=8, framealpha=0.9)
    # --- справа: вероятности GNN (раскрашены САМИ рёбра; высокие — толще и светлее) ---
    ax2.add_collection(LineCollection(segs_all, colors='whitesmoke', linewidths=max(0.4, lw*0.5), zorder=0))
    probs = [op.get(frozenset((u, v)), 0.0) for u, v in G.edges()]
    pmax = max(probs) if probs and max(probs) > 1e-9 else 1.0      # цвет нормирован к реальному max (probs низкие)
    lc = _edges_colored(ax2, G, pos, lambda u, v: op.get(frozenset((u, v)), 0.0),
                        lw + 0.6, vmax=pmax, lw_by_val=True)         # высокая P -> толще + светлее
    _overlay_tree(ax2, G, pos, tedges, max(1.0, lw*0.9), color='black', alpha=0.85, ls=':')
    ax2.scatter([pos[t][0] for t in T], [pos[t][1] for t in T], s=110, c='red',
                edgecolors='black', linewidths=0.7, zorder=5)
    ax2.set_title('PI-GNN: P(ребро войдёт в дерево)', fontsize=12); _clean(ax2)
    ax2.legend(handles=[Line2D([0],[0],color='black',lw=1.4,ls=':',label='истинное дерево'),
                        Line2D([0],[0],marker='o',color='w',markerfacecolor='red',markeredgecolor='black',markersize=10,label='phishing')],
               loc='lower left', fontsize=8, framealpha=0.9)
    cb = fig.colorbar(lc, ax=ax2, fraction=0.046, pad=0.02)
    cb.set_label(f'P(ребро в дереве) по PI-GNN\\n(шкала 0…{pmax:.2f}; толще = выше P)', fontsize=9, labelpad=8)
    nn = d['g']['metrics']['n_nodes']
    fig.suptitle(f'{label} инстанс ({nn} узлов): minimum Steiner tree (слева) vs вероятности PI-GNN (справа); '
                 f'пунктир — истинное дерево', fontsize=12)
    plt.show()

draw_tree_gnn('МАЛЫЙ')""")
code("draw_tree_gnn('СРЕДНИЙ')")
code("draw_tree_gnn('КРУПНЫЙ')")

# ---------------------------------------------------------------- сводка (markdown-таблица)
md("""## Итоговая сводка вердиктов по RQ

QUBO корректна, но метод-солвер (QUBO+PI-GNN) и доменные гипотезы — преимущественно отрицательны.

| RQ | Исследовательский вопрос | Ключевая цифра | Вердикт |
|----|--------------------------|----------------|--------|
| **RQ1** | Корректна ли наша QUBO-формулировка minimum Steiner tree? | глобальный минимум QUBO = точный перебор = ILP на 5/5 графах; закодированный оптимум penalty-feasible с точностью 1e-14 | Подтверждено |
| **RQ2** | Находит ли PI-GNN качественные решения Steiner-дерева? | даже после честного тюнинга median ratio 1.09, оптимум 44%, но строго лучше KMB в 0% и raw-feasibility лишь 4% | Отрицательный: не бьёт тривиальный KMB |
| **RQ3** | Нужен ли GNN, или достаточно simulated annealing на той же QUBO? | тюнингованный GNN превосходит SA (1.09 против 4.30), но и тот и другой проигрывают KMB и точному ILP | Отрицательный: весь QUBO-маршрут уступает классике |
| **RQ4** | Масштабируется ли подход туда, где классика тяжелее? | качество SA падает с размером (ratio 1.4 → 18), ILP решает за <4 c, KMB остаётся около оптимума | Отрицательный: QUBO-солверы не масштабируются |
| **RQ5** | Принципиально ли влияет определение веса ребра на найденный путь? | Jaccard рёбер 0.63 на точных решениях — путь меняется у 75% инстансов | Подтверждено: вес определяет путь |
| **RQ6** | Полезен ли гибридный (многопараметрический) вес ребра? | прокси «правдоподобности» confounded: корреляция с размером дерева −0.70 плюс утечка risk-метки | Отрицательный: пользы гибрида не доказано |
| **RQ7** | Проходит ли Steiner-дерево через скрытые (изъятые) phishing-узлы? | recall скрытых 0.09 против null 0.07, разница незначима и устойчива к кластеризации регионов | Отрицательный: восстановления нет |
| **RQ8** | Обогащены ли найденные посредники относительно честной null-модели? | под configuration-model null z-обогащение betweenness падает до 0.2, доверительный интервал включает 0 | Отрицательный: обогащение исчезает |
| **L2** | Устойчивы ли выводы к неориентированности графа? | согласованность направлений лишь 0.44, направленная арборесценция выполнима у 20% инстансов | Ограничение: directed-интерпретацию делать нельзя |
""")

# ================================================================ сборка и исполнение
nb = new_notebook(cells=C)
nb.metadata['kernelspec'] = dict(name='python3', display_name='Python 3', language='python')
print(f'ячеек: {len(C)}; исполняю...')
client = NotebookClient(nb, timeout=600, kernel_name='python3',
                        resources={'metadata': {'path': '.'}})
client.execute()
nbf.write(nb, 'RESULTS_VISUALIZATION.ipynb')
print('записан RESULTS_VISUALIZATION.ipynb')
