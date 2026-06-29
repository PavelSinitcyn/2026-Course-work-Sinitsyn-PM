# Minimum Steiner Tree на графе транзакций Ethereum (QUBO + PI-GNN)

Объяснимая реконструкция путей отмывания в сети Ethereum через **минимальное дерево Штейнера**,
сформулированное как **QUBO** и решаемое физически вдохновлённой графовой нейронной сетью
(**PI-GNN**). Терминалы дерева - известные фишинговые адреса (набор XBlock «Ethereum Phishing
Transaction Network»).

## Главный результат
QUBO-формулировка **корректна**, но связка QUBO + PI-GNN / имитация отжига **уступает тривиальной
эвристике KMB** и не масштабируется. Из доменных гипотез устойчиво подтвердилась лишь
чувствительность найденного пути к весу ребра; восстановление скрытых фишинговых узлов и
обогащение посредников при честных нуль-моделях **не подтвердились**, а неориентированность графа —
реальное ограничение для потоковых выводов.

## Структура
- **Данные:** `eth_instances.pkl` (200 плотных инстансов), `eth_instances_summary.csv`.
- **Ядро:** `steiner_qubo.py`, `steiner_gnn.py`, `steiner_gnn_tuned.py`, `rq_common.py`, `experiments.py`.
- **Сборка инстансов:** `build_instances_eth.py`, `rebuild_eth_weights.py`, `inspect_muldigraph.py`, `probe_eth_instances.py`, `count_alt_paths.py`.
- **Эксперименты:** `evaluate.py` + `summarize.py`, `evaluate_gnn_tuned.py` + `summarize_gnn_tuned.py`, `rq5_6.py`, `rq7.py`, `rq8.py`, `rq8_configmodel.py`, `clustered_ci.py`, `directed_robustness.py`.
- **Проверки:** `verify_foundations.py`, `verify_gnn.py`, `audit_kmb.py`.
- **Результаты:** `results_eth.csv`, `results_gnn_tuned.csv`, `rq*_results.csv`.
- **Визуализация:** `build_results_notebook.py`, `export_pictures.py`, `RESULTS_VISUALIZATION.ipynb`, `pictures/`.

## Запуск
```bash
python3 evaluate.py        # массовый прогон ILP/SA/PI-GNN -> results_eth.csv
python3 summarize.py       # агрегаты RQ2-RQ4
```
Исходный граф `MulDiGraph.pkl` (XBlock, ~1.2 ГБ) в репозиторий не включён; он нужен только для
пере-сборки `eth_instances.pkl`.

**Зависимости:** torch 2.2, dgl 1.1.3, numpy 1.26, networkx, pulp (+CBC), dimod + dwave-neal,
pandas, plotly, kaleido, matplotlib.
