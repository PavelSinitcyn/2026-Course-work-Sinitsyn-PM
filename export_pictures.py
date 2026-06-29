"""Экспорт всех графиков из ноутбука визуализации в папку pictures/ (Plotly через kaleido, matplotlib через savefig)."""
import os, re, ast, warnings
warnings.filterwarnings('ignore')
import nbformat
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import plotly.graph_objects as go

os.makedirs('pictures', exist_ok=True)
for f in os.listdir('pictures'):           # чистим старое
    if f.endswith('.png'):
        os.remove(os.path.join('pictures', f))

nb = nbformat.read('RESULTS_VISUALIZATION.ipynb', as_version=4)
ns = {}
counter = [0]


def slug(s, n=55):
    s = re.sub(r'<[^>]+>', '', s or '')
    s = re.sub(r'[^0-9A-Za-zА-Яа-яёЁ ._-]+', '', s).strip().replace(' ', '_')
    s = re.sub(r'_+', '_', s)
    return (s[:n].strip('_') or 'figure')


def patched_show(*a, **k):
    counter[0] += 1
    fig = plt.gcf()
    title = fig._suptitle.get_text() if fig._suptitle else ''
    fn = f"pictures/{counter[0]:02d}_{slug(title)}.png"
    fig.savefig(fn, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print('  [mpl]   ', fn, flush=True)


plt.show = patched_show


def exec_cell(src):
    lines = [ln for ln in src.splitlines() if not ln.lstrip().startswith('%')]
    tree = ast.parse('\n'.join(lines))
    if tree.body and isinstance(tree.body[-1], ast.Expr):
        e = tree.body.pop()
        assign = ast.Assign(targets=[ast.Name(id='__last__', ctx=ast.Store())], value=e.value)
        ast.copy_location(assign, e); tree.body.append(assign)
    ast.fix_missing_locations(tree)
    ns['__last__'] = None
    exec(compile(tree, '<cell>', 'exec'), ns)
    return ns.get('__last__')


for i, cell in enumerate(nb.cells):
    if cell.cell_type != 'code':
        continue
    res = exec_cell(cell.source)
    if isinstance(res, go.Figure):
        counter[0] += 1
        title = ''
        try:
            title = res.layout.title.text or ''
        except Exception:
            pass
        fn = f"pictures/{counter[0]:02d}_{slug(title)}.png"
        res.write_image(fn, scale=2)
        print('  [plotly]', fn, flush=True)

print(f'\nГОТОВО: сохранено {counter[0]} картинок в pictures/')
