"""Exact categorical counts of retained raw rows, with honest population scope."""
from io import BytesIO
from uuid import uuid4

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from utils.analysis_charts import ChartPreview, _apply_unicode_font
from utils.analysis_datasets import project_dataset
from utils.analysis_frequency_presentation import draw_categories


def categorical_counts(store, dataset_id, column, *, legend=False,
                       stacked=False, palette='default'):
    info = store.metadata[dataset_id]
    if info.grain != 'raw' or info.aggregation:
        raise ValueError('Retained-row counts require raw rows, not aggregate values')
    values = project_dataset(store, dataset_id, [column])[column]
    counts = values.dropna().value_counts(sort=True)
    if counts.empty:
        raise ValueError('No non-null categorical values in retained rows')
    shown = counts.iloc[:50]
    labels = [str(value)[:120] for value in shown.index]
    heights = [int(value) for value in shown]
    fig = Figure(figsize=(6, 3.5)); ax = fig.subplots()
    presentation = draw_categories(ax, labels, heights, legend=legend,
                                   stacked=stacked, palette=palette)
    ax.set(xlabel=column, ylabel='Count')
    FigureCanvasAgg(fig); _apply_unicode_font(fig); fig.tight_layout()
    buffer = BytesIO(); fig.savefig(buffer, format='png', dpi=110)
    spec = dict(aggregation='count', population='retained_rows', labels=labels,
                counts=heights, total_count=int(counts.sum()),
                has_more_categories=len(counts)>50, null_policy='exclude',
                excluded_nulls=int(values.isna().sum()), **presentation)
    return ChartPreview(str(uuid4()), dataset_id, f'{column} 분포',
        '선택한 보유 행의 범주별 빈도입니다. 결측은 제외했습니다.', 'bar',
        (column,), f'보유 {info.rows:,}행 기준 · {info.coverage}',
        buffer.getvalue(), spec)
