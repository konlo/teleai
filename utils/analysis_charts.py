"""Data-backed previews; no network, model calls or remote profiling."""
from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from uuid import uuid4

import pandas as pd
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg

from utils.analysis_datasets import DatasetStore


@dataclass(frozen=True)
class ChartPreview:
    id: str
    dataset_id: str
    title: str
    reason: str
    kind: str
    columns: tuple[str, ...]
    scope: str
    image: bytes


def recommend_charts(store: DatasetStore, dataset_id: str,
                     columns: list[str] | None = None) -> list[ChartPreview]:
    info = store.metadata[dataset_id]
    source = store.frames[dataset_id]
    selected = columns or list(source.columns)
    if not set(selected).issubset(source.columns):
        raise ValueError("추천 대상 컬럼이 현재 데이터에 없습니다.")
    if source.empty:
        return []
    # Bounded local work. The same deterministic sample is used by the preview
    # and its final displayed image; it is never described as a full-population result.
    frame = source[selected]
    sampled = len(frame) > 20_000
    if sampled:
        frame = frame.sample(n=20_000, random_state=42)
    scope = f"보유 {len(source):,}행 중 {len(frame):,}행 기준 · {info.coverage}"
    numeric = [c for c in selected if pd.api.types.is_numeric_dtype(frame[c])
               and not pd.api.types.is_bool_dtype(frame[c])
               and frame[c].nunique() > 1]
    # Explicitly selected identifiers may still be analyzed. Avoid auto-selecting
    # conventional ID names as measurements; this is a hint, not domain semantics.
    if not columns:
        numeric = [c for c in numeric if not (c.lower() == "id" or c.lower().endswith("_id"))]
    times = [c for c in selected if pd.api.types.is_datetime64_any_dtype(frame[c])]
    categories = [c for c in selected if not pd.api.types.is_numeric_dtype(frame[c])
                  and c not in times and 1 <= frame[c].nunique() <= 50]
    previews = []

    def save(fig, kind, cols, title, reason, extra_scope=""):
        buffer = BytesIO()
        FigureCanvasAgg(fig)
        fig.tight_layout()
        fig.savefig(buffer, format="png", dpi=110)
        previews.append(ChartPreview(str(uuid4()), dataset_id, title, reason, kind,
                                     tuple(cols), scope + extra_scope, buffer.getvalue()))

    if numeric and info.grain == "raw" and not info.aggregation:
        column = numeric[0]
        values = pd.to_numeric(frame[column], errors="coerce").replace([float("inf"), -float("inf")], float("nan")).dropna()
        if len(values) >= 2:
            fig = Figure(figsize=(6, 3.5))
            ax = fig.subplots()
            ax.hist(values, bins=min(60, max(8, int(len(values) ** 0.5))), color="#3278b9", edgecolor="white")
            ax.set(xlabel=column, ylabel="Count")
            save(fig, "histogram", [column], f"{column} 분포",
                 f"유효값 {len(values):,}개에서 중앙값은 {values.median():.4g}입니다.")

    if times and numeric and len(previews) < 3:
        time_col, value_col = times[0], numeric[0]
        values = frame[[time_col, value_col]].dropna().sort_values(time_col)
        # Don't invent an aggregation or connect multiple observations at one time.
        if len(values) >= 2 and values[time_col].is_unique:
            fig = Figure(figsize=(6, 3.5))
            ax = fig.subplots()
            ax.plot(values[time_col], values[value_col], color="#3278b9")
            ax.set(xlabel=time_col, ylabel=value_col)
            fig.autofmt_xdate()
            save(fig, "line", [time_col, value_col], f"{value_col} 시간 변화",
                 "시간순 관측값입니다. 관측 사이의 변화는 추가 확인이 필요합니다.")

    if categories and len(previews) < 3:
        category = categories[0]
        if info.grain == "raw" and not info.aggregation:
            counts = frame[category].value_counts().head(10).sort_values()
            fig = Figure(figsize=(6, 3.5))
            ax = fig.subplots()
            ax.barh(counts.index.astype(str), counts.values, color="#369a86")
            ax.set(xlabel="Count", ylabel=category)
            save(fig, "bar", [category], f"{category}별 건수",
                 "빈도가 높은 최대 10개 범주입니다. 나머지 범주는 표시하지 않습니다.")
        elif numeric and frame[category].is_unique:
            value = numeric[0]
            pairs = frame[[category, value]].dropna().sort_values(value).tail(10)
            fig = Figure(figsize=(6, 3.5))
            ax = fig.subplots()
            ax.barh(pairs[category].astype(str), pairs[value], color="#369a86")
            ax.set(xlabel=value, ylabel=category)
            save(fig, "bar", [category, value], f"{category}별 {value}",
                 f"기존 집계값을 재집계 없이 표시합니다. 집계 정의: {info.aggregation or info.grain}")

    if len(numeric) >= 2 and info.grain == "raw" and not info.aggregation and len(previews) < 3:
        x, y = numeric[:2]
        pairs = frame[[x, y]].replace([float("inf"), -float("inf")], float("nan")).dropna()
        if len(pairs) >= 3:
            render = pairs.sample(n=5000, random_state=42) if len(pairs) > 5000 else pairs
            fig = Figure(figsize=(6, 3.5))
            ax = fig.subplots()
            ax.scatter(render[x], render[y], s=8, alpha=0.35, color="#7450aa")
            ax.set(xlabel=x, ylabel=y)
            save(fig, "scatter", [x, y], f"{x}와 {y}의 관계",
                 "두 값의 관계를 확인합니다. 상관만으로 원인을 판단할 수 없습니다.",
                 f" · 표시점 {len(render):,}개")

    if numeric and info.grain == "raw" and not info.aggregation and len(previews) < 3:
        column = numeric[0]
        values = frame[column].replace([float("inf"), -float("inf")], float("nan")).dropna()
        if len(values) >= 5:
            fig = Figure(figsize=(6, 3.5))
            ax = fig.subplots()
            ax.boxplot(values, vert=False)
            ax.set(xlabel=column)
            save(fig, "boxplot", [column], f"{column} 중앙값과 퍼짐",
                 "중앙값과 사분위 범위를 표시합니다. 바깥 점이 반드시 오류인 것은 아닙니다.")
    return previews


def histogram_from_counts(store, dataset_id, value_column, weight_column):
    """Render complete value-frequency results without expanding source rows."""
    import numpy as np
    info = store.metadata[dataset_id]
    if info.coverage != 'complete':
        raise ValueError('전체 빈도 결과가 필요합니다. 잘린 결과로 전체 분포를 그릴 수 없습니다.')
    frame = store.frames[dataset_id]
    if value_column == weight_column or not {value_column, weight_column}.issubset(frame.columns):
        raise ValueError('값과 빈도 컬럼을 확인해주세요.')
    values = pd.to_numeric(frame[value_column], errors='raise')
    counts = pd.to_numeric(frame[weight_column], errors='raise')
    if not len(values) or not np.isfinite(values).all() or not np.isfinite(counts).all():
        raise ValueError('유효한 수치와 빈도가 필요합니다.')
    if (counts < 0).any() or (counts % 1 != 0).any() or counts.sum() <= 0:
        raise ValueError('빈도는 0 이상의 정수이며 총합은 양수여야 합니다.')
    fig = Figure(figsize=(6, 3.5)); ax = fig.subplots()
    ax.hist(values, weights=counts, bins=min(60, max(8, int(len(values)**0.5))),
            color='#3278b9', edgecolor='white')
    ax.set(xlabel=value_column, ylabel='Count')
    buffer = BytesIO(); FigureCanvasAgg(fig); fig.tight_layout(); fig.savefig(buffer,format='png',dpi=110)
    return ChartPreview(str(uuid4()),dataset_id,f'{value_column} 분포',
        '값별 빈도를 가중치로 사용했습니다. 원본 행을 펼치거나 표본을 만들지 않았습니다.',
        'histogram',(value_column,),f'빈도 합계 {int(counts.sum()):,} · complete · {info.query}',buffer.getvalue())
