"""Data-backed previews; no network, model calls or remote profiling."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
from uuid import uuid4

import pandas as pd
from matplotlib import font_manager
from matplotlib.ft2font import FT2Font
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.text import Text

from utils.analysis_datasets import DatasetStore, project_dataset


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


@lru_cache(maxsize=1)
def _unicode_font():
    """Return an installed font that contains Hangul, without global rc changes."""
    preferred = [
        "/System/Library/Fonts/AppleSDGothicNeo.ttc",
        "/System/Library/Fonts/Supplemental/AppleGothic.ttf",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJKkr-Regular.otf",
        "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
        "C:/Windows/Fonts/malgun.ttf",
    ]
    candidates = [path for path in preferred if Path(path).is_file()]
    candidates.extend(path for path in font_manager.findSystemFonts()
                      if path not in candidates)
    for path in candidates:
        try:
            if ord("한") in FT2Font(path).get_charmap():
                return font_manager.FontProperties(fname=path)
        except (OSError, RuntimeError, ValueError):
            continue
    return None


def _apply_unicode_font(fig: Figure):
    """Apply a Hangul-capable font to chart text when the host provides one."""
    font = _unicode_font()
    if font is not None:
        for item in fig.findobj(match=Text):
            item.set_fontproperties(font)


def recommend_charts(store: DatasetStore, dataset_id: str,
                     columns: list[str] | None = None) -> list[ChartPreview]:
    info = store.metadata[dataset_id]
    selected = columns or list(info.columns)
    if not set(selected).issubset(info.columns):
        raise ValueError("추천 대상 컬럼이 현재 데이터에 없습니다.")
    source = project_dataset(store, dataset_id, selected)
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
        _apply_unicode_font(fig)
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
            ax.boxplot(values, orientation='horizontal')
            ax.set(xlabel=column)
            save(fig, "boxplot", [column], f"{column} 중앙값과 퍼짐",
                 "중앙값과 사분위 범위를 표시합니다. 바깥 점이 반드시 오류인 것은 아닙니다.")
    return previews


def validate_frequency_dataset(store, dataset_id, value_column, weight_column):
    """Validate executed COUNT lineage before trusting a column as frequency."""
    import sqlglot
    from utils.analysis_provenance import count_frequency_columns
    info = store.metadata[dataset_id]
    if info.coverage != 'complete':
        raise ValueError('전체 빈도 결과가 필요합니다. 잘린 결과로 전체 분포를 그릴 수 없습니다.')
    try:
        tree = sqlglot.parse_one(info.query, read='duckdb' if info.parent_id else 'databricks')
    except (sqlglot.errors.ParseError, TypeError):
        tree = None
    if (info.grain != 'aggregate' or not info.aggregation or tree is None
            or count_frequency_columns(tree) != (value_column, weight_column)):
        raise ValueError('값별 COUNT(*) 집계의 실행 출처가 필요합니다. 임의의 수치 컬럼을 빈도로 사용할 수 없습니다.')
    if info.parent_id:
        parent = store.metadata.get(info.parent_id)
        if (parent is None or parent.grain != 'raw' or parent.aggregation
                or not parent.predicate_known or parent.coverage != 'complete'
                or parent.source != info.source):
            raise ValueError('완전한 원본 행에서 계산한 빈도만 사용할 수 있습니다.')
    return info


def histogram_from_counts(store, dataset_id, value_column, weight_column):
    """Render complete value-frequency results without expanding source rows."""
    import numpy as np
    info = validate_frequency_dataset(store, dataset_id, value_column, weight_column)
    if value_column == weight_column or not {value_column, weight_column}.issubset(info.columns):
        raise ValueError('값과 빈도 컬럼을 확인해주세요.')
    frame = project_dataset(store, dataset_id, [value_column, weight_column])
    values = pd.to_numeric(frame[value_column], errors='raise')
    counts = pd.to_numeric(frame[weight_column], errors='raise')
    if not len(values) or not np.isfinite(values).all() or not np.isfinite(counts).all():
        raise ValueError('유효한 수치와 빈도가 필요합니다.')
    if not values.is_unique:
        raise ValueError('값별 빈도에는 같은 값이 중복될 수 없습니다.')
    if (counts < 0).any() or (counts % 1 != 0).any() or counts.sum() <= 0:
        raise ValueError('빈도는 0 이상의 정수이며 총합은 양수여야 합니다.')
    fig = Figure(figsize=(6, 3.5)); ax = fig.subplots()
    ax.hist(values, weights=counts, bins=min(60, max(8, int(len(values)**0.5))),
            color='#3278b9', edgecolor='white')
    ax.set(xlabel=value_column, ylabel='Count')
    buffer = BytesIO(); FigureCanvasAgg(fig); _apply_unicode_font(fig); fig.tight_layout(); fig.savefig(buffer,format='png',dpi=110)
    return ChartPreview(str(uuid4()),dataset_id,f'{value_column} 분포',
        '값별 빈도를 가중치로 사용했습니다. 원본 행을 펼치거나 표본을 만들지 않았습니다.',
        'histogram',(value_column,),f'빈도 합계 {int(counts.sum()):,} · complete · {info.query}',buffer.getvalue())


def render_chart_spec(store: DatasetStore, dataset_id: str, *, kind: str, x: str,
                      y: str = "", category: str = "", aggregation: str = "none",
                      sort: str = "none", top_n: int = 50, bins: int = 20,
                      title: str = "", x_label: str = "", y_label: str = "",
                      orientation: str = "vertical", cumulative: bool = False):
    """Render one bounded, declarative chart from a loaded dataset.

    No expression strings, Python code, file paths, URLs, or arbitrary style
    dictionaries are accepted. The returned digest binds the chart to the exact
    values handed to Matplotlib without returning those values to the model.
    """
    kinds = {"histogram", "bar", "line", "scatter", "boxplot"}
    aggregations = {"none", "count", "sum", "mean", "median", "min", "max"}
    if kind not in kinds or aggregation not in aggregations:
        raise ValueError("지원하는 차트 종류와 집계 방식을 사용해주세요.")
    if sort not in {"none", "ascending", "descending", "calendar_month"}:
        raise ValueError("sort는 none, ascending, descending, calendar_month 중 하나여야 합니다.")
    if orientation not in {"vertical", "horizontal"}:
        raise ValueError("orientation은 vertical 또는 horizontal이어야 합니다.")
    if not 1 <= int(top_n) <= 50 or not 2 <= int(bins) <= 100:
        raise ValueError("top_n은 1~50, bins는 2~100이어야 합니다.")
    labels = {"title": title, "x_label": x_label, "y_label": y_label}
    if any(not isinstance(value, str) or len(value) > 120 for value in labels.values()):
        raise ValueError("차트 제목과 축 라벨은 120자 이하 문자열이어야 합니다.")
    info = store.metadata[dataset_id]
    requested = [column for column in (x, y, category) if column]
    if not x or len(requested) != len(set(requested)) or any(column not in info.columns for column in requested):
        raise ValueError("차트 축은 로딩된 dataset의 중복 없는 실제 컬럼이어야 합니다.")
    source = project_dataset(store, dataset_id, requested)
    if source.empty:
        raise ValueError("빈 데이터로 차트를 만들 수 없습니다.")
    if kind == "boxplot" and (info.grain != "raw" or info.aggregation):
        raise ValueError("박스플롯은 원본 행의 분포에서 만듭니다. 집계 결과의 값만으로는 원본 분포를 복원할 수 없습니다.")
    if aggregation != "none" and (info.grain != "raw" or info.aggregation):
        raise ValueError("이미 집계된 결과를 다시 집계하지 않습니다.")
    if kind != "bar" and orientation != "vertical":
        raise ValueError("orientation은 막대 차트에서만 사용할 수 있습니다.")
    if sort == "calendar_month" and kind != "line":
        raise ValueError("calendar_month 정렬은 선 차트에서만 지원합니다.")
    if cumulative and kind != "line":
        raise ValueError("누적 변환은 선 차트에서만 지원합니다.")
    if kind not in {"scatter", "boxplot", "line"} and category:
        raise ValueError("category는 산점도, 그룹 박스플롯 또는 다중 선 차트에만 사용할 수 있습니다.")

    def numeric(series, name):
        if not pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
            raise ValueError(f"{name} 컬럼은 수치형이어야 합니다.")
        return pd.to_numeric(series, errors="coerce").replace(
            [float("inf"), -float("inf")], float("nan"))

    def digest(frame):
        encoded_columns = json.dumps(list(frame.columns), ensure_ascii=False).encode()
        hashed = pd.util.hash_pandas_object(frame, index=False).values.tobytes()
        return sha256(encoded_columns + hashed).hexdigest()

    fig = Figure(figsize=(6, 3.5))
    ax = fig.subplots()
    plotted = None
    effective_aggregation = aggregation
    reason = ""
    default_title = ""
    sampled = False
    point_value_column = "value"

    if kind == "histogram":
        if y or category or aggregation != "none":
            raise ValueError("히스토그램은 수치 x와 bins만 사용합니다.")
        values = numeric(source[x], x).dropna()
        if len(values) < 2:
            raise ValueError("히스토그램에는 유효한 수치가 2개 이상 필요합니다.")
        plotted = values.to_frame(name=x)
        ax.hist(values, bins=int(bins), color="#3278b9", edgecolor="white")
        ax.set(xlabel=x_label or x, ylabel=y_label or "Count")
        default_title = f"{x} 분포"
        reason = f"유효값 {len(values):,}개를 {int(bins)}개 bin으로 표시했습니다."
    elif kind == "scatter":
        if not y or aggregation != "none":
            raise ValueError("산점도는 수치 x와 y를 집계 없이 사용합니다.")
        frame = pd.DataFrame({x: numeric(source[x], x), y: numeric(source[y], y)})
        if category:
            frame[category] = source[category]
        frame = frame.dropna()
        if len(frame) < 3:
            raise ValueError("산점도에는 완전한 좌표가 3개 이상 필요합니다.")
        if category:
            groups = list(frame[category].astype(str).unique())
            if len(groups) > 20:
                raise ValueError("산점도 category는 고유값이 20개 이하여야 합니다.")
        sampled = len(frame) > 5_000
        plotted = frame.sample(n=5_000, random_state=42) if sampled else frame
        if category:
            for value, group in plotted.groupby(category, sort=False, observed=True):
                ax.scatter(group[x], group[y], s=10, alpha=0.4, label=str(value))
            ax.legend(fontsize=8)
        else:
            ax.scatter(plotted[x], plotted[y], s=10, alpha=0.4, color="#7450aa")
        ax.set(xlabel=x_label or x, ylabel=y_label or y)
        default_title = f"{x}와 {y}의 관계"
        reason = f"완전한 좌표 {len(frame):,}개 중 {len(plotted):,}개를 표시했습니다. 상관만으로 인과를 판단할 수 없습니다."
    elif kind == "boxplot":
        if y or aggregation != "none":
            raise ValueError("박스플롯은 수치 x와 선택적 category를 집계 없이 사용합니다.")
        if category:
            frame = pd.DataFrame({category: source[category], x: numeric(source[x], x)}).dropna()
            groups = sorted(frame[category].unique(), key=lambda value: str(value))
            if not 2 <= len(groups) <= 20:
                raise ValueError("그룹 박스플롯의 category 고유값은 2~20개여야 합니다.")
            arrays = [frame.loc[frame[category] == value, x].to_numpy() for value in groups]
            if any(len(values) < 2 for values in arrays):
                raise ValueError("그룹 박스플롯은 각 category에 유효값이 2개 이상 필요합니다.")
            plotted = pd.concat([
                pd.DataFrame({category: [value] * len(values), x: values})
                for value, values in zip(groups, arrays)
            ], ignore_index=True)
            ax.boxplot(arrays, tick_labels=[str(value) for value in groups])
            ax.set(xlabel=x_label or category, ylabel=y_label or x)
            ax.tick_params(axis="x", rotation=30)
            default_title = f"{category}별 {x} 분포"
            reason = f"{len(groups)}개 그룹의 중앙값과 사분위 범위를 비교합니다. 바깥 점이 반드시 오류인 것은 아닙니다."
        else:
            values = numeric(source[x], x).dropna()
            if len(values) < 2:
                raise ValueError("박스플롯에는 유효한 수치가 2개 이상 필요합니다.")
            plotted = values.to_frame(name=x)
            ax.boxplot(values, orientation="horizontal")
            ax.set(xlabel=x_label or x, ylabel=y_label)
            default_title = f"{x} 중앙값과 퍼짐"
            reason = "중앙값과 사분위 범위를 표시합니다. 바깥 점이 반드시 오류인 것은 아닙니다."
            if len(values) < 5:
                reason += f" 유효값 {len(values)}개라 사분위수 추정이 불안정할 수 있습니다."
    elif kind == "bar":
        if category:
            raise ValueError("막대 차트의 범주는 x로 지정하세요.")
        if not y:
            if aggregation not in {"none", "count"}:
                raise ValueError("y가 없는 막대 차트는 count 집계만 지원합니다.")
            effective_aggregation = "count"
            counts = source[x].dropna().value_counts(sort=False)
            plotted = counts.rename("value").rename_axis(x).reset_index()
            value_label = "Count"
        else:
            values = numeric(source[y], y)
            frame = pd.DataFrame({x: source[x], y: values}).dropna()
            if aggregation == "count":
                raise ValueError("count 막대 차트에는 y를 지정하지 마세요.")
            if aggregation == "none":
                if frame[x].duplicated().any():
                    raise ValueError("집계 없는 막대 차트의 x는 고유해야 합니다.")
                plotted = frame.rename(columns={y: "value"})
            else:
                plotted = (frame.groupby(x, sort=False, observed=True)[y]
                           .agg(aggregation).rename("value").reset_index())
            value_label = y if aggregation == "none" else f"{aggregation}({y})"
        if plotted.empty:
            raise ValueError("막대 차트에 표시할 값이 없습니다.")
        if len(plotted) > int(top_n):
            plotted = plotted.nlargest(int(top_n), "value")
            truncated = True
        else:
            truncated = False
        if sort != "none":
            plotted = plotted.sort_values("value", ascending=sort == "ascending")
        if orientation == "horizontal":
            ax.barh(plotted[x].astype(str), plotted["value"], color="#369a86")
            ax.set(xlabel=y_label or value_label, ylabel=x_label or x)
        else:
            ax.bar(plotted[x].astype(str), plotted["value"], color="#369a86")
            ax.set(xlabel=x_label or x, ylabel=y_label or value_label)
            ax.tick_params(axis="x", rotation=30)
        default_title = f"{x}별 {value_label}"
        reason = f"{effective_aggregation} 집계의 {len(plotted):,}개 범주를 표시했습니다."
        if truncated:
            reason += f" 값 기준 상위 {int(top_n)}개만 포함합니다."
    else:  # line
        if not y:
            if aggregation != "count":
                raise ValueError("y가 없는 선 차트는 count 집계만 지원합니다.")
            point_value_column = "count" if x == "value" else "value"
            if category:
                plotted = (source[[x, category]].dropna()
                           .groupby([x, category], sort=False, observed=True).size()
                           .rename(point_value_column).reset_index())
            else:
                plotted = (source[x].dropna().value_counts(sort=False)
                           .rename(point_value_column).rename_axis(x).reset_index())
            effective_aggregation = "count"
            value_label = "Count"
        else:
            point_value_column = "measure" if x == "value" else "value"
            frame = pd.DataFrame({x: source[x], y: numeric(source[y], y)})
            if category:
                frame[category] = source[category]
            frame = frame.dropna(subset=[x] + ([category] if category else []))
            if aggregation == "count":
                raise ValueError("선 차트의 count 집계에는 y를 지정하지 마세요.")
            if aggregation == "none":
                if frame[y].notna().sum() < 2:
                    raise ValueError("선 차트에는 유효한 y 값이 2개 이상 필요합니다.")
                uniqueness = [x, category] if category else [x]
                if frame.duplicated(uniqueness).any():
                    raise ValueError("집계 없는 선 차트의 시간·category 조합은 고유해야 합니다.")
                plotted = frame.rename(columns={y: point_value_column})
            else:
                frame = frame.dropna(subset=[y])
                group_keys = [x, category] if category else x
                plotted = (frame.groupby(group_keys, sort=False, observed=True)[y]
                           .agg(aggregation).rename(point_value_column).reset_index())
            value_label = y if aggregation == "none" else f"{aggregation}({y})"
        if len(plotted) < 2 or len(plotted) > 5_000:
            raise ValueError("선 차트는 2~5,000개의 점이 필요합니다. 더 큰 데이터는 먼저 집계하세요.")
        if category and not 1 <= plotted[category].nunique(dropna=True) <= 20:
            raise ValueError("다중 선 차트의 category 고유값은 1~20개여야 합니다.")
        if sort == "calendar_month":
            month_number = {
                "jan": 1, "january": 1, "feb": 2, "february": 2,
                "mar": 3, "march": 3, "apr": 4, "april": 4,
                "may": 5, "jun": 6, "june": 6, "jul": 7, "july": 7,
                "aug": 8, "august": 8, "sep": 9, "sept": 9, "september": 9,
                "oct": 10, "october": 10, "nov": 11, "november": 11,
                "dec": 12, "december": 12,
            }
            keys = plotted[x].astype(str).str.strip().str.casefold().map(month_number)
            duplicate_keys = pd.DataFrame({"month": keys})
            if category:
                duplicate_keys[category] = plotted[category].to_numpy()
            if keys.isna().any() or duplicate_keys.duplicated().any():
                raise ValueError("calendar_month 정렬에는 중복되지 않는 영문 월 이름/약어가 필요합니다.")
            sort_columns = ([category] if category else []) + ["_calendar_order"]
            plotted = plotted.assign(_calendar_order=keys).sort_values(
                sort_columns).drop(columns="_calendar_order")
        elif sort != "none":
            sort_columns = ([category] if category else []) + [x]
            plotted = plotted.sort_values(sort_columns, ascending=sort == "ascending")
        if cumulative:
            plotted[point_value_column] = (plotted.groupby(category, sort=False)[point_value_column].cumsum()
                                           if category else plotted[point_value_column].cumsum())
        if category:
            for value, group in plotted.groupby(category, sort=False, observed=True):
                ax.plot(group[x], group[point_value_column], marker="o", label=str(value))
            ax.legend(fontsize=8)
        else:
            ax.plot(plotted[x], plotted[point_value_column], marker="o", color="#3278b9")
        ax.set(xlabel=x_label or x, ylabel=y_label or value_label)
        if pd.api.types.is_datetime64_any_dtype(plotted[x]):
            fig.autofmt_xdate()
        default_title = f"{x}별 {'누적 ' if cumulative else ''}{value_label}"
        reason = f"{effective_aggregation} 기준 {len(plotted):,}개 점을 연결했습니다."
        if cumulative:
            reason += " 정렬 후 누적합을 적용했습니다."

    final_title = title or default_title
    ax.set_title(final_title)
    buffer = BytesIO()
    FigureCanvasAgg(fig)
    _apply_unicode_font(fig)
    fig.tight_layout()
    fig.savefig(buffer, format="png", dpi=110)
    columns = tuple(dict.fromkeys(requested))
    scope = (f"보유 {len(source):,}행 기준 · {info.coverage} · {info.grain} · "
             f"표시 데이터 {len(plotted):,}개 · 집계 {effective_aggregation}")
    card = ChartPreview(str(uuid4()), dataset_id, final_title, reason, kind, columns, scope, buffer.getvalue())
    summary = {
        "source_rows": len(source),
        "rendered_rows": len(plotted),
        "data_sha256": digest(plotted.reset_index(drop=True)),
        "sampled": sampled,
        "aggregation": effective_aggregation,
    }
    if kind in {"bar", "line"}:
        def json_scalar(value):
            if hasattr(value, "item"):
                value = value.item()
            if pd.isna(value):
                return None
            if isinstance(value, pd.Timestamp):
                return value.isoformat()
            return value if value is None or isinstance(value, (str, int, float, bool)) else str(value)
        summary["points"] = []
        for _, row in plotted.iterrows():
            point = {"x": json_scalar(row[x]), "value": json_scalar(row[point_value_column])}
            if category:
                point["category"] = json_scalar(row[category])
            summary["points"].append(point)
    spec = {"kind": kind, "x": x, "y": y, "category": category,
            "aggregation": effective_aggregation, "sort": sort,
            "cumulative": bool(cumulative),
            "top_n": int(top_n), "bins": int(bins), "title": final_title,
            "x_label": ax.get_xlabel(), "y_label": ax.get_ylabel(),
            "orientation": orientation}
    return card, summary, spec


def render_count_rate_chart(store: DatasetStore, dataset_id: str, *,
                            group_column: str, outcome_column: str,
                            success_value, layout: str = "dual_axis",
                            sort: str = "none", top_n: int = 50,
                            title: str = "", x_label: str = "",
                            count_label: str = "", rate_label: str = ""):
    """Render grouped row counts and an explicit success rate.

    The success numerator is bound to ``outcome_column == success_value`` and
    the denominator is the non-null outcome count within each non-null group.
    This avoids asking the model to infer a percentage from plotted pixels or
    silently treating missing outcomes as failures.
    """
    if layout not in {"dual_axis", "split_panel"}:
        raise ValueError("layout은 dual_axis 또는 split_panel이어야 합니다.")
    if sort not in {"none", "group_ascending", "count_descending", "calendar_month"}:
        raise ValueError("지원하는 정렬 방식을 사용해주세요.")
    if not 1 <= int(top_n) <= 50:
        raise ValueError("top_n은 1~50이어야 합니다.")
    labels = (title, x_label, count_label, rate_label)
    if any(not isinstance(value, str) or len(value) > 120 for value in labels):
        raise ValueError("차트 제목과 축 라벨은 120자 이하 문자열이어야 합니다.")
    if not group_column or not outcome_column or group_column == outcome_column:
        raise ValueError("서로 다른 group_column과 outcome_column이 필요합니다.")
    info = store.metadata[dataset_id]
    if info.grain != "raw" or info.aggregation:
        raise ValueError("건수·비율 차트는 집계되지 않은 raw dataset만 사용합니다.")
    if not {group_column, outcome_column}.issubset(info.columns):
        raise ValueError("차트 컬럼은 로딩된 dataset의 실제 컬럼이어야 합니다.")
    source = project_dataset(store, dataset_id, [group_column, outcome_column])
    frame = source[[group_column, outcome_column]].dropna(subset=[group_column]).copy()
    if frame.empty:
        raise ValueError("그룹값이 있는 행이 필요합니다.")
    observed = frame[outcome_column].dropna()
    if observed.empty or not observed.eq(success_value).any():
        raise ValueError("success_value가 outcome 컬럼의 실제 값과 일치해야 합니다.")
    groups = int(frame[group_column].nunique(dropna=True))
    if not 1 <= groups <= 50:
        raise ValueError("그룹 고유값은 1~50개여야 합니다.")

    grouped = frame.groupby(group_column, sort=False, observed=True)[outcome_column]
    plotted = grouped.agg(row_count="size", denominator_count="count").reset_index()
    successes = (frame.assign(_success=frame[outcome_column].eq(success_value))
                 .groupby(group_column, sort=False, observed=True)["_success"].sum()
                 .astype(int).reset_index(name="success_count"))
    plotted = plotted.merge(successes, on=group_column, how="inner", validate="one_to_one")
    if (plotted["denominator_count"] <= 0).any():
        raise ValueError("각 그룹에는 결측이 아닌 outcome 값이 하나 이상 필요합니다.")
    plotted["rate_percent"] = 100.0 * plotted["success_count"] / plotted["denominator_count"]

    if sort == "calendar_month":
        month_number = {
            "jan": 1, "january": 1, "feb": 2, "february": 2,
            "mar": 3, "march": 3, "apr": 4, "april": 4,
            "may": 5, "jun": 6, "june": 6, "jul": 7, "july": 7,
            "aug": 8, "august": 8, "sep": 9, "sept": 9, "september": 9,
            "oct": 10, "october": 10, "nov": 11, "november": 11,
            "dec": 12, "december": 12,
        }
        keys = plotted[group_column].astype(str).str.strip().str.casefold().map(month_number)
        if keys.isna().any() or keys.duplicated().any():
            raise ValueError("calendar_month 정렬에는 중복되지 않는 영문 월 이름/약어가 필요합니다.")
        plotted = plotted.assign(_order=keys).sort_values("_order").drop(columns="_order")
    elif sort == "group_ascending":
        try:
            plotted = plotted.sort_values(group_column)
        except TypeError as exc:
            raise ValueError("그룹값을 오름차순으로 정렬할 수 없습니다.") from exc
    elif sort == "count_descending":
        plotted = (plotted.assign(_group_sort=plotted[group_column].astype(str))
                   .sort_values(["row_count", "_group_sort"], ascending=[False, True],
                                kind="mergesort")
                   .drop(columns="_group_sort"))
    if len(plotted) > int(top_n):
        if sort != "count_descending":
            plotted = plotted.nlargest(int(top_n), "row_count")
        else:
            plotted = plotted.head(int(top_n))
    plotted = plotted.reset_index(drop=True)

    count_axis_label = count_label or "Count"
    rate_axis_label = rate_label or "Rate (%)"
    final_title = title or f"{group_column}별 건수와 {outcome_column} 성공률"
    if layout == "dual_axis":
        fig = Figure(figsize=(8, 4.5))
        ax_count = fig.subplots()
        ax_rate = ax_count.twinx()
        labels_for_axis = plotted[group_column].astype(str)
        ax_count.bar(labels_for_axis, plotted["row_count"], color="#6f9fd8", alpha=0.68)
        ax_rate.plot(labels_for_axis, plotted["rate_percent"], color="#c43b4d",
                     marker="o", linewidth=2)
        ax_count.set(xlabel=x_label or group_column, ylabel=count_axis_label)
        ax_rate.set_ylabel(rate_axis_label)
        ax_count.tick_params(axis="x", rotation=30)
        ax_count.set_title(final_title)
    else:
        fig = Figure(figsize=(10, 4.5))
        ax_count, ax_rate = fig.subplots(1, 2)
        labels_for_axis = plotted[group_column].astype(str)
        ax_count.bar(labels_for_axis, plotted["row_count"], color="#6f9fd8")
        ax_rate.plot(labels_for_axis, plotted["rate_percent"], color="#c43b4d",
                     marker="o", linewidth=2)
        ax_count.set(xlabel=x_label or group_column, ylabel=count_axis_label,
                     title=count_axis_label)
        ax_rate.set(xlabel=x_label or group_column, ylabel=rate_axis_label,
                    title=rate_axis_label)
        ax_count.tick_params(axis="x", rotation=30)
        ax_rate.tick_params(axis="x", rotation=30)
        fig.suptitle(final_title)

    encoded_columns = json.dumps(list(plotted.columns), ensure_ascii=False).encode()
    hashed = pd.util.hash_pandas_object(plotted, index=False).values.tobytes()
    data_sha256 = sha256(encoded_columns + hashed).hexdigest()
    buffer = BytesIO()
    FigureCanvasAgg(fig)
    _apply_unicode_font(fig)
    fig.tight_layout()
    fig.savefig(buffer, format="png", dpi=110)
    missing_outcomes = int(frame[outcome_column].isna().sum())
    reason = (f"{len(plotted):,}개 그룹의 전체 행 수와 {outcome_column}={success_value!r} 비율을 "
              f"결측 outcome을 분모에서 제외해 표시했습니다.")
    scope = (f"보유 {len(source):,}행 기준 · {info.coverage} · raw · "
             f"그룹 {len(plotted):,}개 · outcome 결측 {missing_outcomes:,}행")
    card = ChartPreview(str(uuid4()), dataset_id, final_title, reason, layout,
                        (group_column, outcome_column), scope, buffer.getvalue())

    def scalar(value):
        if hasattr(value, "item"):
            value = value.item()
        if pd.isna(value):
            return None
        return value if isinstance(value, (str, int, float, bool)) else str(value)

    # ``iterrows`` coerces an integer grouping column to float when the same
    # row also contains ``rate_percent``. Records preserve the source dtype so
    # the structured evidence reports group 1 as 1 rather than 1.0.
    points = [{
        "group": scalar(row[group_column]),
        "row_count": int(row["row_count"]),
        "denominator_count": int(row["denominator_count"]),
        "success_count": int(row["success_count"]),
        "rate_percent": float(row["rate_percent"]),
    } for row in plotted.to_dict(orient="records")]
    summary = {
        "source_rows": len(source),
        "grouped_source_rows": len(frame),
        "rendered_groups": len(plotted),
        "missing_outcome_rows": missing_outcomes,
        "data_sha256": data_sha256,
        "points": points,
    }
    spec = {
        "layout": layout,
        "group_column": group_column,
        "outcome_column": outcome_column,
        "success_value": scalar(success_value),
        "sort": sort,
        "top_n": int(top_n),
        "title": final_title,
        "x_label": x_label or group_column,
        "count_label": count_axis_label,
        "rate_label": rate_axis_label,
    }
    return card, summary, spec
