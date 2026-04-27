from dataclasses import asdict, dataclass, field
import re
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd

from utils.eda_validation import choose_distribution_chart
from utils.table_context import coerce_table_context, get_column_names, is_trained_table_context, strip_internal_prompt_context


DEFAULT_CONTROLLED_SQL_LIMIT = 2000


@dataclass(frozen=True)
class FilterCondition:
    column: str
    op: str
    value: Any


def _has_yes_filter(text: str, term: str) -> bool:
    lowered = (text or "").lower()
    separator = r"(?:(?:\s*(?:이|가|은|는)\s*)|(?:\s*(?:==|=)\s*)|\s+)"
    return bool(re.search(rf"(?<![A-Za-z0-9_]){re.escape(term.lower())}{separator}'?yes'?", lowered))


def _has_positive_filter(text: str, term: str) -> bool:
    lowered = (text or "").lower()
    separator = r"(?:\s*(?:이|가|은|는|을|를|의)?\s*)"
    term_pattern = rf"(?<![A-Za-z0-9_]){re.escape(term.lower())}(?![A-Za-z0-9_])"
    positive_pattern = r"(?:가지고\s*있는|보유|있는|있고|yes)"
    return bool(re.search(term_pattern + separator + positive_pattern, lowered))


def _identifier_position(text: str, column: str) -> int:
    match = re.search(rf"(?<![A-Za-z0-9_]){re.escape(column)}(?![A-Za-z0-9_])", text)
    return match.start() if match else -1


@dataclass(frozen=True)
class ControlledPlan:
    intent: str
    task: str
    target_column: str
    filters: Dict[str, Any] = field(default_factory=dict)
    table: str = ""
    rank_column: str = ""
    rank_percent: Optional[float] = None
    rank_direction: str = ""
    filter_conditions: Tuple[FilterCondition, ...] = ()
    target_semantic_type: str = ""
    group_column: str = ""
    group_values: Tuple[Any, ...] = ()
    group_mode: str = ""
    resolution_debug: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class VisualizationConfig:
    plot_type: str
    column: str
    group_column: str = ""
    group_values: Tuple[Any, ...] = ()
    group_mode: str = ""


def _semantic_types(table_context) -> dict[str, str]:
    context = coerce_table_context(table_context)
    if context is None:
        return {}
    return {column.name: column.semantic_type for column in context.columns if column.name}


def _column_terms(table_context) -> dict[str, list[str]]:
    context = coerce_table_context(table_context)
    if context is None:
        return {}
    terms: dict[str, list[str]] = {}
    for column in context.columns:
        column_terms = []
        if column.name:
            column_terms.append(column.name)
        column_terms.extend(column.aliases)
        seen: dict[str, str] = {}
        for term in column_terms:
            text = str(term or "").strip()
            if text:
                seen.setdefault(text.lower(), text)
        if column.name:
            terms[column.name] = list(seen.values())
    return terms


def _find_column_mentions(text: str, table_context, *, excluded_columns: Optional[Iterable[str]] = None) -> list[tuple[int, str, str]]:
    lowered = strip_internal_prompt_context(text or "").lower()
    excluded = {str(column).lower() for column in (excluded_columns or [])}
    mentions: list[tuple[int, str, str]] = []
    for column, terms in _column_terms(table_context).items():
        if column.lower() in excluded:
            continue
        for term in terms:
            position = _identifier_position(lowered, term.lower())
            if position >= 0:
                mentions.append((position, column, term))
    mentions.sort()
    return mentions


def _visual_token_position(text: str) -> int:
    lowered = strip_internal_prompt_context(text or "").lower()
    positions = [
        lowered.find(token)
        for token in ("분포", "본포", "시각화", "그래프", "차트", "그려", "plot", "chart", "graph")
        if lowered.find(token) >= 0
    ]
    return min(positions) if positions else -1


def _unique_mentions(mentions: Iterable[tuple[int, str, str]]) -> list[tuple[int, str, str]]:
    seen: set[str] = set()
    unique: list[tuple[int, str, str]] = []
    for position, column, term in sorted(mentions):
        key = column.lower()
        if key in seen:
            continue
        seen.add(key)
        unique.append((position, column, term))
    return unique


def _parenthetical_target_candidate(
    text: str,
    table_context,
    *,
    excluded_columns: Iterable[str],
) -> str:
    lowered = strip_internal_prompt_context(text or "").lower()
    excluded = {str(column).lower() for column in excluded_columns}
    candidates: list[tuple[int, str]] = []
    for column, terms in _column_terms(table_context).items():
        if column.lower() in excluded:
            continue
        for term in sorted(terms, key=len, reverse=True):
            term_pattern = rf"(?<![A-Za-z0-9_]){re.escape(term.lower())}(?![A-Za-z0-9_])"
            match = re.search(term_pattern + r"\s*\(", lowered)
            if match:
                candidates.append((match.start(), column))
                break
    candidates.sort()
    return candidates[0][1] if candidates else ""


def _resolve_target_column(
    text: str,
    table_context,
    *,
    excluded_columns: Iterable[str],
) -> tuple[str, list[dict[str, Any]]]:
    excluded = {str(column).lower() for column in excluded_columns}
    parenthetical = _parenthetical_target_candidate(
        text,
        table_context,
        excluded_columns=excluded,
    )
    mentions = _unique_mentions(
        _find_column_mentions(text, table_context, excluded_columns=excluded)
    )
    candidates = [
        {
            "column": column,
            "term": term,
            "position": position,
        }
        for position, column, term in mentions
    ]
    if parenthetical:
        return parenthetical, candidates
    if not mentions:
        return "", candidates

    visual_position = _visual_token_position(text)
    if visual_position < 0:
        return mentions[0][1], candidates
    ranked_mentions = sorted(
        mentions,
        key=lambda item: (
            abs(item[0] - visual_position),
            item[0],
        ),
    )
    return ranked_mentions[0][1], candidates


def _parse_group_value_condition(text: str, table_context) -> tuple[str, Tuple[Any, ...]]:
    lowered = strip_internal_prompt_context(text or "").lower()
    for position, column, _ in _find_column_mentions(lowered, table_context):
        window = lowered[position : position + 160]
        if "값" not in window[:40]:
            continue
        raw_values = re.findall(r"(-?\d+(?:\.\d+)?)\s*인\s*사람", window)
        if len(raw_values) < 2:
            raw_values = re.findall(r"(-?\d+(?:\.\d+)?)\s*인", window)
        values: list[Any] = []
        for raw_value in raw_values:
            parsed = _parse_numeric_value(raw_value)
            if parsed not in values:
                values.append(parsed)
        if len(values) >= 2:
            return column, tuple(values)
    return "", ()


def _parse_rank_percent(text: str) -> tuple[str, Optional[float]]:
    lowered = (text or "").lower()
    match = re.search(r"(상위|top)\s*(\d+(?:\.\d+)?)\s*%", lowered)
    if match:
        return "top", float(match.group(2))
    match = re.search(r"(하위|bottom)\s*(\d+(?:\.\d+)?)\s*%", lowered)
    if match:
        return "bottom", float(match.group(2))
    return "", None


def _resolve_rank_column(text: str, candidates: list[str], semantic_types: dict[str, str]) -> str:
    return ""


def _resolve_rank_column_from_context(text: str, table_context, semantic_types: dict[str, str]) -> str:
    for _, column, _ in _find_column_mentions(text, table_context):
        if semantic_types.get(column) == "numeric":
            return column
    return ""


def _parse_numeric_value(value: str) -> Any:
    number = float(value)
    return int(number) if number.is_integer() else number


def _comparison_op_from_word(word: str) -> str:
    normalized = re.sub(r"\s+", "", word or "")
    if normalized in {">", "초과", "넘는", "보다큰", "크다"}:
        return ">"
    if normalized in {">=", "이상", "보다크거나같은"}:
        return ">="
    if normalized in {"<", "미만", "보다작은", "작다"}:
        return "<"
    if normalized in {"<=", "이하", "보다작거나같은"}:
        return "<="
    if normalized in {"=", "==", "같은"}:
        return "="
    return ""


def _parse_comparison_filters(text: str, table_context, semantic_types: dict[str, str]) -> tuple[FilterCondition, ...]:
    lowered = (text or "").lower()
    conditions: list[FilterCondition] = []
    for column, terms in _column_terms(table_context).items():
        if semantic_types.get(column) not in {"numeric", "unknown"}:
            continue
        for term in sorted(terms, key=len, reverse=True):
            term_pattern = rf"(?<![A-Za-z0-9_]){re.escape(term.lower())}(?![A-Za-z0-9_])"
            particle = r"(?:\s*(?:이|가|은|는|을|를|의)?\s*)"
            number = r"(-?\d+(?:\.\d+)?)"
            patterns = [
                rf"{term_pattern}{particle}(>=|<=|>|<|=|==){particle}{number}",
                rf"{term_pattern}{particle}{number}{particle}(이상|이하|초과|미만|넘는|보다\s*큰|보다\s*작은)",
            ]
            for pattern in patterns:
                match = re.search(pattern, lowered)
                if not match:
                    continue
                if match.group(1).replace(" ", "") in {">", ">=", "<", "<=", "=", "=="}:
                    op = _comparison_op_from_word(match.group(1))
                    value_text = match.group(2)
                else:
                    value_text = match.group(1)
                    op = _comparison_op_from_word(match.group(2))
                if op:
                    condition = FilterCondition(column=column, op=op, value=_parse_numeric_value(value_text))
                    if condition not in conditions:
                        conditions.append(condition)
                    break
            if any(condition.column == column for condition in conditions):
                break
    return tuple(conditions)


def _top_value_text(value: Any) -> str:
    if isinstance(value, dict):
        value = value.get("value")
    return str(value or "").strip()


def _find_phrase_positions(text: str, phrase: str) -> list[int]:
    lowered = strip_internal_prompt_context(text or "").lower()
    normalized = str(phrase or "").strip().lower()
    if not normalized:
        return []
    pattern = rf"(?<![A-Za-z0-9_]){re.escape(normalized)}(?![A-Za-z0-9_])"
    return [match.start() for match in re.finditer(pattern, lowered)]


def _parse_categorical_value_filters(
    text: str,
    table_context,
    semantic_types: dict[str, str],
) -> tuple[dict[str, str], list[dict[str, Any]]]:
    context = coerce_table_context(table_context)
    if context is None:
        return {}, []

    mentions_by_column: dict[str, list[tuple[int, str]]] = {}
    for position, column, term in _find_column_mentions(text, context):
        mentions_by_column.setdefault(column, []).append((position, term))

    value_matches_by_text: dict[str, set[str]] = {}
    column_value_candidates: list[dict[str, Any]] = []
    for column in context.columns:
        column_name = str(getattr(column, "name", "") or "").strip()
        if not column_name or semantic_types.get(column_name) != "categorical":
            continue
        if not mentions_by_column.get(column_name):
            continue
        for top_value in getattr(column, "top_values", []) or []:
            value_text = _top_value_text(top_value)
            positions = _find_phrase_positions(text, value_text)
            if not positions:
                continue
            normalized_value = value_text.lower()
            value_matches_by_text.setdefault(normalized_value, set()).add(column_name)
            column_value_candidates.append(
                {
                    "column": column_name,
                    "value": value_text,
                    "value_positions": positions,
                    "column_mentions": [
                        {"position": position, "term": term}
                        for position, term in mentions_by_column.get(column_name, [])
                    ],
                }
            )

    filters: dict[str, str] = {}
    debug: list[dict[str, Any]] = []
    for candidate in column_value_candidates:
        column = candidate["column"]
        value = candidate["value"]
        normalized_value = value.lower()
        matched_columns = sorted(value_matches_by_text.get(normalized_value, set()))
        mention_positions = [item["position"] for item in candidate["column_mentions"]]
        value_positions = candidate["value_positions"]
        distances = [
            value_position - mention_position
            for mention_position in mention_positions
            for value_position in value_positions
        ]
        forward_distances = [distance for distance in distances if 0 <= distance <= 160]
        reverse_distances = [abs(distance) for distance in distances if -40 <= distance < 0]
        accepted = bool(forward_distances or reverse_distances)
        reason = "matched_explicit_column_top_value" if accepted else "value_not_near_column_mention"
        if accepted and len(matched_columns) > 1:
            reason = "matched_explicit_column_top_value_with_shared_value"
        if accepted and column not in filters:
            filters[column] = value
        debug.append(
            {
                "column": column,
                "value": value,
                "matched_columns": matched_columns,
                "accepted": accepted,
                "reason": reason,
            }
        )
    return filters, debug


def build_controlled_plan(
    user_query: str,
    *,
    default_table: str = "",
    table_context=None,
) -> Optional[ControlledPlan]:
    """Build a narrow deterministic plan for common chatbot visualization requests."""

    text = user_query or ""
    lowered = text.lower()
    wants_visual = any(token in lowered for token in ("시각화", "분포", "그래프", "차트", "plot", "chart"))
    if not wants_visual:
        return None

    if not is_trained_table_context(table_context):
        return None

    context = coerce_table_context(table_context)
    candidates = get_column_names(context)
    if not candidates:
        return None
    semantic_types = _semantic_types(context)
    rank_direction, rank_percent = _parse_rank_percent(text)
    rank_column = ""
    if rank_direction and rank_percent is not None:
        rank_column = _resolve_rank_column_from_context(text, context, semantic_types)
    group_column, group_values = _parse_group_value_condition(text, context)
    comparison_filters = _parse_comparison_filters(text, context, semantic_types)
    categorical_filters, categorical_filter_debug = _parse_categorical_value_filters(
        text,
        context,
        semantic_types,
    )
    range_filters = _parse_decade_range_filters(text, context, semantic_types)

    explicit_filter_columns = {
        column
        for column in candidates
        if any(_has_yes_filter(text, term) or _has_positive_filter(text, term) for term in _column_terms(context).get(column, [column]))
    }
    excluded_columns = sorted(
        set(explicit_filter_columns)
        | ({rank_column} if rank_column else set())
        | ({group_column} if group_column else set())
        | {condition.column for condition in comparison_filters}
        | set(categorical_filters.keys())
        | set(range_filters.keys())
    )
    target_column, target_candidates = _resolve_target_column(
        text,
        context,
        excluded_columns=excluded_columns,
    )
    if not target_column:
        return None

    filters: Dict[str, Any] = dict(range_filters)
    filters.update(categorical_filters)
    for column in sorted(explicit_filter_columns):
        if column != target_column:
            filters[column] = "yes"

    task = "distribution"
    if group_column and group_values:
        task = "grouped_distribution"
    elif rank_column and rank_percent is not None:
        task = "ranked_distribution"

    column_mentions = [
        {"position": position, "column": column, "term": term}
        for position, column, term in _find_column_mentions(text, context)
    ]
    resolution_debug = {
        "column_mentions": column_mentions,
        "excluded_columns": excluded_columns,
        "target_candidates": target_candidates,
        "target_column": target_column,
        "group_column": group_column,
        "group_values": list(group_values),
        "categorical_value_filters": categorical_filter_debug,
    }

    return ControlledPlan(
        intent="VISUALIZE",
        task=task,
        target_column=target_column,
        filters=filters,
        table=(default_table or "").strip(),
        rank_column=rank_column,
        rank_percent=rank_percent,
        rank_direction=rank_direction,
        filter_conditions=comparison_filters,
        target_semantic_type=semantic_types.get(target_column, ""),
        group_column=group_column,
        group_values=group_values,
        group_mode="separate" if "각각" in lowered else "grouped",
        resolution_debug=resolution_debug,
    )


def build_sql_from_plan(plan: ControlledPlan, *, limit: Optional[int] = None) -> str:
    if plan.intent != "VISUALIZE" or not plan.target_column or not plan.table:
        raise ValueError("Unsupported controlled plan.")

    selected_columns = required_columns_for_plan(plan)
    if not selected_columns:
        raise ValueError("Controlled plan has no required columns.")

    clauses = _filter_clauses(plan.filters, plan.filter_conditions)
    limit_value = limit if limit is not None else DEFAULT_CONTROLLED_SQL_LIMIT

    if plan.task == "grouped_distribution":
        if not plan.group_column:
            raise ValueError("Grouped distribution plan has no group column.")
        where_clauses = [*clauses]
        if plan.group_values:
            values = ", ".join(_format_sql_value(value) for value in plan.group_values)
            where_clauses.append(f"{plan.group_column} IN ({values})")
        where_sql = f" WHERE {' AND '.join(where_clauses)}" if where_clauses else ""
        return (
            f"SELECT {plan.group_column}, {plan.target_column}, COUNT(*) AS stat_count "
            f"FROM {plan.table}{where_sql} "
            f"GROUP BY {plan.group_column}, {plan.target_column} "
            f"ORDER BY {plan.group_column}, stat_count DESC "
            f"LIMIT {limit_value}"
        )

    should_aggregate = (
        bool(plan.rank_column and plan.rank_percent is not None)
        or (plan.target_semantic_type == "categorical" and bool(clauses))
    )

    if should_aggregate:
        if plan.rank_column and plan.rank_column not in selected_columns:
            selected_columns.append(plan.rank_column)
        where_clauses = [*clauses]
        if plan.rank_column and plan.rank_percent is not None:
            percentile = float(plan.rank_percent) / 100.0
            percentile = max(0.0, min(1.0, percentile))
            quantile = 1.0 - percentile if plan.rank_direction == "top" else percentile
            operator = ">=" if plan.rank_direction == "top" else "<="
            filter_where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
            rank_clause = (
                f"{plan.rank_column} {operator} "
                f"(SELECT percentile_approx({plan.rank_column}, {quantile:g}) "
                f"FROM {plan.table}{filter_where})"
            )
            where_clauses.append(rank_clause)
        where_sql = f" WHERE {' AND '.join(where_clauses)}"
        return (
            f"SELECT {plan.target_column}, COUNT(*) AS stat_count "
            f"FROM {plan.table}{where_sql} "
            f"GROUP BY {plan.target_column} "
            "ORDER BY stat_count DESC "
            f"LIMIT {limit_value}"
        )

    where_sql = f" WHERE {' AND '.join(clauses)}" if clauses else ""
    return f"SELECT {', '.join(selected_columns)} FROM {plan.table}{where_sql} LIMIT {limit_value}"


def _parse_decade_range_filters(
    text: str,
    table_context,
    semantic_types: dict[str, str],
) -> Dict[str, list[int]]:
    match = re.search(r"(\d{2})\s*대.*?(\d{2})\s*대", text or "")
    if not match:
        return {}
    start_decade = int(match.group(1))
    end_decade = int(match.group(2))
    if start_decade > end_decade:
        start_decade, end_decade = end_decade, start_decade

    range_start, range_end = match.span()
    mentioned_numeric_columns = []
    for position, column, _ in _find_column_mentions(text, table_context):
        if not (range_start <= position < range_end):
            continue
        if semantic_types.get(column) in {"numeric", "unknown"} and column not in mentioned_numeric_columns:
            mentioned_numeric_columns.append(column)
    if len(mentioned_numeric_columns) != 1:
        return {}
    return {mentioned_numeric_columns[0]: [start_decade, end_decade]}


def _format_sql_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    escaped = str(value).replace("'", "''")
    return f"'{escaped}'"


def _filter_clauses(filters: Dict[str, Any], filter_conditions: Iterable[FilterCondition] = ()) -> list[str]:
    clauses: List[str] = []
    for column, value in filters.items():
        if isinstance(value, list) and len(value) == 2:
            clauses.append(f"{column} BETWEEN {int(value[0])} AND {int(value[1])}")
        elif isinstance(value, str):
            escaped = value.replace("'", "''")
            clauses.append(f"{column} = '{escaped}'")
    for condition in filter_conditions or ():
        clauses.append(f"{condition.column} {condition.op} {_format_sql_value(condition.value)}")
    return clauses


def required_columns_for_plan(plan: ControlledPlan) -> list[str]:
    columns = [plan.target_column]
    if getattr(plan, "group_column", ""):
        columns.append(plan.group_column)
    columns.extend(str(column) for column in plan.filters.keys())
    columns.extend(condition.column for condition in getattr(plan, "filter_conditions", ()) or ())
    if getattr(plan, "rank_column", ""):
        columns.append(plan.rank_column)
    return list(dict.fromkeys(column for column in columns if column))


def controlled_plan_to_dict(plan: ControlledPlan) -> dict[str, Any]:
    return asdict(plan)


def select_visualization_config(plan: ControlledPlan, df: pd.DataFrame) -> VisualizationConfig:
    if plan.target_column not in df.columns:
        raise ValueError(f"Column not found: {plan.target_column}")

    if plan.task == "grouped_distribution":
        if not plan.group_column or plan.group_column not in df.columns:
            raise ValueError(f"Group column not found: {plan.group_column}")
        return VisualizationConfig(
            plot_type="grouped_bar",
            column=plan.target_column,
            group_column=plan.group_column,
            group_values=tuple(plan.group_values or ()),
            group_mode=plan.group_mode or "grouped",
        )

    chart_type = choose_distribution_chart(df[plan.target_column])
    if chart_type == "hist":
        plot_type = "histogram"
    elif chart_type == "boxplot":
        plot_type = "boxplot"
    elif chart_type == "bar":
        plot_type = "bar"
    else:
        raise ValueError(f"Unsupported plot type for column: {plan.target_column}")
    return VisualizationConfig(plot_type=plot_type, column=plan.target_column)


__all__ = [
    "ControlledPlan",
    "FilterCondition",
    "VisualizationConfig",
    "build_controlled_plan",
    "build_sql_from_plan",
    "controlled_plan_to_dict",
    "required_columns_for_plan",
    "select_visualization_config",
]
