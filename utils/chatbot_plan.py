from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pandas as pd

from utils.eda_validation import choose_distribution_chart


DEFAULT_CONTROLLED_SQL_LIMIT = 2000


@dataclass(frozen=True)
class ControlledPlan:
    intent: str
    task: str
    target_column: str
    filters: Dict[str, Any] = field(default_factory=dict)
    table: str = ""


@dataclass(frozen=True)
class VisualizationConfig:
    plot_type: str
    column: str


def build_controlled_plan(user_query: str, *, default_table: str = "") -> Optional[ControlledPlan]:
    """Build a narrow deterministic plan for common chatbot visualization requests."""

    text = user_query or ""
    lowered = text.lower()
    wants_visual = any(token in lowered for token in ("시각화", "분포", "그래프", "차트", "plot", "chart"))
    if not wants_visual:
        return None

    target_column = ""
    for candidate in ("balance", "age", "job", "loan"):
        if candidate in lowered:
            target_column = candidate
            break
    if not target_column:
        return None

    filters: Dict[str, Any] = {}
    if "20대" in text and "30대" in text:
        filters["age"] = [20, 30]
    if "대출" in text or "loan" in lowered:
        filters["loan"] = "yes"

    return ControlledPlan(
        intent="VISUALIZE",
        task="distribution",
        target_column=target_column,
        filters=filters,
        table=(default_table or "").strip(),
    )


def build_sql_from_plan(plan: ControlledPlan, *, limit: Optional[int] = None) -> str:
    if plan.intent != "VISUALIZE" or not plan.target_column or not plan.table:
        raise ValueError("Unsupported controlled plan.")

    selected_columns = required_columns_for_plan(plan)
    if not selected_columns:
        raise ValueError("Controlled plan has no required columns.")

    clauses: List[str] = []
    age_filter = plan.filters.get("age")
    if isinstance(age_filter, list) and len(age_filter) == 2:
        clauses.append(f"age BETWEEN {int(age_filter[0])} AND {int(age_filter[1])}")
    if plan.filters.get("loan") == "yes":
        clauses.append("loan = 'yes'")

    where_sql = f" WHERE {' AND '.join(clauses)}" if clauses else ""
    limit_value = limit if limit is not None else DEFAULT_CONTROLLED_SQL_LIMIT
    return f"SELECT {', '.join(selected_columns)} FROM {plan.table}{where_sql} LIMIT {limit_value}"


def required_columns_for_plan(plan: ControlledPlan) -> list[str]:
    columns = [plan.target_column]
    columns.extend(str(column) for column in plan.filters.keys())
    return list(dict.fromkeys(column for column in columns if column))


def select_visualization_config(plan: ControlledPlan, df: pd.DataFrame) -> VisualizationConfig:
    if plan.target_column not in df.columns:
        raise ValueError(f"Column not found: {plan.target_column}")

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
    "VisualizationConfig",
    "build_controlled_plan",
    "build_sql_from_plan",
    "required_columns_for_plan",
    "select_visualization_config",
]
