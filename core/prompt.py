import json
from typing import Iterable, Optional

import pandas as pd
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import BaseTool, render_text_description
from utils.session import get_default_sql_limit


def _stringify_cell(value) -> str:
    """Convert complex cell values (e.g. JSON blobs) into a compact string."""

    if isinstance(value, (dict, list)):
        value = json.dumps(value, ensure_ascii=False)
    elif isinstance(value, str):
        stripped = value.strip()
        if stripped:
            if stripped[0] in "[{" and stripped[-1] in "]}":
                try:
                    parsed = json.loads(stripped)
                except json.JSONDecodeError:
                    pass
                else:
                    value = json.dumps(parsed, ensure_ascii=False)

    text = str(value)
    if len(text) > 200:
        return text[:197] + "..."
    return text


def _df_head(df: pd.DataFrame) -> str:
    preview = df.head().copy()
    preview = preview.apply(lambda column: column.map(_stringify_cell))
    with pd.option_context("display.width", None, "display.max_colwidth", 200):
        return preview.to_string(index=False)


def _format_df_columns(df: pd.DataFrame) -> str:
    columns = [str(column) for column in df.columns]
    if not columns:
        return "(no columns)"
    return ", ".join(columns)


def _format_df_dtypes(df: pd.DataFrame) -> str:
    if len(df.columns) == 0:
        return "(no columns)"
    lines = [f"- {column}: {dtype}" for column, dtype in df.dtypes.items()]
    return "\n".join(lines) if lines else "(no columns)"


def build_react_prompt(
    df_a: pd.DataFrame,
    df_b,
    tools: Iterable[BaseTool],
) -> ChatPromptTemplate:
    """Construct the ReAct system prompt with up-to-date dataframe heads."""
    head_a = _df_head(df_a)
    head_b = _df_head(df_b) if df_b is not None else "(df_B not loaded)"

    system_template = (
        "You are an expert data analyst for SSD telemetry and tabular data. "
        "You work with two dataframes: df_A (main, alias: df) and df_B (optional, for comparison/labels).\n\n"
        "When the user asks for outlier-focused EDA, DO NOT ask questions. Immediately run this pipeline:\n"
        "  1) make_timesafe(column='datetime', tz='UTC') if 'datetime' exists\n"
        "  2) describe_columns → select_numeric_candidates → rank_outlier_columns\n"
        "  3) anomaly_iqr on top-N columns\n"
        "  4) stl_decompose on top-1 if time series available\n"
        "  5) anomaly_isoforest on k-best numeric (if sklearn available)\n"
        "  6) cohort_compare(by='model,fw', agg='mean') if columns exist\n"
        "  7) Summarize: top outlier columns + time spikes + cohort outliers + next steps.\n"
        "If df_B is loaded and user mentions comparison, use propose_join_keys then compare_on_keys.\n\n"
        "ALWAYS follow this EXACT format:\n"
        "Question: <restated question>\n"
        "Thought: <brief reasoning>\n"
        "Action: <ONE tool name from {tool_names}>\n"
        "Action Input: <valid input with NO backticks>\n"
        "Observation: <tool result>\n"
        "(Repeat Thought/Action/Action Input/Observation as needed)\n"
        "Thought: I now know the final answer\n"
        "Final Answer: <concise answer>\n\n"
        "If you output anything outside this format, continue immediately by outputting ONLY a valid 'Action' and 'Action Input'.\n\n"
        "Tool routing guide:\n"
        "- Schema/summary → describe_columns, describe_columns_on\n"
        "- File load → load_loading_csv, load_df_b\n"
        "- SQL/join/aggregation → sql_on_dfs\n"
        "- TWO-CSV comparison → propose_join_keys → compare_on_keys('machineID,datetime') → mismatch_report('...')\n"
        "- SSD utilities → make_timesafe, create_features, rolling_stats, stl_decompose, anomaly_iqr, anomaly_isoforest, cohort_compare, topn_machines\n"
        "- Outlier one-click → auto_outlier_eda, then plot_outliers on top columns\n"
        "- Distribution → plot_distribution\n"
        "- Custom compute/plots → python_repl_ast (do complex tasks in ONE call and print results)\n\n"
        "When preparing any time-series or trend visualisation, ALWAYS sort the data by the relevant time column in ascending order before plotting to prevent back-and-forth lines.\n\n"
        "Whenever the user requests a chart, plot, image, or visualisation, you MUST execute the relevant plotting tool (e.g., plot_distribution, plot_outliers, corr_heatmap) or run python_repl_ast with matplotlib code so the figure renders in Streamlit. Do not stop at textual descriptions.\n"
        "When visualising multiple columns or metrics, default to creating separate plots (one figure per column/metric) unless the user explicitly asks for a single combined plot.\n"
        "Example pattern for separate figures:\n"
        "  for col in ['temperature', 'pressure']:\n"
        "      fig, ax = plt.subplots(figsize=(8, 3))\n"
        "      ax.plot(df_A[col])\n"
        "      ax.set_title(col)\n"
        "      plt.tight_layout()\n\n"
        "For tools that take a column name (e.g., anomaly_iqr), pass ONLY the raw column name (e.g., temperature), "
        "not 'column=temperature'. If you accidentally wrote 'column=...', immediately continue by outputting just the raw name.\n\n"
        "Call compare_on_keys with just the keys string (e.g., 'machineID,datetime') or as JSON {{\"keys\":\"machineID,datetime\"}}.\n"
        "Do NOT pass \"keys='...'\" literal unless JSON.\n\n"
        "df_A.head():\n{df_a_head}\n\n"
        "df_B.head():\n{df_b_head}\n"
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_template),
            MessagesPlaceholder("chat_history", optional=True),
            ("human", "{input}"),
            ("assistant", "{agent_scratchpad}"),
        ]
    )

    tool_desc = render_text_description(tools)
    tool_names = ", ".join([tool.name for tool in tools])
    return prompt.partial(
        tools=tool_desc,
        tool_names=tool_names,
        df_a_head=head_a,
        df_b_head=head_b,
    )


def build_sql_prompt(
    tools: Iterable[BaseTool],
    *,
    selected_table: str = "",
    selected_catalog: str = "",
    selected_schema: str = "",
    df_preview: Optional[pd.DataFrame] = None,
    df_name: str = "",
) -> ChatPromptTemplate:
    """Construct the SQL generation prompt."""

    table_hint = selected_table.strip()
    catalog_hint = selected_catalog.strip()
    schema_hint = selected_schema.strip()

    context_lines = ["You are a Databricks SQL expert.\n\n"]

    namespace_hint = []
    if catalog_hint:
        namespace_hint.append(f"catalog `{catalog_hint}`")
    if schema_hint:
        namespace_hint.append(f"schema `{schema_hint}`")
    if namespace_hint:
        context_lines.append(
            "The Streamlit sidebar already selected the working "
            + " and ".join(namespace_hint)
            + " for you. "
        )
    else:
        context_lines.append(
            "The Streamlit sidebar already selected the working catalog and schema for you. "
        )

    if table_hint:
        context_lines.append(
            (
                f"Treat the currently selected table `{table_hint}` as the default table unless the user "
                "explicitly requests another one. Do not fall back to an earlier default. "
            )
        )

    df_label = df_name.strip() or (table_hint if table_hint else "df_A")
    if isinstance(df_preview, pd.DataFrame) and not df_preview.empty:
        df_columns = _format_df_columns(df_preview)
        df_dtypes = _format_df_dtypes(df_preview)
        df_head_text = _df_head(df_preview)
        context_lines.append(
            "\nActive dataframe preview for SQL generation:\n"
            f"- Source: {df_label}\n"
            f"- Shape: {df_preview.shape[0]} rows × {df_preview.shape[1]} columns\n"
            f"- Columns: {df_columns}\n"
            "dtypes:\n"
            f"{df_dtypes}\n"
            "head():\n"
            f"{df_head_text}\n\n"
        )
    else:
        context_lines.append(
            "\nActive dataframe preview for SQL generation: (no dataframe loaded)\n\n"
        )

    context_lines.append(
        "All user questions about 'the data' refer to the currently selected table; never ask the user to choose or confirm a "
        "catalog or schema. If you need that context, rely on session-aware tools instead of questioning the user.\n\n"
    )
    context_lines.append(
        "For each step before the final answer, ALWAYS respond in EXACTLY this format:\n"
        "Thought: <brief reasoning>\n"
        "Action: <ONE tool name from {tool_names}>\n"
        "Action Input: <valid input for that tool, no backticks>\n\n"
    )
    context_lines.append(
        "When you are ready to answer with the final SQL (and NOT call a tool), respond in EXACTLY this format:\n"
        "Thought: I now know the final answer\n"
        "Final Answer: SQL:\n <single SQL statement only, no markdown fences, no explanation>\n\n"
    )
    limit_value = get_default_sql_limit()
    context_lines.append(
        f"Always cap result sets with 'LIMIT {limit_value}' at the outermost query. If a LIMIT clause already exists, replace it with LIMIT {limit_value}.\n\n"
    )
    context_lines.append(
        "Do NOT output any other fields such as Question:, Observation:, Explanation:, or Execution: unless the tool runner provides Observation: back to you.\n"
        "Do NOT include markdown fences."
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "".join(context_lines)),
            MessagesPlaceholder("chat_history", optional=True),
            ("human", "{input}"),
            ("assistant", "{agent_scratchpad}"),
        ]
    )

    tool_desc = render_text_description(tools)
    tool_names = ", ".join([tool.name for tool in tools])
    return prompt.partial(tools=tool_desc, tool_names=tool_names)


__all__ = ["build_react_prompt", "build_sql_prompt"]
