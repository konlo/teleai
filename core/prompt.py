import json
from typing import Iterable, Optional

import pandas as pd
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import BaseTool, render_text_description_and_args
from utils.session import get_default_sql_limit


def escape_braces(text: str) -> str:
    return text.replace("{", "{{").replace("}", "}}")


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
    """Construct the structured-chat system prompt with up-to-date dataframe heads."""
    head_a = escape_braces(_df_head(df_a))
    head_b = escape_braces(_df_head(df_b)) if df_b is not None else "(df_B not loaded)"

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
        "Tool routing guide:\n"
        "- Schema/summary → describe_columns, describe_columns_on\n"
        "- File load → load_df_b\n"
        "- SQL/join/aggregation → sql_on_dfs\n"
        "- TWO-CSV comparison → propose_join_keys → compare_on_keys('machineID,datetime') → mismatch_report('...')\n"
        "- SSD utilities → make_timesafe, create_features, rolling_stats, stl_decompose, anomaly_iqr, anomaly_isoforest, cohort_compare, topn_machines\n"
        "- Outlier one-click → auto_outlier_eda, then plot_outliers on top columns\n"
        "- Custom compute/plots → python_repl_ast (do complex tasks in ONE call and print results)\n\n"
        "When preparing any time-series or trend visualisation, ALWAYS sort the data by the relevant time column in ascending order before plotting to prevent back-and-forth lines.\n\n"
        "Whenever the user requests a chart, plot, image, or visualisation, you MUST execute the relevant plotting tool (e.g., plot_outliers, corr_heatmap) or run python_repl_ast with matplotlib/seaborn code so the figure renders in Streamlit. Do not stop at textual descriptions.\n"
        "### 📊 VISUALIZATION GUIDELINES (STRICT):\n\n"
        "#### Phase 1: DATA TYPE ASSESSMENT (MANDATORY FIRST STEP)\n"
        "Before creating ANY visualization, you MUST determine the nature of the dataset:\n"
        "- **Raw data**: Each row represents an individual observation/record (e.g., one row per device per timestamp). Raw data has many rows and individual-level values.\n"
        "- **Aggregated data**: Each row represents a summary statistic (e.g., COUNT, AVG, SUM grouped by category). Aggregated data typically has few rows with columns like 'count', 'avg', 'sum', 'stat_count', or was produced by a GROUP BY query.\n"
        "To determine this, inspect `df.shape`, `df.columns`, and `df.head()` in your Thought step. Look for clues:\n"
        "  - Column names containing 'count', 'sum', 'avg', 'mean', 'total', 'stat_' → likely aggregated\n"
        "  - Very few rows (< 50) with categorical grouping columns → likely aggregated\n"
        "  - Many rows with individual measurements → likely raw\n\n"
        "#### Phase 2: CHART TYPE SELECTION\n"
        "Choose the visualization type based on the data nature:\n"
        "- **If AGGREGATED data**:\n"
        "  - ❌ DO NOT use: histogram, KDE, boxplot, violin plot, swarm plot (these require raw individual observations)\n"
        "  - ✅ USE instead: bar chart, horizontal bar chart, pie chart, treemap, heatmap, line chart (for trends), table display\n"
        "  - If the user explicitly requests a distribution plot on aggregated data, EXPLAIN the limitation clearly: '현재 데이터는 집계(aggregated) 데이터입니다. 히스토그램/박스플롯은 개별 관측값(raw data)이 필요합니다. 대신 막대 차트로 시각화하겠습니다.'\n"
        "- **If RAW data**:\n"
        "  - ✅ All chart types are valid: histogram, KDE, boxplot, scatter, violin, bar, line, heatmap, etc.\n\n"
        "#### Phase 3: VISUALIZATION BEST PRACTICES\n"
        "1. **Prefer Seaborn**: Use `import seaborn as sns` for statistical plots. It handles categorical data and missing values more gracefully than pandas .plot().\n"
        "2. **Defensive Check**: ALWAYS verify data existence before plotting: `if df.empty: print('No data available to plot'); exit()`.\n"
        "3. **Pre-flight Inspection**: Before `plt.show()`, always print `df.shape` and `df.info()` to ensure the plotting data is valid.\n"
        "4. **Robust Filtering**: When removing 'unknown' or specific values, use row-wise logic: `df[~(df[cols] == 'unknown').any(axis=1)]`.\n"
        "5. **Chain of Thought**: Describe your data processing steps in 'Thought' before writing the 'Action Input'.\n"
        "6. **Error Diagnosis**: If a plot fails, use `df.info()` and `df.describe()` in the next step to diagnose the cause.\n\n"
        "#### Phase 4: POST-VISUALIZATION SELF-REVIEW\n"
        "After generating the visualization code, review whether the visualization is valid:\n"
        "- Does the chart type match the data nature (raw vs aggregated)?\n"
        "- Are axis labels meaningful and in the correct language (Korean preferred for user-facing labels)?\n"
        "- Is the data sorted appropriately (e.g., time series in ascending order)?\n"
        "- Would the chart be misleading with the current data? If so, explain and propose an alternative.\n\n"
        "You have access to the following tools:\n\n{tools}\n\n"
        "Use a json blob to specify a tool by providing an action key (tool name) and an action_input key (tool input).\n\n"
        'Valid "action" values: "Final Answer" or {tool_names}\n\n'
        "Provide only ONE action per $JSON_BLOB, as shown:\n\n"
        "```\n"
        "{{{{\n"
        '  "action": $TOOL_NAME,\n'
        '  "action_input": $INPUT\n'
        "}}}}\n"
        "```\n\n"
        "Follow this format:\n\n"
        "Question: input question to answer\n"
        "Thought: consider previous and subsequent steps\n"
        "Action:\n"
        "```\n"
        "$JSON_BLOB\n"
        "```\n"
        "Observation: action result\n"
        "... (repeat Thought/Action/Observation N times)\n"
        "Thought: I know what to respond\n"
        "Action:\n"
        "```\n"
        "{{{{\n"
        '  "action": "Final Answer",\n'
        '  "action_input": "최종 응답을 한국어로 작성하세요"\n'
        "}}}}\n"
        "```\n\n"
        "df_A.head():\n{df_a_head}\n\n"
        "df_B.head():\n{df_b_head}\n"
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_template),
            MessagesPlaceholder("chat_history", optional=True),
            ("human", "{input}\n\n{agent_scratchpad}"),
        ]
    )

    tool_desc = render_text_description_and_args(tools)
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

    table_hint_raw = (selected_table or "").strip()
    catalog_hint_raw = (selected_catalog or "").strip()
    schema_hint_raw = (selected_schema or "").strip()
    table_hint = escape_braces(table_hint_raw)
    catalog_hint = escape_braces(catalog_hint_raw)
    schema_hint = escape_braces(schema_hint_raw)

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

    df_name_hint = (df_name or "").strip()
    df_label_source = df_name_hint or (table_hint_raw if table_hint_raw else "df_A")
    df_label = escape_braces(df_label_source)
    if isinstance(df_preview, pd.DataFrame) and not df_preview.empty:
        df_columns = escape_braces(_format_df_columns(df_preview))
        df_dtypes = escape_braces(_format_df_dtypes(df_preview))
        df_head_text = escape_braces(_df_head(df_preview))
        context_lines.append(
            "\nActive dataframe preview for SQL generation:\n"
            f"- Source: {df_label}\n"
            f"- Shape: {df_preview.shape[0]} rows × {df_preview.shape[1]} columns\n"
            "- Columns: {df_columns}\n"
            "dtypes:\n"
            "{df_dtypes}\n"
            "head():\n"
            "{df_head}\n\n"
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
        "Generate strictly read-only SELECT queries. Do NOT emit INSERT, UPDATE, DELETE, CREATE, DROP, ALTER, MERGE, TRUNCATE, or any non-SELECT statements under any circumstance.\n\n"
    )
    limit_value = get_default_sql_limit()
    context_lines.append(
        f"Always cap result sets with 'LIMIT {limit_value}' at the outermost query. If a LIMIT clause already exists, replace it with LIMIT {limit_value}.\n\n"
    )
    context_lines.append(
        "Whenever you create a COUNT() aggregation, alias the resulting column as `stat_count` "
        "(e.g., COUNT(*) AS stat_count). Do not invent alternative aliases for COUNT outputs.\n\n"
    )

    try:
        from core.learning_memory import get_successful_examples
        examples = get_successful_examples(intent="sql_execute")
        if not examples:
            examples = get_successful_examples(intent="sql")
        
        if examples:
            context_lines.append("Here are some previous successful queries as examples:\n")
            for ex in examples:
                user_q = escape_braces(ex["original_query"])
                ans_sql = escape_braces(ex["generated_sql"])
                context_lines.append(f"User: {user_q}\nSQL:\n{ans_sql}\n\n")
    except Exception:
        pass

    # JSON-based format instructions for structured chat agent
    context_lines.append(
        "You have access to the following tools:\n\n{tools}\n\n"
    )
    context_lines.append(
        "Use a json blob to specify a tool by providing an action key (tool name) and an action_input key (tool input).\n\n"
        'Valid "action" values: "Final Answer" or {tool_names}\n\n'
        "Provide only ONE action per $JSON_BLOB, as shown:\n\n"
        "```\n"
        "{{{{\n"
        '  "action": $TOOL_NAME,\n'
        '  "action_input": $INPUT\n'
        "}}}}\n"
        "```\n\n"
        "Follow this format:\n\n"
        "Question: input question to answer\n"
        "Thought: consider previous and subsequent steps\n"
        "Action:\n"
        "```\n"
        "$JSON_BLOB\n"
        "```\n"
        "Observation: action result\n"
        "... (repeat Thought/Action/Observation N times)\n"
        "Thought: I know what to respond\n"
        "Action:\n"
        "```\n"
        "{{{{\n"
        '  "action": "Final Answer",\n'
        '  "action_input": "SQL:\\n<single SQL statement, no markdown fences>"\n'
        "}}}}\n"
        "```\n\n"
        "Do NOT output any other fields such as Question:, Observation:, Explanation:, or Execution: unless the tool runner provides Observation: back to you.\n"
        "Do NOT include markdown fences in your SQL output.\n\n"
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "".join(context_lines)),
            MessagesPlaceholder("chat_history", optional=True),
            ("human", "{input}\n\n{agent_scratchpad}"),
        ]
    )

    tool_desc = render_text_description_and_args(tools)
    tool_names = ", ".join([tool.name for tool in tools])
    
    partial_vars = {
        "tools": tool_desc,
        "tool_names": tool_names,
    }
    if isinstance(df_preview, pd.DataFrame) and not df_preview.empty:
        partial_vars["df_columns"] = df_columns
        partial_vars["df_dtypes"] = df_dtypes
        partial_vars["df_head"] = df_head_text

    return prompt.partial(**partial_vars)


__all__ = ["build_react_prompt", "build_sql_prompt"]
