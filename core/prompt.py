from typing import Iterable

import pandas as pd
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import BaseTool, render_text_description


def _df_head(df: pd.DataFrame) -> str:
    return df.head().to_string(index=False)


def build_react_prompt(
    df_a: pd.DataFrame,
    df_b,
    tools: Iterable[BaseTool],
) -> ChatPromptTemplate:
    """Construct the ReAct system prompt with up-to-date dataframe heads."""
    head_a = _df_head(df_a)
    head_b = _df_head(df_b) if df_b is not None else "(df_B not loaded)"

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
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
                "Whenever the user requests a chart, plot, image, or visualisation, you MUST execute the relevant plotting tool (e.g., plot_distribution, plot_outliers, corr_heatmap) or run python_repl_ast with matplotlib code so the figure renders in Streamlit. Do not stop at textual descriptions.\n\n"
                "For tools that take a column name (e.g., anomaly_iqr), pass ONLY the raw column name (e.g., temperature), "
                "not 'column=temperature'. If you accidentally wrote 'column=...', immediately continue by outputting just the raw name.\n\n"
                "Call compare_on_keys with just the keys string (e.g., 'machineID,datetime') or as JSON {{\"keys\":\"machineID,datetime\"}}.\n"
                "Do NOT pass \"keys='...'\" literal unless JSON.\n\n"
                f"df_A.head():\n{head_a}\n\n"
                f"df_B.head():\n{head_b}\n",
            ),
            MessagesPlaceholder("chat_history", optional=True),
            ("human", "{input}"),
            ("assistant", "{agent_scratchpad}"),
        ]
    )

    tool_desc = render_text_description(tools)
    tool_names = ", ".join([tool.name for tool in tools])
    return prompt.partial(tools=tool_desc, tool_names=tool_names)


def build_sql_prompt(tools: Iterable[BaseTool]) -> ChatPromptTemplate:
    """Construct the SQL generation prompt."""
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a Databricks SQL expert.\n\n"
                "When you need schema info, you may call tools.\n\n"
                "For each step before the final answer, ALWAYS respond in EXACTLY this format:\n"
                "Thought: <brief reasoning>\n"
                "Action: <ONE tool name from {tool_names}>\n"
                "Action Input: <valid input for that tool, no backticks>\n\n"
                "When you are ready to answer with the final SQL (and NOT call a tool), respond in EXACTLY this format:\n"
                "Thought: I now know the final answer\n"
                "Final Answer: SQL:\n <single SQL statement only, no markdown fences, no explanation>\n\n"
                "Do NOT output any other fields such as Question:, Observation:, Explanation:, or Execution: unless the tool runner provides Observation: back to you.\n"
                "Do NOT include markdown fences.",
            ),
            MessagesPlaceholder("chat_history", optional=True),
            ("human", "{input}"),
            ("assistant", "{agent_scratchpad}"),
        ]
    )

    tool_desc = render_text_description(tools)
    tool_names = ", ".join([tool.name for tool in tools])
    return prompt.partial(tools=tool_desc, tool_names=tool_names)


__all__ = ["build_react_prompt", "build_sql_prompt"]
