import base64
import re
from typing import Any, Dict, List
from uuid import uuid4

import streamlit as st
import pandas as pd
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler

from core.agent import (
    SimpleCollectCallback,
    StdOutCallbackHandler,
    build_agent,
)
from core.llm import load_llm
from core.prompt import build_react_prompt, build_sql_prompt
from core.sql_tools import build_sql_tools
from core.tools import build_tools
from ui.history import get_history
from ui.sidebar import render_sidebar
from ui.viz import render_visualizations
from utils.session import (
    dataframe_signature,
    ensure_session_state,
    load_preview_from_databricks_query,
)


st.set_page_config(
    page_title="Telemetry Chatbot Telly",
    page_icon="✨",
    layout="wide",
)

st.markdown(
    """
    <style>
    :root {
        font-size: 16px;
    }

    html,
    body,
    [data-testid="stAppViewContainer"] {
        font-size: 16px;
    }

    .block-container {
        max-width: 900px;
        margin: 0 auto;
        width: 100%;
        padding-left: 1.5rem;
        padding-right: 1.5rem;
    }

    [data-testid="stChatInput"] {
        width: 100%;
        max-width: 1000px;
        margin: 0 auto;
        padding-left: 1.5rem;
        padding-right: 1.5rem;
    }

    [data-testid="stChatInput"] > div {
        width: 100%;
        min-height: 5rem;
    }

    [data-testid="stChatInputTextArea"] {
        min-height: 5rem;
    }

    @media (max-width: 1200px) {
        .block-container,
        [data-testid="stChatInput"] {
            padding-left: 1rem;
            padding-right: 1rem;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("✨ Telemetry Chatbot Telly")
st.caption("두 CSV 비교 + 이상점 중심 EDA(원클릭) + SSD Telemetry 유틸")


def _get_dataframes():
    ensure_session_state()
    render_sidebar()
    df_a = st.session_state["df_A_data"]
    df_b = st.session_state["df_B_data"]

    sig_a = dataframe_signature(df_a, st.session_state.get("csv_path", ""))
    sig_a_prev = st.session_state.get("df_A_signature", "")
    dataset_changed = sig_a != sig_a_prev
    st.session_state["df_A_signature"] = sig_a

    sig_b = dataframe_signature(df_b, st.session_state.get("csv_b_path", ""))
    sig_b_prev = st.session_state.get("df_B_signature", "")
    df_b_changed = sig_b != sig_b_prev
    st.session_state["df_B_signature"] = sig_b

    return df_a, df_b, dataset_changed, df_b_changed


def _ensure_conversation_store() -> None:
    st.session_state.setdefault("conversation_log", [])
    st.session_state.setdefault("active_run_id", None)


df_A, df_B, dataset_changed, df_b_changed = _get_dataframes()
df_a_ready = isinstance(df_A, pd.DataFrame)
st.session_state.setdefault("log_has_content", False)

_ensure_conversation_store()

sql_history = get_history("lc_msgs:sql")
eda_history = get_history("lc_msgs:eda")
if dataset_changed or df_b_changed:
    eda_history.clear()

def _render_chat_history(title: str, history) -> None:
    st.markdown(f"#### {title}")
    messages = getattr(history, "messages", []) or []
    if not messages:
        st.info("대화 기록이 없습니다.")
        return
    for msg in messages:
        role = getattr(msg, "type", "assistant")
        if role == "human":
            streamlit_role = "user"
        elif role == "ai":
            streamlit_role = "assistant"
        else:
            streamlit_role = role or "assistant"
        content = msg.content if isinstance(msg.content, str) else str(msg.content)
        with st.chat_message(streamlit_role):
            st.markdown(content)
def _append_user_message(run_id: str, content: str) -> None:
    st.session_state["conversation_log"].append(
        {"run_id": run_id, "role": "user", "content": content}
    )


def _append_assistant_message(run_id: str, content: str, mode: str) -> None:
    st.session_state["conversation_log"].append(
        {
            "run_id": run_id,
            "role": "assistant",
            "mode": mode,
            "content": content,
            "figures": [],
            "figures_attached": False,
        }
    )


def _attach_figures_to_run(run_id: str, figures: List[Dict[str, Any]]) -> None:
    if not run_id or not figures:
        return
    log = st.session_state.get("conversation_log", [])
    for entry in reversed(log):
        if entry.get("run_id") == run_id and entry.get("role") == "assistant":
            if entry.get("figures_attached"):
                return
            entry.setdefault("figures", [])
            entry["figures"].extend(figures)
            entry["figures_attached"] = True
            break


def _append_dataframe_preview_message(label: str, df: pd.DataFrame, key: str) -> None:
    if not isinstance(df, pd.DataFrame) or df.empty:
        return
    preview_df = df.head(10)
    if preview_df.empty:
        return
    dataset_name_key = "df_A_name" if key == "A" else "df_B_name"
    dataset_name = st.session_state.get(dataset_name_key, label)
    message = (
        f"**{label} Preview:** `{dataset_name}` (Shape: {df.shape})"
    )
    run_id = f"preview-{key}-{uuid4()}"
    _append_assistant_message(run_id, message, "Data Preview")
    _attach_figures_to_run(
        run_id,
        [
            {
                "kind": "dataframe",
                "title": f"{label} Preview",
                "data": preview_df,
            }
        ],
    )


if df_a_ready and dataset_changed:
    if not st.session_state.pop("skip_next_df_a_preview", False):
        _append_dataframe_preview_message("df_A", df_A, "A")
if isinstance(df_B, pd.DataFrame) and df_b_changed:
    if not st.session_state.pop("skip_next_df_b_preview", False):
        _append_dataframe_preview_message("df_B", df_B, "B")


def _render_conversation_log(show_header: bool = True) -> None:
    if show_header:
        st.markdown("#### 대화 기록")
    log = st.session_state.get("conversation_log", [])
    if not log:
        st.info("대화 기록이 없습니다.")
        return
    for entry in log:
        role = entry.get("role", "assistant")
        streamlit_role = "assistant" if role == "assistant" else "user"
        with st.chat_message(streamlit_role):
            mode = entry.get("mode")
            if mode and role == "assistant":
                st.caption(mode)
            content = entry.get("content", "")
            if content:
                st.markdown(content)
            for fig in entry.get("figures", []):
                title = fig.get("title")
                if title:
                    st.markdown(f"**{title}**")
                kind = fig.get("kind")
                if kind == "bar_chart":
                    st.bar_chart(fig.get("data"), use_container_width=True)
                elif kind == "line_chart":
                    st.line_chart(fig.get("data"), use_container_width=True)
                elif kind == "dataframe":
                    st.dataframe(fig.get("data"), use_container_width=True)
                elif kind == "json":
                    st.json(fig.get("data"))
                elif kind == "matplotlib":
                    image_b64 = fig.get("image")
                    if image_b64:
                        st.image(base64.b64decode(image_b64), use_column_width=True)


llm = load_llm()

pytool_obj = None
eda_agent_with_history = None
if df_a_ready:
    pytool_obj, eda_tools = build_tools(df_A, df_B)
    eda_prompt = build_react_prompt(df_A, df_B, eda_tools)
    _eda_agent, eda_agent_with_history = build_agent(
        llm,
        eda_tools,
        eda_prompt,
        lambda session_id: eda_history,
    )

sql_tools = build_sql_tools()
sql_prompt = build_sql_prompt(
    sql_tools,
    selected_table=st.session_state.get("databricks_selected_table", ""),
    selected_catalog=st.session_state.get("databricks_selected_catalog", ""),
    selected_schema=st.session_state.get("databricks_selected_schema", ""),
)
_sql_agent, sql_agent_with_history = build_agent(
    llm,
    sql_tools,
    sql_prompt,
    lambda session_id: sql_history,
)


BASE_CHAT_PLACEHOLDER = (
    "SQL) 예: sales_transactions에서 최근 7일간 매출 합계를 위한 SQL 작성해줘 / "
    "EDA) 예: auto_outlier_eda() / plot_outliers('temperature') / compare_on_keys('machineID,datetime')"
)


def _infer_agent(user_message: str) -> str:
    text = (user_message or "").lower()
    last_mode = st.session_state.get("last_agent_mode", "SQL Builder")

    eda_keywords = [
        "eda",
        "이상점",
        "시각화",
        "plot",
        "distribution",
        "auto_outlier",
        "anomaly",
        "stl",
        "cohort",
        "compare_on_keys",
        "rolling_stats",
        "mismatch_report",
        "describe_",
        "heatmap",
    ]
    sql_keywords = [
        "sql",
        "쿼리",
        "select",
        " from ",
        "join",
        "where",
        "catalog",
        "schema",
        "table",
        "run",
        "execute",
        "실행",
        "수행",
        "databricks",
        "조회",
        "load",
    ]

    if any(keyword in text for keyword in eda_keywords):
        return "EDA Analyst"
    if any(keyword in text for keyword in sql_keywords):
        return "SQL Builder"
    if not df_a_ready:
        return "SQL Builder"
    return "EDA Analyst"


def _infer_table_from_sql(sql: str) -> str:
    text = (sql or "").strip()
    if not text:
        return ""
    lowered = text.lower()
    marker = " from "
    idx = lowered.find(marker)
    if idx == -1:
        if lowered.startswith("from "):
            idx = 0
        else:
            return ""
    idx += len(marker)
    remainder = text[idx:].strip()
    if not remainder:
        return ""
    candidate = remainder.split()[0]
    candidate = candidate.rstrip(";,)")
    return candidate.strip()


def _ensure_limit_clause(sql: str, limit: int = 2000) -> str:
    text = (sql or "").strip()
    if not text:
        return sql

    semicolon = "" if not text.endswith(";") else ";"
    body = text[:-1].rstrip() if semicolon else text

    pattern = re.compile(r"(?is)\blimit\s+\d+(\s+offset\s+\d+)?\s*$")
    match = pattern.search(body)
    if match:
        prefix = body[: match.start()].rstrip()
        offset_part = (match.group(1) or "").upper()
        body = f"{prefix} LIMIT {limit}{offset_part}"
    else:
        body = f"{body.rstrip()} LIMIT {limit}"

    return f"{body}{semicolon}"


with st.sidebar:
    st.markdown("#### 원본 LangChain 히스토리")
    with st.expander("SQL Builder History", expanded=False):
        _render_chat_history("SQL Builder History", sql_history)
    with st.expander("EDA Analyst History", expanded=False):
        _render_chat_history("EDA Analyst History", eda_history)

    st.markdown("#### ⚙️ 실시간 실행 로그")
    log_placeholder = st.container()
    if not st.session_state.get("log_has_content"):
        with log_placeholder.container():
            st.info("에이전트 실행 시 이 영역에서 로그가 표시됩니다.")


def _execute_sql_preview(
    run_id: str,
    sql_text: str,
    *,
    log_container,
    auto_trigger: bool = False,
) -> bool:
    sql_to_run = (sql_text or "").strip()
    if not sql_to_run:
        warning_msg = "실행할 SQL이 없습니다. 먼저 SQL Builder로 쿼리를 생성해주세요."
        st.warning(warning_msg)
        _append_assistant_message(run_id, warning_msg, "SQL Execution")
        st.session_state["active_run_id"] = None
        return False

    st.session_state["last_sql_statement"] = sql_to_run
    st.session_state["last_agent_mode"] = "SQL Builder"
    st.session_state["log_has_content"] = True
    log_container.empty()
    with log_container.container():
        st.subheader("실시간 실행 로그")
        status_msg = (
            "SQL Builder가 생성한 쿼리를 Databricks에서 실행합니다."
            if auto_trigger
            else "SQL Builder의 마지막 쿼리를 Databricks에서 실행합니다."
        )
        st.write(status_msg)

    cfg = st.session_state.get("databricks_config", {})
    catalog = cfg.get("catalog") or "hive_metastore"
    schema = cfg.get("schema") or "default"
    cfg["catalog"] = catalog
    cfg["schema"] = schema
    st.session_state["databricks_config"] = cfg
    st.session_state.setdefault("databricks_selected_catalog", catalog)
    st.session_state.setdefault("databricks_selected_schema", schema)

    table_name_input = st.session_state.get("databricks_table_input", "").strip()
    selected_table = st.session_state.get("databricks_selected_table", "").strip()
    table_name_inferred = _infer_table_from_sql(sql_to_run)
    table_name = (
        table_name_input
        or selected_table
        or table_name_inferred
        or st.session_state.get("last_sql_table", "")
    )
    if not table_name:
        warning_msg = (
            "실행할 테이블을 결정할 수 없습니다. SQL Builder에서 사용할 테이블을 지정하거나 Sidebar에서 테이블을 선택해주세요."
        )
        st.warning(warning_msg)
        _append_assistant_message(run_id, warning_msg, "SQL Execution")
        st.session_state["active_run_id"] = None
        return False

    answer_container = st.container()
    with st.spinner("Databricks SQL 실행 중..."):
        success, message = load_preview_from_databricks_query(
            table_name,
            query=sql_to_run,
            target="A",
            limit=10,
        )
    preview_payloads: List[Dict[str, Any]] = []
    with answer_container:
        st.subheader("Answer")
        if success:
            st.success(message)
        else:
            st.error(message)

    if success:
        st.session_state["last_agent_mode"] = "EDA Analyst"
        st.session_state["last_sql_table"] = table_name
        st.session_state["databricks_table_input"] = table_name
        st.session_state["databricks_selected_table"] = table_name
        st.session_state["skip_next_df_a_preview"] = True
        df_latest = st.session_state.get("df_A_data")
        if isinstance(df_latest, pd.DataFrame) and not df_latest.empty:
            preview_payloads.append(
                {
                    "kind": "dataframe",
                    "title": f"df_A Preview — {st.session_state.get('df_A_name', 'df_A')}",
                    "data": df_latest.head(10),
                }
            )

    mode_label = "SQL Execution" if auto_trigger else "SQL Builder"
    _append_assistant_message(run_id, message, mode_label)
    if preview_payloads:
        _attach_figures_to_run(run_id, preview_payloads)
    st.session_state["active_run_id"] = None
    if success:
        rerun_callable = getattr(st, "rerun", None) or getattr(st, "experimental_rerun", None)
        if callable(rerun_callable):
            rerun_callable()
    return success

st.write("---")

_render_conversation_log()

if df_a_ready:
    with st.popover("📊 Data Preview"):
        st.write(
            f"**Loaded file for df_A:** `{st.session_state['df_A_name']}` (Shape: {df_A.shape})"
        )
        st.dataframe(df_A.head(10), width="stretch")
        if isinstance(df_B, pd.DataFrame):
            st.markdown(
                f"**df_B Preview —** `{st.session_state['df_B_name']}` (Shape: {df_B.shape})"
            )
            st.dataframe(df_B.head(10), width="stretch")
else:
    st.info(
        "df_A 데이터가 아직 로드되지 않았습니다. 왼쪽 Databricks Loader 또는 SQL Builder 에이전트를 사용해 데이터를 불러오세요."
    )

chat_placeholder = BASE_CHAT_PLACEHOLDER

chat_input_key = "main_chat_input"
if chat_input_key not in st.session_state:
    st.session_state[chat_input_key] = ""

prefill_value = st.session_state.get("chat_input_prefill", "")
if prefill_value:
    st.session_state[chat_input_key] = prefill_value
    st.session_state["chat_input_prefill"] = ""

user_q = st.chat_input(chat_placeholder, key=chat_input_key)

if user_q:
    run_id = str(uuid4())
    st.session_state["active_run_id"] = run_id
    original_user_q = user_q
    _append_user_message(run_id, original_user_q)

    stripped_for_command = original_user_q.lstrip()
    lowered_for_command = stripped_for_command.lower()
    command_prefix = None
    agent_request = original_user_q

    if lowered_for_command.startswith("%sql"):
        command_prefix = "sql"
        agent_request = stripped_for_command[4:].lstrip()
    elif lowered_for_command.startswith("%eda"):
        command_prefix = "eda"
        agent_request = stripped_for_command[4:].lstrip()

    if command_prefix == "eda":
        st.session_state["chat_input_prefill"] = "%eda "
    else:
        st.session_state["chat_input_prefill"] = ""

    normalized_original = original_user_q.strip().lower()
    if command_prefix is None and normalized_original in {"실행", "수행", "run", "execute"}:
        _execute_sql_preview(
            run_id,
            st.session_state.get("last_sql_statement", ""),
            log_container=log_placeholder,
        )
        st.stop()

    auto_execute_sql = command_prefix == "sql"

    if command_prefix == "sql":
        agent_mode = "SQL Builder"
    elif command_prefix == "eda":
        agent_mode = "EDA Analyst"
    else:
        agent_mode = _infer_agent(original_user_q)
    st.session_state["last_agent_mode"] = agent_mode

    if not agent_request:
        if command_prefix == "sql":
            agent_request = "새로운 SQL 쿼리를 작성해줘."
        elif command_prefix == "eda":
            agent_request = "로드된 데이터프레임에 대해 기본 EDA를 수행해줘."
        else:
            agent_request = original_user_q

    if agent_mode == "EDA Analyst" and not df_a_ready:
        error_msg = (
            "df_A 데이터가 없습니다. 먼저 SQL Builder 에이전트나 Databricks Loader로 데이터를 불러온 뒤 다시 시도하세요."
        )
        st.error(error_msg)
        _append_assistant_message(run_id, error_msg, agent_mode)
        st.session_state["active_run_id"] = None
    else:
        st.session_state["log_has_content"] = True
        log_placeholder.empty()
        with log_placeholder.container():
            st.subheader("실시간 실행 로그")
            log_stream_container = st.container()
        st_cb = StreamlitCallbackHandler(log_stream_container)
        collector = SimpleCollectCallback()
        answer_container = st.container()

        agent_runner = (
            sql_agent_with_history if agent_mode == "SQL Builder" else eda_agent_with_history
        )
        session_id = (
            "databricks_sql_builder"
            if agent_mode == "SQL Builder"
            else "two_csv_compare_and_ssd_eda"
        )
        spinner_text = (
            "Databricks SQL을 구상 중입니다..."
            if agent_mode == "SQL Builder"
            else "Thinking with Gemini..."
        )

        with st.spinner(spinner_text):
            try:
                result = agent_runner.invoke(
                    {"input": agent_request},
                    {
                        "callbacks": [st_cb, collector, StdOutCallbackHandler()],
                        "configurable": {"session_id": session_id},
                    },
                )
            except Exception as exc:
                error_text = str(exc)
                lower_error = error_text.lower()
                if "serviceunavailable" in lower_error or "model is overloaded" in lower_error:
                    friendly = (
                        "Gemini 모델이 일시적으로 과부하 상태입니다. 잠시 후 다시 시도해주세요."
                    )
                    st.warning(friendly)
                    st.info("필요시 같은 요청을 조금 뒤에 다시 보내주세요.")
                    result = {"output": friendly}
                else:
                    st.error(f"Agent 실행 중 오류: {error_text}")
                    result = {"output": f"Agent 실행 중 오류: {error_text}"}

        st.success("Done.")
        final = result.get(
            "output", "Agent가 최종 답변을 생성하지 못했습니다."
        )
        with answer_container:
            st.subheader("Answer")
            final_text = final if isinstance(final, str) else str(final)
            sql_capture = ""
            if agent_mode == "SQL Builder" and "SQL:" in final_text:
                tail = final_text.split("SQL:", 1)[1]
                if "Explanation:" in tail:
                    sql_capture = tail.split("Explanation:", 1)[0].strip()
                elif "Execution:" in tail:
                    sql_capture = tail.split("Execution:", 1)[0].strip()
                else:
                    sql_capture = tail.strip()
                if sql_capture:
                    enforced_sql = _ensure_limit_clause(sql_capture)
                    if enforced_sql != sql_capture:
                        final_text = final_text.replace(sql_capture, enforced_sql, 1)
                    sql_capture = enforced_sql

            final_display = final_text
            if final_text.strip():
                try:
                    translation_prompt = (
                        "다음 분석 결과를 자연스럽고 간결한 한국어로 설명해줘.\n\n"
                        f"{final_text}"
                    )
                    translated_msg = llm.invoke(translation_prompt)
                    translated_text = getattr(translated_msg, "content", None)
                    if translated_text:
                        final_display = translated_text
                except Exception as exc:
                    st.warning(f"한국어 번역 중 오류가 발생했습니다: {exc}")
            st.caption(f"{agent_mode} 응답")
            st.write(final_display)
            _append_assistant_message(run_id, final_display, agent_mode)

            if agent_mode == "SQL Builder" and sql_capture:
                st.session_state["last_sql_statement"] = sql_capture
                st.session_state["last_sql_label"] = original_user_q.strip()[:80] or "SQL Query"
                table_hint = (
                    st.session_state.get("databricks_table_input", "").strip()
                    or st.session_state.get("databricks_selected_table", "").strip()
                    or _infer_table_from_sql(sql_capture)
                    or st.session_state.get("last_sql_table", "")
                )
                if table_hint:
                    st.session_state["last_sql_table"] = table_hint
                    st.session_state["databricks_selected_table"] = table_hint
                if auto_execute_sql:
                    _execute_sql_preview(
                        run_id,
                        sql_capture,
                        log_container=log_placeholder,
                        auto_trigger=True,
                    )

            if agent_mode == "EDA Analyst" and pytool_obj is not None:
                figure_payloads = render_visualizations(pytool_obj)
                _attach_figures_to_run(run_id, figure_payloads)
        st.session_state["active_run_id"] = None
