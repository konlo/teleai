import os
from typing import Optional
from uuid import uuid4
import time
import datetime
import json

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
from core.sql_utils import (
    ensure_limit_clause,
    execute_sql_preview,
    infer_table_from_sql,
)
from core.sql_tools import build_sql_tools
from core.tools import build_tools
from ui.history import get_history
from ui.chat_log import (
    append_assistant_message,
    append_dataframe_preview_message,
    append_user_message,
    attach_figures_to_run,
    ensure_conversation_store,
    render_chat_history,
    render_conversation_log,
)
from ui.sidebar import render_sidebar
from ui.viz import render_visualizations
from utils.turn_logger import log_turn, build_turn_payload
from utils.session import (
    DEFAULT_SQL_LIMIT_MAX,
    DEFAULT_SQL_LIMIT_MIN,
    dataframe_signature,
    ensure_session_state,
    set_default_sql_limit,
)
from utils.prompt_help import (
    BASE_CHAT_PLACEHOLDER,
    CHAT_COMMAND_SPECS,
    COMMAND_EXAMPLE_LINES,
    DATA_LOADING_KEYWORDS,
    build_command_help_message,
    build_command_example_message,
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


def _get_dataframes(debug_mode: bool):
    ensure_session_state()
    render_sidebar(show_debug=debug_mode)
    df_a = st.session_state["df_A_data"]
    df_b = st.session_state["df_B_data"]
    df_init = st.session_state["df_init_data"]

    sig_a = dataframe_signature(df_a, st.session_state.get("csv_path", ""))
    sig_a_prev = st.session_state.get("df_A_signature", "")
    dataset_changed = sig_a != sig_a_prev
    st.session_state["df_A_signature"] = sig_a

    sig_b = dataframe_signature(df_b, st.session_state.get("csv_b_path", ""))
    sig_b_prev = st.session_state.get("df_B_signature", "")
    df_b_changed = sig_b != sig_b_prev
    st.session_state["df_B_signature"] = sig_b

    # df_init은 table loading 될 때 수정 되는 값임 
    return df_a, df_b, dataset_changed, df_b_changed, df_init


st.session_state.setdefault("debug_mode", False)
debug_mode = bool(st.session_state["debug_mode"])
st.session_state.setdefault("conversation_id", str(uuid4()))
st.session_state.setdefault("turn_counter", 0)
st.session_state.setdefault("user_id_hash", os.environ.get("USER_ID_HASH", "konlo.na"))
st.session_state.setdefault("pending_rerun", False)

def _next_turn_id() -> int:
    st.session_state["turn_counter"] = st.session_state.get("turn_counter", 0) + 1
    return st.session_state["turn_counter"]

df_A, df_B, dataset_changed, df_b_changed, df_init = _get_dataframes(debug_mode)
df_a_ready = isinstance(df_A, pd.DataFrame)
st.session_state.setdefault("log_has_content", False)
if not debug_mode:
    st.session_state["log_has_content"] = False

ensure_conversation_store()

sql_history = get_history("lc_msgs:sql")
eda_history = get_history("lc_msgs:eda")
if dataset_changed or df_b_changed:
    eda_history.clear()

def _render_csv_download_button(label: str, df: pd.DataFrame, dataset_name: str) -> None:
    if not isinstance(df, pd.DataFrame) or df.empty:
        return
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    file_name = f"{dataset_name or label}.csv"
    st.download_button(
        label=f"Download {label} CSV",
        data=csv_bytes,
        file_name=file_name,
        mime="text/csv",
        use_container_width=True,
    )


if df_a_ready and dataset_changed:
    if not st.session_state.pop("skip_next_df_a_preview", False):
        append_dataframe_preview_message(
            "df_A",
            df_A,
            "A",
            append_assistant_message,
            attach_figures_to_run,
        )
if isinstance(df_B, pd.DataFrame) and df_b_changed:
    if not st.session_state.pop("skip_next_df_b_preview", False):
        append_dataframe_preview_message(
            "df_B",
            df_B,
            "B",
            append_assistant_message,
            attach_figures_to_run,
        )


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
    df_preview=df_init if isinstance(df_init, pd.DataFrame) else None,
    df_name=st.session_state.get("df_A_name", "df_A"),
)
_sql_agent, sql_agent_with_history = build_agent(
    llm,
    sql_tools,
    sql_prompt,
    lambda session_id: sql_history,
)


log_placeholder = None
if debug_mode:
    with st.sidebar:
        st.markdown("#### 원본 LangChain 히스토리")
        with st.expander("SQL Builder History", expanded=False):
            render_chat_history("SQL Builder History", sql_history)
        with st.expander("EDA Analyst History", expanded=False):
            render_chat_history("EDA Analyst History", eda_history)

        st.markdown("#### ⚙️ 실시간 실행 로그")
        log_placeholder = st.container()
        if not st.session_state.get("log_has_content"):
            with log_placeholder.container():
                st.info("에이전트 실행 시 이 영역에서 로그가 표시됩니다.")


def _render_data_preview_section() -> None:
    if df_a_ready:
        with st.popover("📊 Data Preview"):
            st.write(
                f"**Loaded file for df_A:** `{st.session_state['df_A_name']}` (Shape: {df_A.shape})"
            )
            st.dataframe(df_A.head(10), width="stretch")
            _render_csv_download_button("df_A", df_A, st.session_state.get("df_A_name", "df_A"))
            if isinstance(df_B, pd.DataFrame):
                st.markdown(
                    f"**df_B Preview —** `{st.session_state['df_B_name']}` (Shape: {df_B.shape})"
                )
                st.dataframe(df_B.head(10), width="stretch")
                _render_csv_download_button(
                    "df_B", df_B, st.session_state.get("df_B_name", "df_B")
                )
    else:
        st.info(
            "df_A 데이터가 아직 로드되지 않았습니다. 왼쪽 Databricks Loader 또는 SQL Builder 에이전트를 사용해 데이터를 불러오세요."
        )

st.write("---")

conversation_placeholder = st.empty()

def _display_conversation_log() -> None:
    with conversation_placeholder.container():
        render_conversation_log()

chat_placeholder = BASE_CHAT_PLACEHOLDER

chat_input_key = "main_chat_input"
if chat_input_key not in st.session_state:
    st.session_state[chat_input_key] = ""

user_q = st.chat_input(chat_placeholder, key=chat_input_key)

if user_q:
    turn_id = _next_turn_id()
    turn_started = time.time()
    run_id = str(uuid4())
    st.session_state["active_run_id"] = run_id
    original_user_q = user_q
    append_user_message(run_id, original_user_q)
    _display_conversation_log()

    stripped_for_command = original_user_q.lstrip()
    lowered_for_command = stripped_for_command.lower()
    command_prefix = None
    agent_request = original_user_q
    handled_command = False
    rerun_required = False

    command_spec = next(
        (spec for spec in CHAT_COMMAND_SPECS if lowered_for_command.startswith(spec["trigger"])),
        None,
    )
    command_name = command_spec["name"] if command_spec else None
    assistant_response_for_log: Optional[str] = None
    sql_capture_for_log: str = ""
    sql_execution_status_for_log: Optional[str] = None
    result_row_count: Optional[int] = None
    result_schema_json: Optional[str] = None
    result_sample_json: Optional[str] = None
    intent_for_log: str = ""
    tools_used_for_log = []
    generated_python_for_log: str = ""
    python_status_for_log: Optional[str] = None
    python_error_for_log: str = ""
    python_output_summary_for_log: str = ""

    if command_name == "debug":
        handled_command = True
        trigger_len = len(command_spec["trigger"])
        debug_value = stripped_for_command[trigger_len:].strip().lower()
        current_state = bool(debug_mode)
        if debug_value in {"on", "off"}:
            new_state = debug_value == "on"
            st.session_state["debug_mode"] = new_state
            if new_state == current_state:
                ack_message = f"Debug 모드는 이미 {'ON' if new_state else 'OFF'} 상태입니다."
            else:
                ack_message = (
                    "Debug 모드를 활성화했습니다." if new_state else "Debug 모드를 비활성화했습니다."
                )
                rerun_required = True
        else:
            ack_message = f"{command_spec['usage']} 형태로 사용해주세요."
        append_assistant_message(run_id, ack_message, "Debug Mode")
        st.session_state["active_run_id"] = None
        if rerun_required:
            rerun_callable = getattr(st, "rerun", None) or getattr(
                st, "experimental_rerun", None
            )
            if callable(rerun_callable):
                rerun_callable()
        assistant_response_for_log = ack_message
        intent_for_log = "debug"
    elif command_name == "limit":
        handled_command = True
        trigger_len = len(command_spec["trigger"])
        limit_value_text = stripped_for_command[trigger_len:].strip()
        if not limit_value_text:
            ack_message = (
                f"사용법: {command_spec['usage']} (범위 {DEFAULT_SQL_LIMIT_MIN}~{DEFAULT_SQL_LIMIT_MAX})"
            )
        else:
            sanitized = limit_value_text
            if (
                len(sanitized) >= 2
                and sanitized[0] == sanitized[-1]
                and sanitized[0] in {"'", '"'}
            ):
                sanitized = sanitized[1:-1].strip()
            try:
                candidate = int(sanitized)
                new_limit = set_default_sql_limit(candidate)
            except (TypeError, ValueError) as exc:
                ack_message = f"LIMIT 변경 실패: {exc}"
            else:
                ack_message = (
                    f"SQL LIMIT 값을 {new_limit}으로 설정했습니다. "
                    "새로운 쿼리부터 적용됩니다."
                )
        append_assistant_message(run_id, ack_message, "Settings")
        st.session_state["active_run_id"] = None
        assistant_response_for_log = ack_message
        intent_for_log = "limit"
    elif command_name == "help":
        handled_command = True
        ack_message = build_command_help_message()
        append_assistant_message(run_id, ack_message, "Command Help")
        st.session_state["active_run_id"] = None
        assistant_response_for_log = ack_message
        intent_for_log = "help"
    elif command_name == "example":
        handled_command = True
        ack_message = build_command_example_message()
        append_assistant_message(run_id, ack_message, "Command Examples")
        st.session_state["active_run_id"] = None
        assistant_response_for_log = ack_message
        intent_for_log = "example"
    elif command_name == "sql":
        command_prefix = command_name
        trigger_len = len(command_spec["trigger"])
        agent_request = stripped_for_command[trigger_len:].lstrip()

    if (
        not handled_command
        and command_prefix is None
        and any(keyword in original_user_q for keyword in DATA_LOADING_KEYWORDS)
    ):
        command_prefix = "sql"

    normalized_original = original_user_q.strip().lower()
    if (
        not handled_command
        and command_prefix is None
        and normalized_original in {"실행", "수행", "run", "execute"}
    ):
        exec_success = execute_sql_preview(
            run_id=run_id,
            sql_text=st.session_state.get("last_sql_statement", ""),
            log_container=log_placeholder,
            show_logs=debug_mode,
            append_assistant_message=append_assistant_message,
            attach_figures_to_run=attach_figures_to_run,
        )
        handled_command = True
        sql_execution_status_for_log = "success" if exec_success else "fail"
        assistant_response_for_log = st.session_state.get("last_sql_error", "")
        intent_for_log = "sql_execute"
        tools_used_for_log.append("databricks_preview_sql")

    if not handled_command:
        auto_execute_sql = command_prefix == "sql"

        if command_prefix == "sql":
            agent_mode = "SQL Builder"
            tools_used_for_log.append("sql_builder_agent")
        else:
            agent_mode = "EDA Analyst"
            tools_used_for_log.append("eda_agent")
        st.session_state["last_agent_mode"] = agent_mode

        if not agent_request:
            if command_prefix == "sql":
                agent_request = "새로운 SQL 쿼리를 작성해줘."
            else:
                agent_request = original_user_q or "로드된 데이터프레임에 대해 EDA를 수행해줘."

        if agent_mode == "EDA Analyst" and not df_a_ready:
            error_msg = (
                "df_A 데이터가 없습니다. 먼저 SQL Builder 에이전트나 Databricks Loader로 데이터를 불러온 뒤 다시 시도하세요."
            )
            st.error(error_msg)
            append_assistant_message(run_id, error_msg, agent_mode)
            st.session_state["active_run_id"] = None
            assistant_response_for_log = error_msg
            intent_for_log = "eda"

        else:
            collector = SimpleCollectCallback()
            callbacks = [collector, StdOutCallbackHandler()]
            answer_container = st.container()

            if debug_mode and log_placeholder is not None:
                st.session_state["log_has_content"] = True
                log_placeholder.empty()
                with log_placeholder.container():
                    st.subheader("실시간 실행 로그")
                    log_stream_container = st.container()
                callbacks.insert(0, StreamlitCallbackHandler(log_stream_container))
            else:
                st.session_state["log_has_content"] = False

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
                else "Thinking with AI..."
            )

            with st.spinner(spinner_text):
                try:
                    result = agent_runner.invoke(
                        {"input": agent_request},
                        {
                            "callbacks": callbacks,
                            "configurable": {"session_id": session_id},
                        },
                    )
                except Exception as exc:
                    error_text = str(exc)
                    lower_error = error_text.lower()
                    if "serviceunavailable" in lower_error or "model is overloaded" in lower_error:
                        friendly = (
                            "AI 모델이 일시적으로 과부하 상태입니다. 잠시 후 다시 시도해주세요."
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
            intermediate_steps = result.get("intermediate_steps", [])
            for step in intermediate_steps:
                try:
                    action, observation = step
                except Exception:
                    continue
                tool_name = getattr(action, "tool", "") or ""
                if tool_name != "python_repl_ast":
                    continue
                tools_used_for_log.append("python_repl_ast")
                generated_python_for_log = getattr(action, "tool_input", "") or ""
                observation_text = (
                    observation if isinstance(observation, str) else str(observation)
                )
                python_output_summary_for_log = observation_text[:1000]
                lower_obs = observation_text.lower()
                if "traceback" in lower_obs or "error" in lower_obs:
                    python_status_for_log = "fail"
                    python_error_for_log = observation_text[:1000]
                else:
                    python_status_for_log = "success"
                    python_error_for_log = ""
                # keep last python call in the turn
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
                        enforced_sql = ensure_limit_clause(sql_capture)
                        if enforced_sql != sql_capture:
                            final_text = final_text.replace(sql_capture, enforced_sql, 1)
                        sql_capture = enforced_sql

                final_display = final_text
                st.caption(f"{agent_mode} 응답")
                st.write(final_display)
                append_assistant_message(run_id, final_display, agent_mode)
                assistant_response_for_log = final_display
                sql_capture_for_log = sql_capture
                intent_for_log = "sql" if agent_mode == "SQL Builder" else "eda"

                if agent_mode == "SQL Builder" and sql_capture:
                    st.session_state["last_sql_statement"] = sql_capture
                    st.session_state["last_sql_label"] = original_user_q.strip()[:80] or "SQL Query"
                    table_hint = (
                        st.session_state.get("databricks_table_input", "").strip()
                        or st.session_state.get("databricks_selected_table", "").strip()
                        or infer_table_from_sql(sql_capture)
                        or st.session_state.get("last_sql_table", "")
                    )
                    if table_hint:
                        st.session_state["last_sql_table"] = table_hint
                        st.session_state["databricks_selected_table"] = table_hint
                    if auto_execute_sql:
                        exec_success = execute_sql_preview(
                            run_id=run_id,
                            sql_text=sql_capture,
                            log_container=log_placeholder,
                            show_logs=debug_mode,
                            auto_trigger=True,
                            append_assistant_message=append_assistant_message,
                            attach_figures_to_run=attach_figures_to_run,
                        )
                        sql_execution_status_for_log = "success" if exec_success else "fail"
                        tools_used_for_log.append("databricks_preview_sql")

            if agent_mode == "EDA Analyst" and pytool_obj is not None:
                figure_payloads = render_visualizations(pytool_obj)
                attach_figures_to_run(run_id, figure_payloads)
        st.session_state["active_run_id"] = None

    _display_conversation_log()
    if assistant_response_for_log is not None:
        payload = build_turn_payload(
            llm=llm,
            conversation_id=st.session_state.get("conversation_id"),
            turn_id=turn_id,
            user_message=original_user_q,
            assistant_message=assistant_response_for_log,
            intent=intent_for_log,
            tools_used=tools_used_for_log,
            generated_sql=sql_capture_for_log or st.session_state.get("last_sql_statement", ""),
            sql_execution_status=sql_execution_status_for_log,
            sql_error_message=st.session_state.get("last_sql_error", ""),
            result_row_count=result_row_count,
            result_schema_json=result_schema_json,
            result_sample_json=result_sample_json,
            latency_ms=int((time.time() - turn_started) * 1000),
            df_latest=st.session_state.get("df_A_data"),
            generated_python=generated_python_for_log,
            python_execution_status=python_status_for_log,
            python_error_message=python_error_for_log,
            python_output_summary=python_output_summary_for_log,
        )
        log_turn(payload)

else:
    _display_conversation_log()

_render_data_preview_section()

if st.session_state.get("pending_rerun"):
    st.session_state["pending_rerun"] = False
    rerun_callable = getattr(st, "rerun", None) or getattr(st, "experimental_rerun", None)
    if callable(rerun_callable):
        rerun_callable()
