import time
from typing import Callable, Optional
from uuid import uuid4

import streamlit as st
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler

from core.agent import SimpleCollectCallback, StdOutCallbackHandler
from core.sql_utils import ensure_limit_clause, execute_sql_preview, infer_table_from_sql
from ui.chat_log import (
    append_assistant_message,
    append_user_message,
    attach_figures_to_run,
)
from ui.viz import render_visualizations
from utils.prompt_help import (
    CHAT_COMMAND_SPECS,
    DATA_LOADING_KEYWORDS,
    build_command_example_message,
    build_command_help_message,
)
from utils.session import (
    DEFAULT_SQL_LIMIT_MAX,
    DEFAULT_SQL_LIMIT_MIN,
    set_default_sql_limit,
)
from utils.turn_logger import build_turn_payload, log_turn


def handle_user_query(
    user_q: str,
    *,
    debug_mode: bool,
    df_a_ready: bool,
    log_placeholder,
    sql_agent_with_history,
    eda_agent_with_history,
    pytool_obj,
    llm,
    next_turn_id_fn: Callable[[], int],
    display_conversation_log: Callable[[], None],
) -> None:
    """Handle a single user query turn including commands, agent calls, and logging."""

    turn_id = next_turn_id_fn()
    turn_started = time.time()
    run_id = str(uuid4())
    st.session_state["active_run_id"] = run_id
    original_user_q = user_q
    append_user_message(run_id, original_user_q)
    display_conversation_log()

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
                    st.session_state["last_sql_label"] = (
                        original_user_q.strip()[:80] or "SQL Query"
                    )
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

    display_conversation_log()
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


__all__ = ["handle_user_query"]
