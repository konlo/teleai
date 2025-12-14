import time
from typing import Callable, Dict, List, Optional
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


def _default_log_state() -> Dict[str, Optional[str]]:
    """턴 단위 로깅에 사용할 기본 상태를 생성합니다."""
    return {
        "assistant_response_for_log": None,
        "sql_capture_for_log": "",
        "sql_execution_status_for_log": None,
        "result_row_count": None,
        "result_schema_json": None,
        "result_sample_json": None,
        "intent_for_log": "",
        "tools_used_for_log": [],
        "generated_python_for_log": "",
        "python_status_for_log": None,
        "python_error_for_log": "",
        "python_output_summary_for_log": "",
    }


def _handle_builtin_commands(
    *,
    command_name: Optional[str],
    command_spec,
    stripped_for_command: str,
    debug_mode: bool,
    run_id: str,
    turn_id: int,
) -> Dict[str, Optional[str]]:
    """debug/limit/help/example/SQL 프리픽스와 같은 빌트인 명령을 처리합니다."""

    result = {
        "handled_command": False,
        "command_prefix": None,
        "agent_request": stripped_for_command,
        "assistant_response_for_log": None,
        "intent_for_log": "",
    }

    def _ack_and_finish(message: str, intent: str) -> Dict[str, Optional[str]]:
        """공통 응답/로그 세팅 후 반환."""

        append_assistant_message(
            run_id, message, intent if intent != "debug" else "Debug Mode", turn_id=turn_id
        )
        st.session_state["active_run_id"] = None
        result["handled_command"] = True
        result["assistant_response_for_log"] = message
        result["intent_for_log"] = intent
        return result

    # debug on/off 처리
    if command_name == "debug":
        result["handled_command"] = True
        trigger_len = len(command_spec["trigger"])
        debug_value = stripped_for_command[trigger_len:].strip().lower()
        current_state = bool(debug_mode)
        rerun_required = False
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
        append_assistant_message(run_id, ack_message, "Debug Mode", turn_id=turn_id)
        st.session_state["active_run_id"] = None
        if rerun_required:
            rerun_callable = getattr(st, "rerun", None) or getattr(
                st, "experimental_rerun", None
            )
            if callable(rerun_callable):
                rerun_callable()
        result["assistant_response_for_log"] = ack_message
        result["intent_for_log"] = "debug"
        return result

    # SQL LIMIT 값 변경
    if command_name == "limit":
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
        return _ack_and_finish(ack_message, "limit")

    # 명령어 도움말
    if command_name == "help":
        ack_message = build_command_help_message()
        return _ack_and_finish(ack_message, "help")

    # 예시 안내
    if command_name == "example":
        ack_message = build_command_example_message()
        return _ack_and_finish(ack_message, "example")

    # sql 프리픽스는 뒤를 에이전트 입력으로 넘긴다
    if command_name == "sql":
        result["command_prefix"] = command_name
        trigger_len = len(command_spec["trigger"])
        result["agent_request"] = stripped_for_command[trigger_len:].lstrip()

    return result


def _maybe_execute_last_sql(
    *,
    run_id: str,
    log_placeholder,
    debug_mode: bool,
) -> Dict[str, Optional[str]]:
    """run/execute 입력 시 마지막 SQL을 바로 실행합니다."""

    exec_success = execute_sql_preview(
        run_id=run_id,
        sql_text=st.session_state.get("last_sql_statement", ""),
        log_container=log_placeholder,
        show_logs=debug_mode,
        append_assistant_message=append_assistant_message,
        attach_figures_to_run=attach_figures_to_run,
    )
    return {
        "handled_command": True,
        "sql_execution_status_for_log": "success" if exec_success else "fail",
        "assistant_response_for_log": st.session_state.get("last_sql_error", ""),
        "intent_for_log": "sql_execute",
        "tools_used_for_log": ["databricks_preview_sql"],
    }


def _build_callbacks(debug_mode: bool, log_placeholder):
    """Streamlit/LangChain 콜백 설정을 구성합니다."""

    collector = SimpleCollectCallback()
    callbacks = [collector, StdOutCallbackHandler()]

    log_stream_container = None
    if debug_mode and log_placeholder is not None:
        st.session_state["log_has_content"] = True
        log_placeholder.empty()
        with log_placeholder.container():
            st.subheader("실시간 실행 로그")
            log_stream_container = st.container()
        callbacks.insert(0, StreamlitCallbackHandler(log_stream_container))
    else:
        st.session_state["log_has_content"] = False

    return callbacks


def _invoke_agent_runner(agent_runner, agent_request, callbacks, session_id, spinner_text):
    """에이전트를 실행하고 예외를 사용자 친화적으로 처리합니다."""

    with st.spinner(spinner_text):
        try:
            return agent_runner.invoke(
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
                friendly = "AI 모델이 일시적으로 과부하 상태입니다. 잠시 후 다시 시도해주세요."
                st.warning(friendly)
                st.info("필요시 같은 요청을 조금 뒤에 다시 보내주세요.")
                return {"output": friendly}
            st.error(f"Agent 실행 중 오류: {error_text}")
            return {"output": f"Agent 실행 중 오류: {error_text}"}


def _capture_python_steps(intermediate_steps, tools_used_for_log: List[str], log_updates: Dict):
    """python_repl_ast 실행 결과를 로깅 상태에 반영합니다."""

    for step in intermediate_steps:
        try:
            action, observation = step
        except Exception:
            continue
        tool_name = getattr(action, "tool", "") or ""
        if tool_name != "python_repl_ast":
            continue
        tools_used_for_log.append("python_repl_ast")
        log_updates["generated_python_for_log"] = getattr(action, "tool_input", "") or ""
        observation_text = observation if isinstance(observation, str) else str(observation)
        log_updates["python_output_summary_for_log"] = observation_text[:1000]
        lower_obs = observation_text.lower()
        if "traceback" in lower_obs or "error" in lower_obs:
            log_updates["python_status_for_log"] = "fail"
            log_updates["python_error_for_log"] = observation_text[:1000]
        else:
            log_updates["python_status_for_log"] = "success"
            log_updates["python_error_for_log"] = ""


def _render_agent_answer(
    *,
    agent_mode: str,
    original_user_q: str,
    final_text: str,
    run_id: str,
    log_updates: Dict[str, Optional[str]],
    turn_id: int,
) -> str:
    """에이전트 응답을 렌더링하고 SQL 추출/보정 결과를 반환합니다."""

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

    st.caption(f"{agent_mode} 응답")
    st.write(final_text)
    append_assistant_message(run_id, final_text, agent_mode, turn_id=turn_id)
    log_updates["assistant_response_for_log"] = final_text
    log_updates["sql_capture_for_log"] = sql_capture
    log_updates["intent_for_log"] = "sql" if agent_mode == "SQL Builder" else "eda"
    return sql_capture


def _maybe_auto_execute_sql(
    *,
    auto_execute_sql: bool,
    sql_capture: str,
    run_id: str,
    debug_mode: bool,
    log_placeholder,
    log_updates: Dict[str, Optional[str]],
    tools_used_for_log: List[str],
):
    """SQL Builder가 생성한 SQL을 자동 실행할지 결정하고 실행합니다."""

    if auto_execute_sql and sql_capture:
        exec_success = execute_sql_preview(
            run_id=run_id,
            sql_text=sql_capture,
            log_container=log_placeholder,
            show_logs=debug_mode,
            auto_trigger=True,
            append_assistant_message=append_assistant_message,
            attach_figures_to_run=attach_figures_to_run,
        )
        log_updates["sql_execution_status_for_log"] = "success" if exec_success else "fail"
        tools_used_for_log.append("databricks_preview_sql")


def _run_agent_interaction(
    *,
    agent_mode: str,
    agent_request: str,
    debug_mode: bool,
    df_a_ready: bool,
    run_id: str,
    original_user_q: str,
    log_placeholder,
    sql_agent_with_history,
    eda_agent_with_history,
    pytool_obj,
    llm,
    tools_used_for_log: List[str],
    turn_id: int,
) -> Dict[str, Optional[str]]:
    """SQL/EDA 에이전트를 실행하고 UI/로그 업데이트를 수행합니다."""

    log_updates = _default_log_state()
    auto_execute_sql = agent_mode == "SQL Builder" and st.session_state.get(
        "command_prefix"
    ) == "sql"

    # EDA 모드인데 df_A가 없으면 바로 에러 반환
    if agent_mode == "EDA Analyst" and not df_a_ready:
        error_msg = (
            "df_A 데이터가 없습니다. 먼저 SQL Builder 에이전트나 Databricks Loader로 데이터를 불러온 뒤 다시 시도하세요."
        )
        st.error(error_msg)
        append_assistant_message(run_id, error_msg, agent_mode, turn_id=turn_id)
        st.session_state["active_run_id"] = None
        log_updates["assistant_response_for_log"] = error_msg
        log_updates["intent_for_log"] = "eda"
        return log_updates

    answer_container = st.container()
    callbacks = _build_callbacks(debug_mode, log_placeholder)

    agent_runner = sql_agent_with_history if agent_mode == "SQL Builder" else eda_agent_with_history
    session_id = (
        "databricks_sql_builder" if agent_mode == "SQL Builder" else "two_csv_compare_and_ssd_eda"
    )
    spinner_text = (
        "Databricks SQL을 구상 중입니다..." if agent_mode == "SQL Builder" else "Thinking with AI..."
    )

    result = _invoke_agent_runner(
        agent_runner,
        agent_request,
        callbacks,
        session_id,
        spinner_text,
    )

    st.success("Done.")
    final = result.get("output", "Agent가 최종 답변을 생성하지 못했습니다.")
    intermediate_steps = result.get("intermediate_steps", [])
    _capture_python_steps(intermediate_steps, tools_used_for_log, log_updates)

    with answer_container:
        st.subheader("Answer")
        final_text = final if isinstance(final, str) else str(final)
        sql_capture = _render_agent_answer(
            agent_mode=agent_mode,
            original_user_q=original_user_q,
            final_text=final_text,
            run_id=run_id,
            log_updates=log_updates,
            turn_id=turn_id,
        )

        # SQL Builder일 때 추출한 SQL 메모리/자동 실행 처리
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
            _maybe_auto_execute_sql(
                auto_execute_sql=auto_execute_sql,
                sql_capture=sql_capture,
                run_id=run_id,
                debug_mode=debug_mode,
                log_placeholder=log_placeholder,
                log_updates=log_updates,
                tools_used_for_log=tools_used_for_log,
            )

    if agent_mode == "EDA Analyst" and pytool_obj is not None:
        figure_payloads = render_visualizations(pytool_obj)
        attach_figures_to_run(run_id, figure_payloads)

    st.session_state["active_run_id"] = None
    return log_updates


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
    display_conversation_log: Callable[[bool], None],
) -> None:
    """단일 턴 처리: 명령 파싱 → SQL 실행 여부 판단 → 에이전트 호출 → 로그 기록."""

    turn_id = next_turn_id_fn()
    st.session_state["turn_id"] = turn_id
    turn_started = time.time()
    run_id = str(uuid4())
    st.session_state["active_run_id"] = run_id
    original_user_q = user_q
    append_user_message(run_id, original_user_q)
    display_conversation_log(show_ratings=False)

    stripped_for_command = original_user_q.lstrip()
    lowered_for_command = stripped_for_command.lower()
    # 프리픽스 기반 명령 감지
    command_spec = next(
        (spec for spec in CHAT_COMMAND_SPECS if lowered_for_command.startswith(spec["trigger"])),
        None,
    )
    command_name = command_spec["name"] if command_spec else None
    command_result = _handle_builtin_commands(
        command_name=command_name,
        command_spec=command_spec,
        stripped_for_command=stripped_for_command,
        debug_mode=debug_mode,
        run_id=run_id,
        turn_id=turn_id,
    )

    log_state = _default_log_state()
    tools_used_for_log: List[str] = []

    if command_result["assistant_response_for_log"]:
        log_state["assistant_response_for_log"] = command_result["assistant_response_for_log"]
    if command_result["intent_for_log"]:
        log_state["intent_for_log"] = command_result["intent_for_log"]

    handled_command = command_result["handled_command"]
    command_prefix = command_result["command_prefix"]
    agent_request = command_result["agent_request"] or original_user_q

    # 데이터 로딩 키워드가 있으면 SQL 빌더로 강제 전환
    if (
        not handled_command
        and command_prefix is None
        and any(keyword in original_user_q for keyword in DATA_LOADING_KEYWORDS)
    ):
        command_prefix = "sql"

    normalized_original = original_user_q.strip().lower()
    # 실행/수행/run/execute 입력 시 마지막 SQL 바로 실행
    if (
        not handled_command
        and command_prefix is None
        and normalized_original in {"실행", "수행", "run", "execute"}
    ):
        execute_result = _maybe_execute_last_sql(
            run_id=run_id,
            log_placeholder=log_placeholder,
            debug_mode=debug_mode,
        )
        handled_command = True
        log_state["sql_execution_status_for_log"] = execute_result["sql_execution_status_for_log"]
        log_state["assistant_response_for_log"] = execute_result["assistant_response_for_log"]
        log_state["intent_for_log"] = execute_result["intent_for_log"]
        tools_used_for_log.extend(execute_result.get("tools_used_for_log", []))

    # 빌트인 명령 외의 일반 질의 처리
    if not handled_command:
        if command_prefix == "sql":
            agent_mode = "SQL Builder"
            tools_used_for_log.append("sql_builder_agent")
            st.session_state["command_prefix"] = "sql"
        else:
            agent_mode = "EDA Analyst"
            tools_used_for_log.append("eda_agent")
            st.session_state["command_prefix"] = None

        st.session_state["last_agent_mode"] = agent_mode

        if not agent_request:
            agent_request = (
                "새로운 SQL 쿼리를 작성해줘." if command_prefix == "sql" else original_user_q
            )

        agent_updates = _run_agent_interaction(
            agent_mode=agent_mode,
            agent_request=agent_request,
            debug_mode=debug_mode,
            df_a_ready=df_a_ready,
            run_id=run_id,
            original_user_q=original_user_q,
            log_placeholder=log_placeholder,
            sql_agent_with_history=sql_agent_with_history,
            eda_agent_with_history=eda_agent_with_history,
            pytool_obj=pytool_obj,
            llm=llm,
            tools_used_for_log=tools_used_for_log,
            turn_id=turn_id,
        )
        for key, value in agent_updates.items():
            if key == "tools_used_for_log":
                continue
            if value:
                log_state[key] = value

    display_conversation_log()
    if log_state["assistant_response_for_log"] is not None:
        payload = build_turn_payload(
            llm=llm,
            conversation_id=st.session_state.get("conversation_id"),
            turn_id=turn_id,
            user_message=original_user_q,
            assistant_message=log_state["assistant_response_for_log"],
            intent=log_state["intent_for_log"],
            tools_used=tools_used_for_log,
            generated_sql=log_state["sql_capture_for_log"]
            or st.session_state.get("last_sql_statement", ""),
            sql_execution_status=log_state["sql_execution_status_for_log"],
            sql_error_message=st.session_state.get("last_sql_error", ""),
            result_row_count=log_state["result_row_count"],
            result_schema_json=log_state["result_schema_json"],
            result_sample_json=log_state["result_sample_json"],
            latency_ms=int((time.time() - turn_started) * 1000),
            df_latest=st.session_state.get("df_A_data"),
            generated_python=log_state["generated_python_for_log"],
            python_execution_status=log_state["python_status_for_log"],
            python_error_message=log_state["python_error_for_log"],
            python_output_summary=log_state["python_output_summary_for_log"],
        )
        log_turn(payload)


__all__ = ["handle_user_query"]
