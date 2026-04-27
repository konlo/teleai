import time
import json
from dataclasses import replace
from typing import Callable, Dict, List, Optional
from uuid import uuid4

import pandas as pd
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
from utils.agent_output import detect_agent_parser_loop
from utils.agent_routing import resolve_forced_agent_mode, should_force_sql_from_keywords
from utils.chatbot_plan import (
    build_controlled_plan,
    build_sql_from_plan,
    controlled_plan_to_dict,
    select_visualization_config,
)
from utils.controlled_visualization import (
    collect_matplotlib_figure_payloads,
    plot_controlled_visualization,
)
from utils.data_context import (
    DataReadinessDecision,
    evaluate_data_readiness,
    format_dataframe_state_for_log,
    requirement_from_controlled_plan,
    resolve_source_table,
)
from utils.eda_validation import (
    find_exact_prompt_column,
    validate_eda_visualization_request,
)
from utils.prompt_help import (
    CHAT_COMMAND_SPECS,
    AUTO_SQL_KEYWORDS,
    VIZ_KEYWORDS,
    build_command_example_message,
    build_command_help_message,
)
from utils.runtime_trace import (
    finish_turn_trace,
    record_trace_event,
    snapshot_session_state,
    start_turn_trace,
)
from utils.sql_text import extract_sql_from_text
from utils.table_context import (
    coerce_table_context,
    is_trained_table_context,
    table_context_hash,
    table_training_work_log_fields,
)
from utils.session import (
    DEFAULT_SQL_LIMIT_MAX,
    DEFAULT_SQL_LIMIT_MIN,
    set_default_sql_limit,
    train_selected_table_context,
)
from utils.turn_logger import build_turn_payload, log_turn

import datetime as _dt

_TLOG_PATH = "/tmp/telly_debug.log"

def _tlog(tag: str, msg: str) -> None:
    """Append a timestamped debug line to /tmp/telly_debug.log"""
    line = f"[{_dt.datetime.now().strftime('%H:%M:%S.%f')}] [{tag}] {msg}"
    try:
        with open(_TLOG_PATH, "a") as f:
            f.write(line + "\n")
    except Exception:
        pass


# ── Persistent Thinking Log (session_state 기반) ──────────────────────
_THINKING_ICONS = {
    "CONTEXT": "🔍",
    "ROUTING": "🤖",
    "PLAN": "📋",
    "AGENT_START": "🚀",
    "TOOL_CALL": "🛠️",
    "TOOL_RESULT": "📦",
    "AGENT_DONE": "✅",
    "SQL": "🗃️",
    "EDA": "📊",
    "CHAIN": "🔗",
    "ERROR": "❌",
    "INFO": "ℹ️",
    "TABLE": "🧾",
}


def _thinking_log_init() -> None:
    """세션에 thinking_log 키가 없으면 초기화한다."""
    st.session_state.setdefault("thinking_log", [])


def _thinking_log_clear() -> None:
    """새 프롬프트 시작 시 이전 thinking log를 클리어한다."""
    st.session_state["thinking_log"] = []


def _think(tag: str, msg: str) -> None:
    """Thinking log에 한 줄을 추가한다. UI에서 지속적으로 표시된다."""
    _thinking_log_init()
    icon = _THINKING_ICONS.get(tag, "💭")
    ts = _dt.datetime.now().strftime("%H:%M:%S")
    st.session_state["thinking_log"].append(
        {"ts": ts, "tag": tag, "icon": icon, "msg": msg}
    )
    # 디버그 파일에도 동시 기록
    _tlog(tag, msg)


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
        "tools_used_for_log": [],
        "python_status_for_log": None,
        "python_error_for_log": "",
        "python_output_summary_for_log": "",
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

    if command_name == "table":
        trigger_len = len(command_spec["trigger"])
        table_command = stripped_for_command[trigger_len:].strip().lower()
        if table_command != "training":
            ack_message = f"사용법: {command_spec['usage']}"
            return _ack_and_finish(ack_message, "table")
        selected_table = st.session_state.get("databricks_selected_table", "").strip()
        record_trace_event("table_training_start", table=selected_table)
        _think("TABLE", f"TableContext training 시작: {selected_table or '(no table selected)'}")
        ok, message = train_selected_table_context(selected_table)
        record_trace_event(
            "table_training_result",
            table=selected_table,
            status="success" if ok else "fail",
            message=message,
            active_table_context=st.session_state.get("active_table_context"),
        )
        if ok:
            _think("TABLE", message)
            st.success(message)
        else:
            _think("ERROR", message)
            st.error(message)
        ack_result = _ack_and_finish(message, "table_training")
        ack_result.update(table_training_work_log_fields(selected_table, ok, message))
        return ack_result

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


def _plot_controlled_visualization(df: pd.DataFrame, config) -> str:
    return plot_controlled_visualization(df, config)


def _try_run_controlled_production_flow(
    *,
    user_query: str,
    run_id: str,
    turn_id: int,
    log_placeholder,
    debug_mode: bool,
    pytool_obj,
    tools_used_for_log: List[str],
    log_state: Dict[str, Optional[str]],
) -> bool:
    active_table_context = coerce_table_context(st.session_state.get("active_table_context"))
    default_table = (
        st.session_state.get("databricks_selected_table", "").strip()
        or st.session_state.get("databricks_table_input", "").strip()
        or st.session_state.get("last_sql_table", "").strip()
        or (active_table_context.table_fqn if active_table_context else "")
    )
    training_status = active_table_context.training_status if active_table_context else "none"
    context_source = active_table_context.source if active_table_context else ""
    context_hash = table_context_hash(active_table_context.table_fqn) if active_table_context else ""
    wants_visual = any(token in (user_query or "").lower() for token in VIZ_KEYWORDS)
    if wants_visual and not is_trained_table_context(active_table_context):
        message = (
            "선택한 테이블의 `%table training` 정보가 없어 controlled visualization을 실행하지 않습니다. "
            "먼저 `%table training`을 실행한 뒤 다시 요청해주세요."
        )
        record_trace_event(
            "controlled_plan",
            generated=False,
            reason="trained_table_context_required",
            user_query=user_query,
            table_context_source=context_source,
            training_status=training_status,
            context_hash=context_hash,
            resolved_from_training=False,
        )
        _think("ERROR", message)
        st.warning(message)
        append_assistant_message(run_id, message, "Controlled Executor", turn_id=turn_id)
        log_state["assistant_response_for_log"] = message
        log_state["intent_for_log"] = "table_context_required"
        log_state["python_status_for_log"] = "fail"
        log_state["python_error_for_log"] = message
        return True

    plan = build_controlled_plan(
        user_query,
        default_table=default_table,
        table_context=active_table_context,
    )
    if plan is None:
        record_trace_event(
            "controlled_plan",
            generated=False,
            user_query=user_query,
            table_context_source=context_source,
            training_status=training_status,
            context_hash=context_hash,
            resolved_from_training=False,
        )
        return False

    _think("PLAN", f"Controlled JSON Plan: {json.dumps(controlled_plan_to_dict(plan), ensure_ascii=False)}")
    record_trace_event(
        "controlled_plan",
        generated=True,
        plan=plan,
        table_context_source=context_source,
        training_status=training_status,
        context_hash=context_hash,
        resolved_from_training=True,
        **(getattr(plan, "resolution_debug", {}) or {}),
    )
    if getattr(plan, "resolution_debug", None):
        _think(
            "PLAN",
            (
                "Target resolution: "
                f"target={plan.target_column}, group={plan.group_column or '-'}, "
                f"values={list(plan.group_values or ())}"
            ),
        )
    current_state = st.session_state.get("df_A_state")
    requirement = requirement_from_controlled_plan(plan)
    readiness = evaluate_data_readiness(current_state, requirement)
    record_trace_event(
        "data_readiness",
        decision=readiness.decision,
        reason=readiness.reason,
        required_columns=list(requirement.columns),
        missing_columns=list(readiness.missing_columns),
        df_A_state=current_state,
    )
    _think("CONTEXT", format_dataframe_state_for_log(current_state))
    _think(
        "PLAN",
        (
            "Data readiness: "
            f"{readiness.decision.value} | required={list(requirement.columns)} | "
            f"missing={list(readiness.missing_columns)} | reason={readiness.reason}"
        ),
    )
    tools_used_for_log.extend(["controlled_planner", "data_readiness_gate"])
    st.session_state["last_agent_mode"] = "Controlled Executor"
    st.session_state["last_sql_label"] = user_query.strip()[:80] or "Controlled SQL"

    sql = ""
    if readiness.decision == DataReadinessDecision.FAIL:
        message = f"요청을 수행할 원본 테이블을 결정할 수 없습니다: {readiness.reason}"
        record_trace_event("controlled_result", status="fail", error=message)
        _think("ERROR", message)
        st.error(message)
        append_assistant_message(run_id, message, "Controlled Executor", turn_id=turn_id)
        log_state["assistant_response_for_log"] = message
        log_state["intent_for_log"] = "data_readiness"
        log_state["python_status_for_log"] = "fail"
        log_state["python_error_for_log"] = message
        return True

    if readiness.decision == DataReadinessDecision.RELOAD_REQUIRED:
        source_table = resolve_source_table(
            current_state,
            requirement_source=requirement.source_table,
            last_sql_table=st.session_state.get("last_sql_table", ""),
            selected_table=st.session_state.get("databricks_selected_table", ""),
        )
        if not source_table:
            message = "요청에 필요한 데이터를 다시 로드해야 하지만 원본 테이블을 결정할 수 없습니다."
            record_trace_event("controlled_result", status="fail", error=message)
            _think("ERROR", message)
            st.error(message)
            append_assistant_message(run_id, message, "Controlled Executor", turn_id=turn_id)
            log_state["assistant_response_for_log"] = message
            log_state["intent_for_log"] = "data_readiness"
            log_state["python_status_for_log"] = "fail"
            log_state["python_error_for_log"] = message
            return True

        plan_for_sql = replace(plan, table=source_table)
        sql = build_sql_from_plan(plan_for_sql)
        record_trace_event("controlled_reload_sql", sql=sql, source_table=source_table)
        _think("SQL", f"Deterministic reload SQL 생성: {sql}")
        st.session_state["last_sql_statement"] = sql
        st.session_state["last_sql_table"] = source_table
        st.session_state["databricks_table_input"] = source_table
        st.session_state["databricks_selected_table"] = source_table
        tools_used_for_log.extend(["deterministic_sql_builder", "databricks_preview_sql"])

        success = execute_sql_preview(
            run_id=run_id,
            sql_text=sql,
            log_container=log_placeholder,
            show_logs=debug_mode,
            auto_trigger=False,
            append_assistant_message=append_assistant_message,
            attach_figures_to_run=attach_figures_to_run,
        )
        log_state["sql_execution_status_for_log"] = "success" if success else "fail"
        if not success:
            log_state["assistant_response_for_log"] = st.session_state.get("last_sql_error", "")
            log_state["intent_for_log"] = "sql_execute"
            return True
    else:
        _think("SQL", "현재 df_A가 요청에 필요한 컬럼/행 조건을 만족하여 SQL reload를 생략")
        log_state["sql_execution_status_for_log"] = "skipped"

    df = st.session_state.get("df_A_data")
    validation = validate_eda_visualization_request(
        df,
        plan.target_column,
        table_context=None,
    )
    record_trace_event(
        "eda_validation",
        ok=validation.ok,
        reason=validation.reason,
        column=validation.column,
        dtype=validation.dtype,
        chart_type=validation.chart_type,
        error="" if validation.ok else validation.reason,
    )
    if not validation.ok:
        message = f"Controlled EDA validation 실패: {validation.reason}"
        _think("ERROR", message)
        st.error(message)
        append_assistant_message(run_id, message, "Controlled Executor", turn_id=turn_id)
        log_state["assistant_response_for_log"] = message
        log_state["intent_for_log"] = "eda"
        log_state["python_status_for_log"] = "fail"
        log_state["python_error_for_log"] = message
        return True

    try:
        config = select_visualization_config(plan, df)
        record_trace_event("visualization_config", config=config)
        _think("EDA", f"Controlled Viz Config: {json.dumps(config.__dict__, ensure_ascii=False)}")
        import matplotlib.pyplot as plt

        plt.close("all")
        summary = _plot_controlled_visualization(df, config)
        figure_payloads = collect_matplotlib_figure_payloads()
        if not figure_payloads and pytool_obj is not None and hasattr(pytool_obj, "globals"):
            pytool_obj.globals["df_A"] = df
            pytool_obj.globals["df"] = df
            pytool_obj.globals["plt"] = plt
            figure_payloads = render_visualizations(pytool_obj)
        message = (
            "Controlled production flow로 실행했습니다. "
            "LLM은 계획만 만들고 SQL/시각화 실행은 코드로 처리했습니다. "
            f"({summary})"
        )
        st.success(message)
        append_assistant_message(run_id, message, "Controlled Executor", turn_id=turn_id)
        attach_figures_to_run(run_id, figure_payloads)
        tools_used_for_log.extend(["rule_based_visualization_selector", "deterministic_python_plot"])
        log_state["assistant_response_for_log"] = message
        log_state["intent_for_log"] = "eda"
        log_state["sql_capture_for_log"] = sql
        log_state["generated_python_for_log"] = summary
        log_state["python_status_for_log"] = "success"
        record_trace_event(
            "controlled_result",
            status="success",
            summary=summary,
            figure_count=len(figure_payloads),
        )
        return True
    except Exception as exc:
        message = f"Controlled visualization 실행 실패: {exc}"
        record_trace_event("controlled_result", status="fail", error=message)
        _think("ERROR", message)
        st.error(message)
        append_assistant_message(run_id, message, "Controlled Executor", turn_id=turn_id)
        log_state["assistant_response_for_log"] = message
        log_state["intent_for_log"] = "eda"
        log_state["python_status_for_log"] = "fail"
        log_state["python_error_for_log"] = message
        return True


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

    from utils.perf_monitor import TimeTracker
    _think("AGENT_START", f"{spinner_text}")
    status_label = f"🚀 {spinner_text}"
    with st.status(status_label, expanded=True) as status:
        with TimeTracker("agent_execution"):
            try:
                st.write("🛠️ 에이전트가 도구를 사용하여 작업을 수행 중입니다...")
                t0 = time.time()
                result = agent_runner.invoke(
                    {"input": agent_request},
                    {
                        "callbacks": callbacks,
                        "configurable": {"session_id": session_id},
                    },
                )
                elapsed = time.time() - t0
                _think("AGENT_DONE", f"에이전트 실행 완료 ({elapsed:.1f}초 소요)")
                # intermediate_steps가 있으면 도구 호출 내역 기록
                for step in result.get("intermediate_steps", []):
                    try:
                        action, observation = step
                        tool_name = getattr(action, "tool", "unknown")
                        tool_input_raw = getattr(action, "tool_input", "")
                        tool_input_str = tool_input_raw if isinstance(tool_input_raw, str) else str(tool_input_raw)
                        obs_str = observation if isinstance(observation, str) else str(observation)
                        _think("TOOL_CALL", f"도구 `{tool_name}` 호출 → 입력: {tool_input_str[:120]}")
                        _think("TOOL_RESULT", f"도구 `{tool_name}` 결과: {obs_str[:200]}")
                    except Exception:
                        pass
                status.update(label=f"✅ {spinner_text} 완료 ({elapsed:.1f}s)", state="complete")
                return result
            except Exception as exc:
                error_text = str(exc)
                _think("ERROR", f"에이전트 실행 오류: {error_text[:200]}")
                lower_error = error_text.lower()
                if "serviceunavailable" in lower_error or "model is overloaded" in lower_error:
                    friendly = "AI 모델이 일시적으로 과부하 상태입니다. 잠시 후 다시 시도해주세요."
                    st.warning(friendly)
                    st.info("필요시 같은 요청을 조금 뒤에 다시 보내주세요.")
                    status.update(label="❌ 실행 중단됨 (AI 과부하)", state="error")
                    return {"output": friendly}
                st.error(f"Agent 실행 중 오류: {error_text}")
                status.update(label="❌ 실행 중 오류 발생", state="error")
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
    if agent_mode == "SQL Builder":
        sql_capture = extract_sql_from_text(final_text)
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


def _is_visualization_request(text: str) -> bool:
    lowered = (text or "").lower()
    return any(keyword.lower() in lowered for keyword in VIZ_KEYWORDS)


def _render_eda_parser_fallback(
    *,
    prompt: str,
    run_id: str,
    log_updates: Dict[str, Optional[str]],
    tools_used_for_log: List[str],
    turn_id: int,
    pytool_obj,
) -> bool:
    """Render a narrow deterministic chart when the EDA agent loops on JSON parsing."""

    if not _is_visualization_request(prompt):
        return False

    df = st.session_state.get("df_A_data")
    if not isinstance(df, pd.DataFrame) or df.empty:
        return False

    column = find_exact_prompt_column(df, prompt)
    if not column:
        return False

    try:
        import matplotlib.pyplot as plt

        series = df[column].dropna()
        if series.empty:
            return False

        plt.figure(figsize=(10, 5))
        if pd.api.types.is_numeric_dtype(series):
            if len(series) > 100:
                series.plot(kind="hist", bins=min(50, max(10, int(len(series) ** 0.5))))
                plt.ylabel("Frequency")
                chart_desc = f"`{column}` 컬럼의 히스토그램"
                generated_code = (
                    f"df_A[{column!r}].dropna().plot(kind='hist', "
                    "bins=min(50, max(10, int(len(df_A) ** 0.5))))"
                )
            else:
                series.plot(kind="box")
                plt.ylabel(column)
                chart_desc = f"`{column}` 컬럼의 박스플롯"
                generated_code = f"df_A[{column!r}].dropna().plot(kind='box')"
        else:
            counts = series.astype(str).value_counts().head(30)
            counts.plot(kind="bar")
            plt.ylabel("Count")
            chart_desc = f"`{column}` 컬럼의 상위 값 빈도 막대 차트"
            generated_code = f"df_A[{column!r}].dropna().astype(str).value_counts().head(30).plot(kind='bar')"
        plt.title(f"{column} distribution")
        plt.xlabel(column)
        plt.tight_layout()

        if pytool_obj is not None and hasattr(pytool_obj, "globals"):
            pytool_obj.globals["df_A"] = df
            pytool_obj.globals["df"] = df
            pytool_obj.globals["plt"] = plt

        message = (
            "Agent 응답 형식 오류로 자동 시각화 fallback을 사용했습니다. "
            f"{chart_desc}을(를) 생성했습니다."
        )
        _think("ERROR", "EDA Agent 파싱 루프 감지 → deterministic visualization fallback 사용")
        st.warning(message)
        append_assistant_message(run_id, message, "EDA Analyst", turn_id=turn_id)
        tools_used_for_log.append("deterministic_visualization_fallback")
        log_updates["assistant_response_for_log"] = message
        log_updates["intent_for_log"] = "eda"
        log_updates["generated_python_for_log"] = generated_code
        log_updates["python_status_for_log"] = "success"
        log_updates["python_error_for_log"] = ""
        return True
    except Exception as exc:
        _think("ERROR", f"자동 시각화 fallback 실패: {str(exc)[:160]}")
        return False


def _validate_eda_before_agent(
    *,
    agent_mode: str,
    agent_request: str,
    run_id: str,
    log_updates: Dict[str, Optional[str]],
    turn_id: int,
) -> bool:
    if agent_mode != "EDA Analyst" or not _is_visualization_request(agent_request):
        return True

    validation = validate_eda_visualization_request(
        st.session_state.get("df_A_data"),
        agent_request,
        table_context=st.session_state.get("active_table_context"),
    )
    record_trace_event(
        "eda_validation",
        ok=validation.ok,
        reason=validation.reason,
        column=validation.column,
        dtype=validation.dtype,
        chart_type=validation.chart_type,
        error="" if validation.ok else validation.reason,
    )
    if validation.ok:
        if validation.column is not None:
            _think(
                "EDA",
                f"시각화 전 검증 통과: column={validation.column}, dtype={validation.dtype}, chart={validation.chart_type}",
            )
        else:
            _think("EDA", validation.reason)
        return True

    message = f"EDA 실행 전 검증 실패: {validation.reason}"
    _think("ERROR", message)
    st.error(message)
    append_assistant_message(run_id, message, agent_mode, turn_id=turn_id)
    st.session_state["active_run_id"] = None
    log_updates["assistant_response_for_log"] = message
    log_updates["intent_for_log"] = "eda"
    log_updates["python_status_for_log"] = "fail"
    log_updates["python_error_for_log"] = message
    return False


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
        record_trace_event(
            "agent_result",
            agent_mode=agent_mode,
            status="fail",
            error=error_msg,
        )
        st.error(error_msg)
        append_assistant_message(run_id, error_msg, agent_mode, turn_id=turn_id)
        st.session_state["active_run_id"] = None
        log_updates["assistant_response_for_log"] = error_msg
        log_updates["intent_for_log"] = "eda"
        return log_updates

    if not _validate_eda_before_agent(
        agent_mode=agent_mode,
        agent_request=agent_request,
        run_id=run_id,
        log_updates=log_updates,
        turn_id=turn_id,
    ):
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

    record_trace_event(
        "agent_start",
        agent_mode=agent_mode,
        agent_request=agent_request,
        auto_execute_sql=auto_execute_sql,
        df_a_ready=df_a_ready,
    )
    agent_started = time.time()
    result = _invoke_agent_runner(
        agent_runner,
        agent_request,
        callbacks,
        session_id,
        spinner_text,
    )

    st.success("Done.")
    agent_elapsed_ms = int((time.time() - agent_started) * 1000)
    final = result.get("output", "Agent가 최종 답변을 생성하지 못했습니다.")
    intermediate_steps = result.get("intermediate_steps", [])
    _capture_python_steps(intermediate_steps, tools_used_for_log, log_updates)

    with answer_container:
        st.subheader("Answer")
        if agent_mode == "EDA Analyst" and detect_agent_parser_loop(result):
            if _render_eda_parser_fallback(
                prompt=agent_request,
                run_id=run_id,
                log_updates=log_updates,
                tools_used_for_log=tools_used_for_log,
                turn_id=turn_id,
                pytool_obj=pytool_obj,
            ):
                if pytool_obj is not None:
                    figure_payloads = render_visualizations(pytool_obj)
                    attach_figures_to_run(run_id, figure_payloads)
                record_trace_event(
                    "agent_result",
                    agent_mode=agent_mode,
                    elapsed_ms=agent_elapsed_ms,
                    parser_loop=True,
                    fallback_used=True,
                    generated_python=log_updates.get("generated_python_for_log", ""),
                    python_execution_status=log_updates.get("python_status_for_log"),
                    error=log_updates.get("python_error_for_log", ""),
                )
                st.session_state["active_run_id"] = None
                return log_updates

            message = (
                "EDA Agent 응답 형식 오류가 반복되어 실행을 중단했습니다. "
                "단일 컬럼명이 포함된 시각화 요청은 자동 fallback으로 처리할 수 있습니다."
            )
            _think("ERROR", "EDA Agent 파싱 루프 감지 → fallback 불가, 실행 중단")
            st.error(message)
            append_assistant_message(run_id, message, agent_mode, turn_id=turn_id)
            log_updates["assistant_response_for_log"] = message
            log_updates["intent_for_log"] = "eda"
            record_trace_event(
                "agent_result",
                agent_mode=agent_mode,
                elapsed_ms=agent_elapsed_ms,
                parser_loop=True,
                fallback_used=False,
                error=message,
            )
            st.session_state["active_run_id"] = None
            return log_updates

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
            record_trace_event(
                "agent_result",
                agent_mode=agent_mode,
                elapsed_ms=agent_elapsed_ms,
                sql=sql_capture,
                generated_python=log_updates.get("generated_python_for_log", ""),
                python_execution_status=log_updates.get("python_status_for_log"),
                error=log_updates.get("python_error_for_log", ""),
            )

    if agent_mode == "SQL Builder" and not log_updates.get("sql_capture_for_log"):
        record_trace_event(
            "agent_result",
            agent_mode=agent_mode,
            elapsed_ms=agent_elapsed_ms,
            sql="",
            generated_python=log_updates.get("generated_python_for_log", ""),
            python_execution_status=log_updates.get("python_status_for_log"),
            error=log_updates.get("python_error_for_log", ""),
        )
    elif agent_mode != "SQL Builder":
        record_trace_event(
            "agent_result",
            agent_mode=agent_mode,
            elapsed_ms=agent_elapsed_ms,
            generated_python=log_updates.get("generated_python_for_log", ""),
            python_execution_status=log_updates.get("python_status_for_log"),
            error=log_updates.get("python_error_for_log", ""),
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

    _tlog("ENTRY", f"========== 새 턴 시작 ==========")
    _tlog("ENTRY", f"user_q={user_q[:80]}")
    _tlog("ENTRY", f"df_a_ready={df_a_ready}, debug_mode={debug_mode}")
    _tlog("ENTRY", f"session: last_sql_status={st.session_state.get('last_sql_status')}, llm_router_suggested_chaining={st.session_state.get('llm_router_suggested_chaining')}, auto_eda_pending={st.session_state.get('auto_eda_pending')}")

    # 새 프롬프트 시작 시 이전 thinking log를 클리어 (chaining rerun 제외)
    if st.session_state.get("auto_eda_pending") is None:
        _thinking_log_clear()
    _thinking_log_init()

    # 자동 연쇄 분석 처리: SQL 로딩 후 EDA가 필요한 경우를 위해 저장해둔 질문 복구
    auto_pending = st.session_state.pop("auto_eda_pending", None)
    _tlog("AUTO_PENDING", f"auto_pending popped = {auto_pending}")
    if auto_pending and not user_q:
        user_q = auto_pending

    turn_id = next_turn_id_fn()
    st.session_state["turn_id"] = turn_id
    turn_started = time.time()
    run_id = str(uuid4())
    st.session_state["active_run_id"] = run_id
    original_user_q = user_q
    start_turn_trace(
        conversation_id=st.session_state.get("conversation_id", ""),
        turn_id=turn_id,
        run_id=run_id,
        user_message=original_user_q,
        df_a_ready=df_a_ready,
        debug_mode=debug_mode,
        auto_eda_pending=auto_pending,
        session=snapshot_session_state(),
    )
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
    record_trace_event(
        "command_detected",
        command_name=command_name,
        command_prefix=command_result.get("command_prefix"),
        handled_command=command_result.get("handled_command"),
        trigger=command_spec.get("trigger") if command_spec else "",
        actual_command_prefix=bool(command_spec),
    )

    log_state = _default_log_state()
    tools_used_for_log: List[str] = []

    tools_used_for_log.extend(command_result.get("tools_used_for_log", []) or [])
    for key in log_state:
        if key == "tools_used_for_log":
            continue
        value = command_result.get(key)
        if value not in (None, "", []):
            log_state[key] = value

    handled_command = command_result["handled_command"]
    command_prefix = command_result["command_prefix"]
    agent_request = command_result["agent_request"] or original_user_q

    # 데이터 로딩 키워드가 있으면 SQL 빌더로 강제 전환
    # (단, auto_eda_pending 으로 연쇄 실행 중일 때는 건너뛴다)
    _kw_matched = [kw for kw in AUTO_SQL_KEYWORDS if kw in original_user_q]
    _tlog("KEYWORD", f"handled_command={handled_command}, command_prefix={command_prefix}, auto_pending={auto_pending}, matched_keywords={_kw_matched}")
    command_prefix_before_keyword = command_prefix
    keyword_forced_sql = should_force_sql_from_keywords(
        matched_keywords=_kw_matched,
        is_visualization_request=_is_visualization_request(original_user_q),
        handled_command=handled_command,
        command_prefix=command_prefix,
        auto_pending=auto_pending,
    )
    if keyword_forced_sql:
        command_prefix = "sql"
        _tlog("KEYWORD", f"→ command_prefix 강제 sql 전환")
    record_trace_event(
        "keyword_forced_sql",
        matched_keywords=_kw_matched,
        keyword_forced_sql=keyword_forced_sql,
        command_prefix_before=command_prefix_before_keyword,
        command_prefix_after=command_prefix,
        actual_command_prefix=bool(command_spec),
        auto_pending=auto_pending,
    )

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
        record_trace_event("last_sql_execute", result=execute_result)

    # Production path: LLM/heuristic plan only, deterministic SQL + deterministic plotting.
    if (
        not handled_command
        and command_prefix is None
        and auto_pending is None
    ):
        handled_command = _try_run_controlled_production_flow(
            user_query=original_user_q,
            run_id=run_id,
            turn_id=turn_id,
            log_placeholder=log_placeholder,
            debug_mode=debug_mode,
            pytool_obj=pytool_obj,
            tools_used_for_log=tools_used_for_log,
            log_state=log_state,
        )

    # 프리픽스가 없고 빌트인 명령도 아닌 경우: 지능형 라우팅 (SQL Builder vs EDA Analyst)
    if not handled_command:
        df_a_data = st.session_state.get("df_A_data")
        is_preview_state = (df_a_data is None) or (isinstance(df_a_data, pd.DataFrame) and len(df_a_data) <= 10)
        
        force_eda_due_to_chaining = (auto_pending is not None)
        _tlog("ROUTING", f"is_preview_state={is_preview_state}, force_eda_due_to_chaining={force_eda_due_to_chaining}, df_a shape={df_a_data.shape if isinstance(df_a_data, pd.DataFrame) else 'None'}")
        
        if force_eda_due_to_chaining:
            agent_mode = "EDA Analyst"
            tools_used_for_log.append("eda_agent")
            st.session_state["command_prefix"] = None
            st.session_state["llm_router_suggested_chaining"] = False
            record_trace_event(
                "router_result",
                route_source="auto_eda_pending",
                agent_mode=agent_mode,
                suggested_agents=["EDA Analyst"],
                llm_router_suggested_chaining=False,
            )
            _think("CHAIN", "SQL 데이터 로딩 완료 → EDA Analyst 자동 시각화 단계로 진입")
            st.info("🔗 SQL 데이터 로딩 완료 → **EDA Analyst** 자동 시각화 단계로 진입합니다.")
        else:
            forced_agent_mode = resolve_forced_agent_mode(command_prefix)
            if forced_agent_mode:
                agent_mode = forced_agent_mode
            else:
                agent_mode = None

        if not force_eda_due_to_chaining and agent_mode == "SQL Builder":
            tools_used_for_log.append("sql_builder_agent")
            st.session_state["command_prefix"] = "sql"
            st.session_state["llm_router_suggested_chaining"] = False
            record_trace_event(
                "router_result",
                route_source="forced_sql",
                agent_mode=agent_mode,
                command_prefix=command_prefix,
                suggested_agents=["SQL Builder"],
                llm_router_suggested_chaining=False,
            )
            _think("ROUTING", "%sql 명령 감지 → LLM Router를 건너뛰고 SQL Builder로 직접 진입")
            st.info("`%sql` 명령을 감지하여 **SQL Builder**로 직접 실행합니다.")
        elif not force_eda_due_to_chaining and agent_mode is None:
            from core.llm_router import route_query
            data_context_message = format_dataframe_state_for_log(
                st.session_state.get("df_A_state")
            )
            if data_context_message == "현재 데이터: none":
                data_context_message = (
                    f"데이터 상태: {'미리보기 (샘플 10행)' if is_preview_state else '전체 데이터 로드됨'}"
                )
            _think("CONTEXT", data_context_message)
            _think("ROUTING", "사용자 질문 의도를 분석하여 최적의 분석 엔진(Agent) 선정 중...")

            with st.status("🧠 Telly가 의도를 분석 중입니다...", expanded=True) as status:
                st.write("🔍 **1단계: 현재 데이터 컨텍스트 확인**")
                st.write(f"   - 메모리 상태: `{data_context_message}`")
                
                st.write("🤖 **2단계: 최적의 분석 엔진(Agent) 선정**")
                st.write("   - 사용자의 질문 의도를 분석하여 `SQL Builder` 또는 `EDA Analyst` 중 가장 적합한 경로를 결정하고 있습니다.")
                st.caption("💡 로컬 AI 모델을 사용 중인 경우 이 과정에서 약 5~10초 정도 소요될 수 있습니다.")
                
                plan = route_query(llm, original_user_q, is_preview_state)
                
                st.write(f"✅ **3단계: 분석 전략 수립 완료**")
                st.write(f"   - **분류된 의도**: `{plan.intent_type}`")
                st.write(f"   - **AI 추론**: {plan.reasoning}")
                st.write(f"   - **실행 계획**: `{' → '.join(plan.suggested_agents)}` 요청 처리 예정")
                
                # Follow suggested_agents
                first_agent = plan.suggested_agents[0] if plan.suggested_agents else "EDA Analyst"
                agent_mode = first_agent
                
                if len(plan.suggested_agents) > 1:
                    st.write(f"🔗 **연쇄 실행 모드**: `{plan.suggested_agents[0]}` 로딩 후 자동으로 `{plan.suggested_agents[1]}` 시각화 단계로 이어집니다.")
                
                status.update(label=f"✅ 의도 분석 완료 → **{agent_mode}** 실행 단계로 진입합니다.", state="complete")

            _think("PLAN", f"의도: {plan.intent_type} | 추론: {plan.reasoning}")
            _think("PLAN", f"실행 계획: {' → '.join(plan.suggested_agents)}")
            
            st.session_state["command_prefix"] = "sql" if agent_mode == "SQL Builder" else None
            tools_used_for_log.append("sql_builder_agent" if agent_mode == "SQL Builder" else "eda_agent")
            
            # Setup for Chaining
            if len(plan.suggested_agents) > 1 and plan.suggested_agents[1] == "EDA Analyst":
                st.session_state["llm_router_suggested_chaining"] = True
                _think("CHAIN", "연쇄 실행 모드 활성화: SQL Builder → EDA Analyst")
            else:
                st.session_state["llm_router_suggested_chaining"] = False
            record_trace_event(
                "router_result",
                route_source="llm_router",
                intent_type=plan.intent_type,
                reasoning=plan.reasoning,
                suggested_agents=plan.suggested_agents,
                agent_mode=agent_mode,
                is_preview_state=is_preview_state,
                llm_router_suggested_chaining=st.session_state.get("llm_router_suggested_chaining"),
            )
            _tlog("ROUTING", f"LLM Router result: intent={plan.intent_type}, agents={plan.suggested_agents}, reasoning={plan.reasoning[:80]}")
            _tlog("ROUTING", f"→ agent_mode={agent_mode}, chaining={st.session_state.get('llm_router_suggested_chaining')}")

        st.session_state["last_agent_mode"] = agent_mode

        if not agent_request or force_eda_due_to_chaining:
            if force_eda_due_to_chaining:
                # EDA 체이닝 시, SQL 결과 df_A의 실제 컬럼 정보를 포함하여 에이전트에게 전달
                _df_now = st.session_state.get("df_A_data")
                _cols_info = ""
                if _df_now is not None and hasattr(_df_now, "columns"):
                    _cols_info = f"\n\n[중요 컨텍스트] 현재 df_A에는 SQL 실행 결과가 이미 로드되어 있습니다. df_A의 칼럼은 {list(_df_now.columns)} 이며, {len(_df_now)}개의 행이 있습니다. SQL에서 이미 필터링이 완료되었으므로, df_A 데이터를 추가 필터링 없이 그대로 사용하여 시각화해주세요."
                agent_request = original_user_q + _cols_info
            else:
                agent_request = "새로운 SQL 쿼리를 작성해줘." if command_prefix == "sql" else original_user_q

        _tlog("AGENT_REQ", f"agent_mode={agent_mode}, command_prefix={command_prefix}")
        _tlog("AGENT_REQ", f"agent_request={agent_request[:120]}")

        _think("AGENT_START", f"{agent_mode} 에이전트에 요청 전달: {agent_request[:80]}...")

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
        _tlog("AGENT_DONE", f"agent_mode={agent_mode} 실행 완료")
        _tlog("AGENT_DONE", f"sql_capture={agent_updates.get('sql_capture_for_log', '')[:80]}")
        _tlog("AGENT_DONE", f"python_code={agent_updates.get('generated_python_for_log', '')[:120]}")
        _tlog("AGENT_DONE", f"python_status={agent_updates.get('python_status_for_log')}")
        _tlog("AGENT_DONE", f"python_error={agent_updates.get('python_error_for_log', '')[:120]}")

        # Thinking log에 결과 요약 기록
        _sql = agent_updates.get('sql_capture_for_log', '')
        _py_status = agent_updates.get('python_status_for_log')
        if _sql:
            _think("SQL", f"생성된 SQL: {_sql[:150]}")
        if _py_status:
            _think("EDA", f"Python 실행 결과: {_py_status} | 코드: {agent_updates.get('generated_python_for_log', '')[:100]}")
        _think("AGENT_DONE", f"{agent_mode} 작업 완료")

        for key, value in agent_updates.items():
            if key == "tools_used_for_log":
                continue
            if value:
                log_state[key] = value

    # Thinking log를 session_state에 보존 (다음 렌더 때 표시하기 위해)
    st.session_state["thinking_log_for_display"] = list(st.session_state.get("thinking_log", []))

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

    # SQL 로딩 후 LLM 라우터가 분석/시각화를 제안했다면 데이터를 로드한 직후 자동으로 EDA 트리거
    _chaining_flag = st.session_state.get("llm_router_suggested_chaining", False)
    _sql_status = st.session_state.get("last_sql_status")
    _tlog("CHAIN_CHECK", f"handled_command={handled_command}, last_sql_status={_sql_status}, chaining_flag={_chaining_flag}, auto_pending_was={auto_pending}")
    chain_triggered = (
        not handled_command
        and _sql_status == "success"
        and _chaining_flag
    )
    record_trace_event(
        "chain_check",
        handled_command=handled_command,
        last_sql_status=_sql_status,
        llm_router_suggested_chaining=_chaining_flag,
        auto_pending_was=auto_pending,
        chain_triggered=chain_triggered,
    )
    if chain_triggered:
        _tlog("CHAIN_CHECK", f"✅ 체이닝 트리거! auto_eda_pending에 질문 저장 후 st.rerun() 호출")
        _tlog("CHAIN_CHECK", f"원래 질문: {original_user_q}")
        st.session_state["auto_eda_pending"] = original_user_q
        finish_turn_trace(
            final_status="rerun_for_auto_eda",
            assistant_response=log_state.get("assistant_response_for_log"),
            sql=log_state.get("sql_capture_for_log") or st.session_state.get("last_sql_statement", ""),
            sql_execution_status=log_state.get("sql_execution_status_for_log"),
            python_execution_status=log_state.get("python_status_for_log"),
            error=log_state.get("python_error_for_log") or st.session_state.get("last_sql_error", ""),
            figure_count=sum(
                len(entry.get("figures", []))
                for entry in st.session_state.get("conversation_log", [])
                if entry.get("run_id") == run_id
            ),
        )
        st.rerun()
    else:
        _tlog("CHAIN_CHECK", f"❌ 체이닝 미트리거. 조건 미충족: handled={handled_command}, sql_status={_sql_status}, flag={_chaining_flag}")

    # 사후 결과 판단 및 검증 UI 추가 (Human-in-the-Loop)
    if not handled_command and st.session_state.get("auto_eda_pending") is None:
        st.markdown("---")
        st.markdown(f"**🤖 이 결과가 처음 의도하셨던 '{original_user_q[:20]}...'와(과) 정확히 일치하나요?**")
        st.caption("여러분의 피드백은 Telly가 도메인 지식을 학습하는 데 활용됩니다.")
        
        # Record interaction into the local learning DB
        from core.learning_memory import record_interaction
        from core.llm_router import ExecutionPlan
        
        # Determine intent name
        intent_val = log_state.get("intent_for_log", "OTHER")
        sql_val = log_state.get("sql_capture_for_log")
        py_val = log_state.get("generated_python_for_log")
        
        # Save to SQLite and store the row ID in session state for rating updates
        # Only record if we actually executed an agent
        if st.session_state.get("active_run_id") is None:
            db_row_id = record_interaction(
                original_query=original_user_q,
                classified_intent=intent_val,
                generated_sql=sql_val if sql_val else None,
                generated_python=py_val if py_val else None,
            )

    finish_turn_trace(
        final_status="completed",
        assistant_response=log_state.get("assistant_response_for_log"),
        sql=log_state.get("sql_capture_for_log") or st.session_state.get("last_sql_statement", ""),
        sql_execution_status=log_state.get("sql_execution_status_for_log"),
        python_execution_status=log_state.get("python_status_for_log"),
        error=log_state.get("python_error_for_log") or st.session_state.get("last_sql_error", ""),
        figure_count=sum(
            len(entry.get("figures", []))
            for entry in st.session_state.get("conversation_log", [])
            if entry.get("run_id") == run_id
        ),
    )


__all__ = ["handle_user_query"]
