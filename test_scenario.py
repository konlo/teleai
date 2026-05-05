import json
import os
import sys
import tempfile
import types

# 프로젝트 루트를 경로에 추가
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd

from utils.agent_output import (
    detect_agent_parser_loop,
    has_invalid_action_schema,
    unwrap_final_answer_payload,
)
from utils.agent_routing import (
    resolve_agent_mode_for_input,
    should_force_sql_builder,
    should_force_sql_from_keywords,
)
from utils.chat_turn import resolve_chat_turn_query, should_process_chat_turn
from utils.chatbot_plan import (
    build_controlled_plan,
    build_sql_from_plan,
    controlled_plan_to_dict,
    explain_controlled_plan_failure,
    required_columns_for_plan,
    select_visualization_config,
    validate_condition_coverage,
)
from utils.config import SQL_LIMIT_DEFAULT
from utils.conversation_figures import attach_figures_to_log
from utils.controlled_visualization import (
    collect_matplotlib_figure_payloads,
    plot_controlled_visualization,
)
from utils.data_context import (
    DataFrameState,
    DataReadinessDecision,
    DataRequirement,
    build_reload_sql_for_requirement,
    evaluate_data_readiness,
    make_dataframe_state,
    normalize_columns,
    requirement_from_controlled_plan,
)
from utils.eda_validation import (
    choose_distribution_chart,
    strip_internal_eda_context,
    validate_data_sufficiency,
    validate_eda_visualization_request,
)
from utils.runtime_trace import (
    build_trace_event,
    finish_turn_trace,
    read_latest_trace,
    read_recent_traces,
    record_trace_event,
    runtime_trace_fixture_for_keyword_sql,
    sanitize_for_trace,
    start_turn_trace,
)
from utils.prompt_help import CHAT_COMMAND_SPECS
from utils.sql_text import extract_sql_from_text
from utils.table_context import (
    TableContext,
    apply_table_context_overrides,
    build_schema_only_context,
    build_trained_context,
    contains_raw_sample_rows,
    ensure_table_context_override_file,
    load_table_context_for_selection,
    load_table_context_overrides,
    resolve_column_from_prompt,
    save_table_context,
    table_context_override_path,
    table_context_summary,
    table_context_to_dict,
    table_training_work_log_fields,
)
from utils.table_training_sql import build_bulk_profile_stats_sql, build_bulk_top_values_sql


EXTERNAL_LLM_PROVIDERS = {"google", "azure"}


def _bank_schema_context() -> TableContext:
    return build_schema_only_context(
        "workspace.default.bank_loan",
        columns=[
            "id",
            "age",
            "job",
            "marital",
            "education",
            "default",
            "balance",
            "housing",
            "loan",
            "contact",
            "day",
            "month",
            "duration",
            "campaign",
            "pdays",
            "previous",
            "poutcome",
            "y",
        ],
        dtypes={
            "age": "int64",
            "balance": "int64",
            "job": "object",
            "education": "object",
            "housing": "object",
            "loan": "object",
            "duration": "int64",
        },
    )


def _bank_table_context() -> TableContext:
    """Return a trained-context fixture with manual aliases, like %table training overrides."""

    trained = build_trained_context(
        _bank_schema_context(),
        row_count=750000,
        column_profiles={
            "age": {"null_count": 0, "distinct_count": 70, "min_value": 18, "max_value": 95},
            "job": {
                "null_count": 0,
                "distinct_count": 12,
                "top_values": [
                    {"value": "management", "count": 100},
                    {"value": "technician", "count": 80},
                ],
            },
            "education": {"null_count": 0, "distinct_count": 4, "top_values": [{"value": "secondary", "count": 100}]},
            "balance": {"null_count": 0, "distinct_count": 5000, "min_value": -8019, "max_value": 102127},
            "housing": {
                "null_count": 0,
                "distinct_count": 2,
                "top_values": [{"value": "yes", "count": 100}, {"value": "no", "count": 90}],
            },
            "loan": {
                "null_count": 0,
                "distinct_count": 2,
                "top_values": [{"value": "no", "count": 100}, {"value": "yes", "count": 30}],
            },
            "duration": {"null_count": 0, "distinct_count": 1500, "min_value": 0, "max_value": 4918},
        },
    )
    return apply_table_context_overrides(
        trained,
        overrides={
            "age": ["나이", "연령", "20대", "30대"],
            "job": ["직업", "직업군", "Job"],
            "loan": ["대출"],
            "housing": ["주택", "housing"],
        },
    )


def _bank_table_context_without_aliases() -> TableContext:
    """Return a trained-context fixture shaped like live training without manual aliases."""

    return build_trained_context(
        _bank_schema_context(),
        row_count=750000,
        column_profiles={
            "age": {"null_count": 0, "distinct_count": 70, "min_value": 18, "max_value": 95},
            "balance": {"null_count": 0, "distinct_count": 5000, "min_value": -8019, "max_value": 102127},
            "loan": {
                "null_count": 0,
                "distinct_count": 2,
                "top_values": [{"value": "no", "count": 100}, {"value": "yes", "count": 30}],
            },
        },
    )


def _titanic_schema_context() -> TableContext:
    return build_schema_only_context(
        "workspace.default.titanic",
        columns=[
            "PassengerId",
            "Survived",
            "Pclass",
            "Name",
            "Sex",
            "Age",
            "SibSp",
            "Parch",
            "Ticket",
            "Fare",
            "Cabin",
            "Embarked",
        ],
        dtypes={
            "PassengerId": "int64",
            "Survived": "int64",
            "Pclass": "int64",
            "Name": "object",
            "Sex": "object",
            "Age": "float64",
            "SibSp": "int64",
            "Parch": "int64",
            "Ticket": "object",
            "Fare": "float64",
            "Cabin": "object",
            "Embarked": "object",
        },
    )


def _titanic_table_context() -> TableContext:
    return build_trained_context(
        _titanic_schema_context(),
        row_count=891,
        column_profiles={
            "PassengerId": {"null_count": 0, "distinct_count": 891, "min_value": 1, "max_value": 891},
            "Survived": {
                "null_count": 0,
                "distinct_count": 2,
                "top_values": [{"value": "0", "count": 549}, {"value": "1", "count": 342}],
                "min_value": 0,
                "max_value": 1,
            },
            "Pclass": {"null_count": 0, "distinct_count": 3, "min_value": 1, "max_value": 3},
            "Name": {"null_count": 0, "distinct_count": 891},
            "Sex": {
                "null_count": 0,
                "distinct_count": 2,
                "top_values": [{"value": "male", "count": 577}, {"value": "female", "count": 314}],
            },
            "Age": {"null_count": 177, "distinct_count": 88, "min_value": 0.42, "max_value": 80},
            "SibSp": {"null_count": 0, "distinct_count": 7, "min_value": 0, "max_value": 8},
            "Parch": {"null_count": 0, "distinct_count": 7, "min_value": 0, "max_value": 6},
            "Ticket": {"null_count": 0, "distinct_count": 681},
            "Fare": {"null_count": 0, "distinct_count": 248, "min_value": 0, "max_value": 512.3292},
            "Cabin": {"null_count": 687, "distinct_count": 147},
            "Embarked": {"null_count": 2, "distinct_count": 3},
        },
    )


def external_llm_config_errors(provider=None, env=None):
    """Return human-readable config errors for external LLM test mode."""

    provider_name = (provider or os.environ.get("LLM_PROVIDER", "google")).strip().lower()
    env_map = env if env is not None else os.environ

    if provider_name == "ollama":
        return ["LLM_PROVIDER=ollama is local. Use google or azure for --external-llm."]
    if provider_name not in EXTERNAL_LLM_PROVIDERS:
        return [f"Unsupported external LLM_PROVIDER={provider_name!r}. Use google or azure."]

    if provider_name == "google":
        return [] if env_map.get("GOOGLE_API_KEY") else ["GOOGLE_API_KEY is required."]

    missing = [
        name
        for name in (
            "AZURE_OPENAI_API_KEY",
            "AZURE_OPENAI_ENDPOINT",
            "AZURE_OPENAI_DEPLOYMENT",
        )
        if not env_map.get(name)
    ]
    return [f"{name} is required." for name in missing]


def run_static_tests():
    print("🧪 체이닝 진입 조건 테스트 실행...\n")

    test_cases = [
        {
            "description": "새 사용자 입력이 있으면 일반 턴을 처리한다",
            "user_q": "balance를 시각화해줘",
            "auto_eda_pending": None,
            "expected_process": True,
            "expected_query": "balance를 시각화해줘",
        },
        {
            "description": "rerun 후 새 입력이 없어도 auto_eda_pending이 있으면 EDA 체이닝 턴을 처리한다",
            "user_q": None,
            "auto_eda_pending": "20대에서 30대 사이의 대출을 가지고 있는 사람들의 balance에 대해서 시각화해줘",
            "expected_process": True,
            "expected_query": "20대에서 30대 사이의 대출을 가지고 있는 사람들의 balance에 대해서 시각화해줘",
        },
        {
            "description": "새 입력과 체이닝 대기가 모두 없으면 대화 로그만 렌더링한다",
            "user_q": None,
            "auto_eda_pending": None,
            "expected_process": False,
            "expected_query": "",
        },
        {
            "description": "새 사용자 입력과 체이닝 대기가 동시에 있으면 새 입력을 우선한다",
            "user_q": "새로운 질문",
            "auto_eda_pending": "이전 체이닝 질문",
            "expected_process": True,
            "expected_query": "새로운 질문",
        },
    ]

    passed = 0
    failed = 0

    for idx, tc in enumerate(test_cases, 1):
        print(f"[static-{idx}] {tc['description']}")
        process = should_process_chat_turn(tc["user_q"], tc["auto_eda_pending"])
        query = resolve_chat_turn_query(tc["user_q"], tc["auto_eda_pending"])
        if process == tc["expected_process"] and query == tc["expected_query"]:
            print("   ✅ PASS\n")
            passed += 1
        else:
            print(
                "   ❌ FAIL "
                f"(Expected process={tc['expected_process']}, query={tc['expected_query']!r}; "
                f"Got process={process}, query={query!r})\n"
            )
            failed += 1

    print("-" * 50)
    print(f"🎯 체이닝 진입 조건 테스트 결과: {passed} 통과, {failed} 실패\n")
    failed += run_agent_output_tests()
    failed += run_eda_validation_tests()
    failed += run_controlled_plan_tests()
    failed += run_prompt_input_controlled_flow_tests()
    failed += run_data_context_tests()
    failed += run_figure_attachment_tests()
    failed += run_explicit_sql_routing_tests()
    failed += run_external_llm_config_tests()
    failed += run_sql_prompt_tests()
    failed += run_table_context_tests()
    failed += run_table_sample_tests()
    failed += run_runtime_trace_tests()
    return failed


def run_prompt_input_controlled_flow_tests():
    print("🧪 Prompt input controlled flow 시나리오 테스트 실행...\n")

    os.environ.setdefault("XDG_CACHE_HOME", "/tmp/teleai_xdg_cache")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/teleai_mplconfig")
    os.environ.setdefault("MPLBACKEND", "Agg")

    import matplotlib.pyplot as plt
    import pandas as pd
    import tempfile

    prompt = "job column이 technician 인 사람들의 housing 여부를 시각화 해줘"
    with tempfile.TemporaryDirectory() as temp_dir:
        save_table_context(_bank_table_context(), storage_dir=temp_dir)
        active_context = load_table_context_for_selection(
            "workspace.default.bank_loan",
            storage_dir=temp_dir,
        )
        current_state = DataFrameState(
            role="query_result",
            source_table="workspace.default.bank_loan",
            query="SELECT housing FROM workspace.default.bank_loan LIMIT 2000",
            columns=("housing",),
            row_count=2000,
            created_by="prompt_input_test",
        )

        start_turn_trace(
            conversation_id="prompt-input",
            turn_id=1,
            run_id="run-prompt",
            user_message=prompt,
            selected_table="workspace.default.bank_loan",
            storage_dir=temp_dir,
        )
        plan = build_controlled_plan(
            prompt,
            default_table="workspace.default.bank_loan",
            table_context=active_context,
        )
        record_trace_event(
            "controlled_plan",
            generated=plan is not None,
            plan=controlled_plan_to_dict(plan) if plan else {},
            table_context_source=active_context.source,
            training_status=active_context.training_status,
            resolved_from_training=plan is not None,
            **(getattr(plan, "resolution_debug", {}) if plan else {}),
        )
        requirement = requirement_from_controlled_plan(plan) if plan else DataRequirement()
        readiness = evaluate_data_readiness(current_state, requirement)
        record_trace_event(
            "data_readiness",
            decision=readiness.decision,
            reason=readiness.reason,
            required_columns=list(requirement.columns),
            missing_columns=list(readiness.missing_columns),
            df_A_state=current_state,
        )
        reload_sql = ""
        if plan and readiness.decision == DataReadinessDecision.RELOAD_REQUIRED:
            reload_sql = build_sql_from_plan(plan)
            record_trace_event(
                "controlled_reload_sql",
                sql=reload_sql,
                source_table="workspace.default.bank_loan",
            )
        result_df = pd.DataFrame({"housing": ["yes", "no"], "stat_count": [120, 80]})
        plt.close("all")
        config = select_visualization_config(plan, result_df) if plan else None
        summary = plot_controlled_visualization(result_df, config) if config else ""
        figure_count = len(plt.get_fignums())
        record_trace_event(
            "visualization_config",
            config=config,
        )
        record_trace_event(
            "controlled_result",
            status="success" if figure_count > 0 else "fail",
            summary=summary,
            figure_count=figure_count,
        )
        plt.close("all")
        finish_turn_trace(final_status="completed")
        latest_trace = read_latest_trace(storage_dir=temp_dir) or {}
        events = latest_trace.get("events", [])
        controlled_event = next(
            (event for event in events if event.get("event_type") == "controlled_plan"),
            {},
        )
        readiness_event = next(
            (event for event in events if event.get("event_type") == "data_readiness"),
            {},
        )
        reload_event = next(
            (event for event in events if event.get("event_type") == "controlled_reload_sql"),
            {},
        )
        result_event = next(
            (event for event in events if event.get("event_type") == "controlled_result"),
            {},
        )

        marital_prompt = "전체 marital 분포를 시각화 해줘"
        start_turn_trace(
            conversation_id="prompt-input-marital",
            turn_id=2,
            run_id="run-prompt-marital",
            user_message=marital_prompt,
            selected_table="workspace.default.bank_loan",
            storage_dir=temp_dir,
        )
        marital_plan = build_controlled_plan(
            marital_prompt,
            default_table="workspace.default.bank_loan",
            table_context=active_context,
        )
        marital_requirement = (
            requirement_from_controlled_plan(marital_plan)
            if marital_plan
            else DataRequirement()
        )
        marital_readiness = evaluate_data_readiness(None, marital_requirement)
        if marital_plan and marital_readiness.decision == DataReadinessDecision.RELOAD_REQUIRED:
            record_trace_event(
                "controlled_reload_sql",
                sql=build_sql_from_plan(marital_plan),
                source_table="workspace.default.bank_loan",
            )
        marital_result_df = pd.DataFrame(
            {"marital": ["married", "single", "divorced"], "stat_count": [120, 60, 20]}
        )
        marital_config = select_visualization_config(marital_plan, marital_result_df) if marital_plan else None
        plt.close("all")
        marital_summary = (
            plot_controlled_visualization(marital_result_df, marital_config)
            if marital_config
            else ""
        )
        marital_payloads = collect_matplotlib_figure_payloads()
        marital_log = [
            {
                "run_id": "run-prompt-marital",
                "role": "assistant",
                "content": "controlled result",
                "figures": [],
                "figures_attached": False,
            }
        ]
        marital_attached = attach_figures_to_log(
            marital_log,
            "run-prompt-marital",
            marital_payloads,
        )
        record_trace_event(
            "controlled_result",
            status="success" if marital_payloads else "fail",
            summary=marital_summary,
            figure_count=len(marital_payloads),
        )
        finish_turn_trace(final_status="completed")
        marital_attached_figures = marital_log[0].get("figures", [])

        housing_loan_prompt = "housing이 yes 인사람들의 loan 분포를 그려줘"
        start_turn_trace(
            conversation_id="prompt-input-housing-loan",
            turn_id=3,
            run_id="run-prompt-housing-loan",
            user_message=housing_loan_prompt,
            selected_table="workspace.default.bank_loan",
            storage_dir=temp_dir,
        )
        housing_loan_plan = build_controlled_plan(
            housing_loan_prompt,
            default_table="workspace.default.bank_loan",
            table_context=active_context,
        )
        record_trace_event(
            "controlled_plan",
            generated=housing_loan_plan is not None,
            plan=controlled_plan_to_dict(housing_loan_plan) if housing_loan_plan else {},
            table_context_source=active_context.source,
            training_status=active_context.training_status,
            resolved_from_training=housing_loan_plan is not None,
            **(getattr(housing_loan_plan, "resolution_debug", {}) if housing_loan_plan else {}),
        )
        housing_loan_requirement = (
            requirement_from_controlled_plan(housing_loan_plan)
            if housing_loan_plan
            else DataRequirement()
        )
        housing_loan_state = DataFrameState(
            role="query_result",
            source_table="workspace.default.bank_loan",
            query=f"SELECT balance FROM workspace.default.bank_loan LIMIT {SQL_LIMIT_DEFAULT}",
            columns=("balance",),
            row_count=750000,
            created_by="prompt_input_test",
        )
        housing_loan_readiness = evaluate_data_readiness(
            housing_loan_state,
            housing_loan_requirement,
        )
        record_trace_event(
            "data_readiness",
            decision=housing_loan_readiness.decision,
            reason=housing_loan_readiness.reason,
            required_columns=list(housing_loan_requirement.columns),
            missing_columns=list(housing_loan_readiness.missing_columns),
            df_A_state=housing_loan_state,
        )
        housing_loan_reload_sql = ""
        if housing_loan_plan and housing_loan_readiness.decision == DataReadinessDecision.RELOAD_REQUIRED:
            housing_loan_reload_sql = build_sql_from_plan(housing_loan_plan)
            record_trace_event(
                "controlled_reload_sql",
                sql=housing_loan_reload_sql,
                source_table="workspace.default.bank_loan",
            )
        housing_loan_result_df = pd.DataFrame(
            {"loan": ["no", "yes"], "stat_count": [345428, 65860]}
        )
        housing_loan_config = (
            select_visualization_config(housing_loan_plan, housing_loan_result_df)
            if housing_loan_plan
            else None
        )
        plt.close("all")
        housing_loan_summary = (
            plot_controlled_visualization(housing_loan_result_df, housing_loan_config)
            if housing_loan_config
            else ""
        )
        housing_loan_payloads = collect_matplotlib_figure_payloads()
        record_trace_event(
            "controlled_result",
            status="success" if housing_loan_payloads else "fail",
            summary=housing_loan_summary,
            figure_count=len(housing_loan_payloads),
        )
        finish_turn_trace(final_status="completed")
        housing_loan_trace = read_latest_trace(storage_dir=temp_dir) or {}
        housing_loan_events = housing_loan_trace.get("events", [])
        housing_loan_controlled_event = next(
            (event for event in housing_loan_events if event.get("event_type") == "controlled_plan"),
            {},
        )
        housing_loan_result_event = next(
            (event for event in housing_loan_events if event.get("event_type") == "controlled_result"),
            {},
        )
        housing_loan_router_events = [
            event for event in housing_loan_events if event.get("event_type") == "router_result"
        ]

        numeric_range_prompt = "age가 20~ 30 사이의 loan 값이 yes인 사람들의 balance에 대해서 시각화해줘"
        start_turn_trace(
            conversation_id="prompt-input-numeric-range",
            turn_id=4,
            run_id="run-prompt-numeric-range",
            user_message=numeric_range_prompt,
            selected_table="workspace.default.bank_loan",
            storage_dir=temp_dir,
        )
        numeric_range_plan = build_controlled_plan(
            numeric_range_prompt,
            default_table="workspace.default.bank_loan",
            table_context=active_context,
        )
        record_trace_event(
            "controlled_plan",
            generated=numeric_range_plan is not None,
            plan=controlled_plan_to_dict(numeric_range_plan) if numeric_range_plan else {},
            table_context_source=active_context.source,
            training_status=active_context.training_status,
            resolved_from_training=numeric_range_plan is not None,
            **(getattr(numeric_range_plan, "resolution_debug", {}) if numeric_range_plan else {}),
        )
        numeric_range_requirement = (
            requirement_from_controlled_plan(numeric_range_plan)
            if numeric_range_plan
            else DataRequirement()
        )
        numeric_range_state = DataFrameState(
            role="query_result",
            source_table="workspace.default.bank_loan",
            query=f"SELECT marital FROM workspace.default.bank_loan LIMIT {SQL_LIMIT_DEFAULT}",
            columns=("marital",),
            row_count=750000,
            created_by="prompt_input_test",
        )
        numeric_range_readiness = evaluate_data_readiness(
            numeric_range_state,
            numeric_range_requirement,
        )
        record_trace_event(
            "data_readiness",
            decision=numeric_range_readiness.decision,
            reason=numeric_range_readiness.reason,
            required_columns=list(numeric_range_requirement.columns),
            missing_columns=list(numeric_range_readiness.missing_columns),
            df_A_state=numeric_range_state,
        )
        numeric_range_reload_sql = ""
        if numeric_range_plan and numeric_range_readiness.decision == DataReadinessDecision.RELOAD_REQUIRED:
            numeric_range_reload_sql = build_sql_from_plan(numeric_range_plan)
            record_trace_event(
                "controlled_reload_sql",
                sql=numeric_range_reload_sql,
                source_table="workspace.default.bank_loan",
            )
        numeric_range_result_df = pd.DataFrame(
            {
                "balance": list(range(101)),
                "age": [20 + (idx % 11) for idx in range(101)],
                "loan": ["yes"] * 101,
            }
        )
        numeric_range_config = (
            select_visualization_config(numeric_range_plan, numeric_range_result_df)
            if numeric_range_plan
            else None
        )
        plt.close("all")
        numeric_range_summary = (
            plot_controlled_visualization(numeric_range_result_df, numeric_range_config)
            if numeric_range_config
            else ""
        )
        numeric_range_payloads = collect_matplotlib_figure_payloads()
        record_trace_event(
            "controlled_result",
            status="success" if numeric_range_payloads else "fail",
            summary=numeric_range_summary,
            figure_count=len(numeric_range_payloads),
        )
        finish_turn_trace(final_status="completed")
        numeric_range_trace = read_latest_trace(storage_dir=temp_dir) or {}
        numeric_range_events = numeric_range_trace.get("events", [])
        numeric_range_controlled_event = next(
            (event for event in numeric_range_events if event.get("event_type") == "controlled_plan"),
            {},
        )
        numeric_range_readiness_event = next(
            (event for event in numeric_range_events if event.get("event_type") == "data_readiness"),
            {},
        )
        numeric_range_reload_event = next(
            (event for event in numeric_range_events if event.get("event_type") == "controlled_reload_sql"),
            {},
        )
        numeric_range_result_event = next(
            (event for event in numeric_range_events if event.get("event_type") == "controlled_result"),
            {},
        )
        numeric_range_router_events = [
            event for event in numeric_range_events if event.get("event_type") == "router_result"
        ]

        aliasless_balance_prompt = "20대에서 30대 사이의 대출을 가지고 있는 사람들의 balance에 대해서 시각화해줘"
        aliasless_context = _bank_table_context_without_aliases()
        start_turn_trace(
            conversation_id="prompt-input-unused-conditions",
            turn_id=5,
            run_id="run-prompt-unused-conditions",
            user_message=aliasless_balance_prompt,
            selected_table="workspace.default.bank_loan",
            storage_dir=temp_dir,
        )
        aliasless_balance_plan = build_controlled_plan(
            aliasless_balance_prompt,
            default_table="workspace.default.bank_loan",
            table_context=aliasless_context,
        )
        record_trace_event(
            "controlled_plan",
            generated=aliasless_balance_plan is not None,
            plan=controlled_plan_to_dict(aliasless_balance_plan) if aliasless_balance_plan else {},
            table_context_source=aliasless_context.source,
            training_status=aliasless_context.training_status,
            resolved_from_training=aliasless_balance_plan is not None,
            **(getattr(aliasless_balance_plan, "resolution_debug", {}) if aliasless_balance_plan else {}),
        )
        aliasless_balance_coverage = validate_condition_coverage(
            aliasless_balance_prompt,
            aliasless_balance_plan,
            aliasless_context,
        )
        if not aliasless_balance_coverage.get("ok", True):
            record_trace_event(
                "controlled_result",
                status="blocked_unused_conditions",
                condition_coverage=aliasless_balance_coverage,
                figure_count=0,
            )
        finish_turn_trace(final_status="completed")
        aliasless_balance_trace = read_latest_trace(storage_dir=temp_dir) or {}
        aliasless_balance_events = aliasless_balance_trace.get("events", [])
        aliasless_balance_controlled_event = next(
            (event for event in aliasless_balance_events if event.get("event_type") == "controlled_plan"),
            {},
        )
        aliasless_balance_result_event = next(
            (event for event in aliasless_balance_events if event.get("event_type") == "controlled_result"),
            {},
        )
        aliasless_balance_reload_events = [
            event for event in aliasless_balance_events if event.get("event_type") == "controlled_reload_sql"
        ]

        unresolved_numeric_range_prompt = "나이가 20~ 30 사이의 loan 값이 yes인 사람들의 balance에 대해서 시각화해줘"
        start_turn_trace(
            conversation_id="prompt-input-unresolved-numeric-range",
            turn_id=6,
            run_id="run-prompt-unresolved-numeric-range",
            user_message=unresolved_numeric_range_prompt,
            selected_table="workspace.default.bank_loan",
            storage_dir=temp_dir,
        )
        unresolved_numeric_range_plan = build_controlled_plan(
            unresolved_numeric_range_prompt,
            default_table="workspace.default.bank_loan",
            table_context=aliasless_context,
        )
        record_trace_event(
            "controlled_plan",
            generated=unresolved_numeric_range_plan is not None,
            plan=controlled_plan_to_dict(unresolved_numeric_range_plan) if unresolved_numeric_range_plan else {},
            table_context_source=aliasless_context.source,
            training_status=aliasless_context.training_status,
            resolved_from_training=unresolved_numeric_range_plan is not None,
            **(
                getattr(unresolved_numeric_range_plan, "resolution_debug", {})
                if unresolved_numeric_range_plan
                else {}
            ),
        )
        unresolved_numeric_range_coverage = validate_condition_coverage(
            unresolved_numeric_range_prompt,
            unresolved_numeric_range_plan,
            aliasless_context,
        )
        if not unresolved_numeric_range_coverage.get("ok", True):
            record_trace_event(
                "controlled_result",
                status="blocked_unused_conditions",
                condition_coverage=unresolved_numeric_range_coverage,
                figure_count=0,
            )
        finish_turn_trace(final_status="completed")
        unresolved_numeric_range_trace = read_latest_trace(storage_dir=temp_dir) or {}
        unresolved_numeric_range_events = unresolved_numeric_range_trace.get("events", [])
        unresolved_numeric_range_result_event = next(
            (
                event
                for event in unresolved_numeric_range_events
                if event.get("event_type") == "controlled_result"
            ),
            {},
        )
        unresolved_numeric_range_reload_events = [
            event
            for event in unresolved_numeric_range_events
            if event.get("event_type") == "controlled_reload_sql"
        ]

    prompt_visual_scenarios = [
        {
            "name": "balance histogram",
            "prompt": "20대에서 30대 사이의 대출을 가지고 있는 사람들의 balance에 대해서 시각화해줘",
            "context": _bank_table_context(),
            "table": "workspace.default.bank_loan",
            "df": pd.DataFrame({"balance": range(101), "age": [20 + (idx % 11) for idx in range(101)], "loan": ["yes"] * 101}),
            "expected": ("balance", {"age": [20, 30], "loan": "yes"}, "histogram"),
        },
        {
            "name": "explicit numeric range balance histogram",
            "prompt": "age가 20~ 30 사이의 loan 값이 yes인 사람들의 balance에 대해서 시각화해줘",
            "context": _bank_table_context(),
            "table": "workspace.default.bank_loan",
            "df": pd.DataFrame(
                {
                    "balance": range(101),
                    "age": [20 + (idx % 11) for idx in range(101)],
                    "loan": ["yes"] * 101,
                }
            ),
            "expected": ("balance", {"age": [20, 30], "loan": "yes"}, "histogram"),
        },
        {
            "name": "housing yes loan bar",
            "prompt": "housing이 yes 인사람들의 loan 분포를 그려줘",
            "context": _bank_table_context(),
            "table": "workspace.default.bank_loan",
            "df": pd.DataFrame({"loan": ["yes", "no"], "stat_count": [30, 70]}),
            "expected": ("loan", {"housing": "yes"}, "bar"),
        },
        {
            "name": "duration job bar",
            "prompt": "duration이 500 넘는 사람들의 직업군이 어떻게 되는지 시각화 해줘",
            "context": _bank_table_context(),
            "table": "workspace.default.bank_loan",
            "df": pd.DataFrame({"job": ["management", "technician"], "stat_count": [20, 10]}),
            "expected": ("job", {}, "bar"),
        },
        {
            "name": "education bar",
            "prompt": "전체 education의 분포를 보고 싶어",
            "context": _bank_table_context(),
            "table": "workspace.default.bank_loan",
            "df": pd.DataFrame({"education": ["secondary", "primary", "tertiary"]}),
            "expected": ("education", {}, "bar"),
        },
        {
            "name": "top balance job bar",
            "prompt": "balance 가 가장 높은 상위 10%의 사람들의 직업이 어떻게 되는지 분포를 그려줘",
            "context": _bank_table_context(),
            "table": "workspace.default.bank_loan",
            "df": pd.DataFrame({"job": ["management", "technician"], "stat_count": [20, 10]}),
            "expected": ("job", {}, "bar"),
        },
        {
            "name": "titanic grouped bar",
            "prompt": "survived 값이 1인사람들과 0인 사람들의 Sex(성별) 분포를 각각 시각화 해줘.",
            "context": _titanic_table_context(),
            "table": "workspace.default.titanic",
            "df": pd.DataFrame(
                {
                    "Survived": [1, 1, 0, 0],
                    "Sex": ["female", "male", "male", "female"],
                    "stat_count": [200, 100, 400, 100],
                }
            ),
            "expected": ("Sex", {}, "grouped_bar"),
        },
        {
            "name": "titanic survived bar",
            "prompt": "Survived 분포를 그려줘",
            "context": _titanic_table_context(),
            "table": "workspace.default.titanic",
            "df": pd.DataFrame({"Survived": [0, 1, 0, 1]}),
            "expected": ("Survived", {}, "boxplot"),
        },
        {
            "name": "titanic sex parenthetical bar",
            "prompt": "Sex(성별) 분포를 보고 싶어",
            "context": _titanic_table_context(),
            "table": "workspace.default.titanic",
            "df": pd.DataFrame({"Sex": ["male", "female", "male"]}),
            "expected": ("Sex", {}, "bar"),
        },
    ]
    prompt_visual_results = []
    for scenario in prompt_visual_scenarios:
        scenario_plan = build_controlled_plan(
            scenario["prompt"],
            default_table=scenario["table"],
            table_context=scenario["context"],
        )
        scenario_config = select_visualization_config(scenario_plan, scenario["df"]) if scenario_plan else None
        plt.close("all")
        scenario_summary = plot_controlled_visualization(scenario["df"], scenario_config) if scenario_config else ""
        scenario_figure_count = len(plt.get_fignums())
        plt.close("all")
        prompt_visual_results.append(
            (
                scenario["name"],
                scenario_plan.target_column if scenario_plan else "",
                scenario_plan.filters if scenario_plan else {},
                scenario_config.plot_type if scenario_config else "",
                scenario_figure_count > 0,
                bool(scenario_summary),
                scenario["expected"],
            )
        )

    test_cases = [
        {
            "description": "실제 사용자 prompt 문자열이 저장된 trained TableContext를 거쳐 controlled plan을 만든다",
            "actual": (
                latest_trace.get("summary", {}).get("user_message"),
                latest_trace.get("summary", {}).get("controlled_plan_generated"),
                latest_trace.get("summary", {}).get("controlled_target_column"),
                controlled_event.get("training_status"),
            ),
            "expected": (prompt, True, "housing", "trained"),
        },
        {
            "description": "prompt 입력 경로의 Controlled JSON Plan에 job=technician filter가 포함된다",
            "actual": (
                plan.target_column if plan else "",
                plan.filters if plan else {},
                controlled_event.get("categorical_value_filters", [{}])[0].get("accepted")
                if controlled_event.get("categorical_value_filters")
                else False,
            ),
            "expected": ("housing", {"job": "technician"}, True),
        },
        {
            "description": "prompt 입력 경로의 readiness는 housing-only df_A를 충분하다고 보지 않는다",
            "actual": (
                readiness_event.get("decision"),
                readiness_event.get("missing_columns"),
                list(requirement.columns),
            ),
            "expected": (
                DataReadinessDecision.RELOAD_REQUIRED,
                ["job"],
                ["housing", "job"],
            ),
        },
        {
            "description": "prompt 입력 경로에서 reload SQL은 job=technician 조건을 포함한다",
            "actual": (
                reload_event.get("sql", "").startswith(
                    "SELECT housing, COUNT(*) AS stat_count FROM workspace.default.bank_loan"
                ),
                "WHERE job = 'technician'" in reload_event.get("sql", ""),
                "GROUP BY housing" in reload_event.get("sql", ""),
            ),
            "expected": (True, True, True),
        },
        {
            "description": "prompt 입력 경로에서 실제 controlled visualization 실행 결과 figure가 생성된다",
            "actual": (
                result_event.get("status"),
                result_event.get("figure_count", 0) > 0,
                "plot_type=bar" in result_event.get("summary", ""),
            ),
            "expected": ("success", True, True),
        },
        {
            "description": "df_A가 없는 실제 prompt 입력 흐름에서도 controlled chart image payload가 chat log에 붙는다",
            "actual": (
                marital_plan.target_column if marital_plan else "",
                marital_readiness.decision,
                len(marital_payloads),
                marital_attached,
                marital_attached_figures[0].get("kind") if marital_attached_figures else "",
                bool(marital_attached_figures[0].get("image")) if marital_attached_figures else False,
            ),
            "expected": (
                "marital",
                DataReadinessDecision.RELOAD_REQUIRED,
                1,
                True,
                "matplotlib",
                True,
            ),
        },
        {
            "description": "housing=yes loan prompt 입력 흐름은 Router로 떨어지지 않고 controlled result를 만든다",
            "actual": (
                housing_loan_controlled_event.get("generated"),
                housing_loan_plan.target_column if housing_loan_plan else "",
                housing_loan_plan.filters if housing_loan_plan else {},
                housing_loan_readiness.decision,
                "WHERE housing = 'yes'" in housing_loan_reload_sql,
                "GROUP BY loan" in housing_loan_reload_sql,
                len(housing_loan_router_events),
                housing_loan_result_event.get("status"),
                housing_loan_result_event.get("figure_count", 0) > 0,
            ),
            "expected": (
                True,
                "loan",
                {"housing": "yes"},
                DataReadinessDecision.RELOAD_REQUIRED,
                True,
                True,
                0,
                "success",
                True,
            ),
        },
        {
            "description": "명시적 numeric range prompt 입력 흐름은 age BETWEEN 조건을 누락하지 않는다",
            "actual": (
                numeric_range_controlled_event.get("generated"),
                numeric_range_plan.target_column if numeric_range_plan else "",
                numeric_range_plan.filters if numeric_range_plan else {},
                numeric_range_controlled_event.get("condition_coverage", {}).get("ok"),
                numeric_range_readiness_event.get("decision"),
                list(numeric_range_requirement.columns),
                "WHERE age BETWEEN 20 AND 30 AND loan = 'yes'" in numeric_range_reload_event.get("sql", ""),
                len(numeric_range_router_events),
                numeric_range_result_event.get("status"),
                numeric_range_result_event.get("figure_count", 0) > 0,
            ),
            "expected": (
                True,
                "balance",
                {"age": [20, 30], "loan": "yes"},
                True,
                DataReadinessDecision.RELOAD_REQUIRED,
                ["balance", "age", "loan"],
                True,
                0,
                "success",
                True,
            ),
        },
        {
            "description": "조건 표현이 미사용이면 prompt 입력 흐름에서 reload/figure 없이 차단한다",
            "actual": (
                aliasless_balance_controlled_event.get("generated"),
                aliasless_balance_plan.target_column if aliasless_balance_plan else "",
                aliasless_balance_plan.filters if aliasless_balance_plan else {},
                aliasless_balance_result_event.get("status"),
                aliasless_balance_result_event.get("condition_coverage", {}).get("unused_conditions"),
                len(aliasless_balance_reload_events),
                aliasless_balance_result_event.get("figure_count"),
            ),
            "expected": (
                True,
                "balance",
                {},
                "blocked_unused_conditions",
                ["20대에서 30대 사이", "대출을 가지고 있는"],
                0,
                0,
            ),
        },
        {
            "description": "컬럼/alias와 연결되지 않은 numeric range는 prompt 입력 흐름에서 차단한다",
            "actual": (
                unresolved_numeric_range_plan.target_column if unresolved_numeric_range_plan else "",
                unresolved_numeric_range_plan.filters if unresolved_numeric_range_plan else {},
                unresolved_numeric_range_result_event.get("status"),
                unresolved_numeric_range_result_event.get("condition_coverage", {}).get("unused_conditions"),
                len(unresolved_numeric_range_reload_events),
                unresolved_numeric_range_result_event.get("figure_count"),
            ),
            "expected": (
                "balance",
                {"loan": "yes"},
                "blocked_unused_conditions",
                ["나이가 20~ 30 사이"],
                0,
                0,
            ),
        },
    ]
    for name, target, filters, plot_type, has_figure, has_summary, expected in prompt_visual_results:
        expected_target, expected_filters, expected_plot_type = expected
        test_cases.append(
            {
                "description": f"실제 prompt 시각화 실행 결과를 검증한다: {name}",
                "actual": (target, filters, plot_type, has_figure, has_summary),
                "expected": (expected_target, expected_filters, expected_plot_type, True, True),
            }
        )

    passed = 0
    failed = 0
    for idx, tc in enumerate(test_cases, 1):
        print(f"[prompt-input-{idx}] {tc['description']}")
        if tc["actual"] == tc["expected"]:
            print("   ✅ PASS\n")
            passed += 1
        else:
            print(f"   ❌ FAIL (Expected: {tc['expected']}, Got: {tc['actual']})\n")
            failed += 1

    print("-" * 50)
    print(f"🎯 Prompt input controlled flow 테스트 결과: {passed} 통과, {failed} 실패\n")
    return failed


def _nested_final_answer(depth: int):
    payload = "최종 응답"
    for _ in range(depth):
        payload = {"action": "Final Answer", "action_input": payload}
    return payload


def run_agent_output_tests():
    print("🧪 Agent 출력 파싱 루프 감지 테스트 실행...\n")

    real_prompt = "20대에서 30대 사이의 대출을 가지고 있는 사람들의 balance에 대해서 시각화해줘"
    malformed_channel_output = """
Parsing error: 올바른 형식으로 다시 응답해주세요. 반드시 아래 JSON 형식을 사용하세요:

Thought
The user wants to visualize the distribution of the 'balance' column from df_A.
<channel|>```python
import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(12, 6))
sns.histplot(df_A['balance'], bins=50, kde=True)
plt.show()
```

Output:
올바른 형식으로 다시 응답해주세요. 반드시 아래 JSON 형식을 사용하세요:
{"action": "도구이름", "action_input": "입력값"}
"""
    invalid_content_payload = {"action": "Final Answer", "content": "잘못된 키 사용"}
    deep_nested_payload = _nested_final_answer(5)
    valid_final_payload = {"action": "Final Answer", "action_input": "정상 최종 응답"}
    exception_steps = [
        (type("Action", (), {"tool": "_Exception", "tool_input": "Invalid Format: Missing 'Action:'"})(), "Parsing error"),
        (type("Action", (), {"tool": "_Exception", "tool_input": "올바른 형식으로 다시 응답해주세요"})(), "Parsing error"),
    ]

    test_cases = [
        {
            "description": "실제 체이닝 prompt가 auto_eda_pending에서 처리 대상이 된다",
            "actual": should_process_chat_turn(None, real_prompt)
            and resolve_chat_turn_query(None, real_prompt) == real_prompt,
            "expected": True,
        },
        {
            "description": "Parsing error + channel marker + standalone python fence output을 루프로 감지한다",
            "actual": detect_agent_parser_loop(malformed_channel_output),
            "expected": True,
        },
        {
            "description": "content 키를 사용한 Final Answer schema를 invalid로 감지한다",
            "actual": has_invalid_action_schema(invalid_content_payload)
            and detect_agent_parser_loop(invalid_content_payload),
            "expected": True,
        },
        {
            "description": "깊게 중첩된 Final Answer는 max depth 초과로 감지한다",
            "actual": unwrap_final_answer_payload(deep_nested_payload)[1]
            and detect_agent_parser_loop(deep_nested_payload),
            "expected": True,
        },
        {
            "description": "정상 Final Answer JSON은 parser-loop로 감지하지 않는다",
            "actual": detect_agent_parser_loop(valid_final_payload),
            "expected": False,
        },
        {
            "description": "_Exception 파싱 오류 intermediate step 2회 이상이면 루프로 감지한다",
            "actual": detect_agent_parser_loop({"output": "", "intermediate_steps": exception_steps}),
            "expected": True,
        },
    ]

    passed = 0
    failed = 0
    for idx, tc in enumerate(test_cases, 1):
        print(f"[agent-output-{idx}] {tc['description']}")
        if tc["actual"] == tc["expected"]:
            print("   ✅ PASS\n")
            passed += 1
        else:
            print(f"   ❌ FAIL (Expected: {tc['expected']}, Got: {tc['actual']})\n")
            failed += 1

    print("-" * 50)
    print(f"🎯 Agent 출력 테스트 결과: {passed} 통과, {failed} 실패\n")
    return failed


def run_eda_validation_tests():
    print("🧪 EDA validation/fallback chart 테스트 실행...\n")

    import pandas as pd

    large_df = pd.DataFrame({"balance": range(101)})
    small_df = pd.DataFrame({"balance": range(100)})
    category_df = pd.DataFrame({"job": ["admin", "tech", "admin"]})
    prompt = "20대에서 30대 사이의 대출을 가지고 있는 사람들의 balance에 대해서 시각화해줘"

    large_validation = validate_eda_visualization_request(large_df, prompt)
    small_validation = validate_eda_visualization_request(small_df, prompt)
    missing_validation = validate_eda_visualization_request(
        pd.DataFrame({"age": [20, 30]}),
        prompt,
    )
    education_chain_prompt = (
        "전체 education의 분포를 보고 싶어"
        "\n\n[중요 컨텍스트] 현재 df_A에는 SQL 실행 결과가 이미 로드되어 있습니다. "
        "df_A의 칼럼은 ['education', 'stat_count'] 이며, 4개의 행이 있습니다. "
        "SQL에서 이미 필터링이 완료되었으므로, df_A 데이터를 추가 필터링 없이 그대로 사용하여 시각화해주세요."
    )
    education_chain_validation = validate_eda_visualization_request(
        pd.DataFrame({"education": ["secondary", "primary"], "stat_count": [401683, 99510]}),
        education_chain_prompt,
        table_context=_bank_table_context(),
    )
    housing_loan_chain_validation = validate_eda_visualization_request(
        pd.DataFrame({"loan": ["no", "yes"], "stat_count": [345428, 65860]}),
        "housing이 yes 인사람들의 loan 분포를 그려줘",
        table_context=_bank_table_context(),
    )

    test_cases = [
        {
            "description": "df_A 행 수가 100개를 초과하면 numeric fallback은 hist를 선택한다",
            "actual": choose_distribution_chart(large_df["balance"]),
            "expected": "hist",
        },
        {
            "description": "df_A 행 수가 100개 이하면 numeric fallback은 boxplot을 선택한다",
            "actual": choose_distribution_chart(small_df["balance"]),
            "expected": "boxplot",
        },
        {
            "description": "categorical 분포는 bar chart로 판단한다",
            "actual": choose_distribution_chart(category_df["job"]),
            "expected": "bar",
        },
        {
            "description": "실제 balance prompt는 컬럼 존재와 dtype 검증 후 hist로 통과한다",
            "actual": large_validation.ok
            and large_validation.column == "balance"
            and large_validation.chart_type == "hist",
            "expected": True,
        },
        {
            "description": "100행 이하 balance prompt는 boxplot으로 통과한다",
            "actual": small_validation.ok
            and small_validation.column == "balance"
            and small_validation.chart_type == "boxplot",
            "expected": True,
        },
        {
            "description": "프롬프트에 언급된 컬럼이 df_A에 없으면 EDA 전 검증에서 실패한다",
            "actual": missing_validation.ok,
            "expected": False,
        },
        {
            "description": "df_A가 로드되어 있어도 요청 컬럼 job이 없으면 충분하지 않다고 판단한다",
            "actual": validate_data_sufficiency(pd.DataFrame({"balance": [100, 200]}), ["job"]).sufficient,
            "expected": False,
        },
        {
            "description": "df_A가 요청 컬럼 balance를 포함하면 충분하다고 판단한다",
            "actual": validate_data_sufficiency(pd.DataFrame({"balance": [100, 200]}), ["balance"]).sufficient,
            "expected": True,
        },
        {
            "description": "자동 EDA 내부 컨텍스트는 컬럼 추출 대상에서 제외한다",
            "actual": strip_internal_eda_context(education_chain_prompt),
            "expected": "전체 education의 분포를 보고 싶어",
        },
        {
            "description": "education SQL 결과 자동 EDA 검증은 내부 SQL 컨텍스트의 `SQL` 토큰을 컬럼으로 오인하지 않는다",
            "actual": (
                education_chain_validation.ok,
                education_chain_validation.column,
                education_chain_validation.chart_type,
            ),
            "expected": (True, "education", "bar"),
        },
        {
            "description": "집계 SQL 결과 자동 EDA 검증은 원문 filter 컬럼 대신 df_A의 target/stat_count를 사용한다",
            "actual": (
                housing_loan_chain_validation.ok,
                housing_loan_chain_validation.column,
                housing_loan_chain_validation.chart_type,
            ),
            "expected": (True, "loan", "bar"),
        },
    ]

    passed = 0
    failed = 0
    for idx, tc in enumerate(test_cases, 1):
        print(f"[eda-validation-{idx}] {tc['description']}")
        if tc["actual"] == tc["expected"]:
            print("   ✅ PASS\n")
            passed += 1
        else:
            print(f"   ❌ FAIL (Expected: {tc['expected']}, Got: {tc['actual']})\n")
            failed += 1

    print("-" * 50)
    print(f"🎯 EDA validation 테스트 결과: {passed} 통과, {failed} 실패\n")
    return failed


def run_controlled_plan_tests():
    print("🧪 Controlled production flow plan 테스트 실행...\n")

    import pandas as pd

    bank_context = _bank_table_context()
    prompt = "20대에서 30대 사이의 대출을 가지고 있는 사람들의 balance에 대해서 시각화해줘"
    plan = build_controlled_plan(
        prompt,
        default_table="workspace.default.bank_loan",
        table_context=bank_context,
    )
    sql = build_sql_from_plan(plan) if plan else ""
    condition_coverage = validate_condition_coverage(prompt, plan, bank_context)
    aliasless_bank_context = _bank_table_context_without_aliases()
    aliasless_plan = build_controlled_plan(
        prompt,
        default_table="workspace.default.bank_loan",
        table_context=aliasless_bank_context,
    )
    aliasless_condition_coverage = validate_condition_coverage(
        prompt,
        aliasless_plan,
        aliasless_bank_context,
    )
    numeric_range_prompt = "age가 20~ 30 사이의 loan 값이 yes인 사람들의 balance에 대해서 시각화해줘"
    numeric_range_plan = build_controlled_plan(
        numeric_range_prompt,
        default_table="workspace.default.bank_loan",
        table_context=bank_context,
    )
    numeric_range_sql = build_sql_from_plan(numeric_range_plan) if numeric_range_plan else ""
    numeric_range_requirement = (
        requirement_from_controlled_plan(numeric_range_plan)
        if numeric_range_plan
        else DataRequirement()
    )
    numeric_range_condition_coverage = validate_condition_coverage(
        numeric_range_prompt,
        numeric_range_plan,
        bank_context,
    )
    unresolved_numeric_range_prompt = "나이가 20~ 30 사이의 loan 값이 yes인 사람들의 balance에 대해서 시각화해줘"
    unresolved_numeric_range_plan = build_controlled_plan(
        unresolved_numeric_range_prompt,
        default_table="workspace.default.bank_loan",
        table_context=aliasless_bank_context,
    )
    unresolved_numeric_range_condition_coverage = validate_condition_coverage(
        unresolved_numeric_range_prompt,
        unresolved_numeric_range_plan,
        aliasless_bank_context,
    )
    large_config = select_visualization_config(plan, pd.DataFrame({"balance": range(101)})) if plan else None
    small_config = select_visualization_config(plan, pd.DataFrame({"balance": range(100)})) if plan else None
    housing_loan_prompt = "housing이 yes 인사람들의 loan 분포를 그려줘"
    housing_loan_plan = build_controlled_plan(
        housing_loan_prompt,
        default_table="workspace.default.bank_loan",
        table_context=bank_context,
    )
    housing_loan_sql = build_sql_from_plan(housing_loan_plan) if housing_loan_plan else ""
    housing_loan_config = (
        select_visualization_config(
            housing_loan_plan,
            pd.DataFrame({"loan": ["yes", "no", "no"], "housing": ["yes", "yes", "yes"]}),
        )
        if housing_loan_plan
        else None
    )
    housing_loan_requirement = (
        requirement_from_controlled_plan(housing_loan_plan)
        if housing_loan_plan
        else DataRequirement()
    )
    balance_only_state = DataFrameState(
        role="query_result",
        source_table="workspace.default.bank_loan",
        query=f"SELECT balance FROM workspace.default.bank_loan LIMIT {SQL_LIMIT_DEFAULT}",
        columns=("balance",),
        row_count=750000,
        created_by="test",
    )
    housing_loan_readiness = evaluate_data_readiness(
        balance_only_state,
        housing_loan_requirement,
    )
    technician_housing_prompt = "job column이 technician 인 사람들의 housing 여부를 시각화 해줘"
    technician_housing_plan = build_controlled_plan(
        technician_housing_prompt,
        default_table="workspace.default.bank_loan",
        table_context=bank_context,
    )
    technician_housing_sql = build_sql_from_plan(technician_housing_plan) if technician_housing_plan else ""
    technician_housing_requirement = (
        requirement_from_controlled_plan(technician_housing_plan)
        if technician_housing_plan
        else DataRequirement()
    )
    housing_only_state = DataFrameState(
        role="query_result",
        source_table="workspace.default.bank_loan",
        query="SELECT housing FROM workspace.default.bank_loan LIMIT 2000",
        columns=("housing",),
        row_count=2000,
        created_by="test",
    )
    technician_housing_readiness = evaluate_data_readiness(
        housing_only_state,
        technician_housing_requirement,
    )
    job_without_technician_context = build_trained_context(
        _bank_schema_context(),
        row_count=750000,
        column_profiles={
            "job": {
                "null_count": 0,
                "distinct_count": 12,
                "top_values": [{"value": "management", "count": 100}],
            },
            "housing": {
                "null_count": 0,
                "distinct_count": 2,
                "top_values": [{"value": "yes", "count": 100}],
            },
        },
    )
    missing_top_value_plan = build_controlled_plan(
        technician_housing_prompt,
        default_table="workspace.default.bank_loan",
        table_context=job_without_technician_context,
    )
    shared_value_context = build_trained_context(
        _bank_schema_context(),
        row_count=750000,
        column_profiles={
            "job": {
                "null_count": 0,
                "distinct_count": 12,
                "top_values": [{"value": "technician", "count": 80}],
            },
            "education": {
                "null_count": 0,
                "distinct_count": 4,
                "top_values": [{"value": "technician", "count": 5}],
            },
            "housing": {
                "null_count": 0,
                "distinct_count": 2,
                "top_values": [{"value": "yes", "count": 100}],
            },
        },
    )
    ambiguous_value_plan = build_controlled_plan(
        "technician 인 사람들의 housing 여부를 시각화 해줘",
        default_table="workspace.default.bank_loan",
        table_context=shared_value_context,
    )
    education_prompt = "전체 education의 분포를 보고 싶어"
    education_plan = build_controlled_plan(
        education_prompt,
        default_table="workspace.default.bank_loan",
        table_context=bank_context,
    )
    education_sql = build_sql_from_plan(education_plan) if education_plan else ""
    education_config = (
        select_visualization_config(
            education_plan,
            pd.DataFrame({"education": ["secondary", "primary", "tertiary"]}),
        )
        if education_plan
        else None
    )
    top_balance_job_prompt = "balance 가 가장 높은 상위 10%의 사람들의 직업이 어떻게 되는지 분포를 그려줘"
    top_balance_job_plan = build_controlled_plan(
        top_balance_job_prompt,
        default_table="workspace.default.bank_loan",
        table_context=bank_context,
    )
    top_balance_job_sql = build_sql_from_plan(top_balance_job_plan) if top_balance_job_plan else ""
    top_balance_job_config = (
        select_visualization_config(
            top_balance_job_plan,
            pd.DataFrame({"job": ["management", "technician"], "stat_count": [20, 10]}),
        )
        if top_balance_job_plan
        else None
    )
    duration_job_prompt = "duration이 500 넘는 사람들의 직업군이 어떻게 되는지 시각화 해줘"
    duration_job_plan = build_controlled_plan(
        duration_job_prompt,
        default_table="workspace.default.bank_loan",
        table_context=bank_context,
    )
    duration_job_sql = build_sql_from_plan(duration_job_plan) if duration_job_plan else ""
    duration_only_state = DataFrameState(
        role="query_result",
        source_table="workspace.default.bank_loan",
        query="SELECT duration FROM workspace.default.bank_loan LIMIT 2000",
        columns=("duration",),
        row_count=2000,
        created_by="test",
    )
    duration_job_requirement = (
        requirement_from_controlled_plan(duration_job_plan) if duration_job_plan else DataRequirement()
    )
    duration_job_readiness = evaluate_data_readiness(duration_only_state, duration_job_requirement)
    duration_job_config = (
        select_visualization_config(
            duration_job_plan,
            pd.DataFrame({"job": ["management", "technician"], "stat_count": [20, 10]}),
        )
        if duration_job_plan
        else None
    )
    no_alias_duration_job_plan = build_controlled_plan(
        duration_job_prompt,
        default_table="workspace.default.bank_loan",
        table_context=_bank_schema_context(),
    )
    titanic_context = _titanic_table_context()
    grouped_prompt = "survived 값이 1인사람들과 0인 사람들의 Sex(성별) 분포를 각각 시각화 해줘."
    grouped_plan = build_controlled_plan(
        grouped_prompt,
        default_table="workspace.default.titanic",
        table_context=titanic_context,
    )
    mismatched_table_grouped_plan = build_controlled_plan(
        grouped_prompt,
        default_table="workspace.default.bank_loan",
        table_context=bank_context,
    )
    grouped_requirement = requirement_from_controlled_plan(grouped_plan) if grouped_plan else DataRequirement()
    grouped_state = DataFrameState(
        role="query_result",
        source_table="workspace.default.titanic",
        query="SELECT * FROM workspace.default.titanic LIMIT 2000",
        columns=tuple(column.name for column in titanic_context.columns),
        row_count=891,
        created_by="test",
    )
    grouped_readiness = evaluate_data_readiness(grouped_state, grouped_requirement)
    grouped_missing_target_state = DataFrameState(
        role="query_result",
        source_table="workspace.default.titanic",
        query="SELECT Survived FROM workspace.default.titanic LIMIT 2000",
        columns=("Survived",),
        row_count=891,
        created_by="test",
    )
    grouped_missing_target_readiness = evaluate_data_readiness(
        grouped_missing_target_state,
        grouped_requirement,
    )
    grouped_wrong_source_state = DataFrameState(
        role="query_result",
        source_table="workspace.default.bank_loan",
        query="SELECT * FROM workspace.default.bank_loan LIMIT 2000",
        columns=tuple(column.name for column in bank_context.columns),
        row_count=2000,
        created_by="test",
    )
    grouped_wrong_source_readiness = evaluate_data_readiness(
        grouped_wrong_source_state,
        grouped_requirement,
    )
    grouped_sql = build_sql_from_plan(grouped_plan) if grouped_plan else ""
    grouped_config = (
        select_visualization_config(
            grouped_plan,
            pd.DataFrame(
                {
                    "Survived": [1, 1, 0, 0],
                    "Sex": ["female", "male", "male", "female"],
                    "stat_count": [200, 100, 400, 100],
                }
            ),
        )
        if grouped_plan
        else None
    )
    grouped_trace_event = (
        build_trace_event(
            {
                "trace_id": "trace",
                "conversation_id": "conv",
                "turn_id": 1,
                "run_id": "run",
                "event_seq": 1,
            },
            "controlled_plan",
            generated=True,
            plan=grouped_plan,
            table_context_source=titanic_context.source,
            training_status=titanic_context.training_status,
            context_hash="context-hash",
            resolved_from_training=True,
            **(grouped_plan.resolution_debug if grouped_plan else {}),
        )
        if grouped_plan
        else {}
    )
    survived_plan = build_controlled_plan(
        "Survived 분포를 그려줘",
        default_table="workspace.default.titanic",
        table_context=titanic_context,
    )
    sex_parenthetical_plan = build_controlled_plan(
        "Sex(성별) 분포를 보고 싶어",
        default_table="workspace.default.titanic",
        table_context=titanic_context,
    )
    untrained_grouped_plan = build_controlled_plan(
        grouped_prompt,
        default_table="workspace.default.titanic",
        table_context=_titanic_schema_context(),
    )
    planner_source = open(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "utils", "chatbot_plan.py"),
        encoding="utf-8",
    ).read()
    missing_target_debug = explain_controlled_plan_failure(
        "분포를 보여줘",
        table_context=bank_context,
    )
    forbidden_planner_literals = [
        '"job"',
        "'job'",
        '"Sex"',
        "'Sex'",
        '"Survived"',
        "'Survived'",
        '"duration"',
        "'duration'",
        '"balance"',
        "'balance'",
        "직업군",
        "성별",
    ]
    leaked_planner_literals = [
        literal for literal in forbidden_planner_literals if literal in planner_source
    ]

    test_cases = [
        {
            "description": "실제 prompt는 VISUALIZE distribution JSON plan으로 변환된다",
            "actual": bool(plan)
            and plan.intent == "VISUALIZE"
            and plan.task == "distribution"
            and plan.target_column == "balance",
            "expected": True,
        },
        {
            "description": "20대~30대 필터와 대출 보유 필터를 JSON plan에 포함한다",
            "actual": bool(plan)
            and plan.filters.get("age") == [20, 30]
            and plan.filters.get("loan") == "yes",
            "expected": True,
        },
        {
            "description": "alias가 있으면 조건 coverage는 20대~30대와 대출 보유 조건을 사용된 조건으로 기록한다",
            "actual": (
                condition_coverage.get("ok"),
                condition_coverage.get("used_conditions"),
                condition_coverage.get("unused_conditions"),
            ),
            "expected": (
                True,
                ["20대에서 30대 사이", "대출을 가지고 있는"],
                [],
            ),
        },
        {
            "description": "live training처럼 alias가 없으면 조건 coverage가 누락 조건을 감지한다",
            "actual": (
                aliasless_plan.target_column if aliasless_plan else "",
                aliasless_plan.filters if aliasless_plan else {},
                aliasless_condition_coverage.get("ok"),
                aliasless_condition_coverage.get("unused_conditions"),
                len(aliasless_condition_coverage.get("missing_context_hints", [])),
            ),
            "expected": (
                "balance",
                {},
                False,
                ["20대에서 30대 사이", "대출을 가지고 있는"],
                2,
            ),
        },
        {
            "description": "요청 대비 충분성 검증을 위해 plan의 target/filter 컬럼 목록을 산출한다",
            "actual": required_columns_for_plan(plan) if plan else [],
            "expected": ["balance", "age", "loan"],
        },
        {
            "description": "deterministic SQL은 age 조건과 loan = 'yes' 조건을 누락하지 않는다",
            "actual": "age BETWEEN 20 AND 30" in sql and "loan = 'yes'" in sql,
            "expected": True,
        },
        {
            "description": "controlled balance SQL은 target/filter 컬럼을 모두 SELECT하여 df_A_state가 요청 대비 충분해지게 한다",
            "actual": sql.startswith("SELECT balance, age, loan FROM workspace.default.bank_loan"),
            "expected": True,
        },
        {
            "description": "명시적 numeric range prompt는 age range와 loan=yes를 JSON plan에 포함한다",
            "actual": (
                numeric_range_plan.target_column if numeric_range_plan else "",
                numeric_range_plan.filters if numeric_range_plan else {},
                numeric_range_condition_coverage.get("ok"),
                numeric_range_condition_coverage.get("used_conditions"),
            ),
            "expected": (
                "balance",
                {"age": [20, 30], "loan": "yes"},
                True,
                ["age가 20~ 30 사이"],
            ),
        },
        {
            "description": "명시적 numeric range SQL은 age BETWEEN과 loan=yes를 누락하지 않는다",
            "actual": (
                list(numeric_range_requirement.columns),
                numeric_range_sql.startswith("SELECT balance, age, loan FROM workspace.default.bank_loan"),
                "WHERE age BETWEEN 20 AND 30 AND loan = 'yes'" in numeric_range_sql,
            ),
            "expected": (
                ["balance", "age", "loan"],
                True,
                True,
            ),
        },
        {
            "description": "컬럼/alias와 연결되지 않은 numeric range 표현은 coverage에서 미사용 조건으로 감지한다",
            "actual": (
                unresolved_numeric_range_plan.target_column if unresolved_numeric_range_plan else "",
                unresolved_numeric_range_plan.filters if unresolved_numeric_range_plan else {},
                unresolved_numeric_range_condition_coverage.get("ok"),
                unresolved_numeric_range_condition_coverage.get("unused_conditions"),
                len(unresolved_numeric_range_condition_coverage.get("missing_context_hints", [])),
            ),
            "expected": (
                "balance",
                {"loan": "yes"},
                False,
                ["나이가 20~ 30 사이"],
                1,
            ),
        },
        {
            "description": "100행 초과 numeric dataframe은 histogram config를 선택한다",
            "actual": large_config.plot_type if large_config else "",
            "expected": "histogram",
        },
        {
            "description": "100행 이하 numeric dataframe은 boxplot config를 선택한다",
            "actual": small_config.plot_type if small_config else "",
            "expected": "boxplot",
        },
        {
            "description": "housing=yes loan 분포 prompt는 loan을 target으로, housing을 filter로 해석한다",
            "actual": (
                housing_loan_plan.target_column if housing_loan_plan else "",
                housing_loan_plan.filters if housing_loan_plan else {},
            ),
            "expected": ("loan", {"housing": "yes"}),
        },
        {
            "description": "housing=yes loan 분포 SQL은 집계 SQL로 생성하고 loan='yes'를 잘못 붙이지 않는다",
            "actual": (
                housing_loan_sql.startswith("SELECT loan, COUNT(*) AS stat_count FROM workspace.default.bank_loan"),
                "housing = 'yes'" in housing_loan_sql,
                "loan = 'yes'" in housing_loan_sql,
                "GROUP BY loan" in housing_loan_sql,
            ),
            "expected": (True, True, False, True),
        },
        {
            "description": "housing=yes loan 분포는 categorical bar config를 선택한다",
            "actual": housing_loan_config.plot_type if housing_loan_config else "",
            "expected": "bar",
        },
        {
            "description": "현재 df_A가 balance만 있으면 housing=yes loan 분포 요청은 controlled reload가 필요하다",
            "actual": (
                list(housing_loan_requirement.columns),
                housing_loan_readiness.decision,
                housing_loan_readiness.missing_columns,
            ),
            "expected": (
                ["loan", "housing"],
                DataReadinessDecision.RELOAD_REQUIRED,
                ("loan", "housing"),
            ),
        },
        {
            "description": "TableContext top_values의 categorical 값은 명시된 컬럼 equality filter로 변환된다",
            "actual": (
                technician_housing_plan.target_column if technician_housing_plan else "",
                technician_housing_plan.filters if technician_housing_plan else {},
            ),
            "expected": ("housing", {"job": "technician"}),
        },
        {
            "description": "categorical value filter SQL은 target categorical count 집계와 WHERE 조건을 생성한다",
            "actual": (
                technician_housing_sql.startswith(
                    "SELECT housing, COUNT(*) AS stat_count FROM workspace.default.bank_loan"
                ),
                "WHERE job = 'technician'" in technician_housing_sql,
                "GROUP BY housing" in technician_housing_sql,
            ),
            "expected": (True, True, True),
        },
        {
            "description": "categorical value filter requirement는 target과 filter 컬럼을 모두 요구한다",
            "actual": list(technician_housing_requirement.columns),
            "expected": ["housing", "job"],
        },
        {
            "description": "현재 df_A가 housing만 갖고 있으면 job=technician 요청은 reload가 필요하다",
            "actual": technician_housing_readiness.decision,
            "expected": DataReadinessDecision.RELOAD_REQUIRED,
        },
        {
            "description": "top_values에 없는 categorical 값은 추측해서 filter로 만들지 않는다",
            "actual": missing_top_value_plan.filters if missing_top_value_plan else {},
            "expected": {},
        },
        {
            "description": "동일 categorical 값이 여러 컬럼에 있어도 컬럼 언급이 없으면 filter로 만들지 않는다",
            "actual": ambiguous_value_plan.filters if ambiguous_value_plan else {},
            "expected": {},
        },
        {
            "description": "전체 education 분포 prompt는 LLM chain 대신 controlled plan으로 변환된다",
            "actual": (
                education_plan.target_column if education_plan else "",
                education_plan.filters if education_plan else {},
            ),
            "expected": ("education", {}),
        },
        {
            "description": "전체 education 분포 SQL은 education만 reload한다",
            "actual": education_sql,
            "expected": f"SELECT education FROM workspace.default.bank_loan LIMIT {SQL_LIMIT_DEFAULT}",
        },
        {
            "description": "전체 education 분포는 categorical bar config를 선택한다",
            "actual": education_config.plot_type if education_config else "",
            "expected": "bar",
        },
        {
            "description": "상위 10% balance 직업 분포 prompt는 job target과 balance rank filter로 변환된다",
            "actual": (
                top_balance_job_plan.task if top_balance_job_plan else "",
                top_balance_job_plan.target_column if top_balance_job_plan else "",
                top_balance_job_plan.rank_column if top_balance_job_plan else "",
                top_balance_job_plan.rank_percent if top_balance_job_plan else None,
                top_balance_job_plan.rank_direction if top_balance_job_plan else "",
            ),
            "expected": ("ranked_distribution", "job", "balance", 10.0, "top"),
        },
        {
            "description": "상위 10% balance 직업 분포 SQL은 percentile_approx와 집계 count를 사용한다",
            "actual": (
                top_balance_job_sql.startswith("SELECT job, COUNT(*) AS stat_count FROM workspace.default.bank_loan"),
                "percentile_approx(balance, 0.9)" in top_balance_job_sql,
                "GROUP BY job" in top_balance_job_sql,
                "ORDER BY stat_count DESC" in top_balance_job_sql,
            ),
            "expected": (True, True, True, True),
        },
        {
            "description": "상위 10% balance 직업 분포 결과는 categorical bar config를 선택한다",
            "actual": top_balance_job_config.plot_type if top_balance_job_config else "",
            "expected": "bar",
        },
        {
            "description": "duration > 500 직업군 분포는 TableContext alias로 target=job, condition=duration > 500을 만든다",
            "actual": (
                duration_job_plan.target_column if duration_job_plan else "",
                [
                    (condition.column, condition.op, condition.value)
                    for condition in (duration_job_plan.filter_conditions if duration_job_plan else ())
                ],
            ),
            "expected": ("job", [("duration", ">", 500)]),
        },
        {
            "description": "duration > 500 직업군 분포 requirement는 target과 filter 컬럼을 모두 요구한다",
            "actual": list(duration_job_requirement.columns),
            "expected": ["job", "duration"],
        },
        {
            "description": "현재 df_A가 duration만 갖고 있으면 job 분포 요청은 reload가 필요하다",
            "actual": duration_job_readiness.decision,
            "expected": DataReadinessDecision.RELOAD_REQUIRED,
        },
        {
            "description": "duration > 500 직업군 분포 SQL은 조건부 categorical 집계 SQL을 생성한다",
            "actual": (
                duration_job_sql.startswith("SELECT job, COUNT(*) AS stat_count FROM workspace.default.bank_loan"),
                "WHERE duration > 500" in duration_job_sql,
                "GROUP BY job" in duration_job_sql,
                "ORDER BY stat_count DESC" in duration_job_sql,
            ),
            "expected": (True, True, True, True),
        },
        {
            "description": "duration > 500 직업군 분포 결과는 bar config를 선택한다",
            "actual": duration_job_config.plot_type if duration_job_config else "",
            "expected": "bar",
        },
        {
            "description": "직업군 alias가 없는 TableContext에서는 코드 하드코딩으로 job을 추측하지 않는다",
            "actual": no_alias_duration_job_plan is None,
            "expected": True,
        },
        {
            "description": "trained TableContext가 없으면 prompt의 실제 컬럼명도 controlled resolver가 추측하지 않는다",
            "actual": untrained_grouped_plan is None,
            "expected": True,
        },
        {
            "description": "정확한 survived/Sex(성별) prompt는 선택된 titanic TableContext 기준으로만 plan을 만든다",
            "actual": (
                grouped_prompt,
                grouped_plan.table if grouped_plan else "",
                mismatched_table_grouped_plan is None,
            ),
            "expected": (
                "survived 값이 1인사람들과 0인 사람들의 Sex(성별) 분포를 각각 시각화 해줘.",
                "workspace.default.titanic",
                True,
            ),
        },
        {
            "description": "값이 1/0인 사람들의 target 분포는 group/filter와 target 역할을 분리한다",
            "actual": (
                grouped_plan.task if grouped_plan else "",
                grouped_plan.target_column if grouped_plan else "",
                grouped_plan.group_column if grouped_plan else "",
                list(grouped_plan.group_values) if grouped_plan else [],
                grouped_plan.group_mode if grouped_plan else "",
            ),
            "expected": ("grouped_distribution", "Sex", "Survived", [1, 0], "separate"),
        },
        {
            "description": "grouped distribution requirement는 target과 group 컬럼을 모두 요구한다",
            "actual": list(grouped_requirement.columns),
            "expected": ["Sex", "Survived"],
        },
        {
            "description": "현재 df_A가 grouped distribution 컬럼을 모두 갖고 있으면 USE_CURRENT가 된다",
            "actual": grouped_readiness.decision,
            "expected": DataReadinessDecision.USE_CURRENT,
        },
        {
            "description": "현재 df_A가 group 컬럼만 갖고 target이 없으면 reload가 필요하다",
            "actual": grouped_missing_target_readiness.decision,
            "expected": DataReadinessDecision.RELOAD_REQUIRED,
        },
        {
            "description": "현재 df_A source가 선택된 table과 다르면 selected table 기준으로 reload가 필요하다",
            "actual": grouped_wrong_source_readiness.decision,
            "expected": DataReadinessDecision.RELOAD_REQUIRED,
        },
        {
            "description": "grouped distribution SQL은 group/target COUNT 집계로 생성된다",
            "actual": (
                grouped_sql.startswith("SELECT Survived, Sex, COUNT(*) AS stat_count FROM workspace.default.titanic"),
                "WHERE Survived IN (1, 0)" in grouped_sql,
                "GROUP BY Survived, Sex" in grouped_sql,
                "ORDER BY Survived, stat_count DESC" in grouped_sql,
            ),
            "expected": (True, True, True, True),
        },
        {
            "description": "grouped distribution은 grouped_bar config를 선택한다",
            "actual": (
                grouped_config.plot_type if grouped_config else "",
                grouped_config.column if grouped_config else "",
                grouped_config.group_column if grouped_config else "",
                list(grouped_config.group_values) if grouped_config else [],
                grouped_config.group_mode if grouped_config else "",
            ),
            "expected": ("grouped_bar", "Sex", "Survived", [1, 0], "separate"),
        },
        {
            "description": "단일 컬럼 분포 요청은 group 없이 해당 컬럼을 target으로 유지한다",
            "actual": (
                survived_plan.task if survived_plan else "",
                survived_plan.target_column if survived_plan else "",
                survived_plan.group_column if survived_plan else "",
            ),
            "expected": ("distribution", "Survived", ""),
        },
        {
            "description": "컬럼명(설명) 표현은 trained TableContext에 실제 컬럼명이 있을 때만 target으로 인정된다",
            "actual": sex_parenthetical_plan.target_column if sex_parenthetical_plan else "",
            "expected": "Sex",
        },
        {
            "description": "runtime trace에는 training 기반 target/group 선택 근거가 남는다",
            "actual": (
                grouped_trace_event.get("resolved_from_training"),
                grouped_trace_event.get("training_status"),
                grouped_trace_event.get("target_column"),
                grouped_trace_event.get("group_column"),
                grouped_trace_event.get("group_values"),
                bool(grouped_trace_event.get("column_mentions")),
                bool(grouped_trace_event.get("target_candidates")),
            ),
            "expected": (True, "trained", "Sex", "Survived", [1, 0], True, True),
        },
        {
            "description": "controlled plan 생성 실패 debug에는 target 미해결 사유가 남는다",
            "actual": (
                missing_target_debug.get("failure_reason"),
                missing_target_debug.get("target_column"),
                missing_target_debug.get("target_candidates"),
            ),
            "expected": ("target_column_not_resolved", "", []),
        },
        {
            "description": "planner production code에는 table-specific 컬럼/alias literal을 넣지 않는다",
            "actual": leaked_planner_literals,
            "expected": [],
        },
    ]

    passed = 0
    failed = 0
    for idx, tc in enumerate(test_cases, 1):
        print(f"[controlled-plan-{idx}] {tc['description']}")
        if tc["actual"] == tc["expected"]:
            print("   ✅ PASS\n")
            passed += 1
        else:
            print(f"   ❌ FAIL (Expected: {tc['expected']}, Got: {tc['actual']})\n")
            failed += 1

    print("-" * 50)
    print(f"🎯 Controlled production flow 테스트 결과: {passed} 통과, {failed} 실패\n")
    return failed


def run_data_context_tests():
    print("🧪 Stateful DataContext readiness 테스트 실행...\n")

    import pandas as pd

    balance_state = DataFrameState(
        role="query_result",
        source_table="workspace.default.bank_loan",
        query="SELECT balance FROM workspace.default.bank_loan LIMIT 2000",
        columns=("balance",),
        row_count=2000,
        created_by="databricks_sql",
    )
    unknown_source_state = DataFrameState(
        role="query_result",
        source_table="",
        query="SELECT balance FROM unknown",
        columns=("balance",),
        row_count=10,
        created_by="test",
    )
    preview_full_columns_state = DataFrameState(
        role="preview",
        source_table="workspace.default.bank_loan",
        query="SELECT * FROM workspace.default.bank_loan LIMIT 10",
        columns=("balance", "age", "loan"),
        row_count=10,
        is_preview=True,
        created_by="test",
    )
    job_requirement = DataRequirement(columns=("job",), task="distribution")
    balance_requirement = DataRequirement(columns=("balance",), task="distribution")
    filtered_balance_requirement = DataRequirement(
        columns=("balance", "age", "loan"),
        filters={"age": [20, 30], "loan": "yes"},
        task="distribution",
    )
    ranked_job_requirement = DataRequirement(
        columns=("job", "balance"),
        task="ranked_distribution",
        source_table="workspace.default.bank_loan",
    )
    prompt = "20대에서 30대 사이의 대출을 가지고 있는 사람들의 balance에 대해서 시각화해줘"
    plan = build_controlled_plan(
        prompt,
        default_table="workspace.default.bank_loan",
        table_context=_bank_table_context(),
    )
    plan_requirement = requirement_from_controlled_plan(plan) if plan else DataRequirement()
    plan_sql = build_sql_from_plan(plan) if plan else ""
    reload_sql = build_reload_sql_for_requirement(
        DataRequirement(columns=("job",), source_table="workspace.default.bank_loan"),
        "workspace.default.bank_loan",
    )
    dataframe_state_from_pandas = make_dataframe_state(
        pd.DataFrame({"balance": [100], "job": ["admin"]}),
        role="query_result",
        source_table="workspace.default.bank_loan",
        query="SELECT balance, job FROM workspace.default.bank_loan LIMIT 1",
        created_by="test",
    )

    test_cases = [
        {
            "description": "pandas.Index 컬럼 입력은 truth-value ambiguity 없이 정규화된다",
            "actual": normalize_columns(pd.Index(["balance", "job"])),
            "expected": ("balance", "job"),
        },
        {
            "description": "make_dataframe_state는 DataFrame.columns(pd.Index)를 안전하게 저장한다",
            "actual": dataframe_state_from_pandas.columns,
            "expected": ("balance", "job"),
        },
        {
            "description": "df_A_state.columns=['balance'], requirement=['job']이면 reload가 필요하다",
            "actual": evaluate_data_readiness(balance_state, job_requirement).decision,
            "expected": DataReadinessDecision.RELOAD_REQUIRED,
        },
        {
            "description": "df_A_state.columns=['balance'], requirement=['balance']이면 현재 df_A를 사용한다",
            "actual": evaluate_data_readiness(balance_state, balance_requirement).decision,
            "expected": DataReadinessDecision.USE_CURRENT,
        },
        {
            "description": "df_A_state.columns=['balance'], requirement=['balance','age','loan']이면 reload가 필요하다",
            "actual": evaluate_data_readiness(balance_state, filtered_balance_requirement).decision,
            "expected": DataReadinessDecision.RELOAD_REQUIRED,
        },
        {
            "description": "preview df_A는 필요한 컬럼을 모두 갖고 있어도 전체 분포 요청에는 reload가 필요하다",
            "actual": evaluate_data_readiness(preview_full_columns_state, filtered_balance_requirement).decision,
            "expected": DataReadinessDecision.RELOAD_REQUIRED,
        },
        {
            "description": "source table이 없고 요청 컬럼이 부족하면 EDA 실패 대신 명확한 FAIL decision을 반환한다",
            "actual": evaluate_data_readiness(unknown_source_state, job_requirement).decision,
            "expected": DataReadinessDecision.FAIL,
        },
        {
            "description": "%sql job에 대한 분포 데이타는 readiness와 무관하게 SQL Builder로 강제된다",
            "actual": resolve_agent_mode_for_input("%sql job에 대한 분포 데이타"),
            "expected": ("SQL Builder", "job에 대한 분포 데이타"),
        },
        {
            "description": "job reload SQL은 필요한 컬럼만 원본 테이블에서 다시 로드한다",
            "actual": reload_sql,
            "expected": "SELECT job FROM workspace.default.bank_loan LIMIT 2000",
        },
        {
            "description": "controlled balance prompt는 balance, age, loan requirement를 만든다",
            "actual": list(plan_requirement.columns),
            "expected": ["balance", "age", "loan"],
        },
        {
            "description": "controlled balance SQL은 requirement 컬럼 세 개를 SELECT에 반영한다",
            "actual": plan_sql.startswith("SELECT balance, age, loan FROM workspace.default.bank_loan"),
            "expected": True,
        },
        {
            "description": "ranked_distribution 요청은 현재 df_A가 컬럼을 갖고 있어도 source table에서 재계산한다",
            "actual": evaluate_data_readiness(
                DataFrameState(
                    role="query_result",
                    source_table="workspace.default.bank_loan",
                    query="SELECT job, balance FROM workspace.default.bank_loan LIMIT 2000",
                    columns=("job", "balance"),
                    row_count=2000,
                    created_by="test",
                ),
                ranked_job_requirement,
            ).decision,
            "expected": DataReadinessDecision.RELOAD_REQUIRED,
        },
    ]

    passed = 0
    failed = 0
    for idx, tc in enumerate(test_cases, 1):
        print(f"[data-context-{idx}] {tc['description']}")
        if tc["actual"] == tc["expected"]:
            print("   ✅ PASS\n")
            passed += 1
        else:
            print(f"   ❌ FAIL (Expected: {tc['expected']}, Got: {tc['actual']})\n")
            failed += 1

    print("-" * 50)
    print(f"🎯 Stateful DataContext 테스트 결과: {passed} 통과, {failed} 실패\n")
    return failed


def run_figure_attachment_tests():
    print("🧪 Controlled visualization figure attach 테스트 실행...\n")

    run_id = "run-controlled"
    log = [
        {"run_id": run_id, "role": "user", "content": "20대에서 30대 사이의 대출을 가지고 있는 사람들의 balance에 대해서 시각화해줘"},
        {
            "run_id": run_id,
            "role": "assistant",
            "mode": "SQL Execution",
            "content": "Loaded data",
            "figures": [{"kind": "dataframe", "title": "df_A Preview"}],
            "figures_attached": True,
        },
        {
            "run_id": run_id,
            "role": "assistant",
            "mode": "Controlled Executor",
            "content": "Controlled production flow로 실행했습니다.",
            "figures": [],
            "figures_attached": False,
        },
    ]
    attached = attach_figures_to_log(
        log,
        run_id,
        [{"kind": "matplotlib", "title": "balance distribution", "image": "abc"}],
    )

    test_cases = [
        {
            "description": "SQL preview에 이미 figure가 있어도 최신 Controlled Executor 메시지에 matplotlib figure를 붙인다",
            "actual": attached,
            "expected": True,
        },
        {
            "description": "SQL Execution preview 메시지는 기존 dataframe figure만 유지한다",
            "actual": [fig["kind"] for fig in log[1]["figures"]],
            "expected": ["dataframe"],
        },
        {
            "description": "Controlled Executor 메시지에 matplotlib figure가 붙는다",
            "actual": [fig["kind"] for fig in log[2]["figures"]],
            "expected": ["matplotlib"],
        },
    ]

    passed = 0
    failed = 0
    for idx, tc in enumerate(test_cases, 1):
        print(f"[figure-attach-{idx}] {tc['description']}")
        if tc["actual"] == tc["expected"]:
            print("   ✅ PASS\n")
            passed += 1
        else:
            print(f"   ❌ FAIL (Expected: {tc['expected']}, Got: {tc['actual']})\n")
            failed += 1

    print("-" * 50)
    print(f"🎯 Figure attach 테스트 결과: {passed} 통과, {failed} 실패\n")
    return failed


def run_explicit_sql_routing_tests():
    print("🧪 Explicit %sql routing 테스트 실행...\n")

    test_cases = [
        {
            "description": "%sql 프리픽스는 df_A가 전체 로드 상태여도 EDA가 아니라 SQL Builder로 강제된다",
            "actual": should_force_sql_builder("sql"),
            "expected": True,
        },
        {
            "description": "프리픽스가 없으면 SQL Builder 강제 라우팅을 하지 않는다",
            "actual": should_force_sql_builder(None),
            "expected": False,
        },
        {
            "description": "전체 입력 `%sql job에 대한 분포 데이타`는 agent_mode=SQL Builder와 request=job...으로 해석된다",
            "actual": resolve_agent_mode_for_input("%sql job에 대한 분포 데이타"),
            "expected": ("SQL Builder", "job에 대한 분포 데이타"),
        },
        {
            "description": "자연어 시각화 요청의 `분포` 키워드는 SQL Builder 강제 전환을 하지 않는다",
            "actual": should_force_sql_from_keywords(
                matched_keywords=["분포"],
                is_visualization_request=True,
            ),
            "expected": False,
        },
        {
            "description": "비시각화 데이터 조회 키워드는 기존처럼 SQL Builder 강제 전환을 유지한다",
            "actual": should_force_sql_from_keywords(
                matched_keywords=["찾아"],
                is_visualization_request=False,
            ),
            "expected": True,
        },
        {
            "description": "SQL Builder가 `SQL:` 라벨 없이 plain SELECT만 반환해도 SQL로 추출한다",
            "actual": extract_sql_from_text(
                "SELECT job, COUNT(*) AS job_count FROM workspace.default.bank_loan GROUP BY job LIMIT 2000"
            ),
            "expected": "SELECT job, COUNT(*) AS job_count FROM workspace.default.bank_loan GROUP BY job LIMIT 2000",
        },
    ]

    passed = 0
    failed = 0
    for idx, tc in enumerate(test_cases, 1):
        print(f"[explicit-sql-{idx}] {tc['description']}")
        if tc["actual"] == tc["expected"]:
            print("   ✅ PASS\n")
            passed += 1
        else:
            print(f"   ❌ FAIL (Expected: {tc['expected']}, Got: {tc['actual']})\n")
            failed += 1

    print("-" * 50)
    print(f"🎯 Explicit %sql routing 테스트 결과: {passed} 통과, {failed} 실패\n")
    return failed


def run_external_llm_config_tests():
    print("🧪 External LLM config preflight 테스트 실행...\n")

    test_cases = [
        {
            "description": "google provider는 GOOGLE_API_KEY가 있으면 external LLM 실행 가능 상태다",
            "actual": external_llm_config_errors("google", {"GOOGLE_API_KEY": "test-key"}),
            "expected": [],
        },
        {
            "description": "google provider는 GOOGLE_API_KEY가 없으면 실행 전 명확히 실패한다",
            "actual": external_llm_config_errors("google", {}),
            "expected": ["GOOGLE_API_KEY is required."],
        },
        {
            "description": "azure provider는 필수 Azure OpenAI 환경변수를 모두 검사한다",
            "actual": external_llm_config_errors(
                "azure",
                {
                    "AZURE_OPENAI_API_KEY": "test-key",
                    "AZURE_OPENAI_ENDPOINT": "https://example.openai.azure.com/",
                },
            ),
            "expected": ["AZURE_OPENAI_DEPLOYMENT is required."],
        },
        {
            "description": "ollama provider는 external LLM 테스트에서 거부된다",
            "actual": external_llm_config_errors("ollama", {}),
            "expected": ["LLM_PROVIDER=ollama is local. Use google or azure for --external-llm."],
        },
    ]

    passed = 0
    failed = 0
    for idx, tc in enumerate(test_cases, 1):
        print(f"[external-config-{idx}] {tc['description']}")
        if tc["actual"] == tc["expected"]:
            print("   ✅ PASS\n")
            passed += 1
        else:
            print(f"   ❌ FAIL (Expected: {tc['expected']}, Got: {tc['actual']})\n")
            failed += 1

    print("-" * 50)
    print(f"🎯 External LLM config preflight 테스트 결과: {passed} 통과, {failed} 실패\n")
    return failed


def _format_sql_prompt_system_text(**kwargs):
    from core.prompt import build_sql_prompt

    prompt = build_sql_prompt([], **kwargs)
    messages = prompt.format_messages(
        input="테이블 분포를 보여줘",
        agent_scratchpad="",
    )
    return "\n".join(str(getattr(message, "content", "")) for message in messages)


def run_sql_prompt_tests():
    print("🧪 SQL prompt TableContext construction 테스트 실행...\n")

    import importlib.util
    import pandas as pd

    prompt_source = open(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "core", "prompt.py"),
        encoding="utf-8",
    ).read()
    dependency_available = importlib.util.find_spec("langchain_core") is not None
    if not dependency_available:
        test_cases = [
            {
                "description": "core/prompt.py는 더 이상 df_columns partial var를 조건 없이 주입하지 않는다",
                "actual": 'partial_vars["df_columns"]' in prompt_source,
                "expected": False,
            },
        ]
        passed = 0
        failed = 0
        for idx, tc in enumerate(test_cases, 1):
            print(f"[sql-prompt-{idx}] {tc['description']}")
            if tc["actual"] == tc["expected"]:
                print("   ✅ PASS\n")
                passed += 1
            else:
                print(f"   ❌ FAIL (Expected: {tc['expected']}, Got: {tc['actual']})\n")
                failed += 1
        print("   ℹ️ langchain_core가 없어 runtime prompt 생성 테스트는 venv/full scenario에서 검증합니다.\n")
        print("-" * 50)
        print(f"🎯 SQL prompt 테스트 결과: {passed} 통과, {failed} 실패\n")
        return failed

    preview_df = pd.DataFrame(
        {
            "Survived": [1, 0],
            "Sex": ["female", "male"],
        }
    )
    trained_context = _titanic_table_context()
    schema_only_context = _titanic_schema_context()

    try:
        trained_prompt_text = _format_sql_prompt_system_text(
            selected_table="workspace.default.titanic",
            selected_catalog="workspace",
            selected_schema="default",
            df_preview=preview_df,
            df_name="workspace.default.titanic",
            table_context=trained_context,
        )
        trained_prompt_error = ""
    except Exception as exc:
        trained_prompt_text = ""
        trained_prompt_error = type(exc).__name__

    try:
        schema_only_prompt_text = _format_sql_prompt_system_text(
            selected_table="workspace.default.titanic",
            selected_catalog="workspace",
            selected_schema="default",
            df_preview=preview_df,
            df_name="workspace.default.titanic",
            table_context=schema_only_context,
        )
        schema_only_prompt_error = ""
    except Exception as exc:
        schema_only_prompt_text = ""
        schema_only_prompt_error = type(exc).__name__

    try:
        file_prompt_text = _format_sql_prompt_system_text(
            selected_table="",
            df_preview=preview_df,
            df_name="local preview",
            table_context=None,
        )
        file_prompt_error = ""
    except Exception as exc:
        file_prompt_text = ""
        file_prompt_error = type(exc).__name__

    test_cases = [
        {
            "description": "core/prompt.py는 더 이상 df_columns partial var를 조건 없이 주입하지 않는다",
            "actual": 'partial_vars["df_columns"]' in prompt_source,
            "expected": False,
        },
        {
            "description": "selected table + trained TableContext + df_preview가 있어도 df_columns UnboundLocalError 없이 prompt를 만든다",
            "actual": (
                trained_prompt_error,
                "Selected table context for SQL generation:" in trained_prompt_text,
                "source-table schema/profile must come from the trained TableContext below" in trained_prompt_text,
                "head():" in trained_prompt_text,
            ),
            "expected": ("", True, True, False),
        },
        {
            "description": "selected table + schema_only context는 preview row로 table 컬럼 의미를 추론하지 말고 training 안내를 넣는다",
            "actual": (
                schema_only_prompt_error,
                "%table training" in schema_only_prompt_text,
                "head():" in schema_only_prompt_text,
                "Selected table context for SQL generation:" in schema_only_prompt_text,
            ),
            "expected": ("", True, False, False),
        },
        {
            "description": "selected table이 없는 local dataframe prompt는 기존 preview 정보를 안전하게 포함한다",
            "actual": (
                file_prompt_error,
                "Active dataframe preview for SQL generation:" in file_prompt_text,
                "head():" in file_prompt_text,
            ),
            "expected": ("", True, True),
        },
    ]

    passed = 0
    failed = 0
    for idx, tc in enumerate(test_cases, 1):
        print(f"[sql-prompt-{idx}] {tc['description']}")
        if tc["actual"] == tc["expected"]:
            print("   ✅ PASS\n")
            passed += 1
        else:
            print(f"   ❌ FAIL (Expected: {tc['expected']}, Got: {tc['actual']})\n")
            failed += 1

    print("-" * 50)
    print(f"🎯 SQL prompt 테스트 결과: {passed} 통과, {failed} 실패\n")
    return failed


def run_table_context_tests():
    print("🧪 TableContext training/cache 테스트 실행...\n")

    import pandas as pd
    import tempfile

    with tempfile.TemporaryDirectory() as temp_dir:
        preview_df = pd.DataFrame(
            {
                "education": ["secondary", "primary"],
                "balance": [100, 200],
            }
        )
        schema_context = load_table_context_for_selection(
            "workspace.default.bank_loan",
            preview_df=preview_df,
            storage_dir=temp_dir,
        )
        trained_context = build_trained_context(
            schema_context,
            row_count=750000,
            column_profiles={
                "education": {
                    "null_count": 0,
                    "distinct_count": 4,
                    "top_values": [{"value": "secondary", "count": 401683}],
                },
                "balance": {
                    "null_count": 0,
                    "distinct_count": 1200,
                    "min_value": -8019,
                    "max_value": 102127,
                    "top_values": [],
                },
            },
        )
        saved_path = save_table_context(trained_context, storage_dir=temp_dir)
        reloaded_context = load_table_context_for_selection(
            "workspace.default.bank_loan",
            storage_dir=temp_dir,
        )
        other_table_context = build_schema_only_context(
            "workspace.default.device_telemetry",
            columns=["device_id", "temperature"],
            dtypes={"device_id": "object", "temperature": "float64"},
        )
        saved_payload = table_context_to_dict(reloaded_context)
        alias_trained_context = build_trained_context(
            _bank_schema_context(),
            row_count=750000,
            column_profiles={
                "job": {"null_count": 0, "distinct_count": 12, "top_values": []},
                "duration": {"null_count": 0, "distinct_count": 1500, "min_value": 0, "max_value": 4918},
            },
        )
        save_table_context(alias_trained_context, storage_dir=temp_dir)
        alias_override_path = ensure_table_context_override_file(alias_trained_context, storage_dir=temp_dir)
        alias_override_payload = json.loads(alias_override_path.read_text(encoding="utf-8"))
        alias_override_payload["columns"]["job"]["aliases"] = ["직업", "직업군", "Job"]
        alias_override_path.write_text(
            json.dumps(alias_override_payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        reloaded_alias_context = load_table_context_for_selection(
            "workspace.default.bank_loan",
            storage_dir=temp_dir,
        )
        alias_override_map = load_table_context_overrides(
            "workspace.default.bank_loan",
            storage_dir=temp_dir,
        )
        retrained_alias_context = build_trained_context(
            reloaded_alias_context,
            row_count=760000,
            column_profiles={
                "job": {"null_count": 0, "distinct_count": 12, "top_values": []},
                "duration": {"null_count": 0, "distinct_count": 1500, "min_value": 0, "max_value": 4918},
            },
        )
        alias_context_payload = table_context_to_dict(reloaded_alias_context)
        titanic_trained_context = _titanic_table_context()
        save_table_context(titanic_trained_context, storage_dir=temp_dir)
        selected_titanic_context = load_table_context_for_selection(
            "workspace.default.titanic",
            storage_dir=temp_dir,
        )
        exact_titanic_prompt = "survived 값이 1인사람들과 0인 사람들의 Sex(성별) 분포를 각각 시각화 해줘."
        selected_titanic_plan = build_controlled_plan(
            exact_titanic_prompt,
            default_table="workspace.default.titanic",
            table_context=selected_titanic_context,
        )
        selected_bank_plan_for_titanic_prompt = build_controlled_plan(
            exact_titanic_prompt,
            default_table="workspace.default.bank_loan",
            table_context=reloaded_alias_context,
        )

    table_command_spec = next(
        (spec for spec in CHAT_COMMAND_SPECS if spec["name"] == "table"),
        {},
    )
    context_plan = build_controlled_plan(
        "전체 education의 분포를 보고 싶어",
        default_table="workspace.default.bank_loan",
        table_context=reloaded_context,
    )
    context_summary = table_context_summary(reloaded_context)
    bulk_stats_sql, bulk_stats_aliases = build_bulk_profile_stats_sql(
        "`workspace`.`default`.`bank_loan`",
        ["education", "balance"],
    )
    bulk_top_sql = build_bulk_top_values_sql(
        "`workspace`.`default`.`bank_loan`",
        ["education", "housing"],
    )
    alias_duration_plan = build_controlled_plan(
        "duration이 500 넘는 사람들의 직업군이 어떻게 되는지 시각화 해줘",
        default_table="workspace.default.bank_loan",
        table_context=reloaded_alias_context,
    )
    aliasless_duration_plan = build_controlled_plan(
        "duration이 500 넘는 사람들의 직업군이 어떻게 되는지 시각화 해줘",
        default_table="workspace.default.bank_loan",
        table_context=_bank_schema_context(),
    )
    training_success_log_fields = table_training_work_log_fields(
        "workspace.default.bank_loan",
        True,
        "Table context training 완료",
    )
    training_fail_log_fields = table_training_work_log_fields(
        "",
        False,
        "학습할 테이블이 선택되어 있지 않습니다.",
    )

    test_cases = [
        {
            "description": "training file이 없으면 preview df로 schema_only context를 만든다",
            "actual": (
                schema_context.training_status,
                [column.name for column in schema_context.columns],
            ),
            "expected": ("schema_only", ["education", "balance"]),
        },
        {
            "description": "%table training 명령 spec이 현재 테이블 학습 명령으로 등록된다",
            "actual": (table_command_spec.get("trigger"), table_command_spec.get("usage")),
            "expected": ("%table", "`%table training`"),
        },
        {
            "description": "trained TableContext JSON에는 raw sample row 키를 저장하지 않는다",
            "actual": (saved_path.name.endswith(".json"), contains_raw_sample_rows(saved_payload)),
            "expected": (True, False),
        },
        {
            "description": "education 컬럼은 hardcoded 후보가 아니라 TableContext에서 resolve된다",
            "actual": resolve_column_from_prompt("전체 education의 분포를 보고 싶어", reloaded_context),
            "expected": "education",
        },
        {
            "description": "다른 table context에서는 이전 table 컬럼 education을 재사용하지 않는다",
            "actual": resolve_column_from_prompt("전체 education의 분포를 보고 싶어", other_table_context),
            "expected": None,
        },
        {
            "description": "df_A가 집계 결과여도 원본 TableContext summary는 원본 table/profile을 유지한다",
            "actual": (
                "workspace.default.bank_loan" in context_summary,
                "education: categorical" in context_summary,
                "secondary" in context_summary,
            ),
            "expected": (True, True, True),
        },
        {
            "description": "controlled plan은 TableContext 기반으로 education target을 만든다",
            "actual": (
                context_plan.target_column if context_plan else "",
                context_plan.table if context_plan else "",
            ),
            "expected": ("education", "workspace.default.bank_loan"),
        },
        {
            "description": "table training stats SQL은 컬럼별 쿼리 대신 단일 bulk SELECT로 생성된다",
            "actual": (
                bulk_stats_sql.count("FROM `workspace`.`default`.`bank_loan`"),
                "c0_null_count" in bulk_stats_sql,
                "c1_distinct_count" in bulk_stats_sql,
                sorted(bulk_stats_aliases.keys()),
            ),
            "expected": (1, True, True, ["balance", "education"]),
        },
        {
            "description": "table training top-values SQL은 컬럼별 접속 대신 하나의 UNION 쿼리로 생성된다",
            "actual": (
                bulk_top_sql.count("WITH value_counts AS"),
                bulk_top_sql.count("UNION ALL"),
                "ROW_NUMBER() OVER (PARTITION BY column_name" in bulk_top_sql,
            ),
            "expected": (1, 1, True),
        },
        {
            "description": "%table training override 파일은 table hash 경로에 저장되고 column alias를 관리한다",
            "actual": (
                alias_override_path == table_context_override_path("workspace.default.bank_loan", storage_dir=temp_dir),
                alias_override_map.get("job"),
            ),
            "expected": (True, ["직업", "직업군", "Job"]),
        },
        {
            "description": "training 결과 JSON에는 aliases가 저장되고 raw/sample rows는 저장되지 않는다",
            "actual": (
                any(
                    column.get("name") == "job" and "직업군" in column.get("aliases", [])
                    for column in alias_context_payload.get("columns", [])
                ),
                contains_raw_sample_rows(alias_context_payload),
            ),
            "expected": (True, False),
        },
        {
            "description": "manual override alias는 training 재실행용 context에도 보존된다",
            "actual": any(
                column.name == "job" and "직업군" in column.aliases
                for column in retrained_alias_context.columns
            ),
            "expected": True,
        },
        {
            "description": "직업군 alias가 있는 TableContext에서만 duration 조건 + job target을 resolve한다",
            "actual": (
                alias_duration_plan.target_column if alias_duration_plan else "",
                [
                    (condition.column, condition.op, condition.value)
                    for condition in (alias_duration_plan.filter_conditions if alias_duration_plan else ())
                ],
                aliasless_duration_plan is None,
            ),
            "expected": ("job", [("duration", ">", 500)], True),
        },
        {
            "description": "선택된 table이 titanic이면 저장된 training file을 로딩해 exact survived/Sex prompt를 해석한다",
            "actual": (
                selected_titanic_context.table_fqn,
                selected_titanic_context.training_status,
                selected_titanic_plan.table if selected_titanic_plan else "",
                selected_titanic_plan.target_column if selected_titanic_plan else "",
                selected_titanic_plan.group_column if selected_titanic_plan else "",
            ),
            "expected": (
                "workspace.default.titanic",
                "trained",
                "workspace.default.titanic",
                "Sex",
                "Survived",
            ),
        },
        {
            "description": "선택된 table이 bank_loan이면 titanic prompt를 이전 table context로 오해하지 않는다",
            "actual": selected_bank_plan_for_titanic_prompt is None,
            "expected": True,
        },
        {
            "description": "%table training 성공은 turn work log에 intent/tool/status/summary를 남긴다",
            "actual": (
                training_success_log_fields.get("intent_for_log"),
                training_success_log_fields.get("tools_used_for_log"),
                training_success_log_fields.get("python_status_for_log"),
                "workspace.default.bank_loan" in training_success_log_fields.get("python_output_summary_for_log", ""),
            ),
            "expected": ("table_training", ["table_context_training"], "success", True),
        },
        {
            "description": "%table training 실패도 turn work log에 fail status와 error message를 남긴다",
            "actual": (
                training_fail_log_fields.get("python_status_for_log"),
                training_fail_log_fields.get("python_error_for_log"),
                "(no table selected)" in training_fail_log_fields.get("python_output_summary_for_log", ""),
            ),
            "expected": ("fail", "학습할 테이블이 선택되어 있지 않습니다.", True),
        },
    ]

    passed = 0
    failed = 0
    for idx, tc in enumerate(test_cases, 1):
        print(f"[table-context-{idx}] {tc['description']}")
        if tc["actual"] == tc["expected"]:
            print("   ✅ PASS\n")
            passed += 1
        else:
            print(f"   ❌ FAIL (Expected: {tc['expected']}, Got: {tc['actual']})\n")
            failed += 1

    print("-" * 50)
    print(f"🎯 TableContext 테스트 결과: {passed} 통과, {failed} 실패\n")
    return failed


def run_table_sample_tests():
    print("🧪 Table Sample 분리 테스트 실행...\n")

    if "streamlit" not in sys.modules:
        sys.modules["streamlit"] = types.SimpleNamespace(session_state={})
    if "dotenv" not in sys.modules:
        sys.modules["dotenv"] = types.SimpleNamespace(load_dotenv=lambda: None)
    import utils.session as session_utils

    session_utils.ensure_session_state()
    original_connector_available = session_utils.databricks_connector_available
    original_load_table = session_utils.databricks_load_table
    original_get_credentials = session_utils.get_databricks_credentials

    original_df_a = pd.DataFrame({"current_result": [999]})
    original_df_a_state = DataFrameState(
        role="query_result",
        source_table="workspace.default.bank_loan",
        query="SELECT current_result FROM workspace.default.bank_loan LIMIT 1",
        columns=("current_result",),
        row_count=1,
        created_by="test",
    )
    session_utils.st.session_state["df_A_data"] = original_df_a
    session_utils.st.session_state["df_A_state"] = original_df_a_state
    session_utils.st.session_state["df_A_name"] = "query_result"
    session_utils.st.session_state["csv_path"] = "databricks://query_result"
    session_utils.st.session_state["df_table_sample"] = None
    session_utils.st.session_state["df_table_sample_table"] = ""
    session_utils.st.session_state["df_table_sample_message"] = ""

    sample_by_table = {
        "workspace.default.bank_loan": pd.DataFrame(
            {
                "job": ["technician", "management", "services"],
                "housing": ["yes", "no", "yes"],
            }
        ),
        "workspace.default.titanic": pd.DataFrame(
            {
                "Survived": [1, 0, 1],
                "Sex": ["female", "male", "female"],
            }
        ),
    }
    load_calls = []

    def fake_load_table(table, creds, limit=None):
        load_calls.append((table, limit))
        return sample_by_table[table].head(limit or 10).copy()

    try:
        session_utils.databricks_connector_available = lambda: True
        session_utils.get_databricks_credentials = lambda: types.SimpleNamespace(catalog="", schema="")
        session_utils.databricks_load_table = fake_load_table

        default_keys_present = all(
            key in session_utils.st.session_state
            for key in ("df_table_sample", "df_table_sample_table", "df_table_sample_message")
        )
        should_load_initial = session_utils.should_load_table_sample("workspace.default.bank_loan")
        ok, message = session_utils.load_table_sample_from_databricks(
            "workspace.default.bank_loan",
            limit=10,
        )
        loaded_sample = session_utils.st.session_state.get("df_table_sample")
        df_a_after = session_utils.st.session_state.get("df_A_data")
        df_a_state_after = session_utils.st.session_state.get("df_A_state")
        should_load_same_table = session_utils.should_load_table_sample("workspace.default.bank_loan")
        should_load_other_table = session_utils.should_load_table_sample("workspace.default.titanic")
        ok_changed, changed_message = session_utils.load_table_sample_from_databricks(
            "workspace.default.titanic",
            limit=10,
        )
        changed_sample = session_utils.st.session_state.get("df_table_sample")
        changed_sample_table = session_utils.st.session_state.get("df_table_sample_table", "")
        should_load_changed_same_table = session_utils.should_load_table_sample("workspace.default.titanic")
        df_a_after_changed_sample = session_utils.st.session_state.get("df_A_data")
        df_a_state_after_changed_sample = session_utils.st.session_state.get("df_A_state")

        with tempfile.TemporaryDirectory() as temp_dir:
            context = load_table_context_for_selection(
                "workspace.default.bank_loan",
                preview_df=loaded_sample,
                storage_dir=temp_dir,
            )

        ui_source = open(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "ui", "data_preview.py"),
            encoding="utf-8",
        ).read()
    finally:
        session_utils.databricks_connector_available = original_connector_available
        session_utils.databricks_load_table = original_load_table
        session_utils.get_databricks_credentials = original_get_credentials

    test_cases = [
        {
            "description": "session defaults에는 Table Sample 전용 키가 존재한다",
            "actual": default_keys_present,
            "expected": True,
        },
        {
            "description": "sample loader는 선택 테이블 10행 sample을 df_table_sample에만 저장한다",
            "actual": (
                should_load_initial,
                ok,
                load_calls[:1],
                isinstance(loaded_sample, pd.DataFrame),
                loaded_sample.equals(sample_by_table["workspace.default.bank_loan"])
                if isinstance(loaded_sample, pd.DataFrame)
                else False,
                "workspace.default.bank_loan",
                "sample rows loaded" in message,
            ),
            "expected": (
                True,
                True,
                [("workspace.default.bank_loan", 10)],
                True,
                True,
                "workspace.default.bank_loan",
                True,
            ),
        },
        {
            "description": "sample loader는 현재 작업 데이터 df_A_data/df_A_state를 변경하지 않는다",
            "actual": (
                df_a_after.equals(original_df_a) if isinstance(df_a_after, pd.DataFrame) else False,
                df_a_state_after == original_df_a_state,
                df_a_after_changed_sample.equals(original_df_a)
                if isinstance(df_a_after_changed_sample, pd.DataFrame)
                else False,
                df_a_state_after_changed_sample == original_df_a_state,
                session_utils.st.session_state.get("df_A_name"),
                session_utils.st.session_state.get("csv_path"),
            ),
            "expected": (True, True, True, True, "query_result", "databricks://query_result"),
        },
        {
            "description": "같은 테이블은 sample reload가 불필요하고 다른 테이블은 reload가 필요하다",
            "actual": (should_load_same_table, should_load_other_table),
            "expected": (False, True),
        },
        {
            "description": "sidebar table 선택이 바뀌면 df_table_sample도 새 테이블 sample로 교체된다",
            "actual": (
                ok_changed,
                changed_sample_table,
                changed_sample.equals(sample_by_table["workspace.default.titanic"])
                if isinstance(changed_sample, pd.DataFrame)
                else False,
                load_calls,
                should_load_changed_same_table,
                "sample rows loaded" in changed_message,
            ),
            "expected": (
                True,
                "workspace.default.titanic",
                True,
                [("workspace.default.bank_loan", 10), ("workspace.default.titanic", 10)],
                False,
                True,
            ),
        },
        {
            "description": "TableContext schema-only 생성은 df_table_sample preview columns를 사용할 수 있다",
            "actual": (
                context.training_status,
                [column.name for column in context.columns],
            ),
            "expected": ("schema_only", ["job", "housing"]),
        },
        {
            "description": "UI 렌더링 코드는 Data Preview와 Table Sample 표시 대상을 분리한다",
            "actual": (
                "Current working df_A" in ui_source,
                "Selected table sample" in ui_source,
                "df_table_sample" in ui_source,
                'with st.popover("📊 Preview Data")' in ui_source,
                "sample_col, preview_col, _ = st.columns([1, 1, 12], gap=None)" in ui_source,
                "gap: 3px !important" in ui_source,
                'with st.popover("🔎 Sample Data")' in ui_source
                and ui_source.index('with st.popover("🔎 Sample Data")')
                < ui_source.index('with st.popover("📊 Preview Data")'),
                "df_A.head(10)" in ui_source,
                "table_sample.head(10)" in ui_source,
            ),
            "expected": (True, True, True, True, True, True, True, True, True),
        },
    ]

    passed = 0
    failed = 0
    for idx, tc in enumerate(test_cases, 1):
        print(f"[table-sample-{idx}] {tc['description']}")
        if tc["actual"] == tc["expected"]:
            print("   ✅ PASS\n")
            passed += 1
        else:
            print(f"   ❌ FAIL (Expected: {tc['expected']}, Got: {tc['actual']})\n")
            failed += 1

    print("-" * 50)
    print(f"🎯 Table Sample 테스트 결과: {passed} 통과, {failed} 실패\n")
    return failed


def run_runtime_trace_tests():
    print("🧪 Runtime trace 테스트 실행...\n")

    import pandas as pd
    import tempfile

    context = {
        "trace_id": "trace-1",
        "conversation_id": "conv-1",
        "turn_id": 7,
        "run_id": "run-1",
        "event_seq": 1,
    }
    event = build_trace_event(context, "router_result", intent_type="VISUALIZE")
    df_payload = sanitize_for_trace(pd.DataFrame({"loan": ["yes", "no", "yes"], "count": [2, 1, 3]}))
    redacted_payload = sanitize_for_trace(
        {
            "api_key": "secret-key",
            "nested": {"access_token": "secret-token", "safe": "visible"},
        }
    )
    fixture = runtime_trace_fixture_for_keyword_sql()
    fixed_keyword_decision = should_force_sql_from_keywords(
        matched_keywords=["분포"],
        is_visualization_request=True,
    )

    with tempfile.TemporaryDirectory() as temp_dir:
        start_turn_trace(
            conversation_id="conv-file",
            turn_id=3,
            run_id="run-file",
            user_message="housing이 yes 인사람들의 loan 분포를 그려줘",
            storage_dir=temp_dir,
        )
        record_trace_event(
            "keyword_forced_sql",
            matched_keywords=["분포"],
            keyword_forced_sql=fixed_keyword_decision,
            actual_command_prefix=False,
        )
        finish_turn_trace(final_status="completed", chain_triggered=False)
        recent_events = read_recent_traces(limit=5, storage_dir=temp_dir)
        latest_trace = read_latest_trace(storage_dir=temp_dir)

    sql_command_event = build_trace_event(
        {**context, "event_seq": 2},
        "command_detected",
        command_name="sql",
        actual_command_prefix=True,
        command_prefix="sql",
    )
    natural_keyword_event = fixture["keyword_forced_sql"]

    test_cases = [
        {
            "description": "trace event는 conversation_id, turn_id, run_id, event_type을 포함한다",
            "actual": (
                event.get("conversation_id"),
                event.get("turn_id"),
                event.get("run_id"),
                event.get("event_type"),
            ),
            "expected": ("conv-1", 7, "run-1", "router_result"),
        },
        {
            "description": "DataFrame은 전체 저장 대신 row_count, columns, dtypes, sample로 제한된다",
            "actual": (
                df_payload.get("kind"),
                df_payload.get("row_count"),
                df_payload.get("columns"),
                len(df_payload.get("sample", [])),
            ),
            "expected": ("dataframe", 3, ["loan", "count"], 3),
        },
        {
            "description": "민감 키워드 token/api_key/password/secret 값은 redaction된다",
            "actual": (
                redacted_payload.get("api_key"),
                redacted_payload.get("nested", {}).get("access_token"),
                redacted_payload.get("nested", {}).get("safe"),
            ),
            "expected": ("[REDACTED]", "[REDACTED]", "visible"),
        },
        {
            "description": "과거 housing loan 분포 재현 fixture는 분포 keyword forced SQL과 chaining false를 남긴다",
            "actual": (
                fixture["keyword_forced_sql"].get("matched_keywords"),
                fixture["keyword_forced_sql"].get("keyword_forced_sql"),
                fixture["chain_check"].get("llm_router_suggested_chaining"),
                fixture["chain_check"].get("chain_triggered"),
            ),
            "expected": (["분포"], True, False, False),
        },
        {
            "description": "%sql 명령과 과거 자연어 분포 keyword forced SQL은 trace에서 구분된다",
            "actual": (
                sql_command_event.get("actual_command_prefix"),
                sql_command_event.get("command_name"),
                natural_keyword_event.get("actual_command_prefix"),
                natural_keyword_event.get("keyword_forced_sql"),
            ),
            "expected": (True, "sql", False, True),
        },
        {
            "description": "수정 후 자연어 시각화 분포 trace는 keyword_forced_sql=False로 저장된다",
            "actual": (
                bool(recent_events),
                latest_trace.get("summary", {}).get("keyword_forced_sql"),
                latest_trace.get("summary", {}).get("chain_triggered"),
            ),
            "expected": (True, False, False),
        },
    ]

    passed = 0
    failed = 0
    for idx, tc in enumerate(test_cases, 1):
        print(f"[runtime-trace-{idx}] {tc['description']}")
        if tc["actual"] == tc["expected"]:
            print("   ✅ PASS\n")
            passed += 1
        else:
            print(f"   ❌ FAIL (Expected: {tc['expected']}, Got: {tc['actual']})\n")
            failed += 1

    print("-" * 50)
    print(f"🎯 Runtime trace 테스트 결과: {passed} 통과, {failed} 실패\n")
    return failed


def _load_dotenv_for_cli():
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except Exception:
        pass


def run_external_llm_tests():
    print("🌐 External LLM 라우터 테스트 실행...\n")

    _load_dotenv_for_cli()
    provider = os.environ.get("LLM_PROVIDER", "google").strip().lower()
    errors = external_llm_config_errors(provider)
    if errors:
        print(f"❌ External LLM 설정 오류(provider={provider}):")
        for error in errors:
            print(f"   - {error}")
        print("\n예시:")
        print("   LLM_PROVIDER=google GOOGLE_API_KEY=... python3 test_scenario.py --external-llm")
        print("   LLM_PROVIDER=azure AZURE_OPENAI_API_KEY=... AZURE_OPENAI_ENDPOINT=... AZURE_OPENAI_DEPLOYMENT=... python3 test_scenario.py --external-llm")
        return 1

    from core.llm import load_llm
    from core.llm_router import route_query

    try:
        llm = load_llm(temperature=0.0, max_tokens=512)
        print(f"✅ External LLM 로드 성공(provider={provider})\n")
    except Exception as exc:
        print(f"❌ External LLM 로드 실패(provider={provider}): {exc}")
        return 1

    test_cases = [
        {
            "description": "실제 balance 시각화 prompt는 preview 상태에서 SQL Builder → EDA Analyst 체인으로 분류된다",
            "query": "20대에서 30대 사이의 대출을 가지고 있는 사람들의 balance에 대해서 시각화해줘",
            "is_preview_state": True,
            "expected_intents": {"VISUALIZE"},
            "expected_agents": ["SQL Builder", "EDA Analyst"],
        },
        {
            "description": "전체 데이터가 로드된 상태의 job 분포 요청은 EDA Analyst 단독으로 분류된다",
            "query": "job에 대한 분포 데이타",
            "is_preview_state": False,
            "expected_intents": {"VISUALIZE"},
            "expected_agents": ["EDA Analyst"],
        },
        {
            "description": "단순 데이터 조회 요청은 SQL Builder 단독으로 분류된다",
            "query": "나이가 42세인 사람들을 찾아줘",
            "is_preview_state": True,
            "expected_intents": {"LOAD_DATA"},
            "expected_agents": ["SQL Builder"],
        },
        {
            "description": "전체 데이터가 로드된 상태의 요약 요청은 EDA Analyst 단독으로 분류된다",
            "query": "데이터의 요약 통계를 보여줘",
            "is_preview_state": False,
            "expected_intents": {"SUMMARIZE"},
            "expected_agents": ["EDA Analyst"],
        },
    ]

    passed = 0
    failed = 0
    for idx, tc in enumerate(test_cases, 1):
        print(f"[external-llm-{idx}] {tc['description']}")
        print(f"   질문: {tc['query']}")
        print(f"   미리보기 상태(is_preview_state): {tc['is_preview_state']}")
        try:
            plan = route_query(llm, tc["query"], tc["is_preview_state"])
            intent_ok = plan.intent_type in tc["expected_intents"]
            agents_ok = plan.suggested_agents == tc["expected_agents"]
            print(f"   => 분류된 의도(Intent): {plan.intent_type}")
            print(f"   => 추론(Reasoning): {plan.reasoning}")
            print(f"   => 제안된 에이전트(Agents): {plan.suggested_agents}")
            if intent_ok and agents_ok:
                print("   ✅ PASS\n")
                passed += 1
            else:
                print(
                    "   ❌ FAIL "
                    f"(Expected intents={sorted(tc['expected_intents'])}, agents={tc['expected_agents']}; "
                    f"Got intent={plan.intent_type}, agents={plan.suggested_agents})\n"
                )
                failed += 1
        except Exception as exc:
            print(f"   ❌ ERROR 수행 중 예외 발생: {exc}\n")
            failed += 1

    print("-" * 50)
    print(f"🎯 External LLM 라우터 테스트 결과: {passed} 통과, {failed} 실패\n")
    return failed


def run_tests():
    print("🚀 LLM 라우터 테스트 시나리오 실행 시작...\n")
    args = set(sys.argv[1:])

    if "--external-llm" in args:
        static_failed = 0 if "--skip-static" in args else run_static_tests()
        external_failed = run_external_llm_tests()
        sys.exit(1 if static_failed + external_failed else 0)

    static_failed = run_static_tests()

    if "--static-only" in sys.argv:
        sys.exit(1 if static_failed else 0)

    from core.llm import load_llm
    from core.llm_router import route_query
    
    try:
        llm = load_llm()
        print("✅ LLM 로드 성공\n")
    except Exception as e:
        print(f"❌ LLM 로드 실패: {e}")
        sys.exit(1)

    test_cases = [
        {
            "description": "1. 단순 데이터 로딩 요청 (미리보기 상태)",
            "query": "나이가 42세인 사람들을 찾아줘",
            "is_preview_state": True,
            "expected_agents": ["SQL Builder"]
        },
        {
            "description": "2. 시각화가 필요한 데이터 요청 (미리보기 상태 -> 연쇄 실행 필요)",
            "query": "나이가 42세인 사람들의 직업에 대해서 파이 차트로 그려줘",
            "is_preview_state": True,
            "expected_agents": ["SQL Builder", "EDA Analyst"]
        },
        {
            "description": "3. 이미 데이터가 로드된 상태에서의 시각화 요청 (EDA 단독 실행)",
            "query": "나이가 42세인 사람들의 직업에 대해서 파이 차트로 그려줘",
            "is_preview_state": False,
            "expected_agents": ["EDA Analyst"]
        },
        {
            "description": "4. 이미 데이터가 로드된 상태에서의 단순 요약 요청 (EDA 단독 실행)",
            "query": "데이터의 요약 통계를 보여줘",
            "is_preview_state": False,
            "expected_agents": ["EDA Analyst"]
        },
        {
            "description": "5-1. [다단계-1] 20~30대 데이터 로딩 요청",
            "query": "나이가 20대에서 30대 사이의 데이타를 로딩해줘",
            "is_preview_state": True,
            "expected_agents": ["SQL Builder"]
        },
        {
            "description": "5-2. [다단계-2] 데이터 로드 후 시각화 요청",
            "query": "age에 따른 job 분포를 시각화 해줘",
            "is_preview_state": False,
            "expected_agents": ["EDA Analyst"]
        },
        {
            "description": "5-3. [다단계-3] 필터링 추가 시각화 요청",
            "query": "여기에서 unknown은 제거 해서 그려줘",
            "is_preview_state": False,
            "expected_agents": ["EDA Analyst"]
        }
    ]

    passed = 0
    failed = 0

    for idx, tc in enumerate(test_cases, 1):
        print(f"[{idx}] {tc['description']}")
        print(f"   질문: {tc['query']}")
        print(f"   미리보기 상태(is_preview_state): {tc['is_preview_state']}")
        
        try:
            plan = route_query(llm, tc["query"], tc["is_preview_state"])
            print(f"   => 분류된 의도(Intent): {plan.intent_type}")
            print(f"   => 추론(Reasoning): {plan.reasoning}")
            print(f"   => 제안된 에이전트(Agents): {plan.suggested_agents}")
            
            if plan.suggested_agents == tc["expected_agents"]:
                print("   ✅ PASS\n")
                passed += 1
            else:
                print(f"   ❌ FAIL (Expected: {tc['expected_agents']}, Got: {plan.suggested_agents})\n")
                failed += 1
        except Exception as e:
            print(f"   ❌ ERROR 수행 중 예외 발생: {e}\n")
            failed += 1

    print("-" * 50)
    print(f"🎯 테스트 결과: {passed} 통과, {failed} 실패")
    total_failed = static_failed + failed
    if total_failed > 0:
        sys.exit(1)
    else:
        sys.exit(0)

if __name__ == "__main__":
    run_tests()
