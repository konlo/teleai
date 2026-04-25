import os
import sys

# 프로젝트 루트를 경로에 추가
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.agent_output import (
    detect_agent_parser_loop,
    has_invalid_action_schema,
    unwrap_final_answer_payload,
)
from utils.agent_routing import resolve_agent_mode_for_input, should_force_sql_builder
from utils.chat_turn import resolve_chat_turn_query, should_process_chat_turn
from utils.chatbot_plan import (
    build_controlled_plan,
    build_sql_from_plan,
    required_columns_for_plan,
    select_visualization_config,
)
from utils.conversation_figures import attach_figures_to_log
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
    validate_data_sufficiency,
    validate_eda_visualization_request,
)


EXTERNAL_LLM_PROVIDERS = {"google", "azure"}


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
    failed += run_data_context_tests()
    failed += run_figure_attachment_tests()
    failed += run_explicit_sql_routing_tests()
    failed += run_external_llm_config_tests()
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

    prompt = "20대에서 30대 사이의 대출을 가지고 있는 사람들의 balance에 대해서 시각화해줘"
    plan = build_controlled_plan(prompt, default_table="workspace.default.bank_loan")
    sql = build_sql_from_plan(plan) if plan else ""
    large_config = select_visualization_config(plan, pd.DataFrame({"balance": range(101)})) if plan else None
    small_config = select_visualization_config(plan, pd.DataFrame({"balance": range(100)})) if plan else None

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
            "description": "100행 초과 numeric dataframe은 histogram config를 선택한다",
            "actual": large_config.plot_type if large_config else "",
            "expected": "histogram",
        },
        {
            "description": "100행 이하 numeric dataframe은 boxplot config를 선택한다",
            "actual": small_config.plot_type if small_config else "",
            "expected": "boxplot",
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
    job_requirement = DataRequirement(columns=("job",), task="distribution")
    balance_requirement = DataRequirement(columns=("balance",), task="distribution")
    filtered_balance_requirement = DataRequirement(
        columns=("balance", "age", "loan"),
        filters={"age": [20, 30], "loan": "yes"},
        task="distribution",
    )
    prompt = "20대에서 30대 사이의 대출을 가지고 있는 사람들의 balance에 대해서 시각화해줘"
    plan = build_controlled_plan(prompt, default_table="workspace.default.bank_loan")
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
