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
    page_title="DF Chatbot (Gemini)",
    page_icon="✨",
    layout="wide",
)
st.title("✨ DataFrame Chatbot (Gemini + LangChain)")
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


df_A, df_B, dataset_changed, df_b_changed = _get_dataframes()
df_a_ready = isinstance(df_A, pd.DataFrame)

st.subheader("Preview")
if df_a_ready:
    st.write(
        f"**Loaded file for df_A:** `{st.session_state['df_A_name']}` (Shape: {df_A.shape})"
    )
    st.dataframe(df_A.head(10), width="stretch")
    if isinstance(df_B, pd.DataFrame):
        with st.expander(
            f"df_B Preview — {st.session_state['df_B_name']} (Shape: {df_B.shape})",
            expanded=False,
        ):
            st.dataframe(df_B.head(10), width="stretch")
else:
    st.info(
        "df_A 데이터가 아직 로드되지 않았습니다. 왼쪽 Databricks Loader 또는 SQL Builder 에이전트를 사용해 데이터를 불러오세요."
    )


sql_history = get_history("lc_msgs:sql")
eda_history = get_history("lc_msgs:eda")
if dataset_changed or df_b_changed:
    eda_history.clear()


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
sql_prompt = build_sql_prompt(sql_tools)
_sql_agent, sql_agent_with_history = build_agent(
    llm,
    sql_tools,
    sql_prompt,
    lambda session_id: sql_history,
)


st.write("---")


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


chat_placeholder = (
    "SQL) 예: sales_transactions에서 최근 7일간 매출 합계를 위한 SQL 작성해줘 / "
    "EDA) 예: auto_outlier_eda() / plot_outliers('temperature') / compare_on_keys('machineID,datetime')"
)

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


user_q = st.chat_input(chat_placeholder)

if user_q:
    normalized = user_q.strip().lower()
    if normalized in {"실행", "수행", "run", "execute"}:
        last_sql = st.session_state.get("last_sql_statement", "").strip()
        if not last_sql:
            st.warning("실행할 SQL이 없습니다. 먼저 SQL Builder로 쿼리를 생성해주세요.")
        else:
            st.session_state["last_agent_mode"] = "SQL Builder"
            left, right = st.columns([1, 1])
            with left:
                st.subheader("실시간 실행 로그")
                st.write("SQL Builder의 마지막 쿼리를 Databricks에서 실행합니다.")
            label = st.session_state.get("last_sql_label", "SQL Query")
            cfg = st.session_state.get("databricks_config", {})
            catalog = cfg.get("catalog") or "hive_metastore"
            schema = cfg.get("schema") or "default"
            cfg["catalog"] = catalog
            cfg["schema"] = schema
            st.session_state["databricks_config"] = cfg
            st.session_state.setdefault("databricks_selected_catalog", catalog)
            st.session_state.setdefault("databricks_selected_schema", schema)
            table_name_input = st.session_state.get("databricks_table_input", "").strip()
            table_name_inferred = _infer_table_from_sql(last_sql)
            table_name = table_name_input or table_name_inferred or st.session_state.get(
                "last_sql_table", ""
            )
            if not table_name:
                st.warning(
                    "실행할 테이블을 결정할 수 없습니다. SQL Builder에서 사용할 테이블을 지정하거나 Sidebar에서 테이블을 선택해주세요."
                )
                st.stop()
            with st.spinner("Databricks SQL 실행 중..."):
                success, message = load_preview_from_databricks_query(
                    table_name,
                    query=last_sql,
                    target="A",
                    limit=10,
                )
            with right:
                st.subheader("Answer")
                if success:
                    st.success(message)
                else:
                    st.error(message)
            if success:
                st.info("df_A 미리보기가 업데이트되었습니다. 상단 Preview 섹션을 확인하세요.")
                st.session_state["last_agent_mode"] = "EDA Analyst"
                st.session_state["last_sql_table"] = table_name
                st.session_state["databricks_table_input"] = table_name
        st.stop()

    agent_mode = _infer_agent(user_q)
    st.session_state["last_agent_mode"] = agent_mode

    if agent_mode == "EDA Analyst" and not df_a_ready:
        st.error(
            "df_A 데이터가 없습니다. 먼저 SQL Builder 에이전트나 Databricks Loader로 데이터를 불러온 뒤 다시 시도하세요."
        )
    else:
        left, right = st.columns([1, 1])
        with left:
            st.subheader("실시간 실행 로그")
            st_cb = StreamlitCallbackHandler(st.container())
        collector = SimpleCollectCallback()

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
                    {"input": user_q},
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
        with right:
            st.subheader("Answer")
            final_text = final if isinstance(final, str) else str(final)
            lang_choice = st.session_state.get("explanation_lang", "English")
            final_display = final_text
            if lang_choice == "한국어" and final_text.strip():
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

            if agent_mode == "SQL Builder":
                sql_capture = ""
                if "SQL:" in final_text:
                    tail = final_text.split("SQL:", 1)[1]
                    if "Explanation:" in tail:
                        sql_capture = tail.split("Explanation:", 1)[0].strip()
                    elif "Execution:" in tail:
                        sql_capture = tail.split("Execution:", 1)[0].strip()
                    else:
                        sql_capture = tail.strip()
                if sql_capture:
                    st.session_state["last_sql_statement"] = sql_capture
                    st.session_state["last_sql_label"] = user_q.strip()[:80] or "SQL Query"
                    table_hint = (
                        st.session_state.get("databricks_table_input", "").strip()
                        or _infer_table_from_sql(sql_capture)
                        or st.session_state.get("last_sql_table", "")
                    )
                    if table_hint:
                        st.session_state["last_sql_table"] = table_hint

            if agent_mode == "EDA Analyst" and pytool_obj is not None:
                render_visualizations(pytool_obj)
