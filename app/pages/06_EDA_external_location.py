import streamlit as st
import pandas as pd
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler

from core.agent import (
    SimpleCollectCallback,
    StdOutCallbackHandler,
    build_agent,
)
from core.llm import load_llm
from core.prompt import build_react_prompt
from core.tools import build_tools
from ui.history import get_history
from ui.sidebar import render_sidebar
from ui.viz import render_visualizations
from utils.session import (
    dataframe_signature,
    ensure_session_state,
)


if not st.session_state.get("_page_configured", False):
    st.set_page_config(
        page_title="DF Chatbot (Gemini)",
        page_icon="✨",
        layout="wide",
    )
    st.session_state["_page_configured"] = True
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

if df_A is None or not isinstance(df_A, pd.DataFrame):
    st.error(
        "분석할 DataFrame (df_A)을 로드하지 못했습니다. "
        "유효한 디렉토리와 지원되는 데이터 파일을 선택해주세요."
    )
    st.stop()

st.subheader("Preview")
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


history = get_history()
if dataset_changed or df_b_changed:
    history.clear()


pytool_obj, tools = build_tools(df_A, df_B)
llm = load_llm()
prompt = build_react_prompt(df_A, df_B, tools)

_agent, agent_with_history = build_agent(
    llm,
    tools,
    prompt,
    lambda session_id: history,
)


st.write("---")
user_q = st.chat_input(
    "예) 이상점 EDA 해줘 / auto_outlier_eda() / plot_outliers('temperature') / "
    "propose_join_keys / compare_on_keys('machineID,datetime') / mismatch_report('temperature') / "
    "rolling_stats(cols='temperature,uncorrectable_error_count', window='24H') / stl_decompose('temperature', 24)"
)

if user_q:
    left, right = st.columns([1, 1])
    with left:
        st.subheader("실시간 실행 로그")
        st_cb = StreamlitCallbackHandler(st.container())
    collector = SimpleCollectCallback()

    with st.spinner("Thinking with Gemini..."):
        try:
            result = agent_with_history.invoke(
                {"input": user_q},
                {
                    "callbacks": [st_cb, collector, StdOutCallbackHandler()],
                    "configurable": {"session_id": "two_csv_compare_and_ssd_eda"},
                },
            )
        except Exception as exc:
            st.error(f"Agent 실행 중 오류: {exc}")
            result = {"output": f"Agent 실행 중 오류: {exc}"}

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
        st.write(final_display)

        render_visualizations(pytool_obj)
