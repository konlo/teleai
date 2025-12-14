import os
from uuid import uuid4

import pandas as pd
import streamlit as st

from core.agent import build_agent
from core.chat_flow import handle_user_query
from core.llm import load_llm
from core.prompt import build_react_prompt, build_sql_prompt
from core.sql_tools import build_sql_tools
from core.tools import build_tools
from ui.history import get_history
from ui.chat_log import (
    append_assistant_message,
    append_dataframe_preview_message,
    attach_figures_to_run,
    display_conversation_log,
    ensure_conversation_store,
    next_turn_id,
    render_chat_history,
)
from ui.data_preview import render_data_preview_section
from ui.data_state import load_dataframes
from ui.style import inject_base_styles
from utils.prompt_help import (
    BASE_CHAT_PLACEHOLDER
)


st.set_page_config(
    page_title="Telemetry Chatbot Telly",
    page_icon="✨",
    layout="wide",
)

inject_base_styles()

st.title("✨ Telemetry Chatbot Telly")


st.session_state.setdefault("debug_mode", False)
debug_mode = bool(st.session_state["debug_mode"])
st.session_state.setdefault("conversation_id", str(uuid4()))
st.session_state.setdefault("turn_counter", 0)
st.session_state.setdefault("user_id_hash", os.environ.get("USER_ID_HASH", "konlo.na"))
st.session_state.setdefault("pending_rerun", False)

df_A, df_B, dataset_changed, df_b_changed, df_init = load_dataframes(debug_mode)
df_a_ready = isinstance(df_A, pd.DataFrame)
st.session_state.setdefault("log_has_content", False)
if not debug_mode:
    st.session_state["log_has_content"] = False

ensure_conversation_store()

sql_history = get_history("lc_msgs:sql")
eda_history = get_history("lc_msgs:eda")
if dataset_changed or df_b_changed:
    eda_history.clear()

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


st.write("---")

conversation_placeholder = st.empty()
conversation_log_renderer = lambda: display_conversation_log(conversation_placeholder)

chat_placeholder = BASE_CHAT_PLACEHOLDER

chat_input_key = "main_chat_input"
if chat_input_key not in st.session_state:
    st.session_state[chat_input_key] = ""

user_q = st.chat_input(chat_placeholder, key=chat_input_key)

if user_q:
    handle_user_query(
        user_q,
        debug_mode=debug_mode,
        df_a_ready=df_a_ready,
        log_placeholder=log_placeholder,
        sql_agent_with_history=sql_agent_with_history,
        eda_agent_with_history=eda_agent_with_history,
        pytool_obj=pytool_obj,
        llm=llm,
        next_turn_id_fn=next_turn_id,
        display_conversation_log=conversation_log_renderer,
    )
else:
    conversation_log_renderer()

render_data_preview_section(df_a_ready, df_A, df_B)

if st.session_state.get("pending_rerun"):
    st.session_state["pending_rerun"] = False
    rerun_callable = getattr(st, "rerun", None) or getattr(st, "experimental_rerun", None)
    if callable(rerun_callable):
        rerun_callable()
