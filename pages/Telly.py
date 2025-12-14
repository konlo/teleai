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
    handle_user_query(
        user_q,
        debug_mode=debug_mode,
        df_a_ready=df_a_ready,
        log_placeholder=log_placeholder,
        sql_agent_with_history=sql_agent_with_history,
        eda_agent_with_history=eda_agent_with_history,
        pytool_obj=pytool_obj,
        llm=llm,
        next_turn_id_fn=_next_turn_id,
        display_conversation_log=_display_conversation_log,
    )

else:
    _display_conversation_log()

_render_data_preview_section()

if st.session_state.get("pending_rerun"):
    st.session_state["pending_rerun"] = False
    rerun_callable = getattr(st, "rerun", None) or getattr(st, "experimental_rerun", None)
    if callable(rerun_callable):
        rerun_callable()
