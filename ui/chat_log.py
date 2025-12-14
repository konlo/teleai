import base64
from typing import Any, Dict, List, Optional
from uuid import uuid4

import pandas as pd
import streamlit as st


def ensure_conversation_store() -> None:
    """Initialize conversation-related session state keys."""

    st.session_state.setdefault("conversation_log", [])
    st.session_state.setdefault("active_run_id", None)


def append_user_message(run_id: str, content: str) -> None:
    st.session_state["conversation_log"].append(
        {"run_id": run_id, "role": "user", "content": content}
    )


def append_assistant_message(run_id: str, content: str, mode: str) -> None:
    st.session_state["conversation_log"].append(
        {
            "run_id": run_id,
            "role": "assistant",
            "mode": mode,
            "content": content,
            "figures": [],
            "figures_attached": False,
        }
    )


def attach_figures_to_run(run_id: str, figures: List[Dict[str, Any]]) -> None:
    if not run_id or not figures:
        return
    log = st.session_state.get("conversation_log", [])
    for entry in reversed(log):
        if entry.get("run_id") == run_id and entry.get("role") == "assistant":
            if entry.get("figures_attached"):
                return
            entry.setdefault("figures", [])
            entry["figures"].extend(figures)
            entry["figures_attached"] = True
            break


def render_chat_history(title: str, history) -> None:
    st.markdown(f"#### {title}")
    messages = getattr(history, "messages", []) or []
    if not messages:
        st.info("대화 기록이 없습니다.")
        return
    for msg in messages:
        role = getattr(msg, "type", "assistant")
        if role == "human":
            streamlit_role = "user"
        elif role == "ai":
            streamlit_role = "assistant"
        else:
            streamlit_role = role or "assistant"
        content = msg.content if isinstance(msg.content, str) else str(msg.content)
        with st.chat_message(streamlit_role):
            st.markdown(content)


def render_conversation_log(show_header: bool = True) -> None:
    if show_header:
        st.markdown("#### 대화 기록")
    log = st.session_state.get("conversation_log", [])
    if not log:
        st.info("대화 기록이 없습니다.")
        return
    for entry in log:
        role = entry.get("role", "assistant")
        streamlit_role = "assistant" if role == "assistant" else "user"
        with st.chat_message(streamlit_role):
            mode = entry.get("mode")
            if mode and role == "assistant":
                st.caption(mode)
            content = entry.get("content", "")
            if content:
                st.markdown(content)
            for fig in entry.get("figures", []):
                title = fig.get("title")
                if title:
                    st.markdown(f"**{title}**")
                kind = fig.get("kind")
                if kind == "bar_chart":
                    st.bar_chart(fig.get("data"))
                elif kind == "line_chart":
                    st.line_chart(fig.get("data"))
                elif kind == "dataframe":
                    st.dataframe(fig.get("data"))
                elif kind == "json":
                    st.json(fig.get("data"))
                elif kind == "matplotlib":
                    image_b64 = fig.get("image")
                    if image_b64:
                        st.image(base64.b64decode(image_b64))


def display_conversation_log(placeholder, show_header: bool = True) -> None:
    """Render the conversation log inside a provided Streamlit container placeholder."""

    with placeholder.container():
        render_conversation_log(show_header=show_header)


def append_dataframe_preview_message(
    label: str,
    df: pd.DataFrame,
    key: str,
    append_message_fn,
    attach_figures_fn,
) -> None:
    if not isinstance(df, pd.DataFrame) or df.empty:
        return
    preview_df = df.head(10)
    if preview_df.empty:
        return
    dataset_name_key = "df_A_name" if key == "A" else "df_B_name"
    dataset_name = st.session_state.get(dataset_name_key, label)
    message = f"**{label} Preview:** `{dataset_name}` (Shape: {df.shape})"
    run_id = f"preview-{key}-{uuid4()}"
    append_message_fn(run_id, message, "Data Preview")
    attach_figures_fn(
        run_id,
        [
            {
                "kind": "dataframe",
                "title": f"{label} Preview",
                "data": preview_df,
            }
        ],
    )


def next_turn_id() -> int:
    """Increment and return the conversation turn counter."""

    st.session_state["turn_counter"] = st.session_state.get("turn_counter", 0) + 1
    return st.session_state["turn_counter"]


__all__ = [
    "append_assistant_message",
    "append_dataframe_preview_message",
    "append_user_message",
    "attach_figures_to_run",
    "display_conversation_log",
    "ensure_conversation_store",
    "next_turn_id",
    "render_chat_history",
    "render_conversation_log",
]
