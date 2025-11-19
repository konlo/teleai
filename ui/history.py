from langchain_community.chat_message_histories import StreamlitChatMessageHistory

_MAX_HISTORY_MESSAGES = 24


def _enforce_limit(history: StreamlitChatMessageHistory) -> None:
    """Trim the stored history so it never grows without bound."""

    messages = getattr(history, "messages", None)
    if not isinstance(messages, list):
        return
    total = len(messages)
    if total <= _MAX_HISTORY_MESSAGES:
        return
    history.messages = messages[-_MAX_HISTORY_MESSAGES:]


def get_history(key: str = "lc_msgs:dfchat") -> StreamlitChatMessageHistory:
    """Return (and cache) the Streamlit-backed chat history."""

    history = StreamlitChatMessageHistory(key=key)
    _enforce_limit(history)
    return history

