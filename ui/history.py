from langchain_community.chat_message_histories import StreamlitChatMessageHistory


def get_history(key: str = "lc_msgs:dfchat") -> StreamlitChatMessageHistory:
    """Return (and cache) the Streamlit-backed chat history."""
    return StreamlitChatMessageHistory(key=key)

