from typing import Optional


def should_process_chat_turn(user_q: Optional[str], auto_eda_pending: Optional[str]) -> bool:
    """Return True when a normal or chained chat turn should be processed."""

    return bool(user_q or auto_eda_pending)


def resolve_chat_turn_query(user_q: Optional[str], auto_eda_pending: Optional[str]) -> str:
    """Prefer the fresh user query, falling back to the pending chained EDA query."""

    return user_q or auto_eda_pending or ""


__all__ = ["resolve_chat_turn_query", "should_process_chat_turn"]
