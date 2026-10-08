"""Standalone chart selection belongs only to the latest user turn."""
from langchain_core.messages import HumanMessage


def latest_selected_chart(messages):
    for message in reversed(messages):
        if not isinstance(message, HumanMessage):
            continue
        selected = message.additional_kwargs.get('selected_card')
        if isinstance(selected, str) and selected:
            return selected
        # Preserve legacy selection turns, without scanning past newer requests.
        content = message.content
        if (isinstance(content, str) and content.startswith('선택한 차트:')
                and ', card_id=' in content):
            return content.rsplit(', card_id=', 1)[1]
        return None
    return None
