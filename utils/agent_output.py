import json
from typing import Any, Iterable, Optional, Tuple


PARSING_ERROR_MARKERS = (
    "Parsing error:",
    "올바른 형식으로 다시 응답해주세요",
    "Invalid Format:",
)


def _json_payload(value: Any) -> Optional[Any]:
    if not isinstance(value, str):
        return value if isinstance(value, dict) else None
    text = value.strip()
    if not text.startswith("{"):
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def has_invalid_action_schema(payload: Any) -> bool:
    data = _json_payload(payload)
    if not isinstance(data, dict):
        return False
    if "action" not in data:
        return False
    if "content" in data and "action_input" not in data:
        return True
    return "action_input" not in data


def unwrap_final_answer_payload(payload: Any, max_depth: int = 3) -> Tuple[Any, bool]:
    data = _json_payload(payload)
    if not isinstance(data, dict):
        return payload, False

    current: Any = data
    for _ in range(max_depth):
        if not (
            isinstance(current, dict)
            and current.get("action") == "Final Answer"
            and "action_input" in current
        ):
            return current, False
        next_value = current["action_input"]
        if not (
            isinstance(next_value, dict)
            and next_value.get("action") == "Final Answer"
            and "action_input" in next_value
        ):
            return next_value, False
        current = next_value

    if (
        isinstance(current, dict)
        and current.get("action") == "Final Answer"
        and isinstance(current.get("action_input"), dict)
        and current["action_input"].get("action") == "Final Answer"
    ):
        return current, True
    return current, False


def _count_parser_exception_steps(intermediate_steps: Optional[Iterable[Any]]) -> int:
    count = 0
    for step in intermediate_steps or []:
        try:
            action, observation = step
        except Exception:
            continue
        tool_name = getattr(action, "tool", "") or ""
        text = f"{getattr(action, 'tool_input', '')} {observation}"
        if tool_name == "_Exception" and any(marker in text for marker in PARSING_ERROR_MARKERS):
            count += 1
    return count


def detect_agent_parser_loop(result: Any, intermediate_steps: Optional[Iterable[Any]] = None) -> bool:
    steps = intermediate_steps
    output = result
    if isinstance(result, dict) and (
        "output" in result or "intermediate_steps" in result
    ):
        output = result.get("output", "")
        steps = result.get("intermediate_steps", intermediate_steps)

    if _count_parser_exception_steps(steps) >= 2:
        return True
    if has_invalid_action_schema(output):
        return True
    _, too_deep = unwrap_final_answer_payload(output)
    if too_deep:
        return True

    text = output if isinstance(output, str) else str(output)
    marker_hits = sum(text.count(marker) for marker in PARSING_ERROR_MARKERS)
    if marker_hits >= 2:
        return True
    return "Parsing error:" in text and "올바른 형식으로 다시 응답해주세요" in text


__all__ = [
    "detect_agent_parser_loop",
    "has_invalid_action_schema",
    "unwrap_final_answer_payload",
]
