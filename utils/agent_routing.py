from typing import Optional


def should_force_sql_builder(command_prefix: Optional[str]) -> bool:
    """Return True when an explicit command must bypass LLM routing."""

    return command_prefix == "sql"


def resolve_forced_agent_mode(command_prefix: Optional[str]) -> Optional[str]:
    """Resolve explicit command prefixes before any LLM router decision."""

    if should_force_sql_builder(command_prefix):
        return "SQL Builder"
    return None


def parse_command_prefix(user_input: str) -> tuple[Optional[str], str]:
    """Parse lightweight command prefixes without importing Streamlit chat flow."""

    stripped = (user_input or "").lstrip()
    lowered = stripped.lower()
    if lowered.startswith("%sql"):
        return "sql", stripped[len("%sql"):].lstrip()
    return None, stripped


def resolve_agent_mode_for_input(user_input: str) -> tuple[Optional[str], str]:
    """Return forced agent mode and agent request for a raw user input."""

    command_prefix, agent_request = parse_command_prefix(user_input)
    return resolve_forced_agent_mode(command_prefix), agent_request


__all__ = [
    "parse_command_prefix",
    "resolve_agent_mode_for_input",
    "resolve_forced_agent_mode",
    "should_force_sql_builder",
]
