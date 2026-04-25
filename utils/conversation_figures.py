from typing import Any, Dict, List


def attach_figures_to_log(
    log: List[Dict[str, Any]],
    run_id: str,
    figures: List[Dict[str, Any]],
) -> bool:
    """Attach figures to the latest assistant entry for a run if possible."""

    if not run_id or not figures:
        return False
    for entry in reversed(log):
        if entry.get("run_id") == run_id and entry.get("role") == "assistant":
            if entry.get("figures_attached"):
                return False
            entry.setdefault("figures", [])
            entry["figures"].extend(figures)
            entry["figures_attached"] = True
            return True
    return False


__all__ = ["attach_figures_to_log"]
