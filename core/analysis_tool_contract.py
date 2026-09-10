"""Provider-independent tool contracts. No conversation runtime dependency."""
from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class ToolDefinition:
    name: str
    description: str
    parameters: dict
    run: Callable[..., dict]

    def schema(self) -> dict:
        return {"name": self.name, "description": self.description,
                "parameters": self.parameters}


@dataclass
class AnalysisToolContext:
    """Explicit services supplied by a runtime; never checkpoint this object."""
    datasets: object
    artifacts: dict
    reference_context: list
    propose_query: Callable[..., dict]
