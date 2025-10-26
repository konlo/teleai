from typing import Sequence

from langchain.agents import AgentExecutor, create_react_agent
from langchain.callbacks import StdOutCallbackHandler
from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.tools import BaseTool


class SimpleCollectCallback(BaseCallbackHandler):
    """
    Collect intermediate events from the agent for UI surfacing.
    """

    def __init__(self) -> None:
        self.events = []

    def on_tool_error(self, error, **kwargs):
        self.events.append({"type": "tool_error", "error": str(error)})

    def on_llm_start(self, serialized, prompts, **kwargs):
        self.events.append({"type": "llm_start", "prompts": prompts})

    def on_llm_end(self, response, **kwargs):
        self.events.append({"type": "llm_end"})


def build_agent(
    llm,
    tools: Sequence[BaseTool],
    prompt: ChatPromptTemplate,
    history_getter,
):
    """
    Build an AgentExecutor with message history support.
    """
    react_runnable = create_react_agent(llm, tools, prompt=prompt)
    agent = AgentExecutor(
        agent=react_runnable,
        tools=list(tools),
        verbose=True,
        return_intermediate_steps=True,
        max_iterations=20,
        handle_parsing_errors=(
            "PARSING ERROR. DO NOT APOLOGIZE. Immediately continue by outputting ONLY:\n"
            "Action: describe_columns\n"
            "Action Input: \n"
        ),
    )

    agent_with_history = RunnableWithMessageHistory(
        agent,
        history_getter,
        input_messages_key="input",
        history_messages_key="chat_history",
    )
    return agent, agent_with_history


__all__ = ["SimpleCollectCallback", "build_agent", "StdOutCallbackHandler"]
