import json
from typing import List, Optional

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field

from utils.perf_monitor import track_time

class ExecutionPlan(BaseModel):
    intent_type: str = Field(description="One of: LOAD_DATA, SUMMARIZE, VISUALIZE, DRILL_DOWN, FORECAST, OTHER")
    reasoning: str = Field(description="Short explanation of why this intent was chosen (in Korean). Example: '데이터 분포 확인을 위해 테이블 조회를 진행한 뒤, 시각화를 수행합니다.'")
    suggested_agents: List[str] = Field(description="Ordered list of agents to run, e.g., ['SQL Builder'], or ['SQL Builder', 'EDA Analyst']")

@track_time("llm_intent_routing")
def route_query(llm, user_query: str, is_preview_state: bool) -> ExecutionPlan:
    """Classify the user intent and provide an execution plan."""
    
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are an advanced Intent Router for a Telemetry Chatbot.
Your goal is to parse the user's natural language request and determine the execution plan.

Available Agents:
1. 'SQL Builder': Used to query a Databricks database to LOAD_DATA.
2. 'EDA Analyst': Used to analyze, SUMMARIZE, VISUALIZE, DRILL_DOWN, or FORECAST on data already loaded in memory.

Current State:
- is_preview_state: {is_preview_state} (If true, it means we ONLY have 10 rows of sample data in memory. If they ask for analysis or visualization on the whole dataset, you MUST run 'SQL Builder' FIRST to load data, then 'EDA Analyst').

Intent Types:
- LOAD_DATA: Fetching rows or getting a list of things.
- VISUALIZE: Plotting, drawing a graph, showing distributions.
- DRILL_DOWN: Deep dive into anomalies.
- SUMMARIZE: Basic stats, averages, max/min.

Provide a logical plan. If they want to draw a distribution and `is_preview_state` is True, suggested_agents should be ['SQL Builder', 'EDA Analyst'].
If they just want to write a SQL query, suggested_agents is ['SQL Builder'].
If `is_preview_state` is False, and they want to draw a graph, suggested_agents is ['EDA Analyst'].
""",
            ),
            ("human", "{user_query}"),
        ]
    )

    # Use the LLM to output structured data
    chain = prompt | llm.with_structured_output(ExecutionPlan)
    
    try:
        result = chain.invoke({
            "user_query": user_query,
            "is_preview_state": str(is_preview_state)
        })
        return result
    except Exception as e:
        # Fallback in case of parsing errors
        return ExecutionPlan(
            intent_type="OTHER",
            reasoning="시스템 내부 오류로 기본 분석 모드로 진입합니다.",
            suggested_agents=["EDA Analyst"] if not is_preview_state else ["SQL Builder"]
        )
