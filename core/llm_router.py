import json
from typing import List, Optional

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from utils.perf_monitor import track_time

class ExecutionPlan(BaseModel):
    intent_type: str = Field(description="One of: LOAD_DATA, SUMMARIZE, VISUALIZE, DRILL_DOWN, FORECAST, OTHER")
    reasoning: str = Field(description="Short explanation of why this intent was chosen (in Korean). Example: '데이터 분포 확인을 위해 테이블 조회를 진행한 뒤, 시각화를 수행합니다.'")
    suggested_agents: List[str] = Field(description="Ordered list of agents to run, e.g., ['SQL Builder'], or ['SQL Builder', 'EDA Analyst']")

@track_time("llm_intent_routing")
def route_query(llm, user_query: str, is_preview_state: bool) -> ExecutionPlan:
    """Classify the user intent and provide an execution plan."""
    
    system_text = """You are an advanced Intent Router for a Telemetry Chatbot.
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
If `is_preview_state` is False, and they want to draw a graph, suggested_agents is ['EDA Analyst']."""

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_text),
            ("human", "{user_query}"),
        ]
    )

    try:
        # Use the LLM to output structured data
        chain = prompt | llm.with_structured_output(ExecutionPlan)
        return chain.invoke({
            "user_query": user_query,
            "is_preview_state": str(is_preview_state)
        })
    except NotImplementedError:
        from langchain_core.output_parsers import PydanticOutputParser
        parser = PydanticOutputParser(pydantic_object=ExecutionPlan)
        fallback_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_text + "\n\n{format_instructions}"),
                ("human", "{user_query}"),
            ]
        )
        fallback_chain = fallback_prompt | llm | parser
        try:
            return fallback_chain.invoke({
                "user_query": user_query,
                "is_preview_state": str(is_preview_state),
                "format_instructions": parser.get_format_instructions()
            })
        except Exception as json_exc:
            needs_eda = any(kw in user_query.lower() for kw in ["차트", "그려", "그래프", "plot", "chart", "시각화", "분포", "요약"])
            agents = ["SQL Builder", "EDA Analyst"] if needs_eda and is_preview_state else (["EDA Analyst"] if not is_preview_state else ["SQL Builder"])
            return ExecutionPlan(
                intent_type="OTHER",
                reasoning=f"시스템 내부 파싱 오류로 키워드 기반 라우팅을 수행합니다.",
                suggested_agents=agents
            )
    except Exception as e:
        # Fallback in case of parsing errors
        needs_eda = any(kw in user_query.lower() for kw in ["차트", "그려", "그래프", "plot", "chart", "시각화", "분포", "요약"])
        agents = ["SQL Builder", "EDA Analyst"] if needs_eda and is_preview_state else (["EDA Analyst"] if not is_preview_state else ["SQL Builder"])
        return ExecutionPlan(
            intent_type="OTHER",
            reasoning=f"시스템 내부 파싱 오류로 키워드 기반 라우팅을 수행합니다.",
            suggested_agents=agents
        )
