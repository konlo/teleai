# Live local model evidence - 2026-09-06

아래 증거는 현재 자체 AnalysisSession의 제한된 기준선이다. [LangChain/LangGraph 전환 설계](langchain_migration_design_2026-09-06.md)의 새 런타임으로 성공 판정을 이전하지 않는다. 동일 모델·데이터로 재실행하고 승인·재시작·다중 턴 수용 기준을 별도로 통과해야 한다.

Model: configured local Ollama gemma4:e4b, native /api/chat tool calling, reasoning enabled.
Input: explicitly synthetic fixture, segment A/A/B/B and value 10/30/50/70.

- First user request: compute current value mean. Answer: 40.0. Actual tool: local_analysis_sql.
- Follow-up request: only segment A. Answer: 20.0. Actual tool: local_analysis_sql.
- Same AnalysisSession and transcript used for both turns.
- No Databricks query or remote model service used.

Earlier attempts were unsuccessful: legacy JSON prompt parsing failed; native non-reasoning responses asked redundant permission. Final successful run added actual dataset metadata to context, distinguished local computation from remote approval, and enabled native reasoning/tool calling.

Automated validation: 26 unit/integration tests passed, including actual DuckDB OR calculation and Streamlit proposal/cancel with no DB connections. Legacy static scenarios passed.

Remaining: more multi-turn tasks, real UI visual inspection, supported provider evaluations, data/query scope and cache/context lifetime, approval-bound connection identity, and user-approved live Databricks E2E. This is limited evidence, not full goal completion.
