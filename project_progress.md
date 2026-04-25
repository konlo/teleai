# Project Progress Log

## Current Status
- **Last Updated**: 2026-04-25
- **Status**: In Progress
- **Summary**: Telly chatbot agent chaining and EDA parser-loop reliability fixes are being implemented and regression-tested.
- **Next Session Focus**: Continue hardening Telly agent workflows and keep request/action logs current.

---

## Next Action Items (Pending tasks for the next session)
- [ ] Verify the Streamlit UI flow end-to-end with the SQL Builder -> EDA Analyst chaining scenario.
- [ ] Keep `test_scenario.py --static-only` passing after future agent-output parsing changes.
- [ ] Update this log after each meaningful implementation or debugging task.

---

## Daily Wrap-ups

### 2026-04-25 Daily Summary
- **Key Accomplishments**:
  - Added regression coverage for chat turn chaining and malformed agent output detection.
  - Added parser-loop detection and deterministic EDA visualization fallback.
- **Major Issues Encountered**:
  - EDA Agent could repeatedly emit malformed structured-chat responses, causing parser retry loops.

---

## Log Entries

### [2026-04-25 11:39:18] [Agent: Codex] User Request: Fix Streamlit runtime error from `pandas.Index` truth-value ambiguity
- **Action** [Agent: Codex]: Investigating `ValueError: The truth value of a Index is ambiguous` raised during Databricks sidebar load when `make_dataframe_state()` receives `df.columns`.
- **Decision**: Treat this as a missing runtime-path regression in `test_scenario.py`, not as a valid pass from prior static coverage.
- **Artifact Update**: Fixed `utils/data_context.py` so `normalize_columns()` no longer boolean-evaluates `pandas.Index`; it now only falls back on `None`.
- **Artifact Update**: Added `pandas.Index` and `make_dataframe_state(pd.DataFrame(...))` regression checks to `test_scenario.py`.
- **Action** [Agent: Codex]: Ran `PYTHONPYCACHEPREFIX=/tmp/teleai_pycache python3 -m py_compile utils/data_context.py test_scenario.py`.
- **Action** [Agent: Codex]: Ran `python3 test_scenario.py --static-only`.
- **Outcome**: Static scenario suite passed with the new runtime-path regression covered; Stateful DataContext now has 10 passing checks.

### [2026-04-25 10:44:30] [Agent: Codex] User Request: Create a `test_scenario` path that uses an external LLM
- **Action** [Agent: Codex]: Inspecting current LLM provider loading and router test structure.
- **Planned Artifact Update**: Add an explicit external-LLM scenario mode to `test_scenario.py` with provider/API-key preflight checks and routing assertions.
- **Artifact Update**: Added `--external-llm` mode to `test_scenario.py`; it rejects local `ollama`, validates `google`/`azure` credentials, loads the configured external LLM, and runs structured router assertions for actual Telly prompts.
- **Artifact Update**: Added static preflight tests for external LLM provider configuration.
- **Action** [Agent: Codex]: Ran `PYTHONPYCACHEPREFIX=/tmp/teleai_pycache python3 -m py_compile test_scenario.py`.
- **Action** [Agent: Codex]: Ran `python3 test_scenario.py --static-only`.
- **Outcome**: Static scenario suite passed with 43 checks passing and 0 failures. External LLM live calls are available via `python3 test_scenario.py --external-llm` once provider credentials are configured.

### [2026-04-25 10:43:47] [Agent: Codex] User Request: Run `test_scenario`
- **Action** [Agent: Codex]: Running the static regression scenario suite with `python3 test_scenario.py --static-only`.
- **Outcome**: Static scenario suite passed with 39 checks passing and 0 failures.

### [2026-04-25 10:37:14] [Agent: Codex] User Request: Implement Stateful Data Management plan for df_A request sufficiency
- **Action** [Agent: Codex]: Started implementing a `DataContext` layer so loaded `df_A` is checked against each request's required columns, filters, source table, and minimum rows before EDA/controlled execution.
- **Decision**: Preserve existing `df_A_data` compatibility while adding `df_A_state` metadata and deterministic reload decisions.
- **Planned Artifact Update**: Add `utils/data_context.py`, update Databricks/file loaders to store state, gate controlled/EDA flow with readiness checks, and strengthen `test_scenario.py --static-only`.
- **Artifact Update**: Added `utils/data_context.py` with `DataFrameState`, `DataRequirement`, `DataReadinessDecision`, readiness evaluation, source resolution, reload SQL generation, and state log formatting.
- **Artifact Update**: Updated `utils/session.py` so file loads, Databricks preview/base loads, and SQL query results persist `df_A_state`/`df_B_state` alongside existing dataframe slots.
- **Artifact Update**: Updated `core/chat_flow.py` controlled production flow to run a data readiness gate, skip reload when current data is sufficient, reload deterministically when columns are missing and source is known, and fail clearly when source is unknown.
- **Artifact Update**: Updated `utils/chatbot_plan.py` so deterministic SQL selects target and filter columns, not just the target column.
- **Artifact Update**: Strengthened `test_scenario.py --static-only` with DataContext readiness, reload SQL, `%sql` forced routing, and controlled balance requirement/SQL coverage.
- **Action** [Agent: Codex]: Ran `PYTHONPYCACHEPREFIX=/tmp/teleai_pycache python3 -m py_compile utils/data_context.py utils/session.py utils/chatbot_plan.py core/chat_flow.py test_scenario.py`.
- **Action** [Agent: Codex]: Ran `python3 test_scenario.py --static-only`.
- **Outcome**: Static regression tests and py_compile checks pass; `df_A=['balance']` no longer counts as sufficient for a `job` request unless a reload path/source is available.

### [2026-04-25 10:35:00] [Agent: Codex] User Request: Reflect the principle “do not trust df_A; always validate whether it is sufficient for the request”
- **Action** [Agent: Codex]: Added explicit data sufficiency validation utilities.
- **Artifact Update**: Added `DataSufficiencyResult` and `validate_data_sufficiency()` to `utils/eda_validation.py`.
- **Artifact Update**: Added `required_columns_for_plan()` to `utils/chatbot_plan.py` so controlled plans expose target and filter columns needed for request sufficiency checks.
- **Artifact Update**: Added static tests proving a loaded `df_A` with only `balance` is insufficient for a `job` request and sufficient for a `balance` request.
- **Outcome**: The test suite now encodes the rule that loaded data is not automatically trusted unless it contains the columns required by the request.

### [2026-04-25 10:25:00] [Agent: Codex] User Request: Explain and fix why `%sql job...` routed to EDA and failed on missing `job` column
- **Action** [Agent: Codex]: Inspected command parsing and routing in `core/chat_flow.py`.
- **Action** [Agent: Codex]: Found that `%sql` set `command_prefix=sql`, but the later LLM Router still ran and overwrote the route with `EDA Analyst`.
- **Artifact Update**: Added `utils/agent_routing.py` and changed routing so explicit `%sql` bypasses the LLM Router and goes directly to SQL Builder.
- **Artifact Update**: Added static tests proving `%sql` forces SQL Builder even when `df_A` appears fully loaded, including the full input `%sql job에 대한 분포 데이타` resolving to `agent_mode=SQL Builder`.
- **Outcome**: `%sql job에 대한 분포 데이타` should now use SQL Builder instead of EDA validation against the stale `df_A=['balance']` result.

### [2026-04-25 10:00:00] [Agent: Codex] User Request: Confirm whether the `balance` visualization test actually draws a chart
- **Action** [Agent: Codex]: Inspected the controlled production flow and conversation-log figure attachment path.
- **Action** [Agent: Codex]: Found that matplotlib payloads were attached before the `Controlled Executor` assistant message existed, while the earlier SQL preview message already had figures attached.
- **Artifact Update**: Updated `core/chat_flow.py` to append the `Controlled Executor` message before attaching visualization payloads.
- **Artifact Update**: Added `utils/conversation_figures.py` and static tests proving matplotlib figures attach to the latest `Controlled Executor` message even when the SQL preview message already has dataframe figures.
- **Outcome**: Controlled flow figures now attach to the correct assistant message; static regression tests still pass.

### [2026-04-25 09:45:00] [Agent: Codex] User Request: Refactor toward production structure where LLM decides and code executes SQL/visualization
- **Action** [Agent: Codex]: Added a controlled JSON-style plan path for supported visualization prompts.
- **Artifact Update**: Created `utils/chatbot_plan.py` for deterministic plan parsing, SQL construction, and visualization config selection.
- **Artifact Update**: Updated `core/chat_flow.py` to run the controlled production flow before legacy SQL Builder/EDA Agent routing when the prompt is supported.
- **Artifact Update**: Added static tests proving the actual `balance` prompt maps to `VISUALIZE`, includes `age` and `loan` filters, builds deterministic SQL with `loan = 'yes'`, and selects histogram/boxplot deterministically.
- **Outcome**: Supported prompts now follow Intent/Plan -> deterministic SQL -> DataFrame -> validation -> deterministic Python plot, reducing LLM execution control.

### [2026-04-25 09:30:00] [Agent: Codex] User Request: Reflect JSON-agent unification, EDA prompt hardening, fallback chart selection, validation step, and execution optimization
- **Action** [Agent: Codex]: Inspected current LLM settings, EDA prompt format instructions, and EDA fallback path.
- **Action** [Agent: Codex]: Identified a prompt-contract conflict risk between ReAct `Thought:` instructions and strict JSON-only requirements.
- **Artifact Update**: Strengthened the EDA prompt to require JSON-only action objects and removed conflicting explanatory `Thought:` instructions.
- **Artifact Update**: Added EDA pre-validation for column existence, data type, and distribution chart eligibility.
- **Artifact Update**: Updated fallback chart selection so numeric data uses `hist` when `len(df_A) > 100` and `boxplot` when `len(df_A) <= 100`.
- **Artifact Update**: Lowered the default LLM `max_tokens` to reduce response size while keeping temperature at `0.0`.
- **Outcome**: Added static tests for validation and fallback behavior; `test_scenario.py --static-only` passes.

### [2026-04-25 09:20:00] [Agent: Codex] User Request: Add system prompt update conflict-check guidance to the chatbot project skill
- **Action** [Agent: Codex]: Reviewed `/Users/najongseong/git_repository/skills-registry/project_management/chatbot_project_manager/SKILL.md`.
- **Action** [Agent: Codex]: Identified the workflow gap around system prompt updates, conflict detection, and regression validation.
- **Artifact Update**: Added an `Update System Prompts Safely` workflow section to the chatbot project manager skill.
- **Outcome**: The skill now requires comparing proposed system prompt changes against existing prompt text, resolving conflicts before editing, validating rendered prompt templates, and adding regression tests before applying prompt updates.

### [2026-04-25 09:10:00] [Agent: Codex] User Request: Create a chatbot project management skill file under `skills-registry`
- **Action** [Agent: Codex]: Read the local `skill-creator` guidance and inspected existing `skills-registry` categories.
- **Action** [Agent: Codex]: Chose `project_management/chatbot_project_manager/SKILL.md` as the target path because the skill is for managing chatbot projects.
- **Artifact Update**: Created `/Users/najongseong/git_repository/skills-registry/project_management/chatbot_project_manager/SKILL.md`.
- **Artifact Update**: Updated `AGENTS.md` to reference the new chatbot project management skill together with `project_logger`.
- **Outcome**: Future Telly chatbot work can use a dedicated skill for routing, parser-loop, UI validation, regression, and handoff workflows.

### [2026-04-25 09:00:00] [Agent: Codex] User Request: Create AGENTS.md so this project can use `/Users/najongseong/git_repository/skills-registry/.../SKILL.md`
- **Action** [Agent: Codex]: Located actual skill files under `/Users/najongseong/git_repository/skills-registry`.
- **Action** [Agent: Codex]: Selected `/Users/najongseong/git_repository/skills-registry/project_management/project_logger/SKILL.md` because the current request followed the prior question about project logging skills.
- **Artifact Update**: Created `AGENTS.md` with project-level instructions to use the `project_logger` skill.
- **Artifact Update**: Initialized `project_progress.md` using the logger skill template.
- **Outcome**: Future work in this project has explicit instructions to maintain a persistent progress log.
