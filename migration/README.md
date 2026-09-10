# Migration compatibility and validation history

Latest (2026-09-07): production code lives in `core/analysis_agent/` and `ui/analysis_page.py`. Run `python3 scripts/run_telly.py`; root `requirements-agent.txt` is the production lock. See [current validation](../docs/t07_t08_validation.md) and [setup](../README.md). The sections below preserve historical migration stages, not current rollout status. Development Databricks approval waiting is superseded: connection diagnosis returned OpenSession HTTP 403. Product reload approval remains mandatory.

## Reproduce (Python 3.11)

```sh
python3.11 -m venv .telly_runtime/v1-venv
.telly_runtime/v1-venv/bin/python -m pip install -r migration/requirements-v1.lock
.telly_runtime/v1-venv/bin/python -m pip check
.telly_runtime/v1-venv/bin/python -m unittest migration.test_v1_contracts -v
.telly_runtime/v1-venv/bin/python -m migration.check_live_ollama
```

`requirements-v1.in` expresses the candidate ranges; `requirements-v1.lock` pins all resolved packages for the tested Python 3.11/macOS arm64 environment. Cross-platform installation requires separate validation. Do not install either over the production `.venv`. The lock now includes Streamlit and Databricks for the isolated new entrypoint. Unused Google/Azure/legacy provider packages remain outside this environment; this is not a lock for the old main.py application.

## Contracts

- Existing ToolDefinition/schema/function pairs adapt to StructuredTool without importing the old session loop.
- A scripted BaseChatModel issues a real tool call through create_agent, real local DuckDB executes, and ToolMessage returns to the model.
- SQLite saver closure/reopen restores the conversation; another thread has no messages.
- HITL interrupt survives SQLite reopen; read-only inspection preserves it; rejection runs the local stub zero times and approval once. These are local stub tests, not database execution guarantees.
- Optional actual ChatOllama smoke test uses the same configured local model, temperature 0, reasoning enabled, 16k context, 4096 generation tokens and 60-second client timeout. It records actual SQL observations, answer and duration. No model reasoning text is written to reports.

## Historical T06b scope (superseded by T06c/T07 below)

DataFrames/artifacts still live in memory and the prompt catalog is initial only. Build a durable, owner-scoped asset store, graph runtime interface, dynamic context, context budget, pending-input semantics, durable execution ledger and crash/UNKNOWN recovery. This prototype cannot safely recover a complete analysis after process loss even though its messages restore. Never enable a real DB tool until the execution ledger and approval contract are implemented.

Reference APIs: [create_agent](https://docs.langchain.com/oss/python/langchain/agents), [HITL](https://docs.langchain.com/oss/python/langchain/human-in-the-loop), [persistence](https://docs.langchain.com/oss/python/langgraph/persistence).

## T06c persistent local runtime

`GraphAnalysisRuntime(root, owner, conversation, model)` in `migration/graph_runtime.py` stores graph checkpoints and local assets in a hashed owner/conversation directory. The caller must supply trusted identities: this namespace mechanism does not replace application authentication. `submit`, `inspect`, `events`, `resume`, `select_chart`, and `close` operate on the persisted conversation. The current Streamlit page has not switched to this implementation.

`PersistentDatasets` commits Parquet payload and DatasetInfo metadata together in SQLite. DataFrames load lazily through a bounded LRU cache; returned frames are copies. The cache budget bounds retained frames, not peak memory during SQL, Parquet serialization or copies. `PersistentCharts` stores immutable metadata and PNG bytes alongside dataset IDs. Missing assets raise an error, never trigger remote loading. No pickle deserialization is used.

Graph context refreshes the asset catalog for each model call. A conservative character budget stops before oversized context instead of silently deleting conditions or tool-message pairs. This is not token-accurate compaction: automatic summarization and semantic retention evaluation remain unfinished. A failed node is resumable with `resume()`; new submissions are blocked while graph work remains incomplete. Local replay can create an unused extra derived asset after a crash between asset commit and checkpoint; garbage collection and deterministic artifact IDs remain follow-up work. Remote replay is still impossible because no remote execution tool is registered.

```sh
.telly_runtime/v1-venv/bin/python -m unittest migration.test_persistent_runtime migration.test_v1_contracts -v
```

The T06c tests include a fresh subprocess reopening the conversation, DataFrame and PNG, a subsequent turn, cross-owner isolation, cache eviction, missing assets, local derived-result persistence, concurrent-controller rejection and incomplete-work resume. They use scripted models; the previous actual Ollama smoke result does not constitute a live model evaluation of the entire new persistent runtime. DB submission crashes, UI integration, metadata/training persistence and automatic context compaction remain outside this stage.

## T07/T08 current entrypoint

The optional remote service in GraphAnalysisRuntime is now protected by both HITL and a durable ApprovalLedger. Use `scripts/run_telly_v1.py` to launch the new desktop app on localhost:8502. The app restores local conversations, samples, selected chart images and saved TableContext. Direct DB tools from the legacy runtime are not registered.

Run all migration contracts with:

```sh
.telly_runtime/v1-venv/bin/python -m unittest migration.test_approval_rollout migration.test_persistent_runtime migration.test_v1_contracts -v
```

See `docs/t07_t08_validation.md` for the current release gate: real DB execution is awaiting user approval, and long-context compaction/latency remain limitations. Do not interpret the old T06b/T06c limitations above as the current absence of the approval ledger or UI. They describe the intermediate milestones.
