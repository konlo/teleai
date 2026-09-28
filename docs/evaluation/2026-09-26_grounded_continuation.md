# Grounded local analysis and clarification — 2026-09-26

Decision: **NO-GO for general autonomous analysis**. Local single-user scope only;
separate server deployment remains excluded at the user's request.

## Changes and acceptance evidence

- Numeric scalar planning uses the loaded schema's persisted dtypes to separate
  an identifier from a numeric measure in a compound label. Multiple numeric
  measures, grouping requests and incomplete population coverage do not become
  an arbitrary overall mean. Planning no longer reads the full numeric column
  merely to identify its type.
- A grounded frequency/distribution table uses bounded local SQL with the
  requested predicates and grouping axis. Completion checks the actual GROUP BY;
  a scalar count cannot substitute for the table. Null categories are retained.
- Korean aliases respect word boundaries and ordinary particles, including OR
  particles. A currency word no longer spuriously introduces a date column.
- An unresolved analytical target retains its request and selected dataset in
  the graph checkpoint. An explicit canonical column name for a numeric metric,
  or `column = 'value'` for a qualified count, continues the original analysis.
  Restart continuity, original preservation and zero remote execution are tested.
  A new question, approval word, unknown column or changed selection does not bind.
  This is deliberately a narrow clarification interface, not general semantic
  understanding or inferred business definitions.

## Actual graph evaluation

Fixed original questions, unchanged fixtures/aliases/oracles, Databricks Qwen
provider, no connected Databricks SQL executor:

| Run | PASS | NOT_COMPLETE | FAIL | UNGRADED |
|---|---:|---:|---:|---:|
| Prior final build | 78 | 9 | 0 | 113 |
| New final build | 82 | 5 | 0 | 113 |

82/87 supported oracles = 94.3%. **This is not a 94.3% release-completion score.**
There are 200 reference questions; 113 remain ungraded. Only 8 questions in the
new run actually invoked the model. The others used deterministic local plans.
It is not an official Spider EX or DeepEval score, nor a fresh remote-loading test.

The four recovered questions were L1_025 (compound-label mean, 0.062 s), L1_029
(category frequency, 0.073 s), L1_047 (filtered category distribution, 0.234 s),
and L2_012 (filtered conditional rate, 0.081 s). All used zero model calls.

The remaining L1_023, L1_024, L1_026, L1_028 and L1_030 need grounded business
meaning, e.g. prior versus current outcome. They remain blocked with clarification;
asking a question is not counted as a solved benchmark case.

A separate recovery test disabled deterministic planning, injected an invalid
dataset ID, and used the real Qwen model afterward: 2 real model calls, 3.465 s,
verified mean 9.0, raw unchanged. The reported 3 model-node calls include the one
injected response. This specifically exercises actual model-driven repair.

[Compact per-case evidence](2026-09-26_grounded_continuation_scores.json).
Full local reports: `/tmp/teleai_grounded_release/report.json` and
`/tmp/teleai_grounded_repair.json`.

## Regression and web evidence

- Application unittest: 290/290.
- Migration unittest: 133/133.
- Reference plus Level 3 runner: 217/217; 58 figures. Reference pandas solutions
  are not counted as production agent natural-language successes.
- Compileall and `git diff --check`: pass.
- Fault-injection tests that would now be bypassed by a valid deterministic plan
  explicitly disable that planner within the test. Their error, scope and
  recovery assertions are retained; no failure is relabeled a success.
- Real browser, existing approved bank_loan sample: request the job distribution
  for age >= 60 within the retained 10,000 rows. Output 11 categories totaling
  327, independently matched against the persisted raw Parquet. 0.339 s,
  one local tool, zero model calls. No new SQL approval or remote load.
- Raw SHA256 remains
  `7c4a1c20588261932629087d091d43e1ad2d80f4e36cb3d1acb20eb8462fe10d`;
  approval ledger remains one `completed` query. Streamlit health is `ok`.

## Authentication investigation

The user's reported 401 was not reproduced. Current locally stored Databricks
credentials returned HTTP 200 from the identity API, and real model serving
worked during evaluation. The checked browser showed no current 401. The
structured authentication failure located in stored runtime logs was an older
403. No token was disclosed, reissued or replaced. The exact failing surface
still needs identification; successful authentication is not proof that every
service endpoint or a separate browser session is healthy.

## Remaining release gates

1. Ground natural-language business definitions in maintained external metadata,
   including value meanings and prior/current distinctions; retain clarification
   for ambiguity and expand beyond the narrow reply syntax.
2. Add independent oracles for the remaining 113 reference questions and rerun
   actual model/held-out journeys. Do not promote unsupported cases to PASS.
3. Resolve and retest official Spider SQL dialect, join relationships and
   reference-date failures; update DeepEval on the final actual graph outputs.
4. Validate final-build approved remote loading and large-data transfer/RSS/query
   pushdown end to end. Existing retained-sample reuse does not establish this.
5. Review the draft PR and release evidence before changing the GO decision.
