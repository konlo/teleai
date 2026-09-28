# Semantic batch repair acceptance plan

The remaining L1_023/024/026/028/030 failures concern business meaning, not arithmetic. Fix the common metadata → semantic binding → scope validation → local calculation path. No table-specific routing or question-specific aliases.

Before implementation, acceptance is fixed as follows:
- Preserve table-wide external descriptions across persistence and catalog loading; schema-only, stale, mismatched and conflicting context must not ground a semantic plan.
- Only unresolved single-measure COUNT/AVG requests may enter bounded semantic resolution. Do not override explicit predicates, known columns, source, selected dataset or freshness obligations.
- Derive two independent structured interpretations from the request and bounded descriptions. Require agreement and validate every column, operation, documented value and definition citation. Agreement is a confidence check, not proof of arbitrary natural-language correctness.
- Persist accepted binding with source/schema provenance, account for both model calls/time, and compute using existing scope/lineage/approval checks. A semantic binding alone is never completion evidence.
- Exercise arbitrary renamed columns, reversed binary encoding, prior/current outcomes, absent/stale/wrong-source context, disagreements, invalid JSON, wrong values, original preservation and restart. Keep current clarification continuity.
- Preserve fixed benchmark questions/oracles. Evaluate aliases-only and description-enriched contexts separately; do not conflate metadata enrichment with an unchanged-input score.
- Run full application/migration/fault-injection regressions and real-model evaluation. No new Databricks SQL without exact approval.
