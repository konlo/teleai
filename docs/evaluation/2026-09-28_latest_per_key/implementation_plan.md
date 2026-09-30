# 最新行分布 — acceptance before implementation

Scope for this increment: schema-independent local latest-per-key selection, explicit ordering, immutable raw reuse, verified distribution and chart delivery. No inference from physical row order. Missing order/ties/NULL must request context without publishing numbers. Remote pushdown and large-volume execution remain independent release gates.

- Validate CTE/window alias scopes using the actual dataset schema; missing physical columns and external relations still fail.
- Persist a row-selection contract on derived datasets and prevent implicit ancestor reuse from broadening that population.
- Expose a bounded latest-distribution tool; validate unique keys, ordered selection, count sum, artifact and raw preservation.
- Bind supported explicit natural-language column roles only from observed schema; preserve a pending request through a narrow ordering clarification.
- Make latest selection and visualization separate completion obligations. Wrong GROUP BY/MAX plans cannot satisfy selection.
- Regression: renamed schema, shuffled input, duplicate newest timestamps, null key/order, numeric/category measures, restart/clarification, downstream SQL denominator, raw digest, CTE/missing-column/external source cases. Repeat the original four-case evaluation without repairing runtime state; report actual model calls separately from deterministic routing.
