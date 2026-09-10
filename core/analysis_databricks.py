"""Approved-only remote execution for the new analysis runtime."""
from dataclasses import asdict

import pandas as pd
from sqlglot import exp

from core.analysis_sql import validate_query
from utils.analysis_provenance import raw_conditions


def execute_approved(request, config, datasets, *, max_rows=100_000, connect=None):
    if request.status != "executing":
        raise PermissionError("승인된 조회 실행 상태가 아닙니다.")
    tree = validate_query(request.query)
    if connect is None:
        from databricks import sql
        connect = sql.connect
    with connect(server_hostname=config.server_hostname, http_path=config.http_path,
                 access_token=config.access_token, catalog=config.catalog, schema=config.schema) as connection:
        with connection.cursor() as cursor:
            cursor.execute(request.query)
            rows = cursor.fetchmany(max_rows + 1)
            columns = [item[0] for item in cursor.description]
    truncated = len(rows) > max_rows
    frame = pd.DataFrame.from_records(rows[:max_rows], columns=columns)
    aggregated = bool(tree.args.get("group") or tree.find(exp.AggFunc))
    source_tables = sorted({t.sql(dialect="databricks") for t in tree.find_all(exp.Table)})
    conditions = raw_conditions(tree)
    has_limit = any(node.args.get("limit") is not None for node in tree.walk())
    info = datasets.register(frame, source=" | ".join(source_tables) or request.source,
        query=request.query, coverage="truncated" if truncated else "unknown" if has_limit else "complete",
        predicate_known=conditions is not None, conditions=conditions or (), grain="aggregate" if aggregated else "raw",
        aggregation=tree.sql() if aggregated else "")
    return {"status": "ready", "dataset": asdict(info),
            "preview": frame.head(10).to_dict(orient="records")}
