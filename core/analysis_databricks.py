"""Approved-only remote execution for the new analysis runtime."""
from dataclasses import asdict
from datetime import datetime, timezone

import pandas as pd

from core.analysis_sql import validate_query
from core.analysis_load_plan import source_plan
from utils.analysis_provenance import raw_conditions, query_coverage


def execute_approved(request, config, datasets, *, max_rows=100_000, connect=None):
    if request.status != "executing":
        raise PermissionError("승인된 조회 실행 상태가 아닙니다.")
    if max_rows < 1:
        raise ValueError('조회 행 한도는 1 이상이어야 합니다.')
    tree = validate_query(request.query)
    plan = source_plan(request.source, request.query)
    if connect is None:
        from databricks import sql
        connect = sql.connect
    with connect(server_hostname=config.server_hostname, http_path=config.http_path,
                 access_token=config.access_token, catalog=config.catalog, schema=config.schema) as connection:
        with connection.cursor() as cursor:
            cursor.execute(request.query)
            description = cursor.description
            if description is None:
                raise ValueError('조회 결과의 컬럼 정보를 확인할 수 없습니다.')
            columns = [item[0] for item in description]
            if not columns or any(not isinstance(name, str) or not name for name in columns):
                raise ValueError('조회 결과의 컬럼 이름이 유효하지 않습니다.')
            if len(set(columns)) != len(columns):
                raise ValueError('조회 결과에 중복 컬럼 이름이 있습니다.')
            max_columns = getattr(datasets, 'max_columns', None)
            if max_columns is not None and len(columns) > max_columns:
                raise ValueError('조회 결과의 컬럼 수가 운영 한도를 초과합니다.')
            if (plan.expected_columns and
                    tuple(name.casefold() for name in columns) !=
                    tuple(name.casefold() for name in plan.expected_columns)):
                raise ValueError('조회 결과 컬럼이 승인된 SQL의 출력 컬럼과 다릅니다.')
            batches = []
            row_count = 0
            estimated_bytes = 0
            max_bytes = getattr(datasets, 'max_frame_bytes', None)
            while row_count <= max_rows:
                batch = cursor.fetchmany(min(4096, max_rows + 1 - row_count))
                if not batch:
                    break
                if any(len(row) != len(columns) for row in batch):
                    raise ValueError('조회 결과 행과 컬럼 구조가 일치하지 않습니다.')
                candidate = pd.DataFrame.from_records(batch, columns=columns)
                estimated_bytes += int(candidate.memory_usage(index=True, deep=True).sum())
                if max_bytes is not None and estimated_bytes > max_bytes:
                    raise MemoryError('조회 결과가 데이터 메모리 한도를 초과해 발행하지 않았습니다.')
                batches.append(candidate)
                row_count += len(batch)
    truncated = row_count > max_rows
    frame = (pd.concat(batches, ignore_index=True).head(max_rows).copy()
             if batches else pd.DataFrame(columns=columns))
    conditions = raw_conditions(tree)
    info = datasets.register(frame, source=" | ".join(plan.actual_tables) or request.source,
        query=request.query, coverage=query_coverage(tree, truncated=truncated),
        predicate_known=conditions is not None, conditions=conditions or (), grain=plan.grain,
        aggregation=tree.sql() if plan.grain == 'aggregate' else "",
        snapshot=datetime.now(timezone.utc).isoformat())
    return {"status": "ready", "dataset": asdict(info),
            "preview": frame.head(10).to_dict(orient="records")}
