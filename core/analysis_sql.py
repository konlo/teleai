"""Read-only query validation and bounded local analysis."""
import duckdb
import sqlglot
from threading import Timer
from sqlglot import exp


def validate_query(sql: str, *, dialect="databricks"):
    expressions = sqlglot.parse(sql, read=dialect)
    if len(expressions) != 1 or not isinstance(expressions[0], exp.Query):
        raise ValueError("분석에는 하나의 SELECT 조회만 사용할 수 있습니다.")
    tree = expressions[0]
    forbidden = (exp.Insert, exp.Update, exp.Delete, exp.Create, exp.Drop, exp.Alter,
                 exp.Merge, exp.Command, exp.Into)
    if any(isinstance(node, forbidden) for node in tree.walk()):
        raise ValueError("데이터를 변경하는 문장은 실행할 수 없습니다.")
    return tree


def local_query(frame, sql: str, *, max_rows=20_000):
    tree = validate_query(sql, dialect="duckdb")
    # External access is disabled independently of SQL parsing. Restrict table
    # references to the registered dataframe and in-query CTEs.
    aliases = {cte.alias for cte in tree.find_all(exp.CTE)}
    for table in tree.find_all(exp.Table):
        if table.name not in {"data", *aliases} or table.db or table.catalog:
            raise ValueError("로컬 SQL은 data 테이블만 참조할 수 있습니다.")
    with duckdb.connect(config={"enable_external_access": False, "memory_limit": "512MB", "threads": 2}) as conn:
        conn.register("data", frame)
        timer = Timer(15, conn.interrupt)
        timer.start()
        try:
            result = conn.execute(f"SELECT * FROM ({sql.rstrip().rstrip(';')}) AS result LIMIT {max_rows + 1}").fetchdf()
        finally:
            timer.cancel()
            timer.join()
    return result.head(max_rows), len(result) > max_rows, tree
