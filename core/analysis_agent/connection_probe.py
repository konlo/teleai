"""One read-only connection probe independent of LLM planning and MySQL."""
import time


def probe_databricks(config, *, connect=None):
    started = time.monotonic()
    stage = 'database_configuration'
    try:
        config.validate()
        stage = 'database_driver'
        if connect is None:
            from databricks import sql
            connect = sql.connect
        stage = 'database_open_session'
        with connect(server_hostname=config.server_hostname, http_path=config.http_path,
                     access_token=config.access_token, catalog=config.catalog or None,
                     schema=config.schema or None) as connection:
            stage = 'database_cursor'
            with connection.cursor() as cursor:
                stage = 'database_execute'
                cursor.execute('SELECT 1')
                stage = 'database_fetch'
                if tuple(cursor.fetchone() or ()) != (1,):
                    raise ValueError('Unexpected probe result')
        result = {'status':'PASS', 'stage':'database_fetch'}
    except Exception as exc:
        context = getattr(exc, 'context', {})
        code = (getattr(exc, 'http_status', None) or getattr(exc, 'status_code', None)
                or getattr(getattr(exc, 'response', None), 'status_code', None)
                or (context.get('http-code') if isinstance(context, dict) else None))
        try:
            code = int(code)
        except (ValueError, TypeError):
            code = None
        result = {'status':'FAIL', 'stage':stage, 'error_type':type(exc).__name__,
                  'http_status':code if code is not None and 100 <= code <= 599 else None}
    return {**result, 'elapsed_seconds':round(time.monotonic()-started, 3),
            'model_called':False, 'probe_sql':'SELECT 1'}
