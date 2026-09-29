"""One bounded availability check before costly live evaluations; not a GO gate.

Calls the configured warehouse status API, one minimal production-model request,
and one SELECT 1 connection. No automatic retries, table access or raw error logs.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import sys
import time
from urllib.parse import quote

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def error_result(error, stage):
    from core.analysis_agent.model_errors import model_error_category
    context = getattr(error, 'context', {})
    context = context if isinstance(context, dict) else {}
    status = (getattr(error, 'status_code', None) or context.get('http-code') or
              getattr(getattr(error, 'response', None), 'status_code', None))
    try:
        status = int(status)
    except (TypeError, ValueError):
        status = None
    # Never persist provider response bodies, exception text, URLs or credentials.
    return {'status': 'BLOCKED', 'stage': stage, 'error_type': type(error).__name__,
            'http_status': status,
            'category': model_error_category(error), 'temporary_confirmed': False}


def summarize(checks):
    available = all(checks.get(key, {}).get('status') == 'PASS'
                    for key in ('model', 'sql'))
    return {'status': 'AVAILABLE' if available else 'BLOCKED', 'checks': checks,
            'scope': 'connectivity_only_not_agent_quality_or_release',
            'temporary_confirmed': False,
            'next_action': ('resume_live_evaluation' if available else
                            'inspect_provider_usage_account_and_service_status')}


def probe(config, *, control, invoke_model, connect):
    checks = {}

    def timed(name, callback):
        started = time.monotonic()
        try:
            result = callback()
        except Exception as error:
            result = error_result(error, name)
        checks[name] = {**result, 'timestamp': datetime.now(timezone.utc).isoformat(),
                        'elapsed_seconds': round(time.monotonic()-started, 3)}

    timed('warehouse', control)

    def model():
        response = invoke_model()
        content = getattr(response, 'content', None)
        return {'status': 'PASS' if content else 'FAIL', 'attempts': 1,
                'nonempty_response': bool(content)}
    timed('model', model)

    def sql():
        stage = 'OpenSession'
        try:
            with connect(server_hostname=config.server_hostname, http_path=config.http_path,
                         access_token=config.access_token, _socket_timeout=20,
                         _retry_stop_after_attempts_count=1,
                         _retry_stop_after_attempts_duration=25) as connection:
                stage = 'SELECT 1'
                with connection.cursor() as cursor:
                    cursor.execute('SELECT 1')
                    row = cursor.fetchone()
                    ok = row is not None and len(row) == 1 and row[0] == 1
            return {'status': 'PASS' if ok else 'FAIL', 'query': 'SELECT 1',
                    'attempts': 1}
        except Exception as error:
            return error_result(error, stage)
    timed('sql', sql)
    return summarize(checks)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args(argv)
    from dotenv import load_dotenv
    load_dotenv(ROOT/'.env')
    # Third-party connector logs can include connection details. The report uses
    # only typed, selected fields and is the CLI's sole diagnostic output.
    logging.disable(logging.CRITICAL)
    try:
        from databricks import sql
        import requests
        from core.analysis_agent.databricks import ConnectionConfig
        from core.analysis_agent.model_provider import build_analysis_chat_model
        from core.analysis_agent.policy import RuntimePolicy
        config = ConnectionConfig.from_env()
        if not config.server_hostname or not config.http_path or not config.access_token:
            raise ValueError('Missing connection configuration')
        model = build_analysis_chat_model(RuntimePolicy(model_timeout_seconds=25),
                                          provider='databricks')
        def control():
            warehouse = config.http_path.rstrip('/').split('/')[-1]
            response = requests.get('https://'+config.server_hostname+
                '/api/2.0/sql/warehouses/'+quote(warehouse, safe=''),
                headers={'Authorization': 'Bearer '+config.access_token}, timeout=20)
            response.raise_for_status()
            data = response.json()
            state = data.get('state')
            allowed = {'STARTING', 'RUNNING', 'STOPPING', 'STOPPED', 'DELETING', 'DELETED'}
            return {'status': 'PASS', 'http_status': response.status_code,
                    'state': state if state in allowed else 'UNKNOWN'}
        report = probe(config, control=control,
                       invoke_model=lambda: model.invoke('Reply only OK.', max_tokens=32),
                       connect=sql.connect)
    except Exception as error:
        report = summarize({'configuration': error_result(error, 'configuration')})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(report, ensure_ascii=False))
    return 0 if report['status'] == 'AVAILABLE' else 1


if __name__ == '__main__':
    raise SystemExit(main())
