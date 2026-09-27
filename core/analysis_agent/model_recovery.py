"""Retry only model inference, with durable per-request attempt/cooldown bounds.

This middleware never wraps the graph or a tool executor. Retrying an inference
cannot resubmit an approved SQL statement or mutate a retained dataset.
"""
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import math
import time

from langchain.agents.middleware import AgentMiddleware
from langgraph.errors import GraphBubbleUp


class ModelCoolingDown(RuntimeError):
    pass


class ModelAttemptBudgetExceeded(RuntimeError):
    pass


def transient_model_error(error):
    status = getattr(error, 'status_code', getattr(error, 'http_status', None))
    if status is not None:
        return status in {408, 429, 500, 502, 503, 504}
    from httpx import TimeoutException, NetworkError
    from openai import APIConnectionError
    return isinstance(error, (TimeoutError, TimeoutException, NetworkError, APIConnectionError))


def retry_after(error):
    headers = getattr(getattr(error, 'response', None), 'headers', {}) or {}
    value = headers.get('retry-after')
    if value is None:
        return 0.0
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        try:
            seconds = (parsedate_to_datetime(value) - datetime.now(timezone.utc)).total_seconds()
        except (TypeError, ValueError, OverflowError):
            return 0.0
    return max(0.0, seconds) if math.isfinite(seconds) else 0.0


class ModelAttemptLedger:
    def __init__(self, db):
        self.db = db
        with db.lock, db.conn:
            db.conn.execute('CREATE TABLE IF NOT EXISTS model_recovery '
                '(request_id TEXT PRIMARY KEY, failures INTEGER NOT NULL DEFAULT 0, '
                'failed_seconds REAL NOT NULL DEFAULT 0, retries INTEGER NOT NULL DEFAULT 0, '
                'next_allowed_at REAL NOT NULL DEFAULT 0)')

    def get(self, request_id):
        with self.db.lock:
            row = self.db.conn.execute('SELECT failures, failed_seconds, retries, next_allowed_at '
                'FROM model_recovery WHERE request_id=?', (request_id,)).fetchone()
        return dict(zip(('failures', 'failed_seconds', 'retries', 'next_allowed_at'), row or (0, 0., 0, 0.)))

    def failure(self, request_id, elapsed, cooldown):
        with self.db.lock, self.db.conn:
            self.db.conn.execute('INSERT OR IGNORE INTO model_recovery(request_id) VALUES (?)', (request_id,))
            self.db.conn.execute('UPDATE model_recovery SET failures=failures+1, '
                'failed_seconds=failed_seconds+?, next_allowed_at=? WHERE request_id=?',
                (elapsed, time.time()+cooldown, request_id))

    def reserve_retry(self, request_id, delay, maximum):
        with self.db.lock, self.db.conn:
            changed = self.db.conn.execute('UPDATE model_recovery SET retries=retries+1, '
                'failed_seconds=failed_seconds+? WHERE request_id=? AND retries<?',
                (delay, request_id, maximum)).rowcount
        return bool(changed)

    def sync(self, current):
        observed = self.get(current.get('request_id', ''))
        delta_calls = max(0, observed['failures'] - current.get('accounted_model_failures', 0))
        delta_seconds = max(0., observed['failed_seconds'] - current.get('accounted_model_failure_seconds', 0.))
        current['model_calls'] = current.get('model_calls', 0) + delta_calls
        current['model_seconds'] = current.get('model_seconds', 0.) + delta_seconds
        current['accounted_model_failures'] = observed['failures']
        current['accounted_model_failure_seconds'] = observed['failed_seconds']
        current['model_retries'] = observed['retries']
        current['_new_model_failure_seconds'] = delta_seconds


class ModelRecoveryMiddleware(AgentMiddleware):
    def __init__(self, ledger, diagnostics, policy, max_calls=10, max_retries=2,
                 max_wait=10., sleep=None, on_progress=None):
        self.ledger, self.diagnostics, self.policy = ledger, diagnostics, policy
        self.max_calls, self.max_retries, self.max_wait = max_calls, max_retries, max_wait
        self.sleep, self.on_progress = sleep or time.sleep, on_progress

    def wrap_model_call(self, request, handler):
        current = request.state.get('recovery') or {}
        request_id = current.get('request_id')
        if not request_id:
            return handler(request)
        observed = self.ledger.get(request_id)
        if observed['next_allowed_at'] > time.time():
            raise ModelCoolingDown('Model provider cooldown has not elapsed')
        baseline_failures = current.get('accounted_model_failures', 0)
        deadline = time.monotonic() + max(0., self.policy.turn_slo_seconds
            - current.get('model_seconds', 0.)
            - max(0., time.time() - current.get('model_started_at', time.time())))
        while True:
            observed = self.ledger.get(request_id)
            calls = current.get('model_calls', 0) + observed['failures'] - baseline_failures
            if calls >= self.max_calls or time.monotonic() >= deadline:
                raise ModelAttemptBudgetExceeded('Request inference budget exhausted')
            started = time.monotonic()
            try:
                return handler(request)
            except GraphBubbleUp:
                raise
            except Exception as error:
                # Interrupts/cancellation and programming errors are never
                # classified by a message substring or retried as network I/O.
                transient = transient_model_error(error)
                cooldown = retry_after(error) if transient else 0.
                self.ledger.failure(request_id, time.monotonic()-started, cooldown)
                observed = self.ledger.get(request_id)
                delay = max(cooldown, min(2. ** observed['retries'], self.max_wait))
                remaining_calls = self.max_calls - current.get('model_calls', 0) - observed['failures'] + baseline_failures
                can_retry = (transient and remaining_calls > 0 and delay <= self.max_wait
                    and time.monotonic()+delay+self.policy.model_timeout_seconds < deadline
                    and self.ledger.reserve_retry(request_id, delay, self.max_retries))
                self.diagnostics.emit('model_inference_failed', request_id=request_id,
                    error_type=type(error).__name__, http_status=getattr(error,'status_code',None),
                    retry_scheduled=can_retry, retry_delay_seconds=delay if can_retry else None)
                if not can_retry:
                    raise
                if self.on_progress:
                    self.on_progress('모델 응답이 일시적으로 지연되어 요청을 보존한 채 다시 시도하고 있습니다.')
                self.sleep(delay)
