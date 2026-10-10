"""Bounded local execution diagnostics without prompts, rows or credentials."""
import json
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime, timezone
from pathlib import Path
import traceback
import time
from contextlib import contextmanager
from uuid import uuid4
import os


def process_peak_rss_bytes():
    """Return the process high-water RSS using only the standard library."""
    try:
        import resource
        measured = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # macOS reports bytes; Linux and other common Unix builds report KiB.
        return int(measured if __import__('sys').platform == 'darwin' else measured * 1024)
    except (ImportError, OSError, ValueError):
        return None


class Diagnostics:
    def __init__(self, directory):
        self.run_id = None
        self.path = Path(directory) / 'runtime.jsonl'
        self.logger = logging.getLogger('telly.runtime.' + str(self.path.resolve()))
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False
        if not self.logger.handlers:
            handler = RotatingFileHandler(self.path, maxBytes=2_000_000, backupCount=3, encoding='utf-8')
            handler.setFormatter(logging.Formatter('%(message)s'))
            self.logger.addHandler(handler)
        self.path.chmod(0o600)
        self.instance_id = uuid4().hex[:12]
        self.last_error_id = None

    def emit(self, event, **fields):
        self.logger.info(json.dumps({'time': datetime.now(timezone.utc).isoformat(),
                                    'event': event, 'run_id': self.run_id,
                                    'instance_id': self.instance_id, 'pid': os.getpid(),
                                    **fields}, ensure_ascii=False))

    def failure(self, exc, *, run_id=None, stage='runtime'):
        error_id = uuid4().hex[:12]
        self.last_error_id = error_id
        frames = [{'file': Path(f.filename).name, 'line': f.lineno, 'function': f.name}
                  for f in traceback.extract_tb(exc.__traceback__)]
        from core.analysis_agent.model_errors import model_error_category
        from core.analysis_agent.model_recovery import ModelAttemptBudgetExceeded
        budget = exc.attempt_budget if isinstance(exc, ModelAttemptBudgetExceeded) else None
        self.emit('error', error_category=model_error_category(exc), run_id=run_id or self.run_id, error_id=error_id, stage=stage,
            error_type=type(exc).__name__,
            attempt_budget=budget,
            database_errno=getattr(exc,'errno',None) if not isinstance(exc,OSError) else None,
            os_errno=getattr(exc,'errno',None) if isinstance(exc,OSError) else None,
            winerror=getattr(exc,'winerror',None) if isinstance(exc,OSError) else None,
            http_status=getattr(exc,'http_status',getattr(exc,'status_code',None)) or
                        getattr(getattr(exc,'response',None),'status_code',None), frames=frames[-12:])
        return error_id

    @contextmanager
    def span(self, phase, **fields):
        """Measure a phase using metadata only; never serialize model inputs."""
        span_id = uuid4().hex[:12]
        started = time.monotonic()
        self.emit(phase + '_started', span_id=span_id, **fields)
        details = {}
        try:
            yield details
        except Exception as exc:
            self.emit(phase + '_finished', span_id=span_id, status='error',
                      error_type=type(exc).__name__,
                      elapsed_seconds=round(time.monotonic()-started, 3), **fields)
            raise
        else:
            self.emit(phase + '_finished', span_id=span_id, status='ok',
                      elapsed_seconds=round(time.monotonic()-started, 3), **fields, **details)
