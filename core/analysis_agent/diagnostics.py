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

    def emit(self, event, **fields):
        self.logger.info(json.dumps({'time': datetime.now(timezone.utc).isoformat(),
                                    'event': event, 'run_id': self.run_id, **fields}, ensure_ascii=False))

    def failure(self, exc, *, run_id=None, stage='runtime'):
        error_id = uuid4().hex[:12]
        frames = [{'file': Path(f.filename).name, 'line': f.lineno, 'function': f.name}
                  for f in traceback.extract_tb(exc.__traceback__)]
        self.emit('error', run_id=run_id or self.run_id, error_id=error_id, stage=stage,
                  error_type=type(exc).__name__, http_status=getattr(exc,'http_status',None), frames=frames)
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
