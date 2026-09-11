"""Bounded local execution diagnostics without prompts, rows or credentials."""
import json
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime, timezone
from pathlib import Path
import traceback
from uuid import uuid4


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
