import time
import functools
import streamlit as st
from typing import Optional
from core.learning_memory import record_performance

class TimeTracker:
    """A context manager to track the execution time of a block of code."""
    def __init__(self, operation_name: str, run_id: Optional[str] = None):
        self.operation_name = operation_name
        self.run_id = run_id or st.session_state.get("active_run_id")
        self.start_time = None

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        duration_ms = (time.perf_counter() - self.start_time) * 1000
        record_performance(self.operation_name, duration_ms, self.run_id)

def track_time(operation_name: str):
    """A decorator to track the execution time of a function."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Try to get run_id from kwargs, or fallback to session state
            run_id = kwargs.get("run_id") or st.session_state.get("active_run_id")
            start_time = time.perf_counter()
            try:
                return func(*args, **kwargs)
            finally:
                duration_ms = (time.perf_counter() - start_time) * 1000
                record_performance(operation_name, duration_ms, run_id)
        return wrapper
    return decorator
