"""
Helpers for loading data from Azure Blob storage (or local stand-ins).
"""

from pathlib import Path
from typing import Optional

import pandas as pd

from utils.session import read_table


def load_local_file(path: str) -> pd.DataFrame:
    """
    Temporary helper that reuses the local file loader until Blob integration is wired.
    """
    return read_table(path)


def ensure_directory(path: str) -> Path:
    """
    Return a Path object for listing local blob caches.
    """
    return Path(path).expanduser()
