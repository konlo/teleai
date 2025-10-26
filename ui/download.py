import io
import zipfile
from typing import Iterable, Tuple


def build_zip_from_files(files: Iterable[Tuple[str, bytes]]) -> bytes:
    """Utility to build an in-memory zip archive."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, data in files:
            zf.writestr(name, data)
    return buffer.getvalue()

