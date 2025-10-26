"""
Azure Blob storage client scaffolding.
"""

from typing import List


def list_blobs(container: str, prefix: str = "") -> List[str]:
    raise NotImplementedError("Blob listing requires Azure credentials.")


def download_blob(container: str, blob_name: str) -> bytes:
    raise NotImplementedError("Blob download requires Azure credentials.")

