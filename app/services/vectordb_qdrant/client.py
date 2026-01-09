from __future__ import annotations

import os
from qdrant_client import QdrantClient


def get_qdrant_client() -> QdrantClient:
    """Initialize and return a Qdrant Cloud client (HTTPS).

    Reads QDRANT_URL and QDRANT_API_KEY from environment variables.
    Raises:
        ValueError: if required environment variables are missing.
        Exception: if client initialization fails.
    """
    url = os.getenv("QDRANT_URL")
    api_key = os.getenv("QDRANT_API_KEY")

    if not url:
        raise ValueError("QDRANT_URL is missing")
    if not api_key:
        raise ValueError("QDRANT_API_KEY is missing")
    if not url.startswith("https://"):
        raise ValueError("Qdrant Cloud URL must use HTTPS")

    client = QdrantClient(url=url, api_key=api_key)
    return client
