from __future__ import annotations

from typing import Optional

from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse

from app.core.settings import get_settings


_client: Optional[QdrantClient] = None


def _init_client() -> QdrantClient:
    """Initialize and cache the Qdrant client.

    Returns:
        QdrantClient: Configured client instance.
    """
    global _client
    if _client is not None:
        return _client

    settings = get_settings()
    try:
        _client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)
        # Optional lightweight call to verify connectivity without altering state.
        # If the server is unreachable, this will raise. We swallow and re-raise a clean message.
        try:
            _client.get_collections()
        except UnexpectedResponse as e:
            # Connectivity or auth issues result in an unexpected response.
            raise RuntimeError(f"Failed to connect to Qdrant: {e}") from e
        return _client
    except Exception as e:  # broad by design to catch transport-level errors
        raise RuntimeError(f"Qdrant client initialization failed: {e}") from e


def get_qdrant_client() -> QdrantClient:
    """Return the initialized Qdrant client (singleton).

    Returns:
        QdrantClient: The initialized client.
    """
    return _init_client()
