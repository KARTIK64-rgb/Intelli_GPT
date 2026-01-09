from __future__ import annotations

from typing import Set

from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.models import Distance, VectorParams, PayloadSchemaType, PayloadIndexParams

from app.infra.qdrant_client import get_qdrant_client


TEXT_COLLECTION = "text_chunks"
IMAGE_COLLECTION = "images"

TEXT_VECTOR_SIZE = 1024
IMAGE_VECTOR_SIZE = 768


def _existing_collections(client: QdrantClient) -> Set[str]:
    """Return the set of existing collection names."""
    resp = client.get_collections()
    return {c.name for c in resp.collections}


def _create_text_collection(client: QdrantClient) -> None:
    client.create_collection(
        collection_name=TEXT_COLLECTION,
        vectors_config=VectorParams(size=TEXT_VECTOR_SIZE, distance=Distance.COSINE),
    )


def _create_image_collection(client: QdrantClient) -> None:
    client.create_collection(
        collection_name=IMAGE_COLLECTION,
        vectors_config=VectorParams(size=IMAGE_VECTOR_SIZE, distance=Distance.COSINE),
    )


def create_collections_if_not_exists() -> None:
    """Create required Qdrant collections if they do not already exist.

    Idempotent and safe for repeated calls.
    """
    client = get_qdrant_client()
    names = _existing_collections(client)

    if TEXT_COLLECTION not in names:
        _create_text_collection(client)
    if IMAGE_COLLECTION not in names:
        _create_image_collection(client)


def ensure_payload_indexes() -> None:
    """Ensure payload indexes exist for required fields on both collections.

    Creates indexes only if they do not already exist. Safe and idempotent.
    """
    client = get_qdrant_client()

    def _ensure_index(collection: str, field: str, schema_type: PayloadSchemaType) -> None:
        try:
            client.create_payload_index(
                collection_name=collection,
                field_name=field,
                field_schema=PayloadIndexParams(schema=schema_type),
            )
        except UnexpectedResponse as e:
            # If index already exists or server returns a conflict, ignore; otherwise re-raise.
            message = str(e)
            if "already exists" in message.lower() or "index exists" in message.lower():
                return
            raise RuntimeError(f"Failed to create payload index for '{collection}.{field}': {e}") from e

    # Text collection indexes
    _ensure_index(TEXT_COLLECTION, "source_pdf_id", PayloadSchemaType.KEYWORD)
    _ensure_index(TEXT_COLLECTION, "page_number", PayloadSchemaType.INTEGER)
    _ensure_index(TEXT_COLLECTION, "tags", PayloadSchemaType.KEYWORD)

    # Image collection indexes
    _ensure_index(IMAGE_COLLECTION, "source_pdf_id", PayloadSchemaType.KEYWORD)
    _ensure_index(IMAGE_COLLECTION, "page_number", PayloadSchemaType.INTEGER)
    _ensure_index(IMAGE_COLLECTION, "tags", PayloadSchemaType.KEYWORD)
