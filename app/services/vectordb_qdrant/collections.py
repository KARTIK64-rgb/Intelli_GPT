from __future__ import annotations

import os
from typing import Set

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PayloadSchemaType


def _existing_collections(client: QdrantClient) -> Set[str]:
    resp = client.get_collections()
    return {c.name for c in resp.collections}


def ensure_collections_exist(client: QdrantClient) -> None:
    """Ensure text and image collections exist with correct configs.

    Creates collections only if missing, and adds minimal payload indexes when newly created.
    Raises exceptions on failure.
    """
    text_name = os.getenv("QDRANT_COLLECTION_TEXT", "text_chunks")
    image_name = os.getenv("QDRANT_COLLECTION_IMAGE", "images")

    existing = _existing_collections(client)

    # Text collection
    if text_name not in existing:
        text_dim = int(os.getenv("TEXT_EMBEDDING_DIM", "1024"))
        client.create_collection(
            collection_name=text_name,
            vectors_config=VectorParams(size=text_dim, distance=Distance.COSINE),
        )
        # Add payload indexes for newly created collection
        client.create_payload_index(
            collection_name=text_name,
            field_name="source_pdf_id",
            field_schema=PayloadSchemaType.KEYWORD,
        )
        client.create_payload_index(
            collection_name=text_name,
            field_name="page_number",
            field_schema=PayloadSchemaType.INTEGER,
        )

    # Image collection
    if image_name not in existing:
        client.create_collection(
            collection_name=image_name,
            vectors_config=VectorParams(size=768, distance=Distance.COSINE),
        )
        # Add payload indexes for newly created collection
        client.create_payload_index(
            collection_name=image_name,
            field_name="source_pdf_id",
            field_schema=PayloadSchemaType.KEYWORD,
        )
        client.create_payload_index(
            collection_name=image_name,
            field_name="page_number",
            field_schema=PayloadSchemaType.INTEGER,
        )
