from __future__ import annotations

from typing import List, Optional
from datetime import datetime, timezone


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def build_text_payload(
    *,
    point_id: str,
    source_pdf_id: str,
    page_number: int,
    chunk_id: str,
    chunk_text_preview: str,
    char_start: int,
    char_end: int,
    embedding_model_name: str,
    doc_title: Optional[str] = None,
    tags: Optional[List[str]] = None,
) -> dict:
    """Build payload dict for a text chunk point.

    Matches the agreed schema for the `text_chunks` collection payload.
    """
    return {
        "id": point_id,
        "type": "text",
        "source_pdf_id": source_pdf_id,
        "page_number": int(page_number),
        "section_id": None,
        "chunk_id": chunk_id,
        "chunk_text_preview": chunk_text_preview,
        "chunk_start_char_index": int(char_start),
        "chunk_end_char_index": int(char_end),
        "language": None,
        "doc_title": doc_title,
        "tags": list(tags) if tags else [],
        "modality": "text",
        "embedding_model_name": embedding_model_name,
        # Embedding dim (1024) is characteristic of collection; include for clarity
        "embedding_dim": 1024,
        "created_at": _now_iso(),
        "version": None,
    }


def build_image_payload(
    *,
    point_id: str,
    source_pdf_id: str,
    page_number: int,
    image_s3_key: str,
    mime_type: str,
    width_px: int,
    height_px: int,
    embedding_model_name: str,
    caption_text: Optional[str] = None,
    alt_text: Optional[str] = None,
    doc_title: Optional[str] = None,
    tags: Optional[List[str]] = None,
) -> dict:
    """Build payload dict for an image point.

    Matches the agreed schema for the `images` collection payload.
    """
    return {
        "id": point_id,
        "type": "image",
        "source_pdf_id": source_pdf_id,
        "page_number": int(page_number),
        "figure_id": None,
        "image_s3_key": image_s3_key,
        "image_mime_type": mime_type,
        "width_px": int(width_px),
        "height_px": int(height_px),
        "caption_text": caption_text,
        "alt_text": alt_text,
        "doc_title": doc_title,
        "tags": list(tags) if tags else [],
        "modality": "image",
        "embedding_model_name": embedding_model_name,
        # Embedding dim (768) is characteristic of collection; include for clarity
        "embedding_dim": 768,
        "created_at": _now_iso(),
        "version": None,
    }
