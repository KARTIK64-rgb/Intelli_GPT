from __future__ import annotations

from typing import List, Dict, Optional
import logging
from uuid import uuid4

from app.services.pdf_parser.parser import PDFParser
from app.services.embeddings.text_embeddings import TextEmbeddingService
from app.services.embeddings.image_embeddings import ImageEmbeddingService
from app.services.image_s3.image_store import ImageStore
from app.services.vectordb_qdrant.client import get_qdrant_client
from app.services.vectordb_qdrant.queries import QdrantRepository
from app.services.vectordb_qdrant.collections import ensure_collections_exist
from app.services.vectordb_qdrant.schema import (
    build_text_payload,
    build_image_payload,
)


class IngestionPipeline:
    """End-to-end ingestion pipeline for PDFs (RAM-only).

    Steps:
    - Parse PDF text and images in memory
    - Embed text (page-level chunks) and images
    - Upload images to S3 (private) and upsert vectors to Qdrant
    - No PDF persistence; raises on failure
    """

    def __init__(self) -> None:
        self.pdf_parser = PDFParser()
        self.text_embedder = TextEmbeddingService()
        self.image_embedder = ImageEmbeddingService()
        self.image_store = ImageStore()
        client = get_qdrant_client()
        ensure_collections_exist(client)
        self.qdrant_repo = QdrantRepository(client)

    def ingest_pdf(self, pdf_bytes: bytes, pdf_id: str, doc_title: Optional[str] = None) -> None:
        if not isinstance(pdf_bytes, (bytes, bytearray)):
            raise TypeError("pdf_bytes must be bytes or bytearray")
        if not isinstance(pdf_id, str) or not pdf_id:
            raise ValueError("pdf_id must be a non-empty string")

        # 1) Parse PDF text and images from RAM
        text_blocks: List[Dict] = self.pdf_parser.parse_text(pdf_bytes)
        image_blocks: List[Dict] = self.pdf_parser.parse_images(pdf_bytes)

        # 2) Text embeddings and payloads (one page = one chunk)
        text_points: List[tuple[str, List[float], Dict]] = []
        non_empty_text_blocks = [
            b for b in text_blocks
            if isinstance(b.get("text", ""), str) and b["text"].strip()
        ]
        if non_empty_text_blocks:
            try:
                texts = [b["text"] for b in non_empty_text_blocks]
                vectors = self.text_embedder.embed_texts(texts)
                if len(vectors) != len(non_empty_text_blocks):
                    raise RuntimeError("Mismatch between text blocks and embeddings count")
                for b, vec in zip(non_empty_text_blocks, vectors):
                    page_no = int(b["page_number"])  # 1-based
                    point_id = uuid4().hex
                    chunk_id = f"p{page_no}"
                    payload = build_text_payload(
                        point_id=point_id,
                        source_pdf_id=pdf_id,
                        page_number=page_no,
                        chunk_id=chunk_id,
                        chunk_text_preview=b["text"][:500],
                        char_start=int(b.get("char_start", 0)),
                        char_end=int(b.get("char_end", len(b.get("text", "")))),
                        embedding_model_name=getattr(self.text_embedder, "_model_name", "text-embedding"),
                        doc_title=doc_title,
                        tags=None,
                    )
                    text_points.append((point_id, vec, payload))
            except Exception as e:
                msg = str(e)
                if "GEMINI_API_KEY" in msg or "GOOGLE_API_KEY" in msg:
                    logging.warning("Skipping text embeddings: %s", msg)
                    text_points = []
                else:
                    raise

        # 3) Image upload, embeddings, and payloads
        image_points: List[tuple[str, List[float], Dict]] = []
        if image_blocks:
            # Upload images first to get S3 keys
            s3_keys: List[str] = []
            for img in image_blocks:
                s3_key = self.image_store.upload_image(
                    image_bytes=img["image_bytes"],
                    mime_type=img["mime_type"],
                    pdf_id=pdf_id,
                    page_number=int(img["page_number"]),
                )
                s3_keys.append(s3_key)

            # Create image embeddings in batch
            img_bytes_list = [img["image_bytes"] for img in image_blocks]
            img_vectors = self.image_embedder.embed_images(img_bytes_list)
            if len(img_vectors) != len(image_blocks):
                raise RuntimeError("Mismatch between image blocks and embeddings count")

            for img, vec, s3_key in zip(image_blocks, img_vectors, s3_keys):
                point_id = uuid4().hex
                payload = build_image_payload(
                    point_id=point_id,
                    source_pdf_id=pdf_id,
                    page_number=int(img["page_number"]),
                    image_s3_key=s3_key,
                    mime_type=img["mime_type"],
                    width_px=int(img["width_px"]),
                    height_px=int(img["height_px"]),
                    embedding_model_name="openclip-ViT-L-14-openai",
                    caption_text=None,
                    alt_text=None,
                    doc_title=doc_title,
                    tags=None,
                )
                image_points.append((point_id, vec, payload))

        # 4) Batch upsert into Qdrant
        if text_points:
            self.qdrant_repo.upsert_text_points(text_points)
        if image_points:
            self.qdrant_repo.upsert_image_points(image_points)

        # 5-6) Do not store PDF; return nothing (implicit None)
        return None
