from __future__ import annotations

import os
from typing import Dict, List, Optional, Any

import numpy as np

from app.services.embeddings.text_embeddings import TextEmbeddingService
from app.services.embeddings.image_embeddings import ImageEmbeddingService
from app.services.vectordb_qdrant.client import get_qdrant_client
from app.services.vectordb_qdrant.queries import QdrantRepository
from app.services.vectordb_qdrant.collections import ensure_collections_exist
from app.services.image_s3.presigned_urls import generate_presigned_url


class QueryPipeline:
    """Multimodal query pipeline with deterministic fusion (text 0.6, image 0.4)."""

    def __init__(self) -> None:
        self.text_embedder = TextEmbeddingService()
        self.image_embedder = ImageEmbeddingService()
        client = get_qdrant_client()
        # Ensure collections exist before querying
        try:
            ensure_collections_exist(client)
        except Exception:
            # Non-fatal during query; if creation fails due to perms, search will fail anyway.
            pass
        self.qdrant_repo = QdrantRepository(client)

    # LLM answer generator is configured below with a safe fallback

        # Config
        self.top_k_text = int(os.getenv("TOP_K_TEXT", "5"))
        self.top_k_image = int(os.getenv("TOP_K_IMAGE", "5"))
        self.url_ttl = int(os.getenv("AWS_S3_PRESIGNED_URL_TTL_SECONDS", "900"))
        # We now use CLIP text encoder to embed text in the same space as image embeddings
        self.image_query_model = "openclip-ViT-L-14-openai"

        # If LLM generator module isn't available, use a no-op fallback
        try:
            from app.services.llm_gemini.answer_generator import GeminiAnswerGenerator  # type: ignore
            self.answer_generator = GeminiAnswerGenerator()
        except Exception:
            class _Fallback:
                def generate(self, question: str, context: Dict[str, Any]) -> str:
                    parts = []
                    for t in (context or {}).get("text_chunks", [])[:3]:
                        preview = (t or {}).get("chunk_text_preview") or ""
                        if preview:
                            parts.append(preview.strip())
                    joined = "\n\n".join(parts) or "No relevant context found."
                    return f"Answer (no LLM available). Based only on retrieved context, here are the most relevant excerpts:\n\n{joined}"
            self.answer_generator = _Fallback()

    # --- helpers ---
    @staticmethod
    def _normalize_scores(scores: List[float]) -> List[float]:
        if not scores:
            return []
        arr = np.asarray(scores, dtype=np.float32)
        min_v = float(arr.min())
        max_v = float(arr.max())
        if max_v - min_v <= 1e-12:
            return [1.0 for _ in scores]
        return [float((s - min_v) / (max_v - min_v)) for s in scores]

    def _embed_text_for_image_search(self, question: str) -> List[float]:
        # Use CLIP text encoder so the text embedding is comparable with image embeddings
        return self.image_embedder.embed_text_to_image_space(question)

    # --- main entry ---
    def answer_question(self, question: str) -> Dict[str, Any]:
        if not isinstance(question, str) or not question.strip():
            raise ValueError("question must be a non-empty string")

        # 1) Embed question: text and image-space text
        text_vec = self.text_embedder.embed_text(question)
        image_query_vec = self._embed_text_for_image_search(question)

        # 2) Search Qdrant
        text_hits = self.qdrant_repo.search_text(query_vector=text_vec, top_k=self.top_k_text)
        image_hits = self.qdrant_repo.search_images(query_vector=image_query_vec, top_k=self.top_k_image)

        # 3) Normalize scores in each modality
        text_scores = [float(h.score) for h in text_hits]
        image_scores = [float(h.score) for h in image_hits]
        text_norm = self._normalize_scores(text_scores)
        image_norm = self._normalize_scores(image_scores)

        # 4) Compute fused scores independently per modality (no cross-linking in v1)
        # For text items: image contribution is 0
        text_fused = [0.6 * ts + 0.4 * 0.0 for ts in text_norm]
        # For image items: text contribution is 0
        image_fused = [0.6 * 0.0 + 0.4 * iscore for iscore in image_norm]

        # 5) Select top fused results (keep original order by score descending within each list)
        # We already have modality-level hits; we will present both lists, sorted by fused score.
        text_ranked = [
            (hit, ts, 0.0, fs)
            for hit, ts, fs in zip(text_hits, text_norm, text_fused)
        ]
        text_ranked.sort(key=lambda x: x[3], reverse=True)

        image_ranked = [
            (hit, 0.0, iscore, fs)
            for hit, iscore, fs in zip(image_hits, image_norm, image_fused)
        ]
        image_ranked.sort(key=lambda x: x[3], reverse=True)

        # 6) Generate pre-signed URLs for selected images
        images_resp: List[Dict[str, Any]] = []
        for hit, tscore, iscore, fscore in image_ranked:
            payload = hit.payload or {}
            s3_key = payload.get("image_s3_key")
            if not s3_key:
                # Skip items without S3 key
                continue
            url = generate_presigned_url(s3_key=s3_key, ttl_seconds=self.url_ttl)
            images_resp.append({
                "id": payload.get("id"),
                "presigned_url": url,
                "ttl_seconds": self.url_ttl,
                "page_number": payload.get("page_number"),
                "caption_text": payload.get("caption_text"),
                "alt_text": payload.get("alt_text"),
                "image_mime_type": payload.get("image_mime_type"),
                "s3_key": s3_key,
                "scores": {
                    "image_score": round(float(iscore), 6),
                    "text_score": round(float(tscore), 6),
                    "fused_score": round(float(fscore), 6),
                },
                "embedding_model_name": payload.get("embedding_model_name"),
                "modality": "image",
            })

        # 7) Build context object: text chunks + image references
        text_ctx: List[Dict[str, Any]] = []
        for hit, tscore, iscore, fscore in text_ranked:
            payload = hit.payload or {}
            text_ctx.append({
                "id": payload.get("id"),
                "chunk_text_preview": payload.get("chunk_text_preview"),
                "page_number": payload.get("page_number"),
                "section_id": payload.get("section_id"),
                "scores": {
                    "text_score": round(float(tscore), 6),
                    "image_score": round(float(iscore), 6),
                    "fused_score": round(float(fscore), 6),
                },
                "embedding_model_name": payload.get("embedding_model_name"),
                "modality": "text",
            })

        context = {
            "text_chunks": text_ctx,
            "images": images_resp,
            "fusion": {
                "strategy": "weighted_sum",
                "weights": {"text": 0.6, "image": 0.4},
                "normalized": True,
            },
        }

        # 8) Generate answer using ONLY provided context
        answer_text = self.answer_generator.generate(question=question, context=context)

        # 9) Assemble response (matches agreed response schema)
        response = {
            "request_id": uuid_like(),
            "question": question,
            "answer_text": answer_text,
            "images": images_resp,
            "context": context,
            "meta": {
                "top_k_text": self.top_k_text,
                "top_k_image": self.top_k_image,
                "qdrant_collections": {
                    "text": os.getenv("QDRANT_COLLECTION_TEXT", "text_chunks"),
                    "image": os.getenv("QDRANT_COLLECTION_IMAGE", "images"),
                },
                "embedding_dims": {"text": 1024, "image": 768},
                "metrics": {"text": "cosine", "image": "cosine"},
                "generated_by": {
                    "llm": os.getenv("GEMINI_MODEL_NAME", "gemini-model"),
                },
            },
        }
        return response


def uuid_like() -> str:
    # Local helper to avoid importing uuid if not desired elsewhere
    import uuid as _uuid

    return _uuid.uuid4().hex
