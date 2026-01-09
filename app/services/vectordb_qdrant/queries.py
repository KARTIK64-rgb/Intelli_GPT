from __future__ import annotations

import os
from typing import List, Tuple, Dict, Any, Optional

from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, Filter, FieldCondition, MatchValue
from qdrant_client.http.exceptions import UnexpectedResponse


def _get_collection_names() -> tuple[str, str]:
    text = os.getenv("QDRANT_COLLECTION_TEXT", "text_chunks")
    image = os.getenv("QDRANT_COLLECTION_IMAGE", "images")
    return text, image


class QdrantRepository:
    """Thin repository for upsert/search against Qdrant collections."""

    def __init__(self, client: QdrantClient) -> None:
        if not isinstance(client, QdrantClient):
            raise TypeError("client must be a QdrantClient")
        self.client = client
        self.text_collection, self.image_collection = _get_collection_names()

    # --- internal compatibility layer ---
    def _search(self, collection: str, vector: List[float], top_k: int, flt: Optional[Filter]):
        """Call the appropriate search API depending on qdrant-client version.

        Returns a list-like of ScoredPoint. If the method returns a response object with
        a `.points` attribute, that list is returned.
        """
        # Preferred modern method name
        if hasattr(self.client, "search"):
            try:
                return self.client.search(
                    collection_name=collection,
                    query_vector=vector,
                    limit=top_k,
                    query_filter=flt,
                )
            except (TypeError, UnexpectedResponse, Exception) as e:
                msg = str(e).lower()
                # If server/client complains about query_filter, retry with filter
                if (
                    "query_filter" in msg
                    or "unexpected keyword argument 'query_filter'" in msg
                    or "unknown arguments: ['query_filter']" in msg
                ):
                    return self.client.search(
                        collection_name=collection,
                        query_vector=vector,
                        limit=top_k,
                        filter=flt,  # type: ignore[arg-type]
                    )
                # If it complains about filter, retry with query_filter
                if (
                    "unknown arguments: ['filter']" in msg
                    or "unexpected keyword argument 'filter'" in msg
                ):
                    return self.client.search(
                        collection_name=collection,
                        query_vector=vector,
                        limit=top_k,
                        query_filter=flt,
                    )
                raise

        # Fallback method names in other versions
        if hasattr(self.client, "query_points"):
            # Try with query_filter first (newer SDKs), then with filter
            try:
                resp = self.client.query_points(
                    collection_name=collection,
                    query=vector,
                    limit=top_k,
                    query_filter=flt,
                )
            except (TypeError, UnexpectedResponse, Exception) as e:
                msg = str(e).lower()
                if (
                    "query_filter" in msg
                    or "unexpected keyword argument 'query_filter'" in msg
                    or "unknown arguments: ['query_filter']" in msg
                ):
                    try:
                        resp = self.client.query_points(
                            collection_name=collection,
                            query=vector,
                            limit=top_k,
                            filter=flt,  # type: ignore[arg-type]
                        )
                    except Exception:
                        # Try alt param name 'vector' instead of 'query'
                        resp = self.client.query_points(
                            collection_name=collection,
                            vector=vector,
                            limit=top_k,
                            filter=flt,  # type: ignore[arg-type]
                        )
                elif (
                    "unknown arguments: ['filter']" in msg
                    or "unexpected keyword argument 'filter'" in msg
                ):
                    try:
                        resp = self.client.query_points(
                            collection_name=collection,
                            query=vector,
                            limit=top_k,
                            query_filter=flt,
                        )
                    except Exception:
                        resp = self.client.query_points(
                            collection_name=collection,
                            vector=vector,
                            limit=top_k,
                            query_filter=flt,
                        )
                else:
                    # If it's some other error, re-raise
                    raise
            return getattr(resp, "points", resp)

        if hasattr(self.client, "search_points"):
            try:
                resp = self.client.search_points(
                    collection_name=collection,
                    query=vector,
                    limit=top_k,
                    query_filter=flt,
                )
            except (TypeError, UnexpectedResponse, Exception) as e:
                msg = str(e).lower()
                if (
                    "query_filter" in msg
                    or "unexpected keyword argument 'query_filter'" in msg
                    or "unknown arguments: ['query_filter']" in msg
                ):
                    resp = self.client.search_points(
                        collection_name=collection,
                        query=vector,
                        limit=top_k,
                        filter=flt,  # type: ignore[arg-type]
                    )
                elif (
                    "unknown arguments: ['filter']" in msg
                    or "unexpected keyword argument 'filter'" in msg
                ):
                    resp = self.client.search_points(
                        collection_name=collection,
                        query=vector,
                        limit=top_k,
                        query_filter=flt,
                    )
                else:
                    raise
            return getattr(resp, "points", resp)

        raise RuntimeError("Qdrant client does not support a known search method")

    # Upserts
    def upsert_text_points(self, points: List[Tuple[str, List[float], Dict[str, Any]]]) -> None:
        if not isinstance(points, list):
            raise TypeError("points must be a list")
        qdrant_points = [PointStruct(id=pid, vector=vec, payload=payload) for pid, vec, payload in points]
        self.client.upsert(collection_name=self.text_collection, points=qdrant_points)

    def upsert_image_points(self, points: List[Tuple[str, List[float], Dict[str, Any]]]) -> None:
        if not isinstance(points, list):
            raise TypeError("points must be a list")
        qdrant_points = [PointStruct(id=pid, vector=vec, payload=payload) for pid, vec, payload in points]
        self.client.upsert(collection_name=self.image_collection, points=qdrant_points)

    # Search
    def _build_filter(self, filters: Optional[Dict[str, Any]]) -> Optional[Filter]:
        if not filters:
            return None
        conditions = []
        for field, value in filters.items():
            conditions.append(FieldCondition(key=field, match=MatchValue(value=value)))
        return Filter(must=conditions)

    def search_text(
        self,
        query_vector: List[float],
        top_k: int,
        filters: Optional[Dict[str, Any]] = None,
    ) -> list:
        if not isinstance(query_vector, list) or not query_vector:
            raise ValueError("query_vector must be a non-empty list of floats")
        if not isinstance(top_k, int) or top_k <= 0:
            raise ValueError("top_k must be a positive integer")
        flt = self._build_filter(filters)
        return self._search(self.text_collection, query_vector, top_k, flt)

    def search_images(
        self,
        query_vector: List[float],
        top_k: int,
        filters: Optional[Dict[str, Any]] = None,
    ) -> list:
        if not isinstance(query_vector, list) or not query_vector:
            raise ValueError("query_vector must be a non-empty list of floats")
        if not isinstance(top_k, int) or top_k <= 0:
            raise ValueError("top_k must be a positive integer")
        flt = self._build_filter(filters)
        return self._search(self.image_collection, query_vector, top_k, flt)
