from __future__ import annotations

from typing import List

import os
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
from google import genai


class TextEmbeddingService:
    """Gemini-based text embedding service (text-embedding-004 by default)."""

    def __init__(self) -> None:
        # Model name can be provided as google/text-embedding-004 or text-embedding-004
        raw_model = os.getenv("TEXT_EMBEDDING_MODEL_NAME", "text-embedding-004")
        if "/" in raw_model:
            raw_model = raw_model.split("/")[-1]
        self._model_name = raw_model
        # Defer API key configuration so callers can handle missing keys
        self._api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        self._client = None  # type: ignore[assignment]
        # Expected dimension fixed to 1024 per project setup
        self._expected_dim = 1024

    def _normalize(self, vec: List[float]) -> List[float]:
        arr = np.asarray(vec, dtype=np.float32)
        norm = np.linalg.norm(arr)
        if norm == 0.0:
            raise ValueError("Received zero-norm embedding")
        arr = arr / norm
        return arr.tolist()

    def _ensure_client(self) -> genai.Client:
        if self._client is not None:
            return self._client
        # Resolve API key lazily from env or .env
        api_key = self._api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            load_dotenv(override=False)
            api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            project_root = Path(__file__).resolve().parents[3]
            env_path = project_root / ".env"
            if env_path.exists():
                load_dotenv(dotenv_path=env_path, override=False)
                api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY or GOOGLE_API_KEY is missing")
        self._client = genai.Client(api_key=api_key)
        return self._client

    def _reshape_to_expected(self, vec: List[float]) -> List[float]:
        """Pad with zeros or truncate to match expected dimension, then renormalize."""
        arr = np.asarray(vec, dtype=np.float32)
        if arr.size == self._expected_dim:
            return self._normalize(arr.tolist())
        if arr.size < self._expected_dim:
            pad_width = self._expected_dim - arr.size
            arr = np.pad(arr, (0, pad_width), mode="constant", constant_values=0.0)
        else:
            arr = arr[: self._expected_dim]
        return self._normalize(arr.tolist())

    def _extract_embedding(self, resp) -> List[float]:
        # Supports different response shapes across SDK versions
        # Try: resp.embedding, resp.embeddings[0], resp['embedding'], resp['embeddings'][0], resp['data'][0]['embedding']
        emb = None
        # Attribute-style
        if hasattr(resp, "embedding"):
            emb = getattr(resp, "embedding")
        elif hasattr(resp, "embeddings"):
            try:
                arr = getattr(resp, "embeddings")
                if arr:
                    emb = arr[0]
            except Exception:
                pass
        # Dict-style
        if emb is None and isinstance(resp, dict):
            if "embedding" in resp:
                emb = resp.get("embedding")
            elif "embeddings" in resp:
                try:
                    arr = resp.get("embeddings")
                    if arr:
                        emb = arr[0]
                except Exception:
                    pass
            elif "data" in resp:
                try:
                    data = resp.get("data")
                    if data:
                        item = data[0]
                        emb = item.get("embedding") if isinstance(item, dict) else getattr(item, "embedding", None)
                except Exception:
                    pass

        if emb is None:
            raise RuntimeError("Embedding response missing 'embedding' field")

        # Unwrap to raw float list
        if isinstance(emb, dict) and "values" in emb:
            return list(emb["values"])  # type: ignore
        if hasattr(emb, "values"):
            try:
                return list(getattr(emb, "values"))  # type: ignore
            except Exception:
                pass
        if isinstance(emb, (list, tuple)):
            return list(emb)  # type: ignore
        # Last resort: if emb has attribute 'embedding' nested
        nested = getattr(emb, "embedding", None)
        if nested is not None:
            return self._extract_embedding({"embedding": nested})
        raise RuntimeError("Embedding response missing 'embedding' field")

    def embed_text(self, text: str) -> List[float]:
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        if text == "":
            raise ValueError("text must be non-empty")
        client = self._ensure_client()
        resp = client.models.embed_content(model=self._model_name, contents=text)
        vector = self._extract_embedding(resp)
        normalized = self._normalize(vector)
        if len(normalized) != self._expected_dim:
            normalized = self._reshape_to_expected(normalized)
        return normalized

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        if not isinstance(texts, list):
            raise TypeError("texts must be a list of strings")
        if any(not isinstance(t, str) for t in texts):
            raise TypeError("all items in texts must be strings")
        if len(texts) == 0:
            return []
        client = self._ensure_client()
        results: List[List[float]] = []
        for t in texts:
            resp = client.models.embed_content(model=self._model_name, contents=t)
            vector = self._extract_embedding(resp)
            normalized = self._normalize(vector)
            if len(normalized) != self._expected_dim:
                normalized = self._reshape_to_expected(normalized)
            results.append(normalized)
        return results
