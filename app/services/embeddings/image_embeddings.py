from __future__ import annotations

from typing import List

import io
import numpy as np
import torch
from PIL import Image
import open_clip


class ImageEmbeddingService:
    """OpenCLIP-based image embedding service (ViT-L-14).

    Loads the model and preprocess once (CPU by default) and embeds image bytes.
    """

    def __init__(self) -> None:
        # Load on CPU by default
        self._device = torch.device("cpu")
        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name="ViT-L-14",
            pretrained="openai",
            device=self._device,
        )
        self._model = model.eval()
        self._preprocess = preprocess
        # Tokenizer for CLIP text encoder (maps text into same 768-d space)
        self._tokenizer = open_clip.get_tokenizer("ViT-L-14")

    def embed_images(self, images: List[bytes]) -> List[List[float]]:
        """Embed a list of raw image bytes and return L2-normalized vectors.

        Args:
            images: List of PNG/JPEG bytes.

        Returns:
            List of embeddings, each a list of floats.
        """
        if not isinstance(images, list):
            raise TypeError("images must be a list of bytes")
        if any(not isinstance(b, (bytes, bytearray)) for b in images):
            raise TypeError("all items in images must be bytes or bytearray")
        if len(images) == 0:
            return []

        embeddings: List[List[float]] = []
        with torch.no_grad():
            for data in images:
                # Decode image bytes using PIL in-memory
                img = Image.open(io.BytesIO(data)).convert("RGB")
                # Preprocess via OpenCLIP transform
                tensor = self._preprocess(img).unsqueeze(0).to(self._device)
                # Forward through model to get image features
                feats = self._model.encode_image(tensor)
                feats = feats.detach().cpu().numpy().astype(np.float32).squeeze(0)
                # L2 normalization
                norm = np.linalg.norm(feats)
                if norm == 0.0:
                    raise ValueError("Received zero-norm image embedding")
                feats = feats / norm
                embeddings.append(feats.tolist())

        return embeddings

    def embed_text_to_image_space(self, text: str) -> List[float]:
        """Embed text using CLIP text encoder into the same space as image embeddings.

        Returns a 768-d L2-normalized vector compatible with encode_image outputs.
        """
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        if not text.strip():
            raise ValueError("text must be non-empty")

        with torch.no_grad():
            tokens = self._tokenizer([text])
            if hasattr(tokens, "to"):
                tokens = tokens.to(self._device)
            feats = self._model.encode_text(tokens)
            feats = feats.detach().cpu().numpy().astype(np.float32).squeeze(0)
            norm = np.linalg.norm(feats)
            if norm == 0.0:
                raise ValueError("Received zero-norm CLIP text embedding")
            feats = feats / norm
        return feats.tolist()

    def embed_texts_to_image_space(self, texts: List[str]) -> List[List[float]]:
        """Batch embed texts into CLIP image-text shared space.

        Args:
            texts: list of strings.
        Returns:
            list of 768-d L2-normalized vectors.
        """
        if not isinstance(texts, list):
            raise TypeError("texts must be a list of strings")
        if any(not isinstance(t, str) for t in texts):
            raise TypeError("all items in texts must be strings")
        if len(texts) == 0:
            return []

        with torch.no_grad():
            tokens = self._tokenizer(texts)
            if hasattr(tokens, "to"):
                tokens = tokens.to(self._device)
            feats = self._model.encode_text(tokens)
            feats = feats.detach().cpu().numpy().astype(np.float32)
            # L2 normalize rows
            norms = np.linalg.norm(feats, axis=1, keepdims=True)
            norms[norms == 0.0] = 1.0
            feats = feats / norms
        return [row.astype(np.float32).tolist() for row in feats]
