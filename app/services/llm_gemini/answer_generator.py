from __future__ import annotations

import os
from typing import Any, Dict, List


class GeminiAnswerGenerator:
    """Thin wrapper around Gemini to generate answers grounded in provided context.

    If google.genai is not installed or API key is missing, falls back to a concise
    extractive summary from the provided text chunks.
    """

    def __init__(self) -> None:
        self._model_name = os.getenv("GEMINI_MODEL_NAME", "gemini-1.5-flash")
        self._api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

        self._genai = None
        if self._api_key:
            try:
                import google.genai as genai  # type: ignore
                genai.configure(api_key=self._api_key)
                self._genai = genai
            except Exception:
                self._genai = None

    def _build_prompt(self, question: str, context: Dict[str, Any]) -> str:
        parts: List[str] = [
            "You are a helpful assistant. Answer ONLY using the provided context.",
            "If the answer isn't in the context, say you don't have enough information.",
            "Be concise and cite page numbers when helpful.",
            "",
            f"Question: {question}",
            "",
            "Context snippets:",
        ]
        for t in (context or {}).get("text_chunks", [])[:6]:
            page = (t or {}).get("page_number")
            txt = ((t or {}).get("chunk_text_preview") or "").strip()
            if txt:
                parts.append(f"- [p{page}] {txt}")
        return "\n".join(parts)

    def generate(self, question: str, context: Dict[str, Any]) -> str:
        prompt = self._build_prompt(question, context)

        # Try LLM if available
        if self._genai is not None:
            try:
                model = self._genai.GenerativeModel(self._model_name)
                resp = model.generate_content(prompt)
                text = getattr(resp, "text", None) or ""
                if isinstance(text, str) and text.strip():
                    return text.strip()
            except Exception:
                pass

        # Fallback: extractive summary of top context
        snippets: List[str] = []
        for t in (context or {}).get("text_chunks", [])[:3]:
            preview = (t or {}).get("chunk_text_preview") or ""
            if preview:
                snippets.append(preview.strip())
        body = "\n\n".join(snippets) or "No relevant context found."
        return f"Based on the retrieved context, here are the most relevant excerpts:\n\n{body}"
