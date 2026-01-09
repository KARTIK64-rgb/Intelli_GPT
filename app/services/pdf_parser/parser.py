from __future__ import annotations

from typing import Dict, List

import fitz  # PyMuPDF


class PDFParser:
    """In-memory PDF parser for text and image extraction using PyMuPDF.

    Accepts raw PDF bytes and performs all operations in RAM (no disk usage).
    """

    def parse_text(self, pdf_bytes: bytes) -> List[Dict]:
        """Extract per-page text with character span metadata.

        Args:
            pdf_bytes: Raw PDF bytes.

        Returns:
            List of dicts with keys: page_number, text, char_start, char_end.
        """
        if not isinstance(pdf_bytes, (bytes, bytearray)):
            raise TypeError("pdf_bytes must be bytes or bytearray")

        try:
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        except Exception as e:
            # Propagate exceptions for malformed PDFs
            raise e

        results: List[Dict] = []
        try:
            for page_index in range(doc.page_count):
                page = doc.load_page(page_index)
                text = page.get_text("text") or ""
                # Whole-page span
                item = {
                    "page_number": page_index + 1,
                    "text": text,
                    "char_start": 0,
                    "char_end": len(text),
                }
                results.append(item)
        finally:
            doc.close()

        return results

    def parse_images(self, pdf_bytes: bytes) -> List[Dict]:
        """Extract embedded images (XObject) as bytes with metadata.

        Filters out very small images (width < 64 or height < 64).

        Args:
            pdf_bytes: Raw PDF bytes.

        Returns:
            List of dicts with keys: page_number, image_bytes, mime_type, width_px, height_px.
        """
        if not isinstance(pdf_bytes, (bytes, bytearray)):
            raise TypeError("pdf_bytes must be bytes or bytearray")

        try:
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        except Exception as e:
            raise e

        def _mime_from_ext(ext: str) -> str:
            ext = (ext or "").lower()
            if ext in {"jpg", "jpeg"}:
                return "image/jpeg"
            if ext == "png":
                return "image/png"
            if ext == "tiff" or ext == "tif":
                return "image/tiff"
            if ext == "bmp":
                return "image/bmp"
            if ext == "webp":
                return "image/webp"
            # Fallback
            return f"image/{ext or 'octet-stream'}"

        images: List[Dict] = []
        try:
            for page_index in range(doc.page_count):
                page = doc.load_page(page_index)
                # Get XObject images
                for img in page.get_images(full=True):
                    xref = img[0]
                    img_info = doc.extract_image(xref)
                    if not img_info:
                        continue
                    data = img_info.get("image")
                    width = img_info.get("width")
                    height = img_info.get("height")
                    ext = img_info.get("ext")

                    if not isinstance(data, (bytes, bytearray)):
                        continue
                    if not isinstance(width, int) or not isinstance(height, int):
                        continue
                    # Filter out very small images
                    if width < 64 or height < 64:
                        continue

                    item = {
                        "page_number": page_index + 1,
                        "image_bytes": bytes(data),
                        "mime_type": _mime_from_ext(ext),
                        "width_px": width,
                        "height_px": height,
                    }
                    images.append(item)
        finally:
            doc.close()

        return images
