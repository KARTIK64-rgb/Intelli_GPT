from __future__ import annotations

import os
import uuid
from typing import Final

from app.services.image_s3.s3_client import get_s3_client
from pathlib import Path
from dotenv import load_dotenv


BASE_PREFIX: Final[str] = "images"


class ImageStore:
    def __init__(self) -> None:
        # Resolve bucket name with fallbacks and lazy .env load
        bucket = os.getenv("AWS_S3_BUCKET") or os.getenv("S3_BUCKET_NAME")
        if not bucket:
            # Try loading .env in case not yet loaded in this process
            load_dotenv(override=False)
            bucket = os.getenv("AWS_S3_BUCKET") or os.getenv("S3_BUCKET_NAME")
        if not bucket:
            # Explicitly load project-root .env as last resort
            project_root = Path(__file__).resolve().parents[3]
            env_path = project_root / ".env"
            if env_path.exists():
                load_dotenv(dotenv_path=env_path, override=False)
                bucket = os.getenv("AWS_S3_BUCKET") or os.getenv("S3_BUCKET_NAME")
        if not bucket:
            raise ValueError("AWS_S3_BUCKET is missing")
        self._bucket = bucket
        self._client = get_s3_client()

    def upload_image(
        self,
        image_bytes: bytes,
        mime_type: str,
        pdf_id: str,
        page_number: int,
    ) -> str:
        if not isinstance(image_bytes, (bytes, bytearray)):
            raise TypeError("image_bytes must be bytes or bytearray")
        if not isinstance(mime_type, str) or not mime_type:
            raise ValueError("mime_type must be a non-empty string")
        if not isinstance(pdf_id, str) or not pdf_id:
            raise ValueError("pdf_id must be a non-empty string")
        if not isinstance(page_number, int) or page_number < 1:
            raise ValueError("page_number must be an integer >= 1")

        ext = self._mime_to_ext(mime_type)
        obj_id = uuid.uuid4().hex
        key = f"{BASE_PREFIX}/{pdf_id}/page_{page_number}/{obj_id}.{ext}"

        self._client.put_object(
            Bucket=self._bucket,
            Key=key,
            Body=bytes(image_bytes),
            ContentType=mime_type,
            ACL="private",
        )

        return key

    @staticmethod
    def _mime_to_ext(mime_type: str) -> str:
        mt = mime_type.lower().strip()
        if mt in ("image/jpeg", "image/jpg"):
            return "jpg"
        if mt == "image/png":
            return "png"
        if mt in ("image/webp",):
            return "webp"
        if mt in ("image/tiff", "image/tif"):
            return "tif"
        if mt == "image/bmp":
            return "bmp"
        # Default fallback
        return "bin"
