from __future__ import annotations

from typing import Optional

import boto3
from botocore.exceptions import BotoCoreError, ClientError

from app.core.settings import get_settings


_s3_client: Optional[boto3.session.Session] = None
_bucket_name: Optional[str] = None


def _init_client() -> boto3.client:
    """Initialize and cache the S3 client and bucket name.

    Returns:
        boto3.client: Configured S3 client.
    """
    global _s3_client, _bucket_name
    if _s3_client is not None and _bucket_name is not None:
        return _s3_client

    settings = get_settings()
    try:
        _s3_client = boto3.client(
            "s3",
            region_name=settings.aws_region,
            aws_access_key_id=settings.aws_access_key_id,
            aws_secret_access_key=settings.aws_secret_access_key,
        )
        _bucket_name = settings.s3_bucket_name
        return _s3_client
    except (BotoCoreError, ClientError) as e:
        raise RuntimeError(f"Failed to initialize S3 client: {e}") from e


def upload_bytes(data: bytes, key: str, content_type: str) -> None:
    """Upload raw bytes to S3 with a private ACL.

    Args:
        data: The binary content to upload.
        key: The S3 object key.
        content_type: The MIME type of the object.

    Raises:
        RuntimeError: If the upload fails due to AWS errors or misconfiguration.
    """
    if not isinstance(data, (bytes, bytearray)):
        raise TypeError("data must be bytes or bytearray")
    if not key:
        raise ValueError("key must be a non-empty string")
    if not content_type:
        raise ValueError("content_type must be provided")

    client = _init_client()
    assert _bucket_name is not None

    try:
        client.put_object(
            Bucket=_bucket_name,
            Key=key,
            Body=data,
            ContentType=content_type,
            ACL="private",
        )
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code", "UnknownError")
        message = e.response.get("Error", {}).get("Message", str(e))
        raise RuntimeError(
            f"S3 upload failed (code={code}) for key '{key}': {message}"
        ) from e
    except BotoCoreError as e:
        raise RuntimeError(f"S3 upload failed for key '{key}': {e}") from e


def generate_presigned_url(key: str, expires_in: int = 900) -> str:
    """Generate a time-limited pre-signed URL for a private S3 object.

    Args:
        key: The S3 object key.
        expires_in: Expiry in seconds (default 900).

    Returns:
        str: The pre-signed URL.

    Raises:
        RuntimeError: If URL generation fails.
    """
    if not key:
        raise ValueError("key must be a non-empty string")
    if expires_in <= 0:
        raise ValueError("expires_in must be a positive integer")

    client = _init_client()
    assert _bucket_name is not None

    try:
        url = client.generate_presigned_url(
            ClientMethod="get_object",
            Params={"Bucket": _bucket_name, "Key": key},
            ExpiresIn=expires_in,
        )
        return url
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code", "UnknownError")
        message = e.response.get("Error", {}).get("Message", str(e))
        raise RuntimeError(
            f"Presigned URL generation failed (code={code}) for key '{key}': {message}"
        ) from e
    except BotoCoreError as e:
        raise RuntimeError(
            f"Presigned URL generation failed for key '{key}': {e}"
        ) from e
