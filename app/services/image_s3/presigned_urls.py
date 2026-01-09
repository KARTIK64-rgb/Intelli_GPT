from __future__ import annotations

import os
from app.services.image_s3.s3_client import get_s3_client, get_s3_client_for_bucket


def generate_presigned_url(s3_key: str, ttl_seconds: int) -> str:
    """Generate a read-only pre-signed URL for a private S3 image.

    Args:
        s3_key: The S3 object key.
        ttl_seconds: Time-to-live in seconds for the URL.

    Returns:
        str: The pre-signed URL.

    Raises:
        botocore.exceptions.ClientError / BotoCoreError: Propagated from boto3.
        ValueError: For invalid inputs.
    """
    if not isinstance(s3_key, str) or not s3_key:
        raise ValueError("s3_key must be a non-empty string")
    env_ttl = os.getenv("AWS_S3_PRESIGNED_URL_TTL_SECONDS")
    if ttl_seconds is None:
        if not env_ttl:
            raise ValueError("AWS_S3_PRESIGNED_URL_TTL_SECONDS is missing")
        ttl_seconds = int(env_ttl)
    else:
        if ttl_seconds <= 0:
            if env_ttl:
                ttl_seconds = int(env_ttl)
            else:
                raise ValueError("ttl_seconds must be a positive integer")

    bucket = os.getenv("AWS_S3_BUCKET") or os.getenv("S3_BUCKET_NAME")
    if not bucket:
        raise ValueError("AWS_S3_BUCKET is missing")

    # Use a client configured for the bucket's actual region to avoid signing errors
    try:
        client = get_s3_client_for_bucket(bucket)
    except Exception:
        # Fallback to generic client if region resolution fails
        client = get_s3_client()
    url = client.generate_presigned_url(
        ClientMethod="get_object",
        Params={"Bucket": bucket, "Key": s3_key},
        ExpiresIn=ttl_seconds,
    )
    return url
