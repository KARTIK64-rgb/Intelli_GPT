from __future__ import annotations

import os
import boto3
from botocore.exceptions import ClientError
from pathlib import Path
from dotenv import load_dotenv


def get_s3_client():
    """Return a configured boto3 S3 client using environment variables.

    Requires AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION, AWS_S3_BUCKET.
    Raises ValueError if any are missing.
    """
    aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
    aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
    aws_region = os.getenv("AWS_REGION")
    bucket = os.getenv("AWS_S3_BUCKET") or os.getenv("S3_BUCKET_NAME")

    if not (aws_access_key_id and aws_secret_access_key and aws_region and bucket):
        # Try loading .env lazily
        load_dotenv(override=False)
        aws_access_key_id = aws_access_key_id or os.getenv("AWS_ACCESS_KEY_ID")
        aws_secret_access_key = aws_secret_access_key or os.getenv("AWS_SECRET_ACCESS_KEY")
        aws_region = aws_region or os.getenv("AWS_REGION")
        bucket = bucket or os.getenv("AWS_S3_BUCKET") or os.getenv("S3_BUCKET_NAME")

    if not (aws_access_key_id and aws_secret_access_key and aws_region and bucket):
        # Explicitly load from project root as last resort
        project_root = Path(__file__).resolve().parents[3]
        env_path = project_root / ".env"
        if env_path.exists():
            load_dotenv(dotenv_path=env_path, override=False)
            aws_access_key_id = aws_access_key_id or os.getenv("AWS_ACCESS_KEY_ID")
            aws_secret_access_key = aws_secret_access_key or os.getenv("AWS_SECRET_ACCESS_KEY")
            aws_region = aws_region or os.getenv("AWS_REGION")
            bucket = bucket or os.getenv("AWS_S3_BUCKET") or os.getenv("S3_BUCKET_NAME")

    if not (aws_access_key_id and aws_secret_access_key and aws_region and bucket):
        raise ValueError("Missing required AWS S3 environment variables.")

    return boto3.client(
        "s3",
        region_name=aws_region,
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
    )


def _resolve_bucket_region_with_fallbacks(bucket: str, default_region: str | None, aws_access_key_id: str, aws_secret_access_key: str) -> str:
    """Resolve the actual region for a bucket using GetBucketLocation, with safe fallbacks.

    If the call fails (permissions/network), fall back to provided default_region when available.
    """
    # 0) Honor explicit env override if provided
    explicit = os.getenv("AWS_S3_BUCKET_REGION") or os.getenv("S3_BUCKET_REGION")
    if explicit:
        return explicit
    # Use a generic client; us-east-1 works for GetBucketLocation
    probe_region = default_region or "us-east-1"
    s3_probe = boto3.client(
        "s3",
        region_name=probe_region,
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
    )
    try:
        resp = s3_probe.get_bucket_location(Bucket=bucket)
        loc = resp.get("LocationConstraint")
        # Map special cases
        if not loc:
            return "us-east-1"
        if loc == "EU":
            return "eu-west-1"
        return str(loc)
    except ClientError:
        # Fall back to the configured region to avoid blocking
        # Try HEAD Bucket to capture x-amz-bucket-region header
        try:
            head = s3_probe.head_bucket(Bucket=bucket)
            # Some SDKs expose region in ResponseMetadata headers
            headers = (head.get("ResponseMetadata") or {}).get("HTTPHeaders") or {}
            hb_region = headers.get("x-amz-bucket-region")
            if hb_region:
                return hb_region
        except ClientError as e:
            resp = getattr(e, "response", {}) or {}
            headers = (resp.get("ResponseMetadata") or {}).get("HTTPHeaders") or {}
            hb_region = headers.get("x-amz-bucket-region")
            if hb_region:
                return hb_region
        # Final fallback
        if default_region:
            return default_region
        return "us-east-1"


def get_s3_client_for_bucket(bucket: str):
    """Return an S3 client configured to the bucket's actual region for correct signing."""
    aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
    aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
    aws_region = os.getenv("AWS_REGION")

    if not (aws_access_key_id and aws_secret_access_key):
        load_dotenv(override=False)
        aws_access_key_id = aws_access_key_id or os.getenv("AWS_ACCESS_KEY_ID")
        aws_secret_access_key = aws_secret_access_key or os.getenv("AWS_SECRET_ACCESS_KEY")
        aws_region = aws_region or os.getenv("AWS_REGION")

    if not (aws_access_key_id and aws_secret_access_key):
        # Last resort load from project root
        project_root = Path(__file__).resolve().parents[3]
        env_path = project_root / ".env"
        if env_path.exists():
            load_dotenv(dotenv_path=env_path, override=False)
            aws_access_key_id = aws_access_key_id or os.getenv("AWS_ACCESS_KEY_ID")
            aws_secret_access_key = aws_secret_access_key or os.getenv("AWS_SECRET_ACCESS_KEY")
            aws_region = aws_region or os.getenv("AWS_REGION")

    if not (aws_access_key_id and aws_secret_access_key):
        raise ValueError("Missing AWS credentials for S3 client")

    bucket_region = _resolve_bucket_region_with_fallbacks(
        bucket=bucket,
        default_region=aws_region,
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
    )

    return boto3.client(
        "s3",
        region_name=bucket_region,
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
    )
