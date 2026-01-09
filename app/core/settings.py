from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables.

    Uses Pydantic v2 settings to load from `.env` and the process environment.
    """

    # AWS
    aws_access_key_id: str = Field(..., alias="AWS_ACCESS_KEY_ID")
    aws_secret_access_key: str = Field(..., alias="AWS_SECRET_ACCESS_KEY")
    aws_region: str = Field(..., alias="AWS_REGION")
    s3_bucket_name: str = Field(..., alias="S3_BUCKET_NAME")

    # Qdrant
    qdrant_url: str = Field(..., alias="QDRANT_URL")
    qdrant_api_key: str = Field(..., alias="QDRANT_API_KEY")
    qdrant_text_collection: str = Field(
        default="text_chunks", alias="QDRANT_TEXT_COLLECTION"
    )
    qdrant_image_collection: str = Field(
        default="images", alias="QDRANT_IMAGE_COLLECTION"
    )

    # Embeddings / LLM
    google_api_key: str = Field(..., alias="GOOGLE_API_KEY")
    google_text_embedding_model: str = Field(
        ..., alias="GOOGLE_TEXT_EMBEDDING_MODEL"
    )
    google_image_embedding_model: str = Field(
        ..., alias="GOOGLE_IMAGE_EMBEDDING_MODEL"
    )
    gemini_model_name: str = Field(..., alias="GEMINI_MODEL_NAME")

    # General
    env: str = Field(default="development", alias="ENV")

    # Settings configuration
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        populate_by_name=True,
        extra="ignore",
    )


def get_settings() -> Settings:
    """Return application settings instance.

    Creates a new instance reading from the environment and `.env`.
    """
    return Settings()
