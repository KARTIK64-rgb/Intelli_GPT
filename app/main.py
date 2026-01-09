from typing import Dict

from fastapi import FastAPI
from dotenv import load_dotenv
from pathlib import Path


# Load environment variables explicitly from project root .env and override existing ones
project_root = Path(__file__).resolve().parents[1]
env_path = project_root / ".env"
load_dotenv(dotenv_path=env_path, override=True)


def create_app() -> FastAPI:
    app = FastAPI(title="Multimodal RAG API")

    # Health
    @app.get("/health", response_model=Dict[str, str])
    def health() -> Dict[str, str]:
        return {"status": "ok"}

    # Routers
    from app.api.routes_ingestion import router as ingestion_router
    from app.api.routes_query import router as query_router

    app.include_router(ingestion_router, prefix="/ingest")
    app.include_router(query_router, prefix="/query")

    return app


app = create_app()
