from __future__ import annotations

from uuid import uuid4

from fastapi import APIRouter, UploadFile, File, HTTPException

from app.pipelines.ingestion_pipeline import IngestionPipeline


router = APIRouter()


@router.post("/pdf")
async def ingest_pdf(file: UploadFile = File(...)) -> dict:
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Content-Type must be application/pdf")

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty PDF file")

    pdf_id = uuid4().hex
    try:
        IngestionPipeline().ingest_pdf(pdf_bytes=data, pdf_id=pdf_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {"pdf_id": pdf_id, "status": "ingested"}
