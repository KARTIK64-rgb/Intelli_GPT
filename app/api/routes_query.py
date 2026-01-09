from __future__ import annotations

from typing import Dict

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.pipelines.query_pipeline import QueryPipeline


class QueryRequest(BaseModel):
    question: str


router = APIRouter()


@router.post("/")
def query(req: QueryRequest) -> Dict:
    question = (req.question or "").strip()
    if not question:
        raise HTTPException(status_code=400, detail="question must be a non-empty string")
    try:
        return QueryPipeline().answer_question(question)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
