from __future__ import annotations

from functools import lru_cache

from fastapi import APIRouter, Depends, Request

from backend.core.auth import require_api_key
from backend.core.rate_limit import limiter
from backend.schemas.knowledge import KnowledgeIngestResponse


router = APIRouter(prefix="/v1/knowledge", tags=["knowledge"])


@lru_cache(maxsize=1)
def _get_vector_store():
    from rag.vector_store import VectorStoreService

    return VectorStoreService()


@router.post(
    "/ingest",
    response_model=KnowledgeIngestResponse,
    dependencies=[Depends(require_api_key)],
)
@limiter.limit("5/minute")
def knowledge_ingest(request: Request):
    return _get_vector_store().load_document()
