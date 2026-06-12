from __future__ import annotations

from functools import lru_cache

from fastapi import APIRouter, Depends, Request

from backend.core.auth import require_api_key
from backend.core.rate_limit import limiter
from backend.schemas.rag import RagSummarizeRequest, RagSummarizeResponse


router = APIRouter(prefix="/v1/rag", tags=["rag"])


@lru_cache(maxsize=1)
def _get_rag():
    from rag.rag_service import RagSummarizeService

    return RagSummarizeService()


@router.post(
    "/summarize",
    response_model=RagSummarizeResponse,
    dependencies=[Depends(require_api_key)],
)
@limiter.limit("30/minute")
def rag_summarize(request: Request, req: RagSummarizeRequest):
    return RagSummarizeResponse(answer=_get_rag().rag_summarize(req.query))
