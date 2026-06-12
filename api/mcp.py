from __future__ import annotations

from functools import wraps
from typing import Any, Callable

from fastapi import APIRouter

from backend.services import mcp_bridge


router = APIRouter(prefix="/v1/mcp", tags=["mcp"])


def _endpoint(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Wrap a mcp_bridge function into a route handler that returns {result: ...}."""

    @wraps(fn)
    def _handler(*args: Any, **kwargs: Any) -> dict[str, Any]:
        return {"result": fn(*args, **kwargs)}

    # Preserve signature for OpenAPI generation
    _handler.__annotations__ = fn.__annotations__
    return _handler


# ── health (special — not a bridge dispatch) ──────────────────────────────────

@router.get("/health")
def mcp_health():
    return {
        "ok": True,
        "mode": mcp_bridge.get_mcp_mode(),
        "base_url": mcp_bridge.get_mcp_base_url(),
    }


# ── MCP tool passthrough routes ──────────────────────────────────────────────

@router.post("/rag_summarize")
def rag_summarize(query: str):
    return {"result": mcp_bridge.rag_summarize(query)}


@router.post("/get_user_location")
def get_user_location():
    return {"result": mcp_bridge.get_user_location()}


@router.post("/get_weather")
def get_weather(region: str):
    return {"result": mcp_bridge.get_weather(region)}


@router.post("/get_current_time_all")
def get_current_time_all():
    return {"result": mcp_bridge.get_current_time_all()}


@router.post("/calculate_relative_date")
def calculate_relative_date(days_offset: int):
    return {"result": mcp_bridge.calculate_relative_date(days_offset)}


@router.post("/get_user_id")
def get_user_id():
    return {"result": mcp_bridge.get_user_id()}


@router.post("/fetch_external_data")
def fetch_external_data(user_id: str, month: str):
    return {"result": mcp_bridge.fetch_external_data(user_id, month)}


@router.post("/knowledge_ingest")
def knowledge_ingest():
    return {"result": mcp_bridge.knowledge_ingest()}
