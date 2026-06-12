from __future__ import annotations

import asyncio
from functools import lru_cache
from typing import Any, Callable

from backend.core.settings import settings


class MCPBridgeError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# mode / base URL helpers
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def get_mcp_mode() -> str:
    return settings.mcp_mode


@lru_cache(maxsize=1)
def get_mcp_base_url() -> str:
    return settings.mcp_server_base_url.rstrip("/")


@lru_cache(maxsize=1)
def _local_tools():
    from mcp_server import tools

    return tools


# ---------------------------------------------------------------------------
# async / sync MCP call helpers
# ---------------------------------------------------------------------------

async def _call_mcp_tool(tool_name: str, arguments: dict[str, Any] | None = None) -> Any:
    try:
        from fastmcp import Client
    except Exception as exc:  # pragma: no cover - import guard
        raise MCPBridgeError(f"Cannot import fastmcp client: {exc}") from exc

    args = arguments or {}
    try:
        async with Client(get_mcp_base_url()) as client:
            result = await client.call_tool(tool_name, args)
    except Exception as exc:
        raise MCPBridgeError(f"MCP HTTP call failed: {exc}") from exc

    return result.data if hasattr(result, "data") else result


def _call_mcp_tool_sync(tool_name: str, arguments: dict[str, Any] | None = None) -> Any:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(_call_mcp_tool(tool_name, arguments))

    raise MCPBridgeError(
        "An event loop is already running; cannot call MCP synchronously. "
        "Use an async caller or run in a separate thread."
    )


# ---------------------------------------------------------------------------
# tool registry — each entry maps {tool_name} → (local_callable, param_names)
# ---------------------------------------------------------------------------

def _wrap(tool_name: str, local_fn: Callable[..., Any], *param_names: str):
    """Create a dispatch function that routes via HTTP or local depending on mode."""

    def _fn(*args: Any, **kwargs: Any) -> str:
        if get_mcp_mode() == "http":
            call_kwargs = kwargs if kwargs else dict(zip(param_names, args))
            return str(_call_mcp_tool_sync(tool_name, call_kwargs))
        return str(local_fn(*args, **kwargs))

    _fn.__name__ = tool_name
    _fn.__qualname__ = tool_name
    _fn.__doc__ = f"Dispatch for MCP tool '{tool_name}' (mode={get_mcp_mode()})."
    return _fn


_tools = _local_tools()

rag_summarize         = _wrap("rag_summarize",         _tools.tool_rag_summarize,         "query")
get_user_location     = _wrap("get_user_location",     _tools.tool_get_user_location)
get_weather           = _wrap("get_weather",           _tools.tool_get_weather,           "region")
get_current_time_all  = _wrap("get_current_time_all",  _tools.tool_get_current_time_all)
calculate_relative_date = _wrap("calculate_relative_date", _tools.tool_calculate_relative_date, "days_offset")
get_user_id           = _wrap("get_user_id",           _tools.tool_get_user_id)
fetch_external_data   = _wrap("fetch_external_data",   _tools.tool_fetch_external_data,   "user_id", "month")
knowledge_ingest      = _wrap("knowledge_ingest",      _tools.tool_knowledge_ingest)
