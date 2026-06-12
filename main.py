from __future__ import annotations

import os as _os
from contextlib import asynccontextmanager

from dotenv import load_dotenv

# ⚠️ 必须在 import backend.* 之前加载 .env，因为 settings 在 import 时就读取环境变量
_env_path = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".env")
load_dotenv(_env_path)

from fastapi import Depends, FastAPI, Request
from fastapi.responses import JSONResponse

from backend.api.chat import router as chat_router
from backend.api.health import router as health_router
from backend.api.knowledge import router as knowledge_router
from backend.api.mcp import router as mcp_router
from backend.api.rag import router as rag_router
from backend.api.vision import router as vision_router
from backend.core.auth import require_api_key
from backend.core.errors import AppError
from backend.core.logging import configure_logging, new_request_id, reset_request_id, set_request_id
from backend.core.metrics import metrics_middleware, render_metrics
from backend.core.rate_limit import limiter, rate_limit_exceeded_handler
from backend.core.runtime_mode import apply_runtime_mode
from backend.core.settings import settings
from backend.core.startup_checks import run_startup_self_check
from slowapi.errors import RateLimitExceeded


def create_app() -> FastAPI:
    mode_state = apply_runtime_mode()
    configure_logging(settings.log_level)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.runtime_mode = mode_state
        app.state.limiter = limiter
        if settings.startup_self_check:
            app.state.startup_status = run_startup_self_check()
        else:
            app.state.startup_status = {"enabled": False}
        yield

    app = FastAPI(
        title="Agent Backend",
        version="0.1.0",
        description="FastAPI backend for agent + agentic RAG, with SSE streaming.",
        lifespan=lifespan,
    )

    app.add_exception_handler(RateLimitExceeded, rate_limit_exceeded_handler)
    app.middleware("http")(metrics_middleware)

    @app.middleware("http")
    async def request_id_middleware(request: Request, call_next):
        rid = request.headers.get("X-Request-Id") or new_request_id()
        token = set_request_id(rid)
        try:
            resp = await call_next(request)
            resp.headers["X-Request-Id"] = rid
            return resp
        finally:
            reset_request_id(token)

    @app.exception_handler(AppError)
    async def app_error_handler(_: Request, exc: AppError):
        return JSONResponse(status_code=exc.status_code, content=exc.detail)

    @app.get("/", tags=["health"])
    def root():
        return {
            "message": "Agent Backend is running",
            "docs": "/docs",
            "health": "/healthz",
        }

    @app.get("/metrics", tags=["observability"], dependencies=[Depends(require_api_key)])
    def metrics():
        return render_metrics()

    app.include_router(health_router)
    app.include_router(chat_router)
    app.include_router(rag_router)
    app.include_router(knowledge_router)
    app.include_router(mcp_router)
    app.include_router(vision_router)
    return app


app = create_app()

