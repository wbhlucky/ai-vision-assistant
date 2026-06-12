from __future__ import annotations

from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from starlette.responses import JSONResponse


limiter = Limiter(key_func=get_remote_address, default_limits=["60/minute"])


def rate_limit_exceeded_handler(_request: object, exc: RateLimitExceeded) -> JSONResponse:
    return JSONResponse(
        status_code=429,
        content={"code": "RATE_LIMITED", "message": "Too many requests"},
        headers=getattr(exc, "headers", None) or {},
    )
