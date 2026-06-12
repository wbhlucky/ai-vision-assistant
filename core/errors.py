from __future__ import annotations

from functools import partial

from fastapi import HTTPException


class AppError(HTTPException):
    """Application-level error with a machine-readable `code`."""

    def __init__(self, *, status_code: int, code: str, message: str) -> None:
        super().__init__(status_code=status_code, detail={"code": code, "message": message})


bad_request = partial(AppError, status_code=400)
unauthorized = partial(AppError, status_code=401)
internal_error = partial(AppError, status_code=500)
