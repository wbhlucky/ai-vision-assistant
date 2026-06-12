from fastapi import Header

from backend.core.errors import unauthorized
from backend.core.settings import settings


def require_api_key(x_api_key: str | None = Header(default=None, alias="X-API-Key")) -> None:
    """
    简历演示级鉴权：如果未配置 API_KEY，则不强制；配置后要求请求携带 X-API-Key。
    """
    if not settings.api_key:
        return
    if not x_api_key or x_api_key != settings.api_key:
        raise unauthorized(code="UNAUTHORIZED", message="Invalid or missing X-API-Key")

