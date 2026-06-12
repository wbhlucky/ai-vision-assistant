import os

from fastapi import APIRouter, Request


router = APIRouter(tags=["health"])


@router.get("/healthz")
def healthz(request: Request):
    """
    轻量健康检查：返回进程就绪状态 + 关键环境变量是否已配置（不泄露值）。
    """

    required_env = {
        "DASHSCOPE_API_KEY": bool(os.getenv("DASHSCOPE_API_KEY")),
        "AMAP_KEY": bool(os.getenv("AMAP_KEY")),
        "TAVILY_API_KEY": bool(os.getenv("TAVILY_API_KEY")),
    }
    runtime_mode = getattr(request.app.state, "runtime_mode", {})
    startup_status = getattr(request.app.state, "startup_status", {})
    return {
        "ok": True,
        "env": required_env,
        "runtime_mode": runtime_mode,
        "startup_status": startup_status,
    }

