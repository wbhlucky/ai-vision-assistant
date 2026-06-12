from __future__ import annotations

import os
from typing import Any

from backend.core.settings import settings
from utils.logger_handler import logger


def _safe_vector_status() -> dict[str, Any]:
    try:
        from rag.vector_store import VectorStoreService

        vs = VectorStoreService()
        bm25_exists = os.path.exists(vs.bm25_path)

        return {
            "vector_store_ready": True,
            "bm25_cache_path": vs.bm25_path,
            "bm25_cache_exists": bm25_exists,
            "hf_rerank_enabled": vs.enable_hf_rerank,
        }
    except Exception as e:
        return {
            "vector_store_ready": False,
            "error": str(e),
        }


def _safe_multimodal_status() -> dict[str, Any]:
    """检测多模态服务的可用性。"""
    try:
        # 检查是否有可用的视觉 Provider
        if settings.vision_provider in ("deepseek", "openai"):
            try:
                import openai  # noqa: F401
                vision_ok = True
            except ImportError:
                vision_ok = False
        else:
            try:
                import dashscope  # noqa: F401
                vision_ok = True
            except ImportError:
                vision_ok = False

        # 检查 STT
        try:
            import dashscope  # noqa: F401
            stt_ok = True
        except ImportError:
            stt_ok = False

        return {
            "vision_provider": settings.vision_provider,
            "vision_api_available": vision_ok,
            "stt_api_available": stt_ok,
        }
    except Exception as e:
        return {"error": str(e)}


def run_startup_self_check() -> dict[str, Any]:
    """
    启动自检：只做轻量探测，不做重型入库。
    """
    status = {
        "run_mode": settings.run_mode,
        "api_key_configured": bool(settings.api_key),
        "external_request_timeout_sec": settings.external_request_timeout_sec,
        "env": {
            "DASHSCOPE_API_KEY": bool(os.getenv("DASHSCOPE_API_KEY")),
            "AMAP_KEY": bool(os.getenv("AMAP_KEY")),
            "TAVILY_API_KEY": bool(os.getenv("TAVILY_API_KEY")),
        },
        "vector": _safe_vector_status(),
        "multimodal": _safe_multimodal_status(),
    }

    logger.info(f"[startup_self_check] {status}")
    return status
