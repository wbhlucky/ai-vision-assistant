"""OCR 文字提取服务 — 支持 DeepSeek / DashScope 双 Provider。"""

from __future__ import annotations

import base64
import time

from backend.core.settings import settings
from backend.services.vision_service import _call_deepseek, _call_dashscope, compress_frame


def _ocr_call(image_base64: str, prompt: str) -> str:
    """统一 OCR 调用入口，根据 vision_provider 路由。"""
    compressed, _ = compress_frame(image_base64)

    if settings.vision_provider in ("deepseek", "openai"):
        return _call_deepseek(compressed, prompt)
    else:
        return _call_dashscope(compressed, prompt)


# ── OCR 文字提取 ────────────────────────────────────────────

def extract_text(image_base64: str) -> dict:
    """从图像中提取文字（OCR）。"""
    t0 = time.perf_counter()

    try:
        text = _ocr_call(
            image_base64,
            "请提取并输出这张图片中的所有文字内容。"
            "按原文顺序输出，保留段落结构。"
            "如果图片中没有文字，请回复「[无文字]」。"
            "只输出文字内容，不要加任何解释。",
        )
        elapsed = (time.perf_counter() - t0) * 1000
        return {
            "text": text.strip(),
            "model": settings.vision_model,
            "latency_ms": round(elapsed, 2),
        }
    except Exception as exc:
        elapsed = (time.perf_counter() - t0) * 1000
        return {
            "text": f"[OCR error: {exc}]",
            "model": f"{settings.vision_model} (error)",
            "latency_ms": round(elapsed, 2),
        }


# ── OCR + 总结 ──────────────────────────────────────────────

def ocr_and_summarize(image_base64: str) -> dict:
    """OCR 提取 → Agent 总结。"""
    t0 = time.perf_counter()
    ocr_result = extract_text(image_base64)
    ocr_text = ocr_result["text"]

    if ocr_text.startswith("[OCR") or ocr_text == "[无文字]":
        elapsed = (time.perf_counter() - t0) * 1000
        return {"ocr_text": ocr_text, "summary": "未能提取到有效文字", "latency_ms": round(elapsed, 2)}

    try:
        from backend.services.agent_service import answer
        summary = answer(
            f"以下是从图片中 OCR 提取的文字内容:\n\n{ocr_text}\n\n"
            f"请用中文简要总结这段文字的核心内容（3-5句话）。"
        )
    except Exception:
        summary = "[Agent 总结失败]"

    elapsed = (time.perf_counter() - t0) * 1000
    return {"ocr_text": ocr_text, "summary": summary, "latency_ms": round(elapsed, 2)}


# ── OCR + 翻译 ──────────────────────────────────────────────

def ocr_and_translate(image_base64: str, target_lang: str = "中文") -> dict:
    """OCR 提取 → 翻译。"""
    t0 = time.perf_counter()
    ocr_result = extract_text(image_base64)
    ocr_text = ocr_result["text"]

    if ocr_text.startswith("[OCR") or ocr_text == "[无文字]":
        elapsed = (time.perf_counter() - t0) * 1000
        return {"ocr_text": ocr_text, "translated": "无可翻译文字", "latency_ms": round(elapsed, 2)}

    try:
        from backend.services.agent_service import answer
        translated = answer(f"请将以下文字翻译为{target_lang}:\n\n{ocr_text}")
    except Exception:
        translated = "[翻译失败]"

    elapsed = (time.perf_counter() - t0) * 1000
    return {"ocr_text": ocr_text, "translated": translated, "latency_ms": round(elapsed, 2)}
