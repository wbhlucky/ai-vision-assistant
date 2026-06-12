from __future__ import annotations

import base64
import time
from functools import lru_cache

from backend.core.settings import settings


# ── 帧压缩 ──────────────────────────────────────────────────

def compress_frame(image_base64: str) -> tuple[str, float]:
    """解码 base64 图像，缩放到设定分辨率，以 JPEG 重新编码。"""
    raw = base64.b64decode(image_base64)
    original_size = len(raw)

    try:
        from io import BytesIO
        from PIL import Image
    except ImportError:
        return image_base64, 1.0

    img = Image.open(BytesIO(raw))
    w, h = img.size

    if w > settings.vision_max_width or h > settings.vision_max_height:
        img.thumbnail((settings.vision_max_width, settings.vision_max_height), Image.LANCZOS)

    buf = BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=settings.vision_jpeg_quality)
    compressed = base64.b64encode(buf.getvalue()).decode("utf-8")
    ratio = len(buf.getvalue()) / max(original_size, 1)
    return compressed, ratio


# ── 启发式降级描述 ─────────────────────────────────────────

def _heuristic_describe(image_base64: str) -> str:
    """轻量级降级：仅报告分辨率和主色调。"""
    try:
        from io import BytesIO
        from PIL import Image

        raw = base64.b64decode(image_base64)
        img = Image.open(BytesIO(raw))
        img_small = img.resize((1, 1))
        r, g, b = img_small.getpixel((0, 0))  # type: ignore[arg-type]
        return (
            f"[heuristic] 图像分辨率 {img.width}×{img.height}, "
            f"主色调 RGB({r},{g},{b})"
        )
    except Exception:
        return "[heuristic] 无法解码图像"


# ── Provider: DeepSeek / OpenAI 兼容 ───────────────────────

def _call_deepseek(image_base64: str, prompt: str) -> str:
    """通过 OpenAI 兼容接口调用 DeepSeek 等多模态模型。"""
    try:
        import openai  # type: ignore[import-untyped]
    except ImportError:
        raise RuntimeError("openai 包未安装，请 pip install openai")

    client = openai.OpenAI(
        api_key=settings.dashscope_api_key,
        base_url=settings.deepseek_base_url,
    )

    resp = client.chat.completions.create(
        model=settings.vision_model,
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"},
                },
                {"type": "text", "text": prompt},
            ],
        }],
        max_tokens=512,
        temperature=0.7,
        timeout=settings.external_request_timeout_sec,
    )

    return resp.choices[0].message.content or "[deepseek] 空回复"


# ── Provider: DashScope ─────────────────────────────────────

@lru_cache(maxsize=1)
def _get_dashscope_client():
    try:
        import dashscope  # type: ignore[import-untyped]
        return dashscope
    except ImportError:
        return None


def _call_dashscope(image_base64: str, prompt: str) -> str:
    """通过 DashScope MultiModalConversation 调用 Qwen-VL。"""
    client = _get_dashscope_client()
    if client is None:
        raise RuntimeError("dashscope 包未安装，请 pip install dashscope")

    from dashscope import MultiModalConversation  # type: ignore[import-untyped]

    messages = [{
        "role": "user",
        "content": [
            {"image": f"data:image/jpeg;base64,{image_base64}"},
            {"text": prompt},
        ],
    }]
    resp = MultiModalConversation.call(
        model=settings.vision_model,
        messages=messages,
        api_key=settings.dashscope_api_key or None,
    )

    if resp and resp.status_code == 200 and resp.output:
        for choice in resp.output.get("choices", []):
            for item in choice.get("message", {}).get("content", []):
                if isinstance(item, dict) and "text" in item:
                    return item["text"]

    msg = getattr(resp, "message", "") if resp else ""
    code = getattr(resp, "code", "Unknown") if resp else "Unknown"
    raise RuntimeError(f"DashScope 返回 {resp.status_code if resp else 'None'}: {msg} (code={code})")


# ── 统一视觉分析入口 ──────────────────────────────────────

def analyze_frame(image_base64: str, prompt: str | None = None) -> dict:
    """对单帧图像进行内容分析。

    根据 settings.vision_provider 自动选择调用链路：
    - "deepseek" → DeepSeek / OpenAI 兼容接口
    - "dashscope" → 阿里云百炼 MultiModalConversation
    失败时自动降级为 Pillow 启发式描述。
    """
    t0 = time.perf_counter()
    compressed, ratio = compress_frame(image_base64)
    default_prompt = "请用中文简要描述这张图片的内容，包括画面中的主要物体、人物和场景。"

    try:
        if settings.vision_provider in ("deepseek", "openai"):
            description = _call_deepseek(compressed, prompt or default_prompt)
            model = settings.vision_model
        else:
            description = _call_dashscope(compressed, prompt or default_prompt)
            model = settings.vision_model

        elapsed = (time.perf_counter() - t0) * 1000
        return {
            "description": description,
            "model": model,
            "latency_ms": round(elapsed, 2),
        }

    except Exception as exc:
        # 降级：API 不可用时走启发式
        elapsed = (time.perf_counter() - t0) * 1000
        error_msg = str(exc)[:200]
        return {
            "description": f"[fallback] {_heuristic_describe(image_base64)}  (API 错误: {error_msg})",
            "model": f"{settings.vision_model} (fallback)",
            "latency_ms": round(elapsed, 2),
        }
