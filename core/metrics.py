from __future__ import annotations

import time
from typing import Callable

from fastapi import Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest


http_requests_total = Counter(
    "http_requests_total",
    "Total HTTP requests",
    ["method", "path", "status"],
)

http_request_duration_seconds = Histogram(
    "http_request_duration_seconds",
    "HTTP request duration in seconds",
    ["method", "path"],
)

# ── 多模态 / 视觉对话 ──────────────────────────────────────

vision_frames_processed = Counter(
    "vision_frames_processed_total",
    "Vision frames analyzed",
    ["model"],
)

vision_api_errors = Counter(
    "vision_api_errors_total",
    "Vision API call failures",
    ["error_type"],
)

speech_chunks_processed = Counter(
    "speech_chunks_processed_total",
    "Audio chunks transcribed (post-VAD)",
)

vad_silence_skipped = Counter(
    "vad_silence_skipped_total",
    "Audio chunks skipped by VAD (silence detected)",
)

websocket_connections = Counter(
    "websocket_connections_total",
    "WebSocket connections",
    ["status"],  # "connected" | "disconnected" | "auth_failed" | "error"
)


async def metrics_middleware(request: Request, call_next: Callable[[Request], Response]):
    start = time.perf_counter()
    response = await call_next(request)
    elapsed = time.perf_counter() - start

    # 注意：使用原始 path（不展开 query），可避免高基数
    path = request.url.path
    http_requests_total.labels(request.method, path, str(response.status_code)).inc()
    http_request_duration_seconds.labels(request.method, path).observe(elapsed)
    return response


def render_metrics() -> Response:
    data = generate_latest()
    return Response(content=data, media_type=CONTENT_TYPE_LATEST)

