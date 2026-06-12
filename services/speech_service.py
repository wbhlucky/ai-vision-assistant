from __future__ import annotations

import base64
import math
import struct
import time
from functools import lru_cache

from backend.core.settings import settings


@lru_cache(maxsize=1)
def _get_stt_client():
    """Lazy-import DashScope STT client. Returns None if unavailable."""
    try:
        import dashscope  # type: ignore[import-untyped]
        return dashscope
    except ImportError:
        return None


# ── 音频解码 ────────────────────────────────────────────────

def _decode_audio_bytes(audio_base64: str) -> bytes:
    """解码 base64 音频，自动剥离 WAV 头（若存在），返回原始 PCM16 字节。"""
    raw = base64.b64decode(audio_base64)
    # 检测 RIFF/WAV 头，定位 "data" chunk
    if raw[:4] == b"RIFF" and len(raw) > 44:
        data_idx = raw.find(b"data")
        if data_idx != -1:
            # "data" 标记(4字节) + 长度(4字节) 之后是原始 PCM
            return raw[data_idx + 8:]
    return raw


# ── VAD（能量端点检测）─────────────────────────────────────

def _compute_rms(audio_bytes: bytes) -> float:
    """计算 16-bit PCM 音频的 RMS 能量值。范围 0.0 ~ 32768.0。"""
    if len(audio_bytes) < 2:
        return 0.0
    count = len(audio_bytes) // 2
    try:
        samples = struct.unpack(f"<{count}h", audio_bytes[: count * 2])
    except struct.error:
        return 0.0
    if count == 0:
        return 0.0
    sum_sq = sum(s * s for s in samples)
    return math.sqrt(sum_sq / count)


def detect_speech(audio_base64: str) -> bool:
    """基于 RMS 能量阈值的语音活动检测 (VAD)。

    当 VAD 被配置禁用时始终返回 True（视为始终有语音）。
    """
    if not settings.vad_enabled:
        return True
    raw = _decode_audio_bytes(audio_base64)
    rms = _compute_rms(raw)
    return rms >= settings.vad_energy_threshold


# ── 语音转录 ────────────────────────────────────────────────

def transcribe(audio_base64: str, sample_rate: int = 16000) -> dict:
    """将 base64 音频转录为文本。

    优先使用 DashScope Paraformer；SDK 不可用时返回占位文本。

    Args:
        audio_base64: Base64 编码的 PCM16/WAV 音频。
        sample_rate: 音频采样率 (Hz)。

    Returns:
        {"text": str, "is_speech": bool, "confidence": float, "latency_ms": float}
    """
    t0 = time.perf_counter()

    # VAD 快速判断
    is_speech = detect_speech(audio_base64)
    if not is_speech:
        elapsed = (time.perf_counter() - t0) * 1000
        return {
            "text": "",
            "is_speech": False,
            "confidence": 0.0,
            "latency_ms": round(elapsed, 2),
        }

    client = _get_stt_client()
    if client is None:
        elapsed = (time.perf_counter() - t0) * 1000
        return {
            "text": "[STT unavailable — dashscope not installed]",
            "is_speech": True,
            "confidence": 0.0,
            "latency_ms": round(elapsed, 2),
        }

    # 解码原始 PCM 字节
    try:
        raw_pcm = _decode_audio_bytes(audio_base64)
    except Exception:
        elapsed = (time.perf_counter() - t0) * 1000
        return {
            "text": "[audio decode error]",
            "is_speech": True,
            "confidence": 0.0,
            "latency_ms": round(elapsed, 2),
        }

    # 调用 DashScope Paraformer（通过 REST API，避免 SDK callback 兼容问题）
    try:
        import urllib.request
        import json as _json

        url = "https://dashscope.aliyuncs.com/api/v1/services/audio/asr/transcription"
        headers = {
            "Authorization": f"Bearer {settings.dashscope_api_key}",
            "Content-Type": "application/octet-stream",
            "X-DashScope-Model": settings.speech_model,
            "X-DashScope-SampleRate": str(sample_rate),
            "X-DashScope-Format": "pcm",
        }
        req = urllib.request.Request(url, data=raw_pcm, headers=headers, method="POST")
        resp = urllib.request.urlopen(req, timeout=settings.external_request_timeout_sec)
        body = _json.loads(resp.read())

        text = ""
        confidence = 0.0
        if body.get("output") and body["output"].get("sentence"):
            text = body["output"]["sentence"].get("text", "")
            confidence = body["output"]["sentence"].get("confidence", 0.0)

        if not text:
            text = "[silence or unrecognized]"

        elapsed = (time.perf_counter() - t0) * 1000
        return {
            "text": text,
            "is_speech": True,
            "confidence": round(confidence, 4),
            "latency_ms": round(elapsed, 2),
        }

    except Exception as exc:
        elapsed = (time.perf_counter() - t0) * 1000
        return {
            "text": f"[STT error: {exc}]",
            "is_speech": True,
            "confidence": 0.0,
            "latency_ms": round(elapsed, 2),
        }
