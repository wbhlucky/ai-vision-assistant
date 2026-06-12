from __future__ import annotations

from pydantic import BaseModel, Field


# ── HTTP 单帧分析 ──────────────────────────────────────────

class VisionFrameRequest(BaseModel):
    """单帧图像分析请求（HTTP）。"""
    image_base64: str = Field(min_length=1, description="Base64 编码的 JPEG/PNG 图像")
    prompt: str | None = Field(default=None, description="可选引导提示词，用于指导视觉模型关注特定内容")


class VisionFrameResponse(BaseModel):
    description: str
    model: str
    latency_ms: float


# ── HTTP 语音转录 ──────────────────────────────────────────

class AudioTranscribeRequest(BaseModel):
    """语音转录请求（HTTP）。"""
    audio_base64: str = Field(min_length=1, description="Base64 编码的 PCM16/WAV 音频数据")
    sample_rate: int = Field(default=16000, ge=8000, le=48000, description="音频采样率 (Hz)")


class AudioTranscribeResponse(BaseModel):
    text: str
    is_speech: bool
    confidence: float
    latency_ms: float


# ── HTTP OCR ────────────────────────────────────────────────

class OcrRequest(BaseModel):
    """OCR 文字提取请求（HTTP）。"""
    image_base64: str = Field(min_length=1, description="Base64 编码的 JPEG/PNG 图像")
    mode: str = Field(default="extract", description="'extract' | 'summarize' | 'translate'")
    target_lang: str = Field(default="中文", description="翻译目标语言，仅 mode=translate 时有效")


class OcrResponse(BaseModel):
    text: str
    summary: str = ""
    translated: str = ""
    model: str
    latency_ms: float


# ── HTTP 场景记忆查询 ───────────────────────────────────────

class SceneMemoryQuery(BaseModel):
    """场景记忆查询请求（HTTP）。"""
    question: str = Field(default="最近有什么变化？", description="自然语言问题")


class SceneMemoryQueryResponse(BaseModel):
    answer: str
    event_count: int
    snapshot_count: int


# ── WebSocket 实时对话 ─────────────────────────────────────

class DialogueMessageIn(BaseModel):
    """客户端 → 服务端 WebSocket 消息。"""
    type: str = Field(
        description="消息类型: 'frame' | 'audio' | 'ocr' | 'ping' | 'end'"
    )
    data: str = Field(default="", description="Base64 载荷（frame/audio/ocr 时有效）")
    metadata: dict = Field(default_factory=dict, description="附加元数据")


class DialogueMessageOut(BaseModel):
    """服务端 → 客户端 WebSocket 消息。"""
    type: str  # "vision_description" | "transcript" | "agent_answer" | "proactive_suggestion" | "ocr_text" | "scene_event" | "error" | "pong" | "cost_summary"
    content: str
    timestamp_ms: int
    metadata: dict = Field(default_factory=dict)
