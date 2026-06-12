from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import cached_property


def _env(name: str, default: str = "") -> str:
    """Read an environment variable, returning *default* when unset or empty."""
    v = os.getenv(name)
    return v if v else default


@dataclass(frozen=True)
class Settings:
    """Lightweight configuration — all values sourced from environment variables."""

    # ── security ──────────────────────────────────────────────
    api_key: str = field(default_factory=lambda: _env("API_KEY"))

    # ── server ────────────────────────────────────────────────
    host: str = field(default_factory=lambda: _env("FASTAPI_HOST", "0.0.0.0"))
    port: int = field(default_factory=lambda: int(_env("FASTAPI_PORT", "8000")))
    log_level: str = field(default_factory=lambda: _env("LOG_LEVEL", "INFO"))

    # ── runtime mode ──────────────────────────────────────────
    # "learn" = stability first (default)  |  "prod" = quality first
    run_mode: str = field(default_factory=lambda: _env("RUN_MODE", "learn").lower())

    # ── external calls ────────────────────────────────────────
    external_request_timeout_sec: float = field(
        default_factory=lambda: float(_env("EXTERNAL_REQUEST_TIMEOUT_SEC", "8"))
    )

    # ── startup ───────────────────────────────────────────────
    startup_self_check: bool = field(
        default_factory=lambda: _env("STARTUP_SELF_CHECK", "1") == "1"
    )

    # ── MCP bridge ────────────────────────────────────────────
    mcp_mode: str = field(default_factory=lambda: _env("MCP_MODE", "local").lower())
    mcp_server_base_url: str = field(
        default_factory=lambda: _env("MCP_SERVER_BASE_URL", "http://127.0.0.1:8001")
    )

    # ── multimodal / vision dialogue ──────────────────────────
    dashscope_api_key: str = field(
        default_factory=lambda: _env("DASHSCOPE_API_KEY")
    )
    vision_provider: str = field(
        default_factory=lambda: _env("VISION_PROVIDER", "deepseek").lower()
    )  # "dashscope" | "deepseek" | "openai"
    deepseek_base_url: str = field(
        default_factory=lambda: _env("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    )
    vision_model: str = field(default_factory=lambda: _env("VISION_MODEL", "deepseek-chat"))
    speech_model: str = field(default_factory=lambda: _env("SPEECH_MODEL", "paraformer-v2"))
    vision_frame_interval_ms: int = field(
        default_factory=lambda: int(_env("VISION_FRAME_INTERVAL_MS", "2000"))
    )
    vision_max_width: int = field(
        default_factory=lambda: int(_env("VISION_MAX_WIDTH", "640"))
    )
    vision_max_height: int = field(
        default_factory=lambda: int(_env("VISION_MAX_HEIGHT", "480"))
    )
    vision_jpeg_quality: int = field(
        default_factory=lambda: int(_env("VISION_JPEG_QUALITY", "60"))
    )
    vad_enabled: bool = field(
        default_factory=lambda: _env("VAD_ENABLED", "1") == "1"
    )
    vad_energy_threshold: int = field(
        default_factory=lambda: int(_env("VAD_ENERGY_THRESHOLD", "300"))
    )
    max_session_duration_sec: int = field(
        default_factory=lambda: int(_env("MAX_SESSION_DURATION_SEC", "300"))
    )

    # ── proactive agent triggers ─────────────────────────────
    proactive_cooldown_sec: float = field(
        default_factory=lambda: float(_env("PROACTIVE_COOLDOWN_SEC", "15"))
    )
    proactive_person_appeared: bool = field(
        default_factory=lambda: _env("PROACTIVE_PERSON_APPEARED", "1") == "1"
    )
    proactive_book_held_up: bool = field(
        default_factory=lambda: _env("PROACTIVE_BOOK_HELD_UP", "1") == "1"
    )
    proactive_screen_error: bool = field(
        default_factory=lambda: _env("PROACTIVE_SCREEN_ERROR", "1") == "1"
    )
    proactive_text_visible: bool = field(
        default_factory=lambda: _env("PROACTIVE_TEXT_VISIBLE", "1") == "1"
    )
    proactive_scene_changed: bool = field(
        default_factory=lambda: _env("PROACTIVE_SCENE_CHANGED", "1") == "1"
    )
    dashboard_report_interval_sec: float = field(
        default_factory=lambda: float(_env("DASHBOARD_REPORT_INTERVAL_SEC", "10"))
    )


settings = Settings()
