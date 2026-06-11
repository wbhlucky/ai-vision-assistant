from __future__ import annotations

import time
from dataclasses import dataclass, field

from backend.core.settings import settings


# ── 帧率门控 ────────────────────────────────────────────────

@dataclass
class FrameRateGate:
    """按最小间隔控制帧捕获频率，避免无效的云端调用。

    用法:
        gate = FrameRateGate()
        if gate.should_capture():
            process_frame(...)
            gate.mark_captured()
    """

    interval_ms: int = field(default_factory=lambda: settings.vision_frame_interval_ms)
    _last_capture: float = 0.0

    def should_capture(self) -> bool:
        """距离上次捕获是否已达最小间隔。"""
        now = time.monotonic()
        return (now - self._last_capture) * 1000 >= self.interval_ms

    def mark_captured(self) -> None:
        """标记一次捕获时间。"""
        self._last_capture = time.monotonic()

    @property
    def seconds_since_last(self) -> float:
        """距离上次捕获过去的秒数。"""
        return time.monotonic() - self._last_capture


# ── API 调用预算 ────────────────────────────────────────────

@dataclass
class ApiCallBudget:
    """限制单次会话的云端 API 调用次数，控制运营成本。

    默认配额：50 次视觉调用 + 100 次语音调用。
    配额耗尽后 allow_vision / allow_stt 将返回 False。
    """

    max_vision_calls: int = 50
    max_stt_calls: int = 100
    vision_calls: int = 0
    stt_calls: int = 0

    def allow_vision(self) -> bool:
        """返回 True 表示还有视觉调用配额（同时消耗一次）。"""
        if self.vision_calls >= self.max_vision_calls:
            return False
        self.vision_calls += 1
        return True

    def allow_stt(self) -> bool:
        """返回 True 表示还有语音转录配额（同时消耗一次）。"""
        if self.stt_calls >= self.max_stt_calls:
            return False
        self.stt_calls += 1
        return True

    def summary(self) -> dict:
        """返回当前预算使用概览。"""
        return {
            "vision_calls": self.vision_calls,
            "stt_calls": self.stt_calls,
            "max_vision": self.max_vision_calls,
            "max_stt": self.max_stt_calls,
            "vision_exhausted": self.vision_calls >= self.max_vision_calls,
            "stt_exhausted": self.stt_calls >= self.max_stt_calls,
        }


# ── 自适应分辨率 ────────────────────────────────────────────

@dataclass
class ResolutionController:
    """根据预算消耗比例动态下调图像分辨率，成本压力越大分辨率越低。

    阶梯：
      - 使用率 > 80%：320×240
      - 使用率 > 50%：480×360
      - 其他：使用 settings 中的配置值
    """

    def get_dimensions(self, budget: ApiCallBudget) -> tuple[int, int]:
        """返回当前应使用的 (width, height)。"""
        usage_ratio = budget.vision_calls / max(budget.max_vision_calls, 1)
        if usage_ratio > 0.8:
            return (320, 240)
        elif usage_ratio > 0.5:
            return (480, 360)
        return (settings.vision_max_width, settings.vision_max_height)
