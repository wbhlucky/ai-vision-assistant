"""Unit tests for services/cost_controller.py — pure data logic, zero deps."""

from __future__ import annotations

import time

import pytest

from backend.services.cost_controller import ApiCallBudget, FrameRateGate, ResolutionController


class TestFrameRateGate:
    """帧率门控单元测试。"""

    def test_initial_should_capture_true(self):
        """首次调用 should_capture 应返回 True。"""
        gate = FrameRateGate(interval_ms=500)
        assert gate.should_capture() is True

    def test_immediate_second_call_blocked(self):
        """间隔不足时应返回 False。"""
        gate = FrameRateGate(interval_ms=500)
        gate.mark_captured()
        assert gate.should_capture() is False

    def test_after_interval_should_capture_true(self):
        """等待足够间隔后应返回 True。"""
        gate = FrameRateGate(interval_ms=100)
        gate.mark_captured()
        time.sleep(0.15)
        assert gate.should_capture() is True

    def test_seconds_since_last(self):
        """seconds_since_last 返回正数。"""
        gate = FrameRateGate()
        gate.mark_captured()
        time.sleep(0.05)
        assert gate.seconds_since_last > 0


class TestApiCallBudget:
    """API 调用预算单元测试。"""

    def test_initial_budget_available(self):
        """初始预算应可用。"""
        budget = ApiCallBudget(max_vision_calls=10, max_stt_calls=20)
        assert budget.allow_vision() is True
        assert budget.allow_stt() is True

    def test_vision_quota_consumed(self):
        """allow_vision 消耗配额。"""
        budget = ApiCallBudget(max_vision_calls=2, max_stt_calls=10)
        assert budget.allow_vision() is True
        assert budget.allow_vision() is True
        assert budget.vision_calls == 2
        assert budget.allow_vision() is False  # 已耗尽

    def test_stt_quota_consumed(self):
        """allow_stt 消耗配额。"""
        budget = ApiCallBudget(max_vision_calls=10, max_stt_calls=2)
        assert budget.allow_stt() is True
        assert budget.allow_stt() is True
        assert budget.allow_stt() is False

    def test_quota_independent(self):
        """视觉和语音配额互不影响。"""
        budget = ApiCallBudget(max_vision_calls=0, max_stt_calls=5)
        assert budget.allow_vision() is False
        assert budget.allow_stt() is True  # 语音仍可用

    def test_summary(self):
        """summary 返回正确的字典结构。"""
        budget = ApiCallBudget(max_vision_calls=5, max_stt_calls=10)
        budget.allow_vision()
        budget.allow_vision()
        summary = budget.summary()
        assert summary["vision_calls"] == 2
        assert summary["vision_exhausted"] is False
        assert summary["max_vision"] == 5
        assert "stt_calls" in summary


class TestResolutionController:
    """自适应分辨率控制器单元测试。"""

    def test_default_high_resolution(self):
        """预算使用率低时返回默认分辨率。"""
        controller = ResolutionController()
        budget = ApiCallBudget(max_vision_calls=100)
        budget.allow_vision()  # 仅 1/100 = 1%
        w, h = controller.get_dimensions(budget)
        from backend.core.settings import settings
        assert w == settings.vision_max_width
        assert h == settings.vision_max_height

    def test_medium_usage(self):
        """使用率 >50% 降到 480×360。"""
        controller = ResolutionController()
        budget = ApiCallBudget(max_vision_calls=10)
        for _ in range(6):  # 60%
            budget.allow_vision()
        w, h = controller.get_dimensions(budget)
        assert (w, h) == (480, 360)

    def test_high_usage(self):
        """使用率 >80% 降到 320×240。"""
        controller = ResolutionController()
        budget = ApiCallBudget(max_vision_calls=10)
        for _ in range(9):  # 90%
            budget.allow_vision()
        w, h = controller.get_dimensions(budget)
        assert (w, h) == (320, 240)
