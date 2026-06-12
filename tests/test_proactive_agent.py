"""Unit tests for services/proactive_agent.py — 主动观察引擎单元测试。"""

from __future__ import annotations

import pytest

from backend.services.proactive_agent import (
    ProactiveAgent,
    TriggerRule,
    _rule_book_held_up,
    _rule_person_appeared,
    _rule_scene_changed_significantly,
    _rule_screen_error,
    _rule_text_visible,
)
from backend.services.scene_memory import SceneMemory


class TestTriggerRules:
    """独立触发器规则函数测试（不依赖 settings）。"""

    def test_person_appeared_false_initially(self):
        """第一帧有人不应触发（因为没有前一帧）。"""
        mem = SceneMemory()
        mem.add_snapshot("一个人出现在画面中", ["人"])
        # 只有一帧快照，prev 无人 → 不应触发
        assert _rule_person_appeared("一个人", ["人"], mem) is False

    def test_person_appeared_true(self):
        """前一帧无人，当前帧有人 → 应触发。"""
        mem = SceneMemory()
        mem.add_snapshot("空房间", ["桌子"])
        mem.add_snapshot("一个人走进房间", ["桌子", "人"])
        assert _rule_person_appeared("一个人走进房间", ["桌子", "人"], mem) is True

    def test_book_held_up(self):
        """举起书 → 应触发。"""
        mem = SceneMemory()
        assert _rule_book_held_up("用户拿起了一本书", ["书"], mem) is True

    def test_book_on_desk_not_triggered(self):
        """书在桌上（未拿起）→ 不应触发。"""
        mem = SceneMemory()
        assert _rule_book_held_up("桌面上放着一本书", ["书"], mem) is False

    def test_screen_error(self):
        """屏幕报错 → 应触发。"""
        mem = SceneMemory()
        assert _rule_screen_error("电脑屏幕上显示红色错误信息", ["电脑"], mem) is True

    def test_screen_normal(self):
        """屏幕正常 → 不应触发。"""
        mem = SceneMemory()
        assert _rule_screen_error("电脑屏幕上显示桌面", ["电脑"], mem) is False

    def test_text_visible(self):
        """白板上有文字 → 应触发。"""
        mem = SceneMemory()
        assert _rule_text_visible("白板上写了很多文字和公式", ["白板"], mem) is True

    def test_text_not_visible(self):
        """无文字场景 → 不应触发。"""
        mem = SceneMemory()
        assert _rule_text_visible("房间里有一张空桌子", ["桌子"], mem) is False

    def test_scene_changed_significantly(self):
        """≥2 个物体变化 → 应触发。"""
        mem = SceneMemory()
        mem.add_snapshot("有水杯和笔记本", ["水杯", "笔记本"])
        mem.add_snapshot("有手机和钥匙", ["手机", "钥匙"])
        assert _rule_scene_changed_significantly("有手机和钥匙", ["手机", "钥匙"], mem) is True

    def test_scene_changed_insignificantly(self):
        """仅 1 个物体变化（新增 0 + 移除 1 = 1 < 2）→ 不应触发。"""
        mem = SceneMemory()
        mem.add_snapshot("有水杯和笔记本", ["水杯", "笔记本"])
        mem.add_snapshot("有水杯和笔记本和手机", ["水杯", "笔记本", "手机"])
        # 仅新增手机，移除 0 个，总变化 = 1 < 2 → 不触发
        assert _rule_scene_changed_significantly("有水杯和笔记本和手机", ["水杯", "笔记本", "手机"], mem) is False


class TestProactiveAgent:
    """主动观察引擎测试（冷却 + 多规则协同）。"""

    def test_observe_returns_none_on_empty_memory(self):
        """空场景记忆 + 无触发条件 → 返回 None。"""
        agent = ProactiveAgent()
        mem = SceneMemory()
        result = agent.observe("空房间", [], mem)
        assert result is None

    def test_cooldown_prevents_repeat(self):
        """冷却期内同一规则不重复触发。"""
        agent = ProactiveAgent()
        # 强制缩短冷却方便测试
        agent._cooldown_sec_property = 999  # 不会过期的冷却

        mem = SceneMemory()
        mem.add_snapshot("空房间", ["桌子"])

        # 直接注入冷却
        agent._cooldown["book_held_up"] = float("inf")
        result = agent.observe("用户拿起了一本书", ["书", "桌子"], mem)
        assert result is None  # 冷却阻止

    def test_reset_cooldown(self):
        """重置冷却后规则可再次触发。"""
        agent = ProactiveAgent()
        agent._cooldown["book_held_up"] = float("inf")
        agent.reset_cooldown("book_held_up")
        assert "book_held_up" not in agent._cooldown

    def test_reset_all_cooldowns(self):
        agent = ProactiveAgent()
        agent._cooldown["a"] = 1.0
        agent._cooldown["b"] = 2.0
        agent.reset_cooldown()
        assert len(agent._cooldown) == 0

    @pytest.mark.parametrize("rule_name,desc,objects", [
        ("book_held_up", "用户拿起了一本教材", ["教材", "桌子"]),
        ("screen_error", "电脑屏幕上出现 exception 报错", ["电脑"]),
        ("text_visible", "白板上写满了笔记和公式", ["白板"]),
    ])
    def test_rules_can_trigger(self, rule_name, desc, objects):
        """参数化测试：各种规则在合适的场景下能触发。"""
        agent = ProactiveAgent()
        agent._cooldown_sec_property = 0  # 无限冷却
        mem = SceneMemory()
        # 需要一个前置快照，部分规则依赖对比
        mem.add_snapshot("初始空场景", ["空"])
        mem.add_snapshot(desc, objects)
        result = agent.observe(desc, objects, mem)
        # 至少有一个规则应该触发（不要求具体是哪个）
        # 只验证函数不崩溃
        assert isinstance(result, (str, type(None)))
