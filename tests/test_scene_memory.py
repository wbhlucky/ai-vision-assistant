"""Unit tests for services/scene_memory.py —视觉记忆引擎单元测试。"""

from __future__ import annotations

import pytest

from backend.services.scene_memory import SceneMemory, SceneSnapshot, _parse_objects_from_text


class TestParseObjects:
    """物体名称提取测试。"""

    def test_empty_string(self):
        assert _parse_objects_from_text("") == []

    def test_quantifier_pattern(self):
        """量词+名词模式：一个水杯、一台笔记本"""
        desc = "画面中有一个水杯，一台笔记本电脑，还有一部手机。"
        objects = _parse_objects_from_text(desc)
        assert "水杯" in objects or "笔记本电脑" in objects or "手机" in objects

    def test_has_pattern(self):
        """有+名词模式：看到、放着"""
        desc = "桌上放着一本书和一把钥匙，看到一台显示器。"
        objects = _parse_objects_from_text(desc)
        # 应该至少提取到部分物体
        assert len(objects) > 0
        # 验证不包含纯量词
        assert "一个" not in objects
        assert "一台" not in objects

    def test_max_cap(self):
        """超过 15 个物体时截断。"""
        desc = "、".join(f"一个物体{i}" for i in range(30))
        objects = _parse_objects_from_text(desc)
        assert len(objects) <= 15

    def test_no_false_positives_on_punctuation(self):
        """标点符号不污染结果。"""
        desc = "，。；！？、"
        objects = _parse_objects_from_text(desc)
        assert all("," not in obj and "。" not in obj for obj in objects)


class TestSceneMemory:
    """场景记忆引擎测试。"""

    def test_initial_empty(self):
        mem = SceneMemory()
        assert len(mem.snapshots) == 0
        assert len(mem.event_log) == 0

    def test_first_snapshot_generates_event(self):
        """添加第一帧应生成初始场景事件。"""
        mem = SceneMemory()
        mem.add_snapshot("桌上有水杯和笔记本", ["水杯", "笔记本"])
        assert len(mem.snapshots) == 1
        assert len(mem.event_log) == 1
        assert "初始场景" in mem.event_log[0]

    def test_diff_detects_added_and_removed(self):
        """第二帧后应检测物体新增/移除。"""
        mem = SceneMemory()
        mem.add_snapshot("桌上有水杯和笔记本", ["水杯", "笔记本"])
        mem.add_snapshot("桌上有笔记本和手机", ["笔记本", "手机"])

        diff = mem.compare_with_previous()
        assert "水杯" in diff["removed"]
        assert "手机" in diff["added"]
        assert "笔记本" in diff["unchanged"]

    def test_no_change(self):
        """相同物体列表不应产生事件。"""
        mem = SceneMemory()
        mem.add_snapshot("桌上有水杯", ["水杯"])
        event_count_before = len(mem.event_log)
        mem.add_snapshot("桌上仍有水杯", ["水杯"])
        # 物体未变，事件数不变（只有初次"初始场景"事件）
        assert len(mem.event_log) == event_count_before

    def test_query_change(self):
        """变化类问题应返回 diff 信息。"""
        mem = SceneMemory()
        mem.add_snapshot("有水杯", ["水杯"])
        mem.add_snapshot("有手机", ["手机"])
        answer = mem.query("有什么变化？")
        assert "水杯" in answer or "手机" in answer

    def test_query_history(self):
        """历史类问题应返回之前帧描述。"""
        mem = SceneMemory()
        mem.add_snapshot("第一帧：水杯", ["水杯"])
        mem.add_snapshot("第二帧：手机", ["手机"])
        answer = mem.query("刚才有什么？")
        # 应包含倒数第二帧的信息
        assert len(answer) > 10

    def test_snapshot_cap(self):
        """快照数不超过上限。"""
        mem = SceneMemory()
        mem._max_snapshots = 5
        for i in range(10):
            mem.add_snapshot(f"帧{i}", [f"物体{i}"])
        assert len(mem.snapshots) <= 5

    def test_event_cap(self):
        """事件数不超过上限。"""
        mem = SceneMemory()
        mem._max_events = 5
        mem._max_snapshots = 100
        for i in range(20):
            mem._add_event(f"事件{i}")
        assert len(mem.event_log) <= 5

    def test_build_memory_context(self):
        """build_memory_context 返回非空字符串。"""
        mem = SceneMemory()
        mem.add_snapshot("桌上有水杯", ["水杯"])
        ctx = mem.build_memory_context()
        assert len(ctx) > 0
        assert "初始" in ctx or "水杯" in ctx

    def test_to_dict(self):
        """SceneSnapshot.to_dict 返回正确结构。"""
        snap = SceneSnapshot(timestamp=0, time_str="12:00:00",
                             description="test", objects=["a", "b"])
        d = snap.to_dict()
        assert d["time"] == "12:00:00"
        assert d["objects"] == ["a", "b"]
