"""Unit tests for services/multimodal_memory.py — 跨模态统一记忆引擎."""

from __future__ import annotations

import pytest

from backend.services.multimodal_memory import (
    MemoryEntry,
    MultimodalMemory,
    _parse_modality_preference,
    _parse_time_window,
    _tokenize_query,
)


class TestHelpers:
    def test_tokenize_query_simple(self):
        tokens = _tokenize_query("水杯")
        assert len(tokens) >= 1
        assert "水杯" in tokens

    def test_tokenize_query_longer(self):
        tokens = _tokenize_query("半小时前桌子上有什么")
        assert len(tokens) > 0

    def test_parse_time_window_minutes(self):
        assert _parse_time_window("5分钟前") == 300
        assert _parse_time_window("30分前") == 1800

    def test_parse_time_window_half_hour(self):
        assert _parse_time_window("半小时前") == 1800

    def test_parse_time_window_just_now(self):
        assert _parse_time_window("刚才") == 120
        assert _parse_time_window("刚刚") == 60

    def test_parse_time_window_none(self):
        assert _parse_time_window("今天天气怎么样") is None

    def test_parse_modality_vision(self):
        assert _parse_modality_preference("我看到了什么") == ["vision"]
        assert "vision" in (_parse_modality_preference("桌上有东西") or [])

    def test_parse_modality_ocr(self):
        assert _parse_modality_preference("纸上写了什么") == ["ocr"]

    def test_parse_modality_speech(self):
        assert _parse_modality_preference("我刚才说了什么") == ["speech"]

    def test_parse_modality_mixed(self):
        result = _parse_modality_preference("我看到白板上写了什么文字")
        assert result is not None
        assert "vision" in result
        assert "ocr" in result


class TestMultimodalMemory:
    def test_record_vision(self):
        mem = MultimodalMemory()
        entry = mem.record_vision("桌上有水杯", ["水杯"])
        assert entry.modality == "vision"
        assert "水杯" in entry.content
        assert len(mem.entries) == 1

    def test_record_ocr(self):
        mem = MultimodalMemory()
        entry = mem.record_ocr("Hello World")
        assert entry.modality == "ocr"
        assert len(mem.entries) == 1

    def test_record_speech(self):
        mem = MultimodalMemory()
        entry = mem.record_speech("这是什么")
        assert entry.modality == "speech"
        assert len(mem.entries) == 1

    def test_search_by_keyword(self):
        mem = MultimodalMemory()
        mem.record_vision("桌上有水杯", ["水杯"])
        mem.record_ocr("笔记内容")
        mem.record_speech("这是什么")
        results = mem.search("水杯")
        assert len(results) >= 1
        assert any("水杯" in r.content for r in results)

    def test_search_filter_modality(self):
        mem = MultimodalMemory()
        mem.record_vision("视觉", ["物体"])
        mem.record_ocr("OCR 文字")
        results = mem.search("视觉", modalities=["vision"])
        assert len(results) >= 1
        assert all(r.modality == "vision" for r in results)

    def test_search_time_window(self):
        mem = MultimodalMemory()
        mem.record_vision("水杯在桌子上", ["水杯"])
        # 足够大的时间窗口应能找到
        results = mem.search("水杯", time_window_sec=9999)
        assert len(results) >= 1

    def test_search_time_window_zero_includes_now(self):
        mem = MultimodalMemory()
        mem.record_vision("水杯在桌子上", ["水杯"])
        # time_window_sec=0: 排除 now - timestamp > 0，刚添加的条目 timestamp ≈ now
        # 所以仍在窗口内（差值 ≈ 0，不大于 0）
        results = mem.search("水杯", time_window_sec=0)
        assert len(results) >= 1  # 刚添加的条目未超时

    def test_context_window(self):
        mem = MultimodalMemory()
        mem.record_vision("帧1", ["a"])
        mem.record_ocr("文字1")
        mem.record_speech("话1")
        ctx = mem.get_context_window(minutes=1)
        assert len(ctx) == 3

    def test_query_answer_with_results(self):
        mem = MultimodalMemory()
        mem.record_vision("桌上有水杯、笔记本和手机", ["水杯", "笔记本", "手机"])
        answer = mem.query_answer("刚才桌上有什么")
        assert len(answer) > 0
        assert "水杯" in answer or "笔记本" in answer or "手机" in answer

    def test_query_answer_no_results(self):
        mem = MultimodalMemory()
        answer = mem.query_answer("有什么")
        assert "没有找到" in answer

    def test_summary(self):
        mem = MultimodalMemory()
        mem.record_vision("a", ["x"])
        mem.record_ocr("b")
        s = mem.summary()
        assert s["total_entries"] == 2
        assert "vision" in s["modality_counts"]
        assert "ocr" in s["modality_counts"]

    def test_memory_entry_to_dict(self):
        e = MemoryEntry(0, "12:00", "vision", "test", ["a"], {})
        d = e.to_dict()
        assert d["modality"] == "vision"
        assert d["time"] == "12:00"

    def test_max_entries_cap(self):
        mem = MultimodalMemory()
        mem._max_entries = 5
        for i in range(10):
            mem.record_vision(f"帧{i}", [f"obj{i}"])
        assert len(mem.entries) <= 5
