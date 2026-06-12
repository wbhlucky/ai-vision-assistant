"""预测性主动提醒引擎 — 从规则触发升级为智能预测。

基于多模态记忆中的行为模式，预测用户下一步可能需要的帮助。
这是从"被动提示"到"智能预测"的升级，体现 AI 产品思维。
"""

from __future__ import annotations

import time
from collections import Counter
from dataclasses import dataclass, field

from backend.services.multimodal_memory import MultimodalMemory


# ── 行为追踪 ──────────────────────────────────────────────

@dataclass
class BehaviorTracker:
    """追踪用户行为模式，用于预测下一步动作。"""

    # 物体出现频次
    _object_frequency: Counter = field(default_factory=Counter)
    # 最近的动作序列（取最近 10 个有意义的记忆事件）
    _recent_actions: list[str] = field(default_factory=list)
    _max_actions: int = 10

    def feed(self, modality: str, content: str, objects: list[str]) -> None:
        """喂入一条记忆事件。"""
        for obj in objects:
            self._object_frequency[obj] += 1
        # 记录有意义的动作
        if len(content) > 5:
            self._recent_actions.append(f"[{modality}] {content[:80]}")
            if len(self._recent_actions) > self._max_actions:
                self._recent_actions = self._recent_actions[-self._max_actions:]

    def frequent_objects(self, min_count: int = 3) -> list[tuple[str, int]]:
        """返回出现次数 ≥ min_count 的物体。"""
        return [(obj, count) for obj, count in self._object_frequency.most_common(20)
                if count >= min_count]

    def recent_pattern(self) -> str | None:
        """检测最近动作中的模式。"""
        if len(self._recent_actions) < 2:
            return None
        return self._recent_actions[-1]


# ── 预测规则 ──────────────────────────────────────────────

@dataclass
class PredictionRule:
    """一条预测规则。"""
    name: str
    description: str
    # 检测函数：根据记忆和追踪器判断是否应触发
    detect: callable  # (MultimodalMemory, BehaviorTracker) -> bool
    suggestion: str


def _predict_note_taking(memory: MultimodalMemory, tracker: BehaviorTracker) -> bool:
    """多模态模式：视觉看到书+OCR 提取到文字 → 预测做笔记需求。"""
    recent = memory.get_context_window(minutes=2)
    has_vision_book = any(
        e.modality == "vision" and any(w in e.content for w in ["书", "书本", "教材", "笔记"])
        for e in recent
    )
    has_ocr_text = any(e.modality == "ocr" and len(e.content) > 20 for e in recent)
    return has_vision_book and has_ocr_text


def _predict_water_reminder(memory: MultimodalMemory, tracker: BehaviorTracker) -> bool:
    """重复模式：水杯频繁出现 → 提醒记录饮水。"""
    count = tracker._object_frequency.get("水杯", 0)
    return count >= 5


def _predict_code_help(memory: MultimodalMemory, tracker: BehaviorTracker) -> bool:
    """错误模式：视觉检测到屏幕错误+语音提到代码 → 预测需要调试帮助。"""
    recent = memory.get_context_window(minutes=3)
    has_error = any(
        "错误" in e.content or "报错" in e.content or "error" in e.content.lower()
        for e in recent if e.modality == "vision"
    )
    has_speech_code = any(
        any(w in e.content for w in ["代码", "程序", "bug", "debug", "调试", "报错"])
        for e in recent if e.modality == "speech"
    )
    return has_error and has_speech_code


def _predict_document_summary(memory: MultimodalMemory, tracker: BehaviorTracker) -> bool:
    """文档模式：OCR 多次提取长文本 → 预测需要总结。"""
    recent = memory.get_context_window(minutes=5)
    ocr_entries = [e for e in recent if e.modality == "ocr" and len(e.content) > 50]
    return len(ocr_entries) >= 2


PREDICTION_RULES: list[PredictionRule] = [
    PredictionRule(
        name="note_taking",
        description="检测到书本 + OCR 文字 → 预测做笔记",
        detect=_predict_note_taking,
        suggestion="你正在阅读并提取文字，需要我帮你整理笔记或生成摘要吗？📝",
    ),
    PredictionRule(
        name="water_reminder",
        description="水杯频繁出现 → 提醒记录饮水",
        detect=_predict_water_reminder,
        suggestion="我注意到你经常在画面中出现水杯，需要我帮你记录每日饮水量吗？💧",
    ),
    PredictionRule(
        name="code_help",
        description="屏幕错误 + 语音提到代码 → 预测调试需求",
        detect=_predict_code_help,
        suggestion="检测到你遇到了代码错误，需要我搜索 StackOverflow 或分析错误原因吗？🔧",
    ),
    PredictionRule(
        name="document_summary",
        description="OCR 多次提取长文本 → 预测需要总结",
        detect=_predict_document_summary,
        suggestion="你已经扫描了多段文字，需要我帮你生成一份综合摘要吗？📋",
    ),
]


# ── 预测引擎 ──────────────────────────────────────────────

@dataclass
class PredictiveAgent:
    """预测性主动 Agent：基于行为模式预测用户下一步可能需要的帮助。

    与 ProactiveAgent 的区别：
    - ProactiveAgent: 基于当前帧的即时触发（"看到书 → 问要不要总结"）
    - PredictiveAgent: 基于历史行为的模式预测（"水杯出现了 5 次 → 问要不要记录饮水"）
    """

    tracker: BehaviorTracker = field(default_factory=BehaviorTracker)
    _cooldown: dict[str, float] = field(default_factory=dict)
    _cooldown_sec: float = 60.0  # 预测类建议冷却更久，避免过于侵略

    def feed_and_predict(self, modality: str, content: str, objects: list[str],
                         memory: MultimodalMemory) -> str | None:
        """喂入一条新事件，若满足预测条件则返回建议。

        Args:
            modality: "vision" | "ocr" | "speech"
            content: 原始内容
            objects: 关联物体
            memory: 多模态记忆实例

        Returns:
            预测建议字符串，或 None。
        """
        self.tracker.feed(modality, content, objects)

        for rule in PREDICTION_RULES:
            last_fired = self._cooldown.get(rule.name, 0)
            if time.monotonic() - last_fired < self._cooldown_sec:
                continue
            try:
                if rule.detect(memory, self.tracker):
                    self._cooldown[rule.name] = time.monotonic()
                    return rule.suggestion
            except Exception:
                continue

        return None

    def reset(self) -> None:
        """重置追踪状态。"""
        self.tracker = BehaviorTracker()
        self._cooldown.clear()
