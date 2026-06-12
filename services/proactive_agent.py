from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable

from backend.services.scene_memory import SceneMemory


# ── 触发器规则 ──────────────────────────────────────────────

@dataclass
class TriggerRule:
    """一条主动观察规则。"""

    name: str
    description: str  # 评委文档用
    condition: Callable[[str, list[str], SceneMemory], bool]
    suggestion_template: str


def _rule_person_appeared(desc: str, objects: list[str], memory: SceneMemory) -> bool:
    """检测到有人出现（之前帧无人物，当前帧有人物）。"""
    person_keywords = ["人", "人物", "人脸", "有人", "一个人", "女士", "先生", "男生", "女生"]
    has_person = any(kw in desc for kw in person_keywords)

    # 检查上一帧是否没有人
    if has_person and len(memory.snapshots) >= 2:
        prev_desc = memory.snapshots[-2].description
        prev_has_person = any(kw in prev_desc for kw in person_keywords)
        return not prev_has_person  # 新出现的人
    return False


def _rule_book_held_up(desc: str, objects: list[str], memory: SceneMemory) -> bool:
    """检测到用户举起一本书（含书名关键词）。"""
    book_keywords = ["书", "书本", "书籍", "封面", "教材", "手册", "论文"]
    action_keywords = ["举起", "拿着", "手持", "翻开", "拿起"]
    return (
        any(kw in desc for kw in book_keywords)
        and any(kw in desc for kw in action_keywords)
    )


def _rule_screen_error(desc: str, objects: list[str], memory: SceneMemory) -> bool:
    """检测到屏幕/显示器上出现错误信息。"""
    screen_keywords = ["屏幕", "显示器", "电脑", "笔记本", "代码", "终端"]
    error_keywords = ["错误", "报错", "error", "异常", "红字", "警告", "warning",
                      "exception", "traceback", "失败"]
    return (
        any(kw in desc for kw in screen_keywords)
        and any(kw in desc for kw in error_keywords)
    )


def _rule_text_visible(desc: str, objects: list[str], memory: SceneMemory) -> bool:
    """检测到画面中有大量文字（文档、白板、PPT）。"""
    text_keywords = ["文字", "文本", "白板", "黑板", "PPT", "幻灯片", "文档",
                     "纸上", "笔记", "便签", "标签", "路牌", "菜单"]
    return any(kw in desc for kw in text_keywords)


def _rule_scene_changed_significantly(desc: str, objects: list[str], memory: SceneMemory) -> bool:
    """场景发生显著变化（≥2 个物体增加或移除）。"""
    if len(memory.snapshots) < 2:
        return False
    diff = memory.compare_with_previous()
    return (len(diff["added"]) + len(diff["removed"])) >= 2


# ── 触发器注册表 ────────────────────────────────────────────

TRIGGERS: list[TriggerRule] = [
    TriggerRule(
        name="person_appeared",
        description="画面中出现新人物",
        condition=_rule_person_appeared,
        suggestion_template="我看到有人进入了画面，需要我帮你识别或打招呼吗？",
    ),
    TriggerRule(
        name="book_held_up",
        description="用户举起书本/文档",
        condition=_rule_book_held_up,
        suggestion_template="我看到你拿起了一本书，需要我帮你提取书名、总结目录或搜索相关资料吗？",
    ),
    TriggerRule(
        name="screen_error",
        description="屏幕显示错误信息",
        condition=_rule_screen_error,
        suggestion_template="我注意到你的屏幕上似乎出现了错误信息，需要我帮你分析原因或搜索解决方案吗？",
    ),
    TriggerRule(
        name="text_visible",
        description="画面中存在可读文字",
        condition=_rule_text_visible,
        suggestion_template="画面中有文字内容，需要我帮你识别（OCR）或翻译吗？",
    ),
    TriggerRule(
        name="scene_changed_significantly",
        description="场景发生显著变化",
        condition=_rule_scene_changed_significantly,
        suggestion_template="我注意到场景发生了一些变化，需要我帮你回顾刚才发生了什么吗？",
    ),
]

# 每条规则对应的 settings 开关
_TRIGGER_ENABLE_MAP: dict[str, str] = {
    "person_appeared": "proactive_person_appeared",
    "book_held_up": "proactive_book_held_up",
    "screen_error": "proactive_screen_error",
    "text_visible": "proactive_text_visible",
    "scene_changed_significantly": "proactive_scene_changed",
}


def _get_enabled_triggers() -> list[TriggerRule]:
    """根据 settings 返回当前启用的触发器列表。"""
    from backend.core.settings import settings

    enabled: list[TriggerRule] = []
    for rule in TRIGGERS:
        attr = _TRIGGER_ENABLE_MAP.get(rule.name)
        if attr and not getattr(settings, attr, True):
            continue  # 被配置禁用
        enabled.append(rule)
    return enabled


# ── 主动观察引擎 ────────────────────────────────────────────

@dataclass
class ProactiveAgent:
    """主动观察引擎：分析每一帧画面，在满足触发条件时主动生成建议。

    这不是"用户问→AI 答"的被动模式，而是"AI 发现→主动提示"的 Agent 模式。
    """

    _cooldown: dict[str, float] = field(default_factory=dict)

    @property
    def _cooldown_sec(self) -> float:
        from backend.core.settings import settings
        return settings.proactive_cooldown_sec

    def observe(self, description: str, objects: list[str], memory: SceneMemory) -> str | None:
        """分析当前帧，若满足某触发条件且不在冷却期，返回主动建议；否则返回 None。

        Args:
            description: 当前帧的视觉描述文本。
            objects: 从描述中解析的物体列表。
            memory: 场景记忆实例。

        Returns:
            一条中文建议字符串，或 None（无需主动发言）。
        """
        import time

        enabled = _get_enabled_triggers()
        for rule in enabled:
            # 冷却检查
            last_fired = self._cooldown.get(rule.name, 0)
            if time.monotonic() - last_fired < self._cooldown_sec:
                continue

            try:
                if rule.condition(description, objects, memory):
                    self._cooldown[rule.name] = time.monotonic()
                    return rule.suggestion_template
            except Exception:
                continue

        return None

    def reset_cooldown(self, rule_name: str | None = None) -> None:
        """重置冷却（用于用户已响应某条建议后避免重复）。"""
        if rule_name:
            self._cooldown.pop(rule_name, None)
        else:
            self._cooldown.clear()


# ── 事件描述生成 ────────────────────────────────────────────

def generate_event_description(memory: SceneMemory) -> str | None:
    """根据场景记忆生成人类可读的事件摘要。

    用于回答"刚刚发生了什么"这类回溯性问题。
    """
    events = memory.get_recent_events(5)
    if not events:
        return None
    return "最近 5 条场景事件:\n" + "\n".join(f"  • {e}" for e in events)
